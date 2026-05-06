/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <c10/cuda/CUDAStream.h>
namespace cg = cooperative_groups;

static constexpr float OIT_ALPHA_EPS = 1.0f / 255.0f;

__device__ inline void atomicMaxFloatNonnegative(float* address, float val)
{
	atomicMax(reinterpret_cast<unsigned int*>(address), __float_as_uint(val));
}

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}





// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

__device__ inline int effectiveOITRadius(int radius, float threshold);

// Generates one key/value pair for all Gaussian / tile overlaps, keyed only
// by tile.  The Mobile-GS OIT compositor is order independent, so depth bits
// are deliberately left zero and the following radix pass only groups by tile.
__global__ void duplicateWithTileKeys(
	int P,
	const float2* points_xy,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	const float2* precomp_w_thres,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] > 0)
	{
		const float2 wt = precomp_w_thres[idx];
		if (!(wt.x > 0.0f) || !(wt.y < 0.0f)) return;
		int r = radii[idx];
		if (r <= 0) return;

		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;
		getRect(points_xy[idx], r, rect_min, rect_max, grid);

		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				gaussian_keys_unsorted[off] = key << 32;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
	checkFrustum << <(P + 255) / 256, 256, 0, stream >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);


    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 256); 
    obtain(chunk, geom.point_offsets, P, 128);

	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 256);
	return binning;
}






struct ChunkRenderWorkspace {
	float2* precomp_w_thres;

	static ChunkRenderWorkspace fromChunk(char*& chunk, size_t num_points)
	{
		ChunkRenderWorkspace ws;
		CudaRasterizer::obtain(chunk, ws.precomp_w_thres, num_points, 128);
		return ws;
	}
};

static size_t requiredChunkRenderWorkspace(size_t num_points)
{
	char* size = nullptr;
	ChunkRenderWorkspace::fromChunk(size, num_points);
	return ((size_t)size) + 128;
}












// Kernel 0: precompute Gaussian global data.
__global__ void precomputeGaussianData(
    int P,
    const int* radii,
    const float* depths,
    const float* phis,
    const glm::vec3* scales,
    const float4* conic_opacity,
    float2* precomp_w_thres
) {
    int idx = cg::this_grid().thread_rank();
    if (idx >= P || radii[idx] <= 0) return;

    float d_val = fmaxf(depths[idx], 1e-6f);
    glm::vec3 s = scales[idx];
    float max_s = fmaxf(s.x, fmaxf(s.y, s.z));
    float p_val = phis[idx];

	const float exp_arg = fminf(max_s / d_val, 20.0f);
	float weight = __expf(exp_arg) + p_val / (d_val * d_val) + (p_val * p_val);
	weight = fminf(weight, 1e6f);


    float w_o = conic_opacity[idx].w;
    float threshold = 1.0f;
    // A splat can affect the output either through the weighted foreground
    // average (alpha * MobileGS_weight) or through the transmittance term T
    // (alpha).  Use the looser of the two support tests during binning so the
    // render kernel never silently drops high-alpha / low-weight occluders.
    const float support_opacity = w_o * fmaxf(1.0f, weight);
    if (support_opacity > 0.0f) {
        threshold = __logf(fmaxf(OIT_ALPHA_EPS, 1e-7f) / fmaxf(support_opacity, 1e-7f));
    }

    precomp_w_thres[idx] = make_float2(weight, threshold);
}

__device__ inline int effectiveOITRadius(int radius, float threshold)
{
	if (radius <= 0) return 0;
	if (!(threshold < 0.0f)) return 0;
	// preprocessCUDA stores a conservative 3-sigma radius.  The OIT kernel later
	// rejects pixels whose Gaussian power falls below ``threshold``.  Shrink the
	// tile footprint to the same support before binning so sort-free rendering
	// does not spend most of its time looping over splats that will be discarded
	// by the per-pixel support test.  Never expand beyond the original 3-sigma
	// footprint; high-weight splats keep the same conservative support as before.
	float sigma_radius = float(radius) * (1.0f / 3.0f);
	float support_radius = sigma_radius * sqrtf(fmaxf(0.0f, -2.0f * threshold));
	int out = (int)ceilf(support_radius);
	out = out < 1 ? 1 : out;
	return out > radius ? radius : out;
}

__global__ void updateOITTilesTouched(
	int P,
	const float2* points_xy,
	int* radii,
	uint32_t* tiles_touched,
	const float2* precomp_w_thres,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P) return;
	if (radii[idx] <= 0)
	{
		tiles_touched[idx] = 0;
		return;
	}

	const float2 wt = precomp_w_thres[idx];
	if (!(wt.x > 0.0f) || !(wt.y < 0.0f))
	{
		radii[idx] = 0;
		tiles_touched[idx] = 0;
		return;
	}

	int r = effectiveOITRadius(radii[idx], wt.y);
	if (r <= 0)
	{
		radii[idx] = 0;
		tiles_touched[idx] = 0;
		return;
	}
	radii[idx] = r;
	uint2 rect_min, rect_max;
	getRect(points_xy[idx], r, rect_min, rect_max, grid);
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}













__global__ void countOITPointsPerTile(
    int P, int W, int H,
    const float2* means2D, 
    const int* radii, 
    const float4* conic_opacity,
    const float2* precomp_w_thres,
    int* tile_counts
) {
    int idx = cg::this_grid().thread_rank();
    if (idx >= P || radii[idx] <= 0) return;

    if (conic_opacity[idx].w * fmaxf(1.0f, precomp_w_thres[idx].x) < OIT_ALPHA_EPS) return;

	float2 center = means2D[idx];
	const int r = max(1, radii[idx]);
	uint2 tile_grid = { (unsigned int)((W + 15) / 16), (unsigned int)((H + 15) / 16) };

	int x0 = min((int)tile_grid.x, max(0, (int)((center.x - r) / 16)));
	int y0 = min((int)tile_grid.y, max(0, (int)((center.y - r) / 16)));
	int x1 = min((int)tile_grid.x, max(0, (int)((center.x + r + 15) / 16)));
	int y1 = min((int)tile_grid.y, max(0, (int)((center.y + r + 15) / 16)));
	if (x0 >= x1 || y0 >= y1) return;

    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            atomicAdd(&tile_counts[y * tile_grid.x + x], 1);
        }
    }
}

__global__ void fillOITTiles(
    int P, int W, int H,
    const float2* means2D, 
    const int* radii, 
    const float4* conic_opacity, 
    const float2* precomp_w_thres,
    uint32_t* point_list_unsorted, 
    uint2* ranges,
    int max_slots
) {
    int idx = cg::this_grid().thread_rank();
    if (idx >= P || radii[idx] <= 0) return;

    if (conic_opacity[idx].w * fmaxf(1.0f, precomp_w_thres[idx].x) < OIT_ALPHA_EPS) return;

	float2 center = means2D[idx];
	const int r = max(1, radii[idx]);
	uint2 tile_grid = { (unsigned int)((W + 15) / 16), (unsigned int)((H + 15) / 16) };

	int x0 = min((int)tile_grid.x, max(0, (int)((center.x - r) / 16)));
	int y0 = min((int)tile_grid.y, max(0, (int)((center.y - r) / 16)));
	int x1 = min((int)tile_grid.x, max(0, (int)((center.x + r + 15) / 16)));
	int y1 = min((int)tile_grid.y, max(0, (int)((center.y + r + 15) / 16)));
	if (x0 >= x1 || y0 >= y1) return;

    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            int tile_id = y * tile_grid.x + x;
            int slot = atomicAdd(&ranges[tile_id].y, 1);
            if (slot < max_slots) point_list_unsorted[slot] = idx;
        }
    }
}


template <uint32_t CHANNELS>
__global__ void __launch_bounds__(128) scatterOITGaussians(
    int P,
    int W, int H,
    const float2* __restrict__ means2D,
    const int* __restrict__ radii,
    const float* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    const float2* __restrict__ precomp_w_thres,
    float* __restrict__ out_num,
    float* __restrict__ w_fg,
    float* __restrict__ log_T,
    float* __restrict__ gaussian_scores,
    const bool compute_score_squares,
    const float* __restrict__ score_error_map,
    float* __restrict__ gaussian_score_max_error
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int idx = blockIdx.x * 4 + warp;
    if (idx >= P) return;

    int r = radii[idx];
    if (r <= 0) return;

    const float2 center = means2D[idx];
    const float4 con_o = conic_opacity[idx];
    const float2 wt = precomp_w_thres[idx];
    if (!(wt.x > 0.0f) || !(wt.y < 0.0f)) return;

    const int x0 = max(0, (int)floorf(center.x - (float)r));
    const int y0 = max(0, (int)floorf(center.y - (float)r));
    const int x1 = min(W, (int)ceilf(center.x + (float)r + 1.0f));
    const int y1 = min(H, (int)ceilf(center.y + (float)r + 1.0f));
    const int bw = x1 - x0;
    const int bh = y1 - y0;
    if (bw <= 0 || bh <= 0) return;

    float3 feat = make_float3(0.0f, 0.0f, 0.0f);
    if (CHANNELS == 3) {
        feat = reinterpret_cast<const float3*>(features)[idx];
    }

    float local_score = 0.0f;
    float local_max_err = 0.0f;
    const int total = bw * bh;
    for (int linear = lane; linear < total; linear += 32) {
        const int lx = linear - (linear / bw) * bw;
        const int ly = linear / bw;
        const int px = x0 + lx;
        const int py = y0 + ly;
        const float dx = center.x - (float)px;
        const float dy = center.y - (float)py;

        const float power = -0.5f * con_o.x * dx * dx - con_o.y * dx * dy - 0.5f * con_o.z * dy * dy;
        if (power > 0.0f || power < wt.y) continue;

        const float alpha = fminf(0.99f, con_o.w * __expf(power));
        const float blend_weight = alpha * wt.x;
        if (alpha < OIT_ALPHA_EPS && blend_weight < OIT_ALPHA_EPS) continue;

        const int pix = py * W + px;
        atomicAdd(out_num + 0 * W * H + pix, feat.x * blend_weight);
        atomicAdd(out_num + 1 * W * H + pix, feat.y * blend_weight);
        atomicAdd(out_num + 2 * W * H + pix, feat.z * blend_weight);
        atomicAdd(w_fg + pix, blend_weight);
        atomicAdd(log_T + pix, __logf(fmaxf(1.0f - alpha, 1e-6f)));

        if (gaussian_scores != nullptr) {
            local_score += compute_score_squares ? blend_weight * blend_weight : blend_weight;
        }
        if (gaussian_score_max_error != nullptr && score_error_map != nullptr) {
            local_max_err = fmaxf(local_max_err, score_error_map[pix]);
        }
    }

    if (gaussian_scores != nullptr) {
        atomicAdd(gaussian_scores + idx, local_score);
    }
    if (gaussian_score_max_error != nullptr && score_error_map != nullptr) {
        atomicMaxFloatNonnegative(gaussian_score_max_error + idx, local_max_err);
    }
}

__global__ void __launch_bounds__(256) finalizeOITPixels(
    int N,
    const float* __restrict__ background,
    float* __restrict__ out_color,
    float* __restrict__ w_fg,
    float* __restrict__ log_T
) {
    int pix = cg::this_grid().thread_rank();
    if (pix >= N) return;
    const float final_T = __expf(log_T[pix]);
    const float inv_w = 1.0f / fmaxf(w_fg[pix], 1e-5f);
    out_color[0 * N + pix] = (out_color[0 * N + pix] * inv_w) * (1.0f - final_T) + final_T * background[0];
    out_color[1 * N + pix] = (out_color[1 * N + pix] * inv_w) * (1.0f - final_T) + final_T * background[1];
    out_color[2 * N + pix] = (out_color[2 * N + pix] * inv_w) * (1.0f - final_T) + final_T * background[2];
    log_T[pix] = final_T;
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(256) renderTileOIT(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ means2D,
    const float* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    const float2* __restrict__ precomp_w_thres,
    float* __restrict__ out_color,
    float* __restrict__ w_fg, 
	float* __restrict__ Ts,
	float* __restrict__ accum_alpha,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ background,
	float* __restrict__ gaussian_scores,
	const bool compute_score_squares,
	const float* __restrict__ score_error_map,
	float* __restrict__ gaussian_score_max_error
) {
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + 15) / 16;
    int tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
    
    uint2 pix = { block.group_index().x * 16 + block.thread_index().x, 
                  block.group_index().y * 16 + block.thread_index().y };
    
    uint32_t tr = block.thread_rank(); 
    bool inside = (pix.x < W && pix.y < H);

    float px_f = (float)pix.x;
    float py_f = (float)pix.y;

    float c[CHANNELS] = {0.0f};
    float w_fg_acc = 0.0f;
    float Ts_acc = 0.0f; 

    uint2 tile_range = ranges[tile_id];
    int count = tile_range.y - tile_range.x;
    int rounds = (count + 255) / 256;

    // Double Buffered Shared Memory
    __shared__ float4 s_geom[2][256];
    __shared__ float4 s_weight[2][256];
    __shared__ float4 s_color[2][256];
    __shared__ int s_id[2][256];

    // Initial preload for Buffer 0
    if (tr < count) {
        int p_idx = point_list[tile_range.x + tr];
        s_id[0][tr] = p_idx;
        float2 center = means2D[p_idx];
        float4 con_o = conic_opacity[p_idx];
        float2 pt = precomp_w_thres[p_idx];

        s_geom[0][tr] = make_float4(center.x, center.y, -0.5f * con_o.x, -con_o.y);
        s_weight[0][tr] = make_float4(-0.5f * con_o.z, con_o.w, pt.x, pt.y);

        if (CHANNELS == 3) {
            float3 feat = reinterpret_cast<const float3*>(features)[p_idx];
            s_color[0][tr] = make_float4(feat.x * pt.x, feat.y * pt.x, feat.z * pt.x, 0.0f);
        }
    }
    block.sync(); 

    int db = 0; // Current buffer index
    
    for (int r = 0; r < rounds; r++) {
        int next_r = r + 1;
        int next_db = db ^ 1;

        // PREFETCH: Load next block from global memory into the background buffer
        if (next_r < rounds) {
            int load_idx = next_r * 256 + tr;
            if (load_idx < count) {
                int p_idx = point_list[tile_range.x + load_idx];
                s_id[next_db][tr] = p_idx;

                float2 center = means2D[p_idx];
                float4 con_o = conic_opacity[p_idx];
                float2 pt = precomp_w_thres[p_idx];
                
                s_geom[next_db][tr] = make_float4(center.x, center.y, -0.5f * con_o.x, -con_o.y);
                s_weight[next_db][tr] = make_float4(-0.5f * con_o.z, con_o.w, pt.x, pt.y);
                
                if (CHANNELS == 3) {
                    float3 feat = reinterpret_cast<const float3*>(features)[p_idx];
                    s_color[next_db][tr] = make_float4(feat.x * pt.x, feat.y * pt.x, feat.z * pt.x, 0.0f);
                }
            }
        }
        
        int limit = min(256, count - r * 256);
        
        // COMPUTE: Process current buffer while prefetching happens in background
        #pragma unroll 4 
        for (int i = 0; i < limit; i++) {
            if (!inside) continue;
            float4 g = s_geom[db][i];
            float4 w = s_weight[db][i];
            
            float dx = g.x - px_f;
            float dy = g.y - py_f;
            
            float power = g.z * dx * dx + dy * (g.w * dx + w.x * dy);
            
            if (power > 0.0f || power < w.w) continue;
            
            float alpha = fminf(0.99f, w.y * __expf(power));
            const float blend_weight = alpha * w.z;
            if (alpha < OIT_ALPHA_EPS && blend_weight < OIT_ALPHA_EPS) continue;
            
            float4 col = s_color[db][i];

            c[0] += col.x * alpha;
            c[1] += col.y * alpha;
            c[2] += col.z * alpha;

			if (gaussian_scores != nullptr) {
				int p_idx = s_id[db][i];
				// Sort-free rendering uses the order-independent MobileGS weight.
				const float score = compute_score_squares ? blend_weight * blend_weight : blend_weight;
				atomicAdd(&gaussian_scores[p_idx], score);
			}
			if (gaussian_score_max_error != nullptr && score_error_map != nullptr) {
				int p_idx = s_id[db][i];
				atomicMaxFloatNonnegative(&gaussian_score_max_error[p_idx], score_error_map[pix.y * W + pix.x]);
			}
            
            w_fg_acc += blend_weight;
            Ts_acc += __logf(max(1.0f - alpha, 1e-6f));
        }
        
        block.sync(); // Crucial: syncs both the prefetch and the compute
        db = next_db; // Swap buffers
    }

    if (inside) {
        int p_id = pix.y * W + pix.x;
        float final_T = __expf(Ts_acc);
        float inv_w = 1.0f / max(w_fg_acc, 1e-5f);
        
        out_color[0 * W * H + p_id] = (c[0] * inv_w) * (1.0f - final_T) + final_T * background[0];
        out_color[1 * W * H + p_id] = (c[1] * inv_w) * (1.0f - final_T) + final_T * background[1];
        out_color[2 * W * H + p_id] = (c[2] * inv_w) * (1.0f - final_T) + final_T * background[2];
        
		if (w_fg) {
			w_fg[p_id] = w_fg_acc;
		}
		if (Ts) Ts[p_id] = final_T;
		if (accum_alpha) accum_alpha[p_id] = final_T;
		if (n_contrib) n_contrib[p_id] = count;
    }
}

















// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* theta,
	const float* phi,
	float* w_fg,
	float* Ts,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* gaussian_scores,
	const bool compute_score_squares,
	const float* score_error_map,
	float* gaussian_score_max_error,
	float* out_color,
	const bool use_scatter_order_independent,

	float* kernel_times,
	int* radii,
	bool debug)
{
	cudaStream_t stream = c10::cuda::getCurrentCUDAStream();



	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	size_t chunk_size = required<GeometryState>(P) + requiredChunkRenderWorkspace(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);
	ChunkRenderWorkspace workspace = ChunkRenderWorkspace::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,

		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Precompute Mobile-GS weights before binning.  This gives us the same
	// support threshold used in renderTileOIT, so we can shrink low-opacity
	// splat footprints before the prefix scan and radix grouping.
	precomputeGaussianData<<<(P + 255) / 256, 256, 0, stream>>>(
        P, radii, geomState.depths, phi, (glm::vec3*)scales,
        geomState.conic_opacity, workspace.precomp_w_thres
    );
	CHECK_CUDA(, debug)

	updateOITTilesTouched<<<(P + 255) / 256, 256, 0, stream>>>(
		P,
		geomState.means2D,
		radii,
		geomState.tiles_touched,
		workspace.precomp_w_thres,
		tile_grid);
	CHECK_CUDA(, debug)

	if (use_scatter_order_independent)
	{
		const int num_pixels = width * height;
		CHECK_CUDA(cudaMemsetAsync(out_color, 0, NUM_CHANNELS * num_pixels * sizeof(float), stream), debug);
		CHECK_CUDA(cudaMemsetAsync(w_fg, 0, num_pixels * sizeof(float), stream), debug);
		CHECK_CUDA(cudaMemsetAsync(Ts, 0, num_pixels * sizeof(float), stream), debug);

		const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
		scatterOITGaussians<3><<<(P + 3) / 4, 128, 0, stream>>>(
			P,
			width, height,
			geomState.means2D,
			radii,
			feature_ptr,
			geomState.conic_opacity,
			workspace.precomp_w_thres,
			out_color,
			w_fg,
			Ts,
			gaussian_scores,
			compute_score_squares,
			score_error_map,
			gaussian_score_max_error);
		CHECK_CUDA(, debug)

		finalizeOITPixels<<<(num_pixels + 255) / 256, 256, 0, stream>>>(
			num_pixels, background, out_color, w_fg, Ts);
		CHECK_CUDA(, debug)
		if (kernel_times != nullptr) {
			kernel_times[0] = 0.0f;
		}
		return P;
	}

	// Reuse the standard 3DGS binning path instead of the previous two-pass
	// atomic count/fill path.  Sort-free rendering does not need per-tile depth
	// order, but it still needs a compact per-tile range list.  Sorting only the
	// tile-id bits groups instances by tile while skipping the expensive 32 depth
	// radix passes used by alpha blending.
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		geomState.scanning_space,
		geomState.scan_size,
		geomState.tiles_touched,
		geomState.point_offsets,
		P,
		stream), debug)

	int total_instances = 0;
	CHECK_CUDA(cudaMemcpyAsync(&total_instances, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost, stream), debug);
	CHECK_CUDA(cudaStreamSynchronize(stream), debug);

	size_t binning_chunk_size = required<BinningState>(total_instances);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, total_instances);

	duplicateWithTileKeys << <(P + 255) / 256, 256, 0, stream >> > (
		P,
		geomState.means2D,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		workspace.precomp_w_thres,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		total_instances, 32, 32 + bit, stream), debug)

	CHECK_CUDA(cudaMemsetAsync(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2), stream), debug);

	if (total_instances > 0)
		identifyTileRanges << <(total_instances + 255) / 256, 256, 0, stream >> > (
			total_instances,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

    const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    
    renderTileOIT<3><<<tile_grid, block, 0, stream>>>(
		imgState.ranges,
		binningState.point_list,
        width, height, 
        geomState.means2D, 
        feature_ptr, 
        geomState.conic_opacity,
        workspace.precomp_w_thres,
		out_color, w_fg, Ts, imgState.accum_alpha, imgState.n_contrib, background, gaussian_scores, compute_score_squares, score_error_map, gaussian_score_max_error
    );
	CHECK_CUDA(, debug)
	
	if (kernel_times != nullptr) {
		kernel_times[0] = 0.0f;
	}

    return total_instances;
}






	

int CudaRasterizer::Rasterizer::forward_depth(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* theta,
	const float* phi,
	float* w_fg,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_pts,

	float* out_depth,
	float* accum_alpha,
	int* gidx,
	float* discriminants,

	int* radii,
	bool debug)
{
	cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}


	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,

		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		geomState.scanning_space,
		geomState.scan_size,
		geomState.tiles_touched,
		geomState.point_offsets,
		P,
		stream), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpyAsync(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost, stream), debug);
	CHECK_CUDA(cudaStreamSynchronize(stream), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256, 0, stream >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit, stream), debug)

	CHECK_CUDA(cudaMemsetAsync(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2), stream), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256, 0, stream >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	imgState.w_fg = w_fg;
	CHECK_CUDA(FORWARD::render_depth(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		geomState.depths,
		feature_ptr,

		geomState.conic_opacity,
	    theta,
		phi,
		imgState.w_fg,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_pts,
		
		out_depth,
		accum_alpha,
		gidx,
		discriminants,

		means3D,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,

		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos
		), debug)



	return num_rendered;
}







// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* out_colors,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* theta,
	const float* phi,
	const float* w_fg,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dtheta,
	float* dL_dphi,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_ddepth,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		(glm::vec3*)scales,
		geomState.depths,
		geomState.conic_opacity,
		theta,
		phi,
		w_fg,
		out_colors,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dtheta,
		dL_dphi,
		dL_dcolor,
		dL_ddepth,
		(glm::vec3*)dL_dscale), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		dL_ddepth), debug)
}