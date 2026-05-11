#include "cuda_common.wgsl"

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<GaussianRecord>;
@group(0) @binding(2) var<storage, read> appearance: array<f32>;
@group(0) @binding(3) var<storage, read> keyframes: array<f32>;
@group(0) @binding(4) var<storage, read> masks: array<u32>;
@group(0) @binding(5) var<storage, read_write> prep: array<PreprocessedGaussian>;
@group(0) @binding(6) var<storage, read_write> tiles_touched: array<u32>;

fn coeff(base: u32, i: u32, max_coeff: u32) -> vec3<f32> {
  if (i >= max_coeff) { return vec3<f32>(0.0); }
  let o = base + i * 3u;
  return vec3<f32>(appearance[o], appearance[o + 1u], appearance[o + 2u]);
}

fn eval_sh(g: GaussianRecord, pos: vec3<f32>, timestamp: f32, frame: FrameUniforms) -> vec3<f32> {
  let max_coeff = min(g.appearance_len / 3u, frame.params3.z);
  if (max_coeff <= 1u) {
    return max(coeff(g.appearance_offset, 0u, max_coeff), vec3<f32>(0.0));
  }
  var dir = pos - frame.camera_pos_time.xyz;
  dir = dir / max(length(dir), 1e-8);
  let x = dir.x; let y = dir.y; let z = dir.z;
  let deg = frame.params3.x;
  let deg_t = frame.params3.y;
  let force_sh3d = frame.params3.w != 0u;
  let base = g.appearance_offset;
  var result = SH_C0 * coeff(base, 0u, max_coeff);
  var l1m1 = 0.0; var l1m0 = 0.0; var l1p1 = 0.0;
  var l2m2 = 0.0; var l2m1 = 0.0; var l2m0 = 0.0; var l2p1 = 0.0; var l2p2 = 0.0;
  var l3m3 = 0.0; var l3m2 = 0.0; var l3m1 = 0.0; var l3m0 = 0.0; var l3p1 = 0.0; var l3p2 = 0.0; var l3p3 = 0.0;
  if (deg > 0u) {
    l1m1 = -SH_C1 * y; l1m0 = SH_C1 * z; l1p1 = -SH_C1 * x;
    result = result + l1m1 * coeff(base, 1u, max_coeff) + l1m0 * coeff(base, 2u, max_coeff) + l1p1 * coeff(base, 3u, max_coeff);
    if (deg > 1u) {
      let xx=x*x; let yy=y*y; let zz=z*z; let xy=x*y; let yz=y*z; let xz=x*z;
      l2m2 = SH_C2_0 * xy; l2m1 = SH_C2_1 * yz; l2m0 = SH_C2_2 * (2.0*zz - xx - yy); l2p1 = SH_C2_3 * xz; l2p2 = SH_C2_4 * (xx - yy);
      result = result + l2m2*coeff(base,4u,max_coeff) + l2m1*coeff(base,5u,max_coeff) + l2m0*coeff(base,6u,max_coeff) + l2p1*coeff(base,7u,max_coeff) + l2p2*coeff(base,8u,max_coeff);
      if (deg > 2u) {
        l3m3 = SH_C3_0 * y * (3.0*xx - yy);
        l3m2 = SH_C3_1 * xy * z;
        l3m1 = SH_C3_2 * y * (4.0*zz - xx - yy);
        l3m0 = SH_C3_3 * z * (2.0*zz - 3.0*xx - 3.0*yy);
        l3p1 = SH_C3_4 * x * (4.0*zz - xx - yy);
        l3p2 = SH_C3_5 * z * (xx - yy);
        l3p3 = SH_C3_6 * x * (xx - 3.0*yy);
        result = result + l3m3*coeff(base,9u,max_coeff) + l3m2*coeff(base,10u,max_coeff) + l3m1*coeff(base,11u,max_coeff) + l3m0*coeff(base,12u,max_coeff) + l3p1*coeff(base,13u,max_coeff) + l3p2*coeff(base,14u,max_coeff) + l3p3*coeff(base,15u,max_coeff);
        if (!force_sh3d && deg_t > 0u) {
          let dt = g.mean4.w - timestamp;
          let duration = max(frame.params4.x, 1e-8);
          let t1 = cos(2.0 * PI * dt / duration);
          result = result + t1 * (SH_C0*coeff(base,16u,max_coeff) + l1m1*coeff(base,17u,max_coeff) + l1m0*coeff(base,18u,max_coeff) + l1p1*coeff(base,19u,max_coeff) + l2m2*coeff(base,20u,max_coeff) + l2m1*coeff(base,21u,max_coeff) + l2m0*coeff(base,22u,max_coeff) + l2p1*coeff(base,23u,max_coeff) + l2p2*coeff(base,24u,max_coeff) + l3m3*coeff(base,25u,max_coeff) + l3m2*coeff(base,26u,max_coeff) + l3m1*coeff(base,27u,max_coeff) + l3m0*coeff(base,28u,max_coeff) + l3p1*coeff(base,29u,max_coeff) + l3p2*coeff(base,30u,max_coeff) + l3p3*coeff(base,31u,max_coeff));
          if (deg_t > 1u) {
            let t2 = cos(2.0 * PI * dt * 2.0 / duration);
            result = result + t2 * (SH_C0*coeff(base,32u,max_coeff) + l1m1*coeff(base,33u,max_coeff) + l1m0*coeff(base,34u,max_coeff) + l1p1*coeff(base,35u,max_coeff) + l2m2*coeff(base,36u,max_coeff) + l2m1*coeff(base,37u,max_coeff) + l2m0*coeff(base,38u,max_coeff) + l2p1*coeff(base,39u,max_coeff) + l2p2*coeff(base,40u,max_coeff) + l3m3*coeff(base,41u,max_coeff) + l3m2*coeff(base,42u,max_coeff) + l3m1*coeff(base,43u,max_coeff) + l3m0*coeff(base,44u,max_coeff) + l3p1*coeff(base,45u,max_coeff) + l3p2*coeff(base,46u,max_coeff) + l3p3*coeff(base,47u,max_coeff));
          }
        }
      }
    }
  }
  result = result + vec3<f32>(0.5);
  return max(result, vec3<f32>(0.0));
}

fn temporal_mask_active(idx: u32, frame: FrameUniforms) -> bool {
  if (frame.params5.w == 0u) { return true; }
  let word = idx / 32u;
  let bit = idx & 31u;
  let mask = 1u << bit;
  let words = frame.params5.x;
  let lo = frame.params5.y;
  let hi = frame.params5.z;
  let a = masks[lo * words + word];
  let b = masks[hi * words + word];
  return ((a | b) & mask) != 0u;
}


fn linear_id(gid: vec3<u32>) -> u32 {
  return gid.x + gid.y * 65535u * 256u;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = linear_id(gid);
  let P = frame.params0.z;
  if (idx >= P) { return; }
  tiles_touched[idx] = 0u;
  var empty: PreprocessedGaussian;
  empty.xy_depth_radius = vec4<f32>(0.0);
  empty.conic_opacity = vec4<f32>(0.0);
  empty.color = vec4<f32>(0.0);
  empty.rect = vec4<u32>(0u);
  prep[idx] = empty;
  if (!temporal_mask_active(idx, frame)) { return; }

  let g = gaussians[idx];
  var p_orig = g.mean4.xyz;
  var opacity = g.opacity;
  var cov3: mat3x3<f32>;
  let scale_modifier = frame.params2.z;
  let prefilter_var = frame.params2.w;
  let timestamp = frame.camera_pos_time.w;

  if ((g.flags & FLAG_ROT_4D) != 0u) {
    let Sigma = cov4_from_scale_rot(g, scale_modifier);
    let cov_t = max(Sigma[3][3], 1e-8);
    let dt = timestamp - g.mean4.w;
    let marginal_t = exp(-0.5 * dt * dt / select(cov_t, cov_t + prefilter_var, prefilter_var > 0.0));
    if (marginal_t <= 0.05) { return; }
    opacity = opacity * marginal_t;
    let cov12 = vec3<f32>(Sigma[0][3], Sigma[1][3], Sigma[2][3]);
    p_orig = p_orig + cov12 / cov_t * dt;
    let c0 = vec3<f32>(Sigma[0][0], Sigma[0][1], Sigma[0][2]) - cov12.x * cov12 / cov_t;
    let c1 = vec3<f32>(Sigma[1][0], Sigma[1][1], Sigma[1][2]) - cov12.y * cov12 / cov_t;
    let c2 = vec3<f32>(Sigma[2][0], Sigma[2][1], Sigma[2][2]) - cov12.z * cov12 / cov_t;
    cov3 = mat3x3<f32>(c0, c1, c2);
  } else {
    cov3 = cov3_from_scale_rot(g.scale4.xyz, scale_modifier, g.q_left);
    let sigma = max(g.scale4.w * scale_modifier, 1e-8);
    let dt = g.mean4.w - timestamp;
    let marginal_t = exp(-0.5 * dt * dt / select(sigma, sigma + prefilter_var, prefilter_var > 0.0));
    if (marginal_t <= 0.05) { return; }
    opacity = opacity * marginal_t;
  }

  let p_view = transform_point_4x3(frame.view, p_orig);
  if (p_view.z <= 0.2) { return; }
  let p_hom = transform_point_4x4(frame.proj, p_orig);
  let p_w = 1.0 / (p_hom.w + 0.0000001);
  let p_proj = p_hom.xyz * p_w;

  let cov = cov2d_cuda(p_orig, cov3, frame);
  let det = cov.x * cov.z - cov.y * cov.y;
  if (det == 0.0) { return; }
  let det_inv = 1.0 / det;
  let conic = vec3<f32>(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);
  let mid = 0.5 * (cov.x + cov.z);
  let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
  let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
  let radius_f = ceil(3.0 * sqrt(max(lambda1, lambda2)));
  if (i32(radius_f) <= 0) { return; }
  let xy = vec2<f32>(ndc2pix(p_proj.x, frame.params0.x), ndc2pix(p_proj.y, frame.params0.y));
  let rect = rect_from_xy_radius(xy, u32(radius_f), frame.params1.x, frame.params1.y);
  let touched = (rect.z - rect.x) * (rect.w - rect.y);
  if (touched == 0u) { return; }
  let color = eval_sh(g, g.mean4.xyz, timestamp, frame);
  var out: PreprocessedGaussian;
  out.xy_depth_radius = vec4<f32>(xy.x, xy.y, p_view.z, radius_f);
  out.conic_opacity = vec4<f32>(conic, opacity);
  out.color = vec4<f32>(color, 0.0);
  out.rect = rect;
  prep[idx] = out;
  tiles_touched[idx] = touched;
}
