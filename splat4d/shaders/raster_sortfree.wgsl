#include "common.wgsl"
#include "opacity_mlp.wgsl"

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<GaussianRecord>;
@group(0) @binding(2) var<storage, read> appearance: array<f32>;
@group(0) @binding(5) var<storage, read> conditioned: array<ConditionedGaussian>;
@group(0) @binding(6) var<storage, read> visible_ids: array<atomic<u32>>;
@group(0) @binding(13) var<storage, read_write> accum_rgba: array<atomic<u32>>;

fn q16(v: f32) -> u32 {
  return u32(clamp(v * 65535.0, 0.0, 4294967295.0));
}

fn eval_alpha_sf(c: ConditionedGaussian, p: vec2<f32>) -> f32 {
  let d = p - c.mean3.xy;
  let sigma = max(c.mean3.z, 1e-4);
  let power = -0.5 * dot(d, d) / (sigma * sigma);
  if (power < -12.0) { return 0.0; }
  return clamp(c.conic_opacity.w * exp(power), 0.0, 0.999);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let compact_idx = gid.x + 1u;
  let visible_count = atomicLoad(&visible_ids[0]);
  if (compact_idx > visible_count) { return; }
  let id = atomicLoad(&visible_ids[compact_idx]);
  let c = conditioned[id];
  if (c.radius_flags.y == 0u) { return; }
  let g = gaussians[id];
  let cam_vec = normalize(c.mean3.xyz - frame.camera_pos_time.xyz);
  let time_feat = vec3<f32>(frame.camera_pos_time.w - g.mean4.w, c.mean3.w, g.scale4.w);
  let po = mobilegs_opacity_phi(id, g.appearance_offset, g.appearance_len, cam_vec, g.scale4.xyz, g.q_left, time_feat);
  let phi = po.x;
  let learned_opacity = po.y;
  let depth = max(c.color_depth.w, 1e-4);
  let smax = max(max(g.scale4.x, g.scale4.y), g.scale4.z);
  let w = phi * phi + phi / (depth * depth) + exp(clamp(smax / depth, -20.0, 20.0));
  let radius = c.radius_flags.x;
  let x0 = max(i32(c.mean3.x) - i32(radius), 0);
  let y0 = max(i32(c.mean3.y) - i32(radius), 0);
  let x1 = min(i32(c.mean3.x) + i32(radius), i32(frame.params0.x) - 1);
  let y1 = min(i32(c.mean3.y) + i32(radius), i32(frame.params0.y) - 1);
  var y = y0;
  loop {
    if (y > y1) { break; }
    var x = x0;
    loop {
      if (x > x1) { break; }
      let a = eval_alpha_sf(c, vec2<f32>(f32(x) + 0.5, f32(y) + 0.5)) * learned_opacity;
      if (a > 0.0) {
        let aw = a * w;
        let pix = u32(y) * frame.params0.x + u32(x);
        atomicAdd(&accum_rgba[pix * 4u + 0u], q16(c.color_depth.x * aw));
        atomicAdd(&accum_rgba[pix * 4u + 1u], q16(c.color_depth.y * aw));
        atomicAdd(&accum_rgba[pix * 4u + 2u], q16(c.color_depth.z * aw));
        atomicAdd(&accum_rgba[pix * 4u + 3u], q16(aw));
      }
      x = x + 1;
    }
    y = y + 1;
  }
}
