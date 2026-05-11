#include "cuda_common.wgsl"
@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(5) var<storage, read> prep: array<PreprocessedGaussian>;
@group(0) @binding(11) var<storage, read> point_list: array<u32>;
@group(0) @binding(12) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(13) var output_tex: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tile_id = wid.y * frame.params1.x + wid.x;
  let pix = vec2<u32>(wid.x * BLOCK_X + lid.x, wid.y * BLOCK_Y + lid.y);
  if (pix.x >= frame.params0.x || pix.y >= frame.params0.y) { return; }
  let range = ranges[tile_id];
  let pixf = vec2<f32>(f32(pix.x), f32(pix.y));
  var Cw = vec3<f32>(0.0);
  var Wsum = 0.0;
  var logT = 0.0;
  for (var ii = range.x; ii < range.y; ii = ii + 1u) {
    let idx = point_list[ii];
    let p = prep[idx];
    let dxy = p.xy_depth_radius.xy - pixf;
    let con_o = p.conic_opacity;
    let power = -0.5 * (con_o.x * dxy.x * dxy.x + con_o.z * dxy.y * dxy.y) - con_o.y * dxy.x * dxy.y;
    if (power > 0.0) { continue; }
    let alpha = min(0.99, con_o.w * exp(power));
    if (alpha < 1.0 / 255.0) { continue; }
    let depth = max(p.xy_depth_radius.z, 1e-6);
    let max_s = max(1e-6, p.xy_depth_radius.w / 3.0);
    let phi = 1.0;
    let weight = exp(min(max_s / depth, 20.0)) + phi / (depth * depth) + phi * phi;
    let val = alpha * weight;
    Cw = Cw + p.color.rgb * val;
    Wsum = Wsum + val;
    logT = logT + log(max(1.0 - alpha, 1e-6));
  }
  let T = exp(logT);
  var color = frame.background.rgb;
  if (Wsum > 0.0) { color = (1.0 - T) * (Cw / Wsum) + T * frame.background.rgb; }
  textureStore(output_tex, vec2<i32>(i32(pix.x), i32(pix.y)), vec4<f32>(color, 1.0));
}
