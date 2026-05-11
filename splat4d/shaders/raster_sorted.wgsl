#include "common.wgsl"

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(5) var<storage, read> conditioned: array<ConditionedGaussian>;
@group(0) @binding(7) var<storage, read> tile_counts: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> tile_items: array<u32>;
@group(0) @binding(9) var output_tex: texture_storage_2d<rgba16float, write>;

fn eval_alpha(c: ConditionedGaussian, p: vec2<f32>) -> f32 {
  let d = p - c.mean3.xy;
  let power = -0.5 * (c.conic_opacity.x * d.x * d.x + c.conic_opacity.z * d.y * d.y) - c.conic_opacity.y * d.x * d.y;
  if (power > 0.0 || power < -12.0) { return 0.0; }
  return min(0.99, c.conic_opacity.w * exp(power));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= frame.params0.x || y >= frame.params0.y) { return; }
  let tile_size = 16u;
  let tx = x / tile_size;
  let ty = y / tile_size;
  let tile = ty * frame.params1.x + tx;
  let max_per = frame.params1.z;
  let count = min(atomicLoad(&tile_counts[tile]), max_per);
  let tile_count = frame.params1.x * frame.params1.y;
  let id_base = tile_count * max_per + tile * max_per;
  let pixel = vec2<f32>(f32(x), f32(y));
  var rgb = vec3<f32>(0.0);
  var T = 1.0;
  var i = 0u;
  loop {
    if (i >= count || T < 0.0001) { break; }
    let id = tile_items[id_base + i];
    let c = conditioned[id];
    let a = eval_alpha(c, pixel);
    if (a > 0.0) { rgb = rgb + T * a * c.color_depth.xyz; T = T * (1.0 - a); }
    i = i + 1u;
  }
  rgb = rgb + T * frame.background.xyz;
  textureStore(output_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(rgb, 1.0));
}
