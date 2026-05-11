#include "common.wgsl"

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(9) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var<storage, read_write> accum_rgba: array<atomic<u32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= frame.params0.x || y >= frame.params0.y) { return; }
  let pix = y * frame.params0.x + x;
  let denom = f32(atomicLoad(&accum_rgba[pix * 4u + 3u])) / 65535.0;
  var rgb = frame.background.xyz;
  if (denom > 1e-6) {
    rgb = vec3<f32>(
      f32(atomicLoad(&accum_rgba[pix * 4u + 0u])) / 65535.0,
      f32(atomicLoad(&accum_rgba[pix * 4u + 1u])) / 65535.0,
      f32(atomicLoad(&accum_rgba[pix * 4u + 2u])) / 65535.0) / denom;
  }
  textureStore(output_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(rgb, 1.0));
  atomicStore(&accum_rgba[pix * 4u + 0u], 0u);
  atomicStore(&accum_rgba[pix * 4u + 1u], 0u);
  atomicStore(&accum_rgba[pix * 4u + 2u], 0u);
  atomicStore(&accum_rgba[pix * 4u + 3u], 0u);
}
