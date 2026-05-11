#include "cuda_common.wgsl"
@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(5) var<storage, read> prep: array<PreprocessedGaussian>;
@group(0) @binding(6) var<storage, read> tiles_touched: array<u32>;
@group(0) @binding(7) var<storage, read> offsets: array<u32>;
@group(0) @binding(9) var<storage, read_write> keys_hi: array<u32>;
@group(0) @binding(10) var<storage, read_write> keys_lo: array<u32>;
@group(0) @binding(11) var<storage, read_write> values: array<u32>;

fn linear_id(gid: vec3<u32>) -> u32 {
  return gid.x + gid.y * 65535u * 256u;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = linear_id(gid);
  if (idx >= frame.params0.z) { return; }
  let touched = tiles_touched[idx];
  if (touched == 0u) { return; }
  let p = prep[idx];
  var off = select(0u, offsets[idx - 1u], idx > 0u);
  let depth_bits = select(bitcast<u32>(p.xy_depth_radius.z), 0u, frame.params0.w == 1u);
  for (var y = p.rect.y; y < p.rect.w; y = y + 1u) {
    for (var x = p.rect.x; x < p.rect.z; x = x + 1u) {
      let tile = y * frame.params1.x + x;
      keys_hi[off] = tile;
      keys_lo[off] = depth_bits;
      values[off] = idx;
      off = off + 1u;
    }
  }
}
