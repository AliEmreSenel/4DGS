#include "common.wgsl"

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(5) var<storage, read> conditioned: array<ConditionedGaussian>;
@group(0) @binding(6) var<storage, read_write> visible_ids: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> tile_counts: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> tile_items: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let compact_idx = gid.x + 1u;
  let visible_count = atomicLoad(&visible_ids[0]);
  if (compact_idx > visible_count) { return; }
  let id = atomicLoad(&visible_ids[compact_idx]);
  let c = conditioned[id];
  if (c.radius_flags.y == 0u) { return; }
  let tile_size = 16u;
  let tx0 = max(i32((c.mean3.x - f32(c.radius_flags.x)) / f32(tile_size)), 0);
  let ty0 = max(i32((c.mean3.y - f32(c.radius_flags.x)) / f32(tile_size)), 0);
  let tx1 = min(i32((c.mean3.x + f32(c.radius_flags.x)) / f32(tile_size)), i32(frame.params1.x) - 1);
  let ty1 = min(i32((c.mean3.y + f32(c.radius_flags.x)) / f32(tile_size)), i32(frame.params1.y) - 1);
  var ty = ty0;
  loop {
    if (ty > ty1) { break; }
    var tx = tx0;
    loop {
      if (tx > tx1) { break; }
      let tile = u32(ty) * frame.params1.x + u32(tx);
      let offset = atomicAdd(&tile_counts[tile], 1u);
      if (offset < frame.params1.z) {
        let key_depth = u32(clamp(c.color_depth.w / max(frame.params2.y, 1e-6), 0.0, 1.0) * 16777215.0);
        tile_items[tile * frame.params1.z + offset] = (key_depth << 8u) | (id & 255u);
        tile_items[tile * frame.params1.z + offset + frame.params1.x * frame.params1.y * frame.params1.z] = id;
      }
      tx = tx + 1;
    }
    ty = ty + 1;
  }
}
