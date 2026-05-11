#include "common.wgsl"

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(7) var<storage, read_write> tile_counts: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> tile_items: array<u32>;

// Per-tile stable insertion sort. This is deterministic and preserves the CUDA-style
// near-to-far alpha order for reference rendering. For very large tiles this can be
// replaced by a tiled radix network without changing the pack format.
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tile = gid.x;
  let tile_count = frame.params1.x * frame.params1.y;
  if (tile >= tile_count) { return; }
  let max_per = frame.params1.z;
  let count = min(atomicLoad(&tile_counts[tile]), max_per);
  let key_base = tile * max_per;
  let id_base = tile_count * max_per + key_base;
  var i = 1u;
  loop {
    if (i >= count) { break; }
    let k = tile_items[key_base + i];
    let id = tile_items[id_base + i];
    var j = i;
    loop {
      if (j == 0u || tile_items[key_base + j - 1u] <= k) { break; }
      tile_items[key_base + j] = tile_items[key_base + j - 1u];
      tile_items[id_base + j] = tile_items[id_base + j - 1u];
      j = j - 1u;
    }
    tile_items[key_base + j] = k;
    tile_items[id_base + j] = id;
    i = i + 1u;
  }
}
