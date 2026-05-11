struct IdentifyUniforms { n: u32, tile_count: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<uniform> ident: IdentifyUniforms;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> ranges: array<vec2<u32>>;

fn linear_id(gid: vec3<u32>) -> u32 {
  return gid.x + gid.y * 65535u * 256u;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = linear_id(gid);
  if (idx >= ident.n) { return; }
  let curr = keys_hi[idx];
  if (curr >= ident.tile_count) { return; }
  if (idx == 0u) {
    ranges[curr].x = 0u;
  } else {
    let prev = keys_hi[idx - 1u];
    if (curr != prev) {
      if (prev < ident.tile_count) { ranges[prev].y = idx; }
      ranges[curr].x = idx;
    }
  }
  if (idx == ident.n - 1u) {
    ranges[curr].y = ident.n;
  }
}
