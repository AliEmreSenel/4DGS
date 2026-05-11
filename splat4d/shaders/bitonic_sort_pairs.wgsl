struct SortUniforms { n: u32, k: u32, j: u32, _pad0: u32 };
@group(0) @binding(0) var<uniform> sortu: SortUniforms;
@group(0) @binding(1) var<storage, read_write> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_lo: array<u32>;
@group(0) @binding(3) var<storage, read_write> values: array<u32>;
fn greater(ahi: u32, alo: u32, bhi: u32, blo: u32) -> bool {
  return (ahi > bhi) || ((ahi == bhi) && (alo > blo));
}

fn linear_id(gid: vec3<u32>) -> u32 {
  return gid.x + gid.y * 65535u * 256u;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = linear_id(gid);
  if (i >= sortu.n) { return; }
  let ixj = i ^ sortu.j;
  if (ixj <= i || ixj >= sortu.n) { return; }
  let ascending = (i & sortu.k) == 0u;
  let ahi = keys_hi[i]; let alo = keys_lo[i]; let av = values[i];
  let bhi = keys_hi[ixj]; let blo = keys_lo[ixj]; let bv = values[ixj];
  let swap_needed = select(greater(bhi, blo, ahi, alo), greater(ahi, alo, bhi, blo), ascending);
  if (swap_needed) {
    keys_hi[i] = bhi; keys_lo[i] = blo; values[i] = bv;
    keys_hi[ixj] = ahi; keys_lo[ixj] = alo; values[ixj] = av;
  }
}
