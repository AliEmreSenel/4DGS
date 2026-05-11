struct ClearUniforms { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<uniform> clear: ClearUniforms;
@group(0) @binding(1) var<storage, read_write> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_lo: array<u32>;
@group(0) @binding(3) var<storage, read_write> values: array<u32>;

fn linear_id(gid: vec3<u32>) -> u32 {
  return gid.x + gid.y * 65535u * 256u;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = linear_id(gid);
  if (i >= clear.n) { return; }
  keys_hi[i] = 0xffffffffu;
  keys_lo[i] = 0xffffffffu;
  values[i] = 0u;
}
