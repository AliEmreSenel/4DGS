struct ClearUniforms { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<uniform> clear: ClearUniforms;
@group(0) @binding(1) var<storage, read_write> ranges: array<vec2<u32>>;

fn linear_id(gid: vec3<u32>) -> u32 {
  return gid.x + gid.y * 65535u * 256u;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = linear_id(gid);
  if (i >= clear.n) { return; }
  ranges[i] = vec2<u32>(0u, 0u);
}
