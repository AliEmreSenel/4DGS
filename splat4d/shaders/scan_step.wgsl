struct ScanUniforms { n: u32, offset: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<uniform> scan: ScanUniforms;
@group(0) @binding(1) var<storage, read> scan_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> scan_out: array<u32>;

fn linear_id(gid: vec3<u32>) -> u32 {
  return gid.x + gid.y * 65535u * 256u;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = linear_id(gid);
  if (i >= scan.n) { return; }
  var v = scan_in[i];
  if (i >= scan.offset) { v = v + scan_in[i - scan.offset]; }
  scan_out[i] = v;
}
