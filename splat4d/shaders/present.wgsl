@group(0) @binding(0) var src_tex: texture_2d<f32>;
struct VsOut { @builtin(position) pos: vec4<f32> };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
  var o: VsOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  return o;
}
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dims = textureDimensions(src_tex);
  let x = min(u32(max(pos.x, 0.0)), dims.x - 1u);
  let y = min(u32(max(pos.y, 0.0)), dims.y - 1u);
  let c = textureLoad(src_tex, vec2<i32>(i32(x), i32(y)), 0);
  return vec4<f32>(c.rgb, 1.0);
}
