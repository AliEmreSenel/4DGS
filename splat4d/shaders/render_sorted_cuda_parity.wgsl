#include "cuda_common.wgsl"
@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(5) var<storage, read> prep: array<PreprocessedGaussian>;
@group(0) @binding(11) var<storage, read> point_list: array<u32>;
@group(0) @binding(12) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(13) var output_tex: texture_storage_2d<rgba16float, write>;

var<workgroup> collected_id: array<u32, 256>;
var<workgroup> collected_xy: array<vec2<f32>, 256>;
var<workgroup> collected_conic_opacity: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tile_x = frame.params1.x;
  let tile_id = wid.y * tile_x + wid.x;
  let pix = vec2<u32>(wid.x * BLOCK_X + lid3.x, wid.y * BLOCK_Y + lid3.y);
  let width = frame.params0.x;
  let height = frame.params0.y;
  let inside = pix.x < width && pix.y < height;
  let local_id = lid3.y * BLOCK_X + lid3.x;
  let range = ranges[tile_id];
  let total = range.y - range.x;
  let rounds = (total + BLOCK_SIZE - 1u) / BLOCK_SIZE;
  var T = 1.0;
  var C = vec3<f32>(0.0);
  var done = !inside;
  let pixf = vec2<f32>(f32(pix.x), f32(pix.y));

  for (var r = 0u; r < rounds; r = r + 1u) {
    if (range.x + r * BLOCK_SIZE + local_id < range.y) {
      let idx = point_list[range.x + r * BLOCK_SIZE + local_id];
      let p = prep[idx];
      collected_id[local_id] = idx;
      collected_xy[local_id] = p.xy_depth_radius.xy;
      collected_conic_opacity[local_id] = p.conic_opacity;
    }
    workgroupBarrier();
    let todo = min(BLOCK_SIZE, total - r * BLOCK_SIZE);
    for (var j = 0u; j < todo; j = j + 1u) {
      if (!done) {
        let idx = collected_id[j];
        let d = collected_xy[j] - pixf;
        let con_o = collected_conic_opacity[j];
        let power = -0.5 * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power <= 0.0) {
          let alpha = min(0.99, con_o.w * exp(power));
          if (alpha >= (1.0 / 255.0)) {
            let test_T = T * (1.0 - alpha);
            // CUDA renderCUDA rejects this contributor when it would push the
            // remaining transmittance below the termination threshold.
            // Keeping it changed saturated/opaque regions and was one source
            // of non-parity with the reference image.
            if (test_T < 0.0001) {
              done = true;
            } else {
              C = C + prep[idx].color.rgb * alpha * T;
              T = test_T;
            }
          }
        }
      }
    }
    workgroupBarrier();
  }
  if (inside) {
    let outc = C + T * frame.background.rgb;
    textureStore(output_tex, vec2<i32>(i32(pix.x), i32(pix.y)), vec4<f32>(outc, 1.0));
  }
}
