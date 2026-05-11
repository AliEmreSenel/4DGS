#include "common.wgsl"

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<GaussianRecord>;
@group(0) @binding(2) var<storage, read> appearance: array<f32>;
@group(0) @binding(3) var<storage, read> keyframes: array<f32>;
@group(0) @binding(4) var<storage, read> masks: array<u32>;
@group(0) @binding(5) var<storage, read_write> conditioned: array<ConditionedGaussian>;
@group(0) @binding(6) var<storage, read_write> visible_ids: array<atomic<u32>>;

const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 = 0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 = 0.5462742152960396;
const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 = 1.445305721320277;
const SH_C3_6: f32 = -0.5900435899266435;
const PI: f32 = 3.141592653589793;

fn coeff(ao: u32, i: u32, max_coeff: u32) -> vec3<f32> {
  if (i >= max_coeff) { return vec3<f32>(0.0); }
  let o = ao + i * 3u;
  return vec3<f32>(appearance[o], appearance[o + 1u], appearance[o + 2u]);
}

fn eval_sh_color(ao: u32, max_coeff: u32, degree: u32, degree_t: u32, force_sh3d: bool, pos: vec3<f32>, campos: vec3<f32>, mu_t: f32, timestamp: f32, duration: f32) -> vec3<f32> {
  var dir = pos - campos;
  dir = dir / max(length(dir), 1e-8);
  let x = dir.x; let y = dir.y; let z = dir.z;
  let l0m0 = SH_C0;
  var result = l0m0 * coeff(ao, 0u, max_coeff);

  var l1m1 = 0.0; var l1m0 = 0.0; var l1p1 = 0.0;
  var l2m2 = 0.0; var l2m1 = 0.0; var l2m0 = 0.0; var l2p1 = 0.0; var l2p2 = 0.0;
  var l3m3 = 0.0; var l3m2 = 0.0; var l3m1 = 0.0; var l3m0 = 0.0; var l3p1 = 0.0; var l3p2 = 0.0; var l3p3 = 0.0;

  if (degree > 0u) {
    l1m1 = -SH_C1 * y; l1m0 = SH_C1 * z; l1p1 = -SH_C1 * x;
    result = result + l1m1 * coeff(ao,1u,max_coeff) + l1m0 * coeff(ao,2u,max_coeff) + l1p1 * coeff(ao,3u,max_coeff);
    if (degree > 1u) {
      let xx=x*x; let yy=y*y; let zz=z*z; let xy=x*y; let yz=y*z; let xz=x*z;
      l2m2 = SH_C2_0 * xy; l2m1 = SH_C2_1 * yz; l2m0 = SH_C2_2 * (2.0*zz - xx - yy); l2p1 = SH_C2_3 * xz; l2p2 = SH_C2_4 * (xx - yy);
      result = result + l2m2*coeff(ao,4u,max_coeff) + l2m1*coeff(ao,5u,max_coeff) + l2m0*coeff(ao,6u,max_coeff) + l2p1*coeff(ao,7u,max_coeff) + l2p2*coeff(ao,8u,max_coeff);
      if (degree > 2u) {
        l3m3 = SH_C3_0 * y * (3.0*xx - yy); l3m2 = SH_C3_1 * xy * z; l3m1 = SH_C3_2 * y * (4.0*zz - xx - yy); l3m0 = SH_C3_3 * z * (2.0*zz - 3.0*xx - 3.0*yy); l3p1 = SH_C3_4 * x * (4.0*zz - xx - yy); l3p2 = SH_C3_5 * z * (xx - yy); l3p3 = SH_C3_6 * x * (xx - 3.0*yy);
        result = result + l3m3*coeff(ao,9u,max_coeff) + l3m2*coeff(ao,10u,max_coeff) + l3m1*coeff(ao,11u,max_coeff) + l3m0*coeff(ao,12u,max_coeff) + l3p1*coeff(ao,13u,max_coeff) + l3p2*coeff(ao,14u,max_coeff) + l3p3*coeff(ao,15u,max_coeff);
        if (!force_sh3d && degree_t > 0u) {
          let dt = mu_t - timestamp;
          let td = max(duration, 1e-8);
          let t1 = cos(2.0 * PI * dt / td);
          result = result + t1 * (l0m0*coeff(ao,16u,max_coeff) + l1m1*coeff(ao,17u,max_coeff) + l1m0*coeff(ao,18u,max_coeff) + l1p1*coeff(ao,19u,max_coeff) + l2m2*coeff(ao,20u,max_coeff) + l2m1*coeff(ao,21u,max_coeff) + l2m0*coeff(ao,22u,max_coeff) + l2p1*coeff(ao,23u,max_coeff) + l2p2*coeff(ao,24u,max_coeff) + l3m3*coeff(ao,25u,max_coeff) + l3m2*coeff(ao,26u,max_coeff) + l3m1*coeff(ao,27u,max_coeff) + l3m0*coeff(ao,28u,max_coeff) + l3p1*coeff(ao,29u,max_coeff) + l3p2*coeff(ao,30u,max_coeff) + l3p3*coeff(ao,31u,max_coeff));
          if (degree_t > 1u) {
            let t2 = cos(2.0 * PI * 2.0 * dt / td);
            result = result + t2 * (l0m0*coeff(ao,32u,max_coeff) + l1m1*coeff(ao,33u,max_coeff) + l1m0*coeff(ao,34u,max_coeff) + l1p1*coeff(ao,35u,max_coeff) + l2m2*coeff(ao,36u,max_coeff) + l2m1*coeff(ao,37u,max_coeff) + l2m0*coeff(ao,38u,max_coeff) + l2p1*coeff(ao,39u,max_coeff) + l2p2*coeff(ao,40u,max_coeff) + l3m3*coeff(ao,41u,max_coeff) + l3m2*coeff(ao,42u,max_coeff) + l3m1*coeff(ao,43u,max_coeff) + l3m0*coeff(ao,44u,max_coeff) + l3p1*coeff(ao,45u,max_coeff) + l3p2*coeff(ao,46u,max_coeff) + l3p3*coeff(ao,47u,max_coeff));
          }
        }
      }
    }
  }
  return max(result + vec3<f32>(0.5), vec3<f32>(0.0));
}

fn active_by_temporal_mask(id: u32, t: f32) -> bool {
  let key_count = frame.params1.w;
  if (key_count == 0u) { return true; }
  var left = 0u; var right = 0u; var i = 0u;
  loop { if (i >= key_count) { break; } if (keyframes[i] <= t) { left = i; } if (keyframes[i] >= t) { right = i; break; } i = i + 1u; }
  let words_per = (frame.params0.z + 31u) / 32u;
  let word_i = id / 32u;
  let mask = 1u << (id & 31u);
  let lword = masks[left * words_per + word_i];
  let rword = masks[right * words_per + word_i];
  return ((lword | rword) & mask) != 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  let n = frame.params0.z;
  if (id >= n) { return; }
  if (!active_by_temporal_mask(id, frame.camera_pos_time.w)) { conditioned[id].radius_flags = vec4<u32>(0u); return; }
  let g = gaussians[id];
  var c = condition_cov_mean(g, frame.camera_pos_time.w, frame.params2.z, frame.params2.w);
  if (c.mean3.w <= 0.05) { conditioned[id].radius_flags = vec4<u32>(0u); return; }

  let p_view = transform_point(frame.view, c.mean3.xyz);
  if (p_view.z <= 0.2) { conditioned[id].radius_flags = vec4<u32>(0u); return; }
  let p_hom = transform_point4(frame.view_proj, c.mean3.xyz);
  let inv_w = 1.0 / (p_hom.w + 0.0000001);
  let ndc = p_hom.xyz * inv_w;

  let cov3 = mat3x3<f32>(
    vec3<f32>(c.conic_opacity.x, c.conic_opacity.y, c.conic_opacity.z),
    vec3<f32>(c.conic_opacity.y, c.color_depth.x, c.color_depth.y),
    vec3<f32>(c.conic_opacity.z, c.color_depth.y, c.color_depth.z)
  );
  // Recompute full cov3 because c used temporary packed fields above cannot hold 6 values.
  var cov_full: mat3x3<f32>;
  if ((g.flags & FLAG_ROT_4D) != 0u) {
    let Sigma = cov4_cuda(g, frame.params2.z);
    let ct = max(Sigma[3][3], 1e-8);
    let cv = vec3<f32>(Sigma[0][3], Sigma[1][3], Sigma[2][3]);
    cov_full = mat3x3<f32>(
      vec3<f32>(Sigma[0][0], Sigma[0][1], Sigma[0][2]) - cv.x * cv / ct,
      vec3<f32>(Sigma[1][0], Sigma[1][1], Sigma[1][2]) - cv.y * cv / ct,
      vec3<f32>(Sigma[2][0], Sigma[2][1], Sigma[2][2]) - cv.z * cv / ct
    );
  } else {
    cov_full = cov3_from_scale_rot(g.scale4.xyz, frame.params2.z, g.q_left);
  }
  let cov2 = cov2d_cuda(c.mean3.xyz, cov_full, frame.view, frame.proj, frame.viewport.xy);
  let det = cov2.x * cov2.z - cov2.y * cov2.y;
  if (det == 0.0) { conditioned[id].radius_flags = vec4<u32>(0u); return; }
  let conic = vec3<f32>(cov2.z / det, -cov2.y / det, cov2.x / det);
  let mid = 0.5 * (cov2.x + cov2.z);
  let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
  let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
  let radius_f = ceil(3.0 * sqrt(max(lambda1, lambda2)));
  if (radius_f <= 0.4) { conditioned[id].radius_flags = vec4<u32>(0u); return; }
  let uv = vec2<f32>(ndc2pix(ndc.x, frame.viewport.x), ndc2pix(ndc.y, frame.viewport.y));

  let deg = min(frame.params3.x, 3u);
  let deg_t = min(frame.params3.y, 2u);
  let coeff_count = min(frame.params3.z, g.appearance_len / 3u);
  let force_sh3d = frame.params3.w != 0u;
  let period = max(1e-8, frame.params4.x);
  let rgb = eval_sh_color(g.appearance_offset, coeff_count, deg, deg_t, force_sh3d, c.mean3.xyz, frame.camera_pos_time.xyz, g.mean4.w, frame.camera_pos_time.w, period);

  c.mean3 = vec4<f32>(uv, radius_f, c.mean3.w);
  c.conic_opacity = vec4<f32>(conic, c.conic_opacity.w);
  c.color_depth = vec4<f32>(rgb, p_view.z);
  c.radius_flags = vec4<u32>(u32(radius_f), 1u, 0u, 0u);
  conditioned[id] = c;
  let slot = atomicAdd(&visible_ids[0], 1u) + 1u;
  visible_ids[slot] = id;
}
