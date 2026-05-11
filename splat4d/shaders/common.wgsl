const BLOCK_X: u32 = 16u;
const BLOCK_Y: u32 = 16u;
const BLOCK_SIZE: u32 = 256u;
const PI: f32 = 3.141592653589793;
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
const FLAG_ISOTROPIC: u32 = 2u;
const FLAG_ROT_4D: u32 = 4u;

struct GaussianRecord {
  mean4: vec4<f32>,
  scale4: vec4<f32>,
  q_left: vec4<f32>,
  q_right: vec4<f32>,
  opacity: f32,
  appearance_offset: u32,
  appearance_len: u32,
  aux_offset: u32,
  aux_len: u32,
  flags: u32,
  _pad0: u32,
  _pad1: u32,
};

struct PreprocessedGaussian {
  xy_depth_radius: vec4<f32>,   // x, y, depth, radius
  conic_opacity: vec4<f32>,     // conic x, y, z, opacity
  color: vec4<f32>,             // rgb, unused
  rect: vec4<u32>,              // minx, miny, maxx, maxy
};

struct FrameUniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  view_proj: mat4x4<f32>,
  inv_view: mat4x4<f32>,
  camera_pos_time: vec4<f32>,
  viewport: vec4<f32>,
  background: vec4<f32>,
  params0: vec4<u32>, // width height gauss_count render_mode
  params1: vec4<u32>, // tile_x tile_y max_unused keyframe_count
  params2: vec4<f32>, // near far scale_modifier prefilter_var
  params3: vec4<u32>, // active_sh_degree active_sh_degree_t coeff_count force_sh_3d
  params4: vec4<f32>, // time_duration time_min time_max unused
  params5: vec4<u32>, // words_per_mask left_mask_idx right_mask_idx has_masks
  params6: vec4<f32>, // focal_x focal_y tan_fovx tan_fovy
};

fn transform_point_4x3(m: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> {
  let h = m * vec4<f32>(p, 1.0);
  return h.xyz;
}

fn transform_point_4x4(m: mat4x4<f32>, p: vec3<f32>) -> vec4<f32> {
  return m * vec4<f32>(p, 1.0);
}

fn ndc2pix(v: f32, s: u32) -> f32 {
  return ((v + 1.0) * f32(s) - 1.0) * 0.5;
}

fn qnorm(q: vec4<f32>) -> vec4<f32> {
  let n = max(length(q), 1e-20);
  return q / n;
}

fn cov3_from_scale_rot(scale: vec3<f32>, scale_modifier: f32, rot: vec4<f32>) -> mat3x3<f32> {
  let s = scale_modifier * scale;
  let q = qnorm(rot);
  let r = q.x; let x = q.y; let y = q.z; let z = q.w;
  let R = mat3x3<f32>(
    vec3<f32>(1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y + r*z),       2.0 * (x*z - r*y)),
    vec3<f32>(2.0 * (x*y - r*z),       1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z + r*x)),
    vec3<f32>(2.0 * (x*z + r*y),       2.0 * (y*z - r*x),       1.0 - 2.0 * (x*x + y*y))
  );
  let S = mat3x3<f32>(vec3<f32>(s.x,0.0,0.0), vec3<f32>(0.0,s.y,0.0), vec3<f32>(0.0,0.0,s.z));
  let M = S * R;
  return transpose(M) * M;
}

fn mat_l_cuda(rot: vec4<f32>) -> mat4x4<f32> {
  let q = qnorm(rot);
  let a=q.x; let b=q.y; let c=q.z; let d=q.w;
  return mat4x4<f32>(
    vec4<f32>( a, -b,  c, -d),
    vec4<f32>( b,  a, -d, -c),
    vec4<f32>(-c,  d,  a, -b),
    vec4<f32>( d,  c,  b,  a)
  );
}

fn mat_r_cuda(rot: vec4<f32>) -> mat4x4<f32> {
  let qv = qnorm(rot);
  let p=qv.x; let q=qv.y; let r=qv.z; let s=qv.w;
  return mat4x4<f32>(
    vec4<f32>( p, -q,  r,  s),
    vec4<f32>( q,  p, -s,  r),
    vec4<f32>(-r,  s,  p,  q),
    vec4<f32>(-s, -r, -q,  p)
  );
}

fn cov4_from_scale_rot(g: GaussianRecord, scale_modifier: f32) -> mat4x4<f32> {
  let sc = scale_modifier * g.scale4;
  let S = mat4x4<f32>(
    vec4<f32>(sc.x,0.0,0.0,0.0),
    vec4<f32>(0.0,sc.y,0.0,0.0),
    vec4<f32>(0.0,0.0,sc.z,0.0),
    vec4<f32>(0.0,0.0,0.0,sc.w)
  );
  let R = mat_r_cuda(g.q_right) * mat_l_cuda(g.q_left);
  let M = S * R;
  return transpose(M) * M;
}

fn cov2d_cuda(mean: vec3<f32>, cov3: mat3x3<f32>, frame: FrameUniforms) -> vec3<f32> {
  var t = transform_point_4x3(frame.view, mean);
  let W = f32(frame.params0.x);
  let H = f32(frame.params0.y);
  let focal_x = frame.params6.x;
  let focal_y = frame.params6.y;
  let tan_fovx = frame.params6.z;
  let tan_fovy = frame.params6.w;
  let limx = 1.3 * tan_fovx;
  let limy = 1.3 * tan_fovy;
  let z = t.z;
  let txtz = t.x / z;
  let tytz = t.y / z;
  t.x = clamp(txtz, -limx, limx) * z;
  t.y = clamp(tytz, -limy, limy) * z;
  let J = mat3x3<f32>(
    vec3<f32>(focal_x / z, 0.0, 0.0),
    vec3<f32>(0.0, focal_y / z, 0.0),
    vec3<f32>(-(focal_x * t.x) / (z*z), -(focal_y * t.y) / (z*z), 0.0)
  );
  let V = frame.view;
  let Wm = mat3x3<f32>(
    vec3<f32>(V[0][0], V[0][1], V[0][2]),
    vec3<f32>(V[1][0], V[1][1], V[1][2]),
    vec3<f32>(V[2][0], V[2][1], V[2][2])
  );
  let T = Wm * J;
  var cov = transpose(T) * transpose(cov3) * T;
  cov[0][0] = cov[0][0] + 0.3;
  cov[1][1] = cov[1][1] + 0.3;
  return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}

fn rect_from_xy_radius(xy: vec2<f32>, radius: u32, tile_x: u32, tile_y: u32) -> vec4<u32> {
  let r = i32(radius);
  let minx = u32(clamp(i32((xy.x - f32(r)) / f32(BLOCK_X)), 0, i32(tile_x)));
  let miny = u32(clamp(i32((xy.y - f32(r)) / f32(BLOCK_Y)), 0, i32(tile_y)));
  let maxx = u32(clamp(i32((xy.x + f32(r) + f32(BLOCK_X) - 1.0) / f32(BLOCK_X)), 0, i32(tile_x)));
  let maxy = u32(clamp(i32((xy.y + f32(r) + f32(BLOCK_Y) - 1.0) / f32(BLOCK_Y)), 0, i32(tile_y)));
  return vec4<u32>(minx, miny, maxx, maxy);
}

