struct MlpLayer {
  in_dim: u32,
  out_dim: u32,
  weight_offset: u32,
  bias_offset: u32,
  activation: u32, // 0 none, 1 relu, 2 sigmoid
};

struct MlpMetaGpu {
  input_dim: u32,
  layer_count: u32,
  sh_dim: u32,
  aux_dim: u32,
  layer0: MlpLayer,
  layer1: MlpLayer,
  layer2: MlpLayer,
  phi: MlpLayer,
  opacity: MlpLayer,
};

@group(0) @binding(10) var<storage, read> mlp_weights: array<f32>;
@group(0) @binding(11) var<storage, read> mlp_meta: MlpMetaGpu;
var<private> mlp_scratch: array<f32, 512>;

fn mlp_weight(layer: MlpLayer, o: u32, i: u32) -> f32 {
  return mlp_weights[layer.weight_offset + o * layer.in_dim + i];
}

fn mlp_bias(layer: MlpLayer, o: u32) -> f32 {
  return mlp_weights[layer.bias_offset + o];
}

fn act(x: f32, a: u32) -> f32 {
  if (a == 1u) { return max(x, 0.0); }
  if (a == 2u) { return 1.0 / (1.0 + exp(-x)); }
  return x;
}

fn mlp_layer_eval(base: u32, src: u32, dst: u32, layer: MlpLayer) {
  var o = 0u;
  loop {
    if (o >= layer.out_dim) { break; }
    var v = mlp_bias(layer, o);
    var i = 0u;
    loop {
      if (i >= layer.in_dim) { break; }
      v = v + mlp_weight(layer, o, i) * mlp_scratch[src + i];
      i = i + 1u;
    }
    mlp_scratch[dst + o] = act(v, layer.activation);
    o = o + 1u;
  }
}

fn load_mobilegs_input(base: u32, sh_start: u32, sh_len: u32, viewdir: vec3<f32>, scale: vec3<f32>, rotation: vec4<f32>, time_features: vec3<f32>) {
  var p = 0u;
  var sh_norm = 1e-8;
  var i = 0u;
  loop {
    if (i >= sh_len || i >= mlp_meta.sh_dim) { break; }
    let v = appearance[sh_start + i];
    sh_norm = sh_norm + v*v;
    i = i + 1u;
  }
  sh_norm = sqrt(sh_norm);
  i = 0u;
  loop {
    if (i >= mlp_meta.sh_dim) { break; }
    var v = 0.0;
    if (i < sh_len) { v = appearance[sh_start + i] / sh_norm; }
    mlp_scratch[base + p] = v;
    p = p + 1u;
    i = i + 1u;
  }
  mlp_scratch[base + p] = viewdir.x; p = p + 1u;
  mlp_scratch[base + p] = viewdir.y; p = p + 1u;
  mlp_scratch[base + p] = viewdir.z; p = p + 1u;
  mlp_scratch[base + p] = log(max(scale.x, 1e-8)); p = p + 1u;
  mlp_scratch[base + p] = log(max(scale.y, 1e-8)); p = p + 1u;
  mlp_scratch[base + p] = log(max(scale.z, 1e-8)); p = p + 1u;
  let r = normalize(rotation);
  mlp_scratch[base + p] = r.x; p = p + 1u;
  mlp_scratch[base + p] = r.y; p = p + 1u;
  mlp_scratch[base + p] = r.z; p = p + 1u;
  mlp_scratch[base + p] = r.w; p = p + 1u;
  mlp_scratch[base + p] = time_features.x; p = p + 1u;
  mlp_scratch[base + p] = time_features.y; p = p + 1u;
  mlp_scratch[base + p] = time_features.z;
}

fn mobilegs_opacity_phi(gaussian_id: u32, sh_start: u32, sh_len: u32, viewdir: vec3<f32>, scale: vec3<f32>, rotation: vec4<f32>, time_features: vec3<f32>) -> vec2<f32> {
  let base = 0u;
  let b0 = base;
  let b1 = base + 192u;
  let b2 = base + 320u;
  let b3 = base + 416u;
  load_mobilegs_input(b0, sh_start, sh_len, viewdir, scale, rotation, time_features);
  mlp_layer_eval(base, b0, b1, mlp_meta.layer0);
  mlp_layer_eval(base, b1, b2, mlp_meta.layer1);
  mlp_layer_eval(base, b2, b3, mlp_meta.layer2);
  var phi = mlp_bias(mlp_meta.phi, 0u);
  var op = mlp_bias(mlp_meta.opacity, 0u);
  var i = 0u;
  loop {
    if (i >= mlp_meta.phi.in_dim) { break; }
    phi = phi + mlp_weight(mlp_meta.phi, 0u, i) * mlp_scratch[b3 + i];
    op = op + mlp_weight(mlp_meta.opacity, 0u, i) * mlp_scratch[b3 + i];
    i = i + 1u;
  }
  return vec2<f32>(max(phi, 0.0), 1.0 / (1.0 + exp(-op)));
}
