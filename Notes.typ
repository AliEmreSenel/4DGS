#set page(
  paper: "a3",
  flipped: true,
  margin: (
    left: 1cm,
    right: 1cm,
    top: 1cm,
    bottom: 1cm,
  ),
)
#show math.equation.where(block: true): set align(left)
#show math.equation.where(block: true): set block(spacing: 0pt)

= Contribution from Papers

#let h1 = [*4DGS-Native*]
#let h2 = [*1000FPS*]
#let h3 = [*Instant4D*]
#let h4 = [*Mobile-3GS*]
#let h5 = [*Usplat4D*]

#let row-summary = [*Summary*]
#let row-encoding = [*Per-Gaussian \ Variables*]
#let row-training = [*Initialization, \ Training*]
#let row-changes = [*Changes  \ to the number \ of gaussians*]
#let row-rendering = [*Rendering*]

#let summary-4dgs = [Train 4D Gaussians directly]
#let summary-1000 = [Faster rendering through one-time prune at train, and visibility masks at render.]
#let summary-instant = [Fast train from having fewer vars and a better initialization.]
#let summary-mobile = [End-to-End training of 3DGS using no-sorting for render, and tiny MLP for fast render and small memory.]
#let summary-usplat = [Added training algorithm for weighing moving gaussians based on high-confidence reference stable gaussians.]

#let encoding-4dgs = [
  $mu in RR^4$ \
  $Sigma = S R in RR^(4 times 4)$ scaling + rotation, 2 x *rotation quaternions* \
  Color from 4D spherindrical harmonics
]
#let encoding-1000 = [
  Same as 4DGS.
  Spatio-Temporal score computed for one-time pruning.
  Bit mask for visibility at each frame. Used during render.

  *Spatio-Temporal score*: $S_i &= sum_t S^T_i S_i^S$ \ \

  $
    S_i^S & = sum_i alpha_i [product_(j=1)^(i-1)(1-alpha_j)] \
          & = "prevalence in time"
  $

  $
    S_i^T & = sum_alpha_i [product_(j=1)^(i-1)(1-alpha_j)] \
          & = "gaussian effect on image"
  $
]
#let encoding-instant = [
  $mu_i in RR^4$

  *Isotropic Gaussians* (in space).

  $Sigma =
  mat(
    Sigma_"xyz", 0;
    0, sigma_t,
  )
  \ "with"
  Sigma_"xyz" = S_"xyz" I_(3 x 3)$


  $Sigma=Sigma^T => Sigma_(x y z,t) = Sigma_(t, x y z)^T$

  Color is $(R,G,B)$ instead of SH.

]
#let encoding-mobile = [
  $mu_i in RR^3$

  $Sigma_i in RR^(3 times 3)$, with scale $s_i in RR^3$ and rotation $r_i in "SO"(3)$

  $o_i in [0,1]$ opacity

  $Y_i$ spherical harmonics coefficients

  View-dependent enhancement uses
  $P_i = (mu_i - t_v) / (||mu_i - t_v||)$,
  together with $s_i$, $r_i$, and $Y_i$.

  The MLP predicts
  $phi_i in RR_(>=0)$ and $o_i in [0,1]$.

  SH features compressed decomposed into
  diffuse $h_d in RR^3$ and view-dependent $h_v in RR^3$,
  then decoded by lightweight MLPs once at inference.
]
#let encoding-usplat = [
  Native 4DGS,

  $q$ quaternion for rotation,

  $alpha$ opacity, $c in RR^(N_k)$ colors.

  $mu_(x y z,t) in RR^3$

  Uncertainty at t: \ $sigma_(i,t)^2 = 1 / (sum_(p in P_t)(T_(i,t)alpha_i)^2)$

  Convergence at t: $II("all pixels converged to color")$.

  Scalar Uncertainty: $u_(i,t) = sigma_(i,t)^2 "if" II_i "else" K >> 1$

  Directional Uncertainty: $U_(i,t) = R_(w,c) U_c R_(w,c)^T$ from world-camera rotation and $U_c = u_(i,t)"diag"(r_x, r_y, r_z)$.
]

#let training-4dgs = [*Batch sampling in time* to reduce jitter]
#let training-1000 = [
  Same as 4DGS.
]
#let training-instant = [
  Use *#link("https://mega-sam.github.io/")[MegaSAM] model* to get camera intrisics, depth map and a good point initializations in space, with color. Use MegaSAM's moving binary label to initialize $Sigma_t^2$ as a large scalar if static, otherwise $S_t = 2 / "fps"$.

  $S_"xyz"$ is initialised to the voxel size.

  Compute during train:

  $mu_(x y z|t) = mu_(x y z) + Sigma_(x y z, t) / Sigma_(t t) (t - mu_t)$
]
#let training-mobile = [
  $L = L_"rgb" + lambda_"distill" L_"distill" + lambda_"depth" L_"depth"$

  The view-dependent enhancement MLP uses as input: \
  $P_i = (mu_i - t_v)/(||mu_i - t_v||) quad s_i in RR^3 quad r_i in "SO"(3) quad Y_i$

  *SH distillation* turns 3rd-order SH to 1st-order with a teacher-student setup for pixel color distillation, plus a scale-invariant depth distillation loss.
  Diffuse and view-dependent 3D components, then decoded by lightweight MLPs.

  *Neural vector quantization* of Gaussian attributes with multiple codebooks.
  Huffman coding is applied to the discrete codes at the end of training.
]
#let training-usplat = [
  $L=L_"RGB" + lambda_"key"L_"key" + lambda_"not-key"L_"not-key"$
  $L_"motion" = "dist." + "rigid SE(3)" + "smooth SO(3)" + "low acceleration"$

  Graph is partitioned into {key, non-key}: deduplicate by voxel
  1. *Deduplicate by voxels*.
  2. Keep top 2% long-living models (>5 frames)
  3. Assign Edge weights between Key Nodes and Attach non-key to its closest key.
  4. Optimize choice of Key nodes, Propagate Motion from key to non-key with DQB. Weigh loss by uncertainty matrix as $||||_U$

  $O = underbrace(N log N, "KD-tree") + underbrace(N T, "non-key assign") + underbrace(N k, "per-item optimise")$
]

#let changes-4dgs = [
  *Pruning and Densification* can both happen based on loss contribution, so gaussians are given more flexibility by splitting, or dropped if they don't contribute to loss.
]
#let changes-1000 = [
  *Spatio-temporal pruning* once. Drops % least important gaussians.

  Finetuning does not change the number of gaussians, only optimises them.
]
#let changes-instant = [
  *Grid Pruning* by placing gaussians into voxels (motion, position, timestamp, temporal scale) and dropping duplicates. 90% reduction.
]
#let changes-mobile = [
  *Contribution-based pruning* runs during training: prune only if it is low in both opacity and maximum spatial scale (quantile threshold). Accumulate pruning votes in train, remove only after repeatedly remaining below threshold.
]
#let changes-usplat = [
  During early train, the certainty estimator is biased, so it could be zero. The $II_i$ term forces high uncertainty, so the points do not become anchors.

  *Monucular has more depth uncertainty*.
]

#let rendering-4dgs = [
  $P(x)=P(x|t)P(t)$, then rasterize to 2D, \
  $C(p,t,v)=sum^N T_i (P,t)alpha_i (p,t)c_i (v, Delta t_i) + T_(N+1)c_(b g)$ for color \
  where contribution is $alpha_i(p,t)=o_i G^(2D)_i (p,t)G^t_i (t)$ and opacity o is learned
]
#let rendering-instant = [
  Compute *visibility masks every N frames*, for each gaussian, render the union of the previous and next known mask at each timeframe.
  Only load visible gaussians for each t being displayed.
]
#let rendering-1000 = [
  *Minimal Opacity Threshold* to drop gaussians.
]
#let rendering-mobile = [
  *Sort-free render* uses depth-aware order-independent blending:

  $
    C = (1 - T) (sum c_i alpha_i w_i) / (sum alpha_i w_i) + T C_"bg"
  $

  where
  $
    alpha_i = o_i exp(-1/2 Delta x_i^T Sigma_i^(-1) Delta x_i)
  $
  with $T = product_j (1 - alpha_j)$ global transmittance.

  $w_i$ is depth-aware and depends on the
  view-dependent $phi_i$ depth, and Gaussian scale.
]
#let rendering-usplat = [
  Assumes pixel color at t: $C_t^p = sum T_(i,t)^P alpha_i c_i$, with L2 loss over image, to $sigma^2_(i,t)$ formula.
]

#table(
  columns: 6,

  [], [#h1], [#h2], [#h3], [#h4], [#h5],

  [#row-summary], [#summary-4dgs], [#summary-1000], [#summary-instant], [#summary-mobile], [#summary-usplat],

  [#row-encoding], [#encoding-4dgs], [#encoding-1000], [#encoding-instant], [#encoding-mobile], [#encoding-usplat],

  [#row-training], [#training-4dgs], [#training-1000], [#training-instant], [#training-mobile], [#training-usplat],

  [#row-changes], [#changes-4dgs], [#changes-1000], [#changes-instant], [#changes-mobile], [#changes-usplat],

  [#row-rendering],
  [#rendering-4dgs],
  [#rendering-instant],
  [#rendering-1000],
  [#rendering-mobile],
  [#rendering-usplat],
)
