#import "template.typ": arkheion, arkheion-appendices

#let abstract = [
  3DGS represents scenes as learnable Gaussians, and 4DGS extends this to dynamic scenes over time. While these models enable high-quality rendering, they are still too heavy for mobile use. Recent work addresses this from different angles: native 4DGS defines the baseline dynamic formulation, 1000FPS speeds up rendering through pruning and visibility masks, Instant4D cuts training cost with a simpler representation and better initialization, MobileGS makes 3D Gaussian rendering compact enough for phones, and Usplat4D improves dynamic modeling with uncertainty-aware motion weighting. We build on these contributions to design a lightweight 4DGS pipeline for mobile devices. We validate our results through extensive ablations, and we describe the best architecture.
]

#show: arkheion.with(
  title: "OMNI-4DGS: Chimera model for fast, light and precise Video-to-Model Reconstruction",
  authors: (
    (name: "Ali Emre Senel", email: "ali.senel*"),
    (name: "Tebe Nigrelli", email: "tebe.nigrelli*"),
    (name: "Stefana Chiriac", email: "stefana.chiriac*"),
  ),

  abstract: abstract,
  keywords: ("4DGS-Native", "Scene Reconstruction", "Mobile Render"),
  date: [\ May 12, 2026],
)
#set cite(style: "chicago-author-date")
#show link: underline

= Introduction
#lorem(60)


#pagebreak()

#bibliography("bibliography.bib")

#show: arkheion-appendices
= Notes

We use ablations extensively to evaluate the performance of the model, since each architecture layer introduces errors which may affect the other components negatively. Consequently, the architecture is developed to be modular and exchangeable, for better awareness of each part.

- *Important Metrics*: measure performance in each architecture. This cannot be plain loss, since it varies between architectures.
  - Color Reconstruction Error
  - Memory Footprint
  - Training Time
  - Other SOTA?

There is a tradeoff between _reconstruction accuracy_ and _speed of training_ with _memory footprint_. If we don't care about training time, we can distill more to have renders that can run on phones.

Fewer parameters reduces train time and memory, at the cost of accuracy.

#pagebreak()

= List of Architecture Choices

#set math.equation(numbering: none)

#set text(10pt)

#let thick = 1.8pt
#let base = 0.6pt

#let g = rgb("#78f082")
#let r = rgb("#fb8686")
#let y = rgb("#fff4bf")

#let M(body, fill: none, top: false, bottom: false) = table.cell(
  fill: fill,
  stroke: (
    left: base,
    right: base,
    top: if top { thick } else { base },
    bottom: if bottom { thick } else { base },
  ),
)[#body]

#let N(body, top: false, bottom: false) = M(body, top: top, bottom: bottom)
#let G(body, top: false, bottom: false) = M(body, fill: g, top: top, bottom: bottom)
#let R(body, top: false, bottom: false) = M(body, fill: r, top: top, bottom: bottom)
#let Y(body, top: false, bottom: false) = M(body, fill: y, top: top, bottom: bottom)

#table(
  columns: (0.5fr, 2fr, 2fr, 1fr, 1fr, 1.2fr, 1fr, 1fr),
  stroke: base,
  inset: 6pt,
  fill: (x, y) => if y == 0 { luma(230) } else { none },

  [],
  [*Specialisation*],
  [*Structure \ Hyperparam.*],
  [*Easy \ Code*],
  [*Good \ Quality*],
  [*Low \ Memory*],
  [*Fast \ Train*],
  [*Fast \ Render*],

  table.cell(rowspan: 1)[#rotate(-90deg, reflow: true)[]],
  [Schedule], [High Number \ of Iterations], N([]), G([]), N([]), R([]), N([]),

  table.cell(rowspan: 7)[#rotate(-90deg, reflow: true)[*1. Gaussians*]],

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[Rotation],
  table.cell(stroke: (top: thick))[Quaternion],
  R([], top: true),
  N([], top: true),
  G([], top: true),
  G([], top: true),
  N([], top: true),

  [Rotation Matrix],
  G([], bottom: true),
  N([], bottom: true),
  R([], bottom: true),
  R([], bottom: true),
  N([], bottom: true),

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[Shape],
  table.cell(stroke: (top: thick))[Isotropic],
  G([], top: true),
  R([], top: true),
  G([], top: true),
  G([], top: true),
  G([], top: true),

  [Anisotropic],
  R([], bottom: true),
  G([], bottom: true),
  R([], bottom: true),
  R([], bottom: true),
  R([], bottom: true),

  table.cell(
    rowspan: 3,
    stroke: (left: thick, top: thick, bottom: thick),
  )[Color Basis],
  table.cell(stroke: (top: thick))[RGB],
  G([], top: true),
  R([], top: true),
  G([], top: true),
  G([], top: true),
  G([], top: true),

  [SH(1)],
  G([], bottom: true),
  N([], bottom: true),
  G([], bottom: true),
  G([], bottom: true),
  G([], bottom: true),

  table.cell(stroke: (bottom: thick))[SH(3)],
  G([]),
  G([]),
  R([]),
  R([]),
  R([]),

  // Force thick bottom border under SH(1), across Code..Render
  table.hline(y: 9, start: 3, end: 8, stroke: thick),

  table.cell(rowspan: 3)[#rotate(-90deg, reflow: true)[*2. Init*]],
  [Segmentation], [MegaSAM initialization], G([]), G([]), N([]), G([]), N([]),
  [Grid], [Voxel Size], G([]), R([]), G([]), G([]), G([]),
  [Confidence], [Uncertainty], R([]), G([]), R([]), R([]), N([]),

  table.cell(rowspan: 2)[#rotate(-90deg, reflow: true)[*3. Compress*]],
  [SH], [Spherical Harmonic Distillation], R([]), G([]), G([]), R([]), G([]),
  [Quantization], [Neural Vector Quantization], R([]), R([]), G([]), N([]), G([]),

  table.cell(rowspan: 3)[#rotate(-90deg, reflow: true)[*4. Train*]],
  [Pruning], [Contribution-based pruning], G([]), G([]), G([]), G([]), G([]),
  [Weighting], [Uncertainty weighing], R([]), G([]), R([]), R([]), N([]),
  [Sampling], [Batch Sampling in Time], G([]), G([]), [], R([]), N([]),

  table.cell(rowspan: 2)[#rotate(-90deg, reflow: true)[*5. Prune*]],
  [Deduplication],
  [Voxel Dedup\
    Spatio-Temporal],
  R([]),
  G([]),
  G([]),
  G([]),
  G([]),
  [Strategy], [One-shot], [], R([]), G([]), G([]), G([]),

  table.cell(rowspan: 4)[#rotate(-90deg, reflow: true)[*6. Render*]],

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[Rasterisation],
  table.cell(stroke: (top: thick))[Sort],
  G([], top: true),
  G([], top: true),
  R([], top: true),
  R([], top: true),
  R([], top: true),

  table.cell(stroke: (bottom: thick))[Sort-free],
  R([], bottom: true),
  R([], bottom: true),
  G([], bottom: true),
  G([], bottom: true),
  G([], bottom: true),

  // Force thick bottom border under Sort-free, across Code..Render
  table.hline(y: 21, start: 3, end: 8, stroke: thick),

  [Thresholding], [Opacity Threshold], G([]), N([]), G([]), [], G([]),
  [Loading], [Visibility Mask Loading], G([]), N([]), R([]), [], G([]),
)

#pagebreak()

#let g = rgb("#78f082")
#let r = rgb("#fb8686")
#let y = rgb("#fff4bf")

#let G(body) = table.cell(fill: g)[#body]
#let Y(body) = table.cell(fill: y)[#body]
#let R(body) = table.cell(fill: r)[#body]
#let GS(body, stroke) = table.cell(fill: g, stroke: stroke)[#body]
#let YS(body, stroke) = table.cell(fill: y, stroke: stroke)[#body]
#let RS(body, stroke) = table.cell(fill: r, stroke: stroke)[#body]

#table(
  columns: (0.5fr, 1fr, 1fr, 0.9fr, 1.1fr),
  stroke: base,
  inset: 6pt,
  fill: (x, y) => if y == 0 { luma(230) } else { none },

  [], [*Specialisation*], [*Structure*], [*Compatibility*], [*Implementation*],

  table.cell(rowspan: 1)[#rotate(-90deg, reflow: true)[]],
  [*Schedule*], G([High Iterations]), G([Much Testing]), G([Hyperparameters]),

  table.cell(rowspan: 7)[#rotate(-90deg, reflow: true)[*1. Gaussians*]],

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Rotation*: can be skipped but would need more gauss.],
  GS([Quaternion], (top: thick)),
  GS([No problem, common choice], (top: thick)),
  GS([Fewer params, faster training], (top: thick)),

  [Rotation Matrix],
  table.cell(stroke: (bottom: thick))[No problem],
  table.cell(stroke: (bottom: thick))[Probably easier],

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Shape*: remain in the computational graph and it's fine],
  YS([Isotropic (but then no rotation either)], (top: thick)),
  YS([Simplification], (top: thick)),
  YS([Fewer Params, faster training], (top: thick)),

  G([Anisotropic]),
  GS([Default Choice], (bottom: thick)),
  GS([Slower training], (bottom: thick)),

  table.cell(
    rowspan: 3,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Color Basis*: initialize from early point cloud ],
  YS([RGB], (top: thick)),
  YS([Simplest], (top: thick)),
  YS([Fewest Params], (top: thick)),

  Y([SH(1)]),
  Y([Simple]), Y([Intermediate]),

  GS([SH(3)], (bottom: thick)),
  GS([More Precise], (bottom: thick)),
  GS([More params], (bottom: thick)),

  table.hline(y: 9, start: 3, end: 5, stroke: thick),

  table.cell(rowspan: 1)[#rotate(-90deg, reflow: true)[*2. Init*]],
  [*Segmentation*:  use model to init gaussians],
  G([MegaSAM initialization]),
  G([Initialization (color + position)]),
  G([Recent models might be better]),

  table.cell(rowspan: 2)[#rotate(-90deg, reflow: true)[*3. Compress*]],
  [*Spherical Harmonics*: compression],
  Y([Spherical Harmonic Distillation]),
  Y([Pipeline Restructure]),
  Y([Additional loss component in the dictionary compression+]),
  [*Quantization*: at rest compression to run faster],
  R([Neural Vector Quantization]),
  R([Pipeline Restructure]),
  R([Train latent features + Time vector, use Huffman Encoding]),

  table.cell(rowspan: 2)[#rotate(-90deg, reflow: true)[*4. Train*: \
    No Densify, out \
    of performance]],
  [*Uncertainty weighing*],
  R([USPLAT Algorithm]),
  R([High: Loss reweighting]),
  R([Many added loss terms, algorithm steps]),
  [*Strategies*: Adam + Batch + slow increase of \#SH],
  G([Batch Sampling in Time]),
  G([Easy]),
  G([Batch training, general choice of optimizer]),

  table.cell(rowspan: 3)[#rotate(-90deg, reflow: true)[*5. Prune*]],

  table.cell(
    rowspan: 2,
    stroke: (left: thick + black, top: thick + black, bottom: thick + black),
  )[*Criterion*: many possibilities. Does not have to be one alone.],
  YS([Contribution-based pruning], (top: thick + black)),
  YS([Threshold tuning at render], (top: thick + black)),
  YS([Hyperparameters], (top: thick + black, right: thick + black)),

  GS(
    [Voxel Dedup\
      Spatio-Temporal],
    (bottom: thick + black),
  ),
  GS([One-shot Pruning], (bottom: thick + black)),
  GS([Write formula from clear instructions], (bottom: thick + black, right: thick + black)),

  [Strategy], G([One-shot]), G([Faster train, but more finetune]), G([Very straightforward]),

  table.cell(rowspan: 4)[#rotate(-90deg, reflow: true)[*6. Render*]],

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Rasterisation*: project 3D to 2D],
  YS([*Sort*: aggregate color back to front], (top: thick)),
  YS([Standard technique], (top: thick)),
  YS([Runtime Bottleneck], (top: thick)),

  GS([*Sort-free*: weighted sum biased by MLP], (bottom: thick)),
  GS([Pipeline Change + time feature], (bottom: thick)),
  GS([Altro train tiny MLPs], (bottom: thick)),

  table.hline(y: 21, start: 3, end: 5),

  [Thresholding],
  G([Opacity Threshold]),
  G([Highly]),
  G([Hyperparameters]),

  [Loading],
  R([Visibility Mask Loading]),
  R([Highly]),
  R([Additional Layer]),
)

#set page(
  paper: "a4",
  flipped: true,
)

#set par(justify: false)

= Contribution

#set text(size: 9pt)
#show math.equation: set text(size: 9pt)

#show math.equation.where(block: true): set align(left)
#show math.equation.where(block: true): set block(spacing: 0pt)

#let h1 = [*4DGS-Native*]
#let h2 = [*1000FPS*]
#let h3 = [*Instant4D*]
#let h4 = [*MobileGS*]
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
  $C(p,t,v)=sum^N T_i alpha_i c_i + T_(N+1)c_(b g)$ for color \
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
    alpha_i = o_i exp(- (Delta x_i^T Sigma_i^(-1) Delta x_i) /2)
  $
  with $T = product_j (1 - alpha_j)$ global transmittance.

  $w_i$ is depth-aware and depends on the
  view-dependent $phi_i$ depth, and Gaussian scale.
]
#let rendering-usplat = [
  Assumes pixel color at t: $C_t^p = sum T_(i,t)^P alpha_i c_i$, with L2 loss over image, to $sigma^2_(i,t)$ formula.
]

#table(
  columns: (0.5fr, 0.5fr, 1.5fr, 1.7fr, 1fr, 1.2fr),
  fill: (x, y) => if x == 0 or y == 0 { luma(230) },

  [], [#row-summary], [#row-encoding], [#row-training], [#row-changes], [#row-rendering],

  [#h1], [#summary-4dgs], [#encoding-4dgs], [#training-4dgs], [#changes-4dgs], [#rendering-4dgs],

  [#h2], [#summary-1000], [#encoding-1000], [#training-1000], [#changes-1000], [#rendering-instant],

  [#h3], [#summary-instant], [#encoding-instant], [#training-instant], [#changes-instant], [#rendering-1000],

  [#h4], [#summary-mobile], [#encoding-mobile], [#training-mobile], [#changes-mobile], [#rendering-mobile],

  [#h5], [#summary-usplat], [#encoding-usplat], [#training-usplat], [#changes-usplat], [#rendering-usplat],
)
