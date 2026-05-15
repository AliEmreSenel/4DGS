#let contrib-table = [
  #let grid = 0.42pt + luma(220)
  #let group = 1.05pt + luma(8.24%)
  #let emph = 1.9pt + luma(45)

  #let neutral = rgb("#9fa6a0")
  #let green = rgb("#42a464")
  #let blue = rgb("#438fd5")
  #let red = rgb("#d76363")

  #let head-blue = rgb("#eaf5ff")
  #let head-orange = rgb("#fff0e3")
  #let head-gray = luma(242)
  #let omni-row = rgb("#eef5ef")

  #let Mark(fill) = table.cell(
    align: center + horizon,
    inset: (x: 0.6pt, y: 3.2pt),
  )[
    #rect(
      width: 7.2pt,
      height: 7.2pt,
      radius: 1.8pt,
      fill: fill,
      stroke: none,
    )
  ]

  #let X = Mark(neutral)
  #let XG = Mark(green)
  #let XB = Mark(blue)
  #let XR = Mark(red)
  #let E = table.cell(align: center + horizon)[]

  #let Pill(fill, body) = rect(
    fill: fill,
    stroke: none,
    radius: 3pt,
    inset: (x: 5.5pt, y: 3.8pt),
  )[
    #text(fill: white, size: 9pt, weight: "bold")[#body]
  ]

  #let KG(body) = Pill(green, body)
  #let KB(body) = Pill(blue, body)
  #let KR(body) = Pill(red, body)

  #let C(body, bg: none) = table.cell(
    align: center + horizon,
    fill: bg,
    inset: (x: 2pt, y: 3.6pt),
  )[
    #text(size: 10pt, weight: "bold")[#body]
  ]

  #let L(body, bg: none) = table.cell(
    align: left + horizon,
    fill: bg,
    inset: (x: 2.5pt, y: 3.6pt),
  )[
    #text(size: 10pt)[#body]
  ]

  #let VH(body, bg: head-gray) = table.cell(
    align: center + horizon,
    fill: bg,
    inset: (x: 0.5pt, y: 4.8pt),
  )[
    #rotate(-90deg, reflow: true)[
      #text(size: 10.5pt)[#body]
    ]
  ]

  #let SEC(rows, body) = table.cell(
    rowspan: rows,
    align: center + horizon,
    fill: head-gray,
    inset: (x: 1pt, y: 4pt),
  )[
    #rotate(-90deg, reflow: true)[
      #text(size: 12pt, weight: "bold")[#body]
    ]
  ]

  #let MGL(rows, body, bg: none) = table.cell(
    rowspan: rows,
    align: center + horizon,
    fill: bg,
    inset: (x: 2pt, y: 3.6pt),
  )[
    #text(size: 10pt, weight: "bold")[#body]
  ]

  #let MGO(body, bg: none) = table.cell(
    align: left + horizon,
    fill: bg,
    inset: (x: 2.5pt, y: 3.6pt),
  )[
    #text(size: 10pt)[#body]
  ]

  #block(width: 100%)[
    #set text(size: 10pt)

    #table(
      columns: (0.62fr, 2.5fr, 3.45fr) + (0.72fr,) * 6,
      stroke: grid,
      align: center + horizon,
      inset: (x: 1.2pt, y: 3.2pt),
      fill: (x, y) => if x == 8 and y > 0 {
        omni-row
      } else if y == 0 and x >= 3 {
        head-gray
      } else if x == 0 and y > 0 {
        head-gray
      } else {
        none
      },

      table.cell(
        colspan: 3,
        inset: (x: 0pt, y: 0pt),
        stroke: none,
        align: left + top,
      )[
        #KG[existing] #KR[heavily modified] #KB[re-implemented]
      ],

      VH([*4DGS-Nat.*]),
      VH([*1000FPS*]),
      VH([*Instant4D*]),
      VH([*MobileGS*]),
      VH([*Usplat4D*]),
      VH([*Omni-4DGS*], bg: omni-row),

      SEC(9, [*Gaussians*]),

      MGL(2, [Gaussians], bg: head-blue),
      MGO([4D], bg: head-blue),
      X, X, X, E, X, XG,

      MGO([3D], bg: head-blue),
      E, E, E, X, E, E,

      MGL(2, [Rotation], bg: head-orange),
      MGO([Quaternion], bg: head-orange),
      X, X, X, E, X, XG,

      MGO([Rotation Matrix], bg: head-orange),
      E, E, E, X, E, E,

      MGL(2, [Shape], bg: head-blue),
      MGO([Isotropic], bg: head-blue),
      E, E, X, E, E, XG,

      MGO([Anisotropic], bg: head-blue),
      X, X, E, X, X, XG,

      MGL(3, [Color \ Basis], bg: head-orange),
      MGO([RGB], bg: head-orange),
      E, E, X, E, E, XG,

      MGO([SH(1)], bg: head-orange),
      E, E, E, X, E, XG,

      MGO([SH(3)], bg: head-orange),
      X, X, E, X, X, XG,

      SEC(2, [*Init*]),

      MGL(2, [Point \ Cloud], bg: head-blue),
      MGO([Random], bg: head-blue),
      X, X, E, E, E, XG,

      MGO([MegaSAM], bg: head-blue),
      E, E, X, E, E, E,

      SEC(3, [*Train*]),

      C([Weighting], bg: head-blue),
      L([Uncertainty], bg: head-blue),
      E, E, E, E, X, XR,

      C([Sampling], bg: head-orange),
      L([Batch in Time], bg: head-orange),
      X, X, E, E, E, XG,

      C([Grid \ Reliance], bg: head-blue),
      L([Voxelization], bg: head-blue),
      E, E, X, X, X, XR,

      SEC(9, [*Prune*]),

      MGL(2, [Criterion], bg: head-orange),
      L([Contribution], bg: head-orange),
      X, E, E, X, E, XG,

      L([Gradient Loss], bg: head-orange),
      X, E, E, E, E, XG,

      MGL(2, [Quantile Filter], bg: head-blue),
      L([Spatio-Temporal], bg: head-blue),
      E, E, X, X, E, XB,

      L([Opacity], bg: head-blue),
      E, E, X, E, E, XG,

      MGL(2, [Strategy], bg: head-orange),
      L([One-shot], bg: head-orange),
      E, X, X, E, X, XG,

      L([Scheduled], bg: head-orange),
      E, E, E, E, E, XB,

      MGL(2, [Increase], bg: head-blue),
      L([Gradient Based], bg: head-blue),
      X, E, E, E, E, XG,

      L([Edge-guided Split], bg: head-blue),
      E, E, E, E, E, XB,

      C([Dropout], bg: head-orange),
      L([Dropout], bg: head-orange),
      E, E, E, E, E, XB,

      SEC(3, [*Render*]),

      C([Loading], bg: head-blue),
      L([Visibility Mask], bg: head-blue),
      E, X, E, E, E, XB,

      MGL(2, [Raster], bg: head-orange),
      MGO([Sort-based], bg: head-orange),
      X, X, X, E, X, XG,

      MGO([Sort-free], bg: head-orange),
      E, E, E, X, E, XR,

      // Header separation.
      table.hline(y: 1, start: 0, end: 9, stroke: group),

      // Main section boundaries.
      table.hline(y: 10, start: 0, end: 9, stroke: group),
      table.hline(y: 12, start: 0, end: 9, stroke: group),
      table.hline(y: 15, start: 0, end: 9, stroke: group),
      table.hline(y: 24, start: 0, end: 9, stroke: group),
      table.hline(y: 27, start: 0, end: 9, stroke: group),

      table.vline(x: 1, start: 1, end: 27, stroke: group),
      table.vline(x: 3, start: 0, end: 27, stroke: group),
      table.vline(x: 9, start: 0, end: 27, stroke: group),

      // Subgroup boundaries.
      table.hline(y: 3, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 5, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 7, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 23, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 25, start: 1, end: 9, stroke: 0.75pt + luma(175)),

      // Strong focus around proposed method.
      table.vline(x: 8, start: 0, end: 27, stroke: emph),
      table.vline(x: 9, start: 0, end: 27, stroke: emph),
      table.hline(y: 0, start: 8, end: 9, stroke: emph),
      table.hline(y: 27, start: 8, end: 9, stroke: emph),
    )
  ]
]
#let contrib-table-large = [
  #let grid = 0.42pt + luma(220)
  #let group = 1.05pt + luma(8.24%)
  #let emph = 1.9pt + luma(45)

  #let neutral = rgb("#9fa6a0")
  #let green = rgb("#42a464")
  #let blue = rgb("#438fd5")
  #let red = rgb("#d76363")

  #let head-blue = rgb("#eaf5ff")
  #let head-orange = rgb("#fff0e3")
  #let head-gray = luma(242)
  #let omni-row = rgb("#eef5ef")

  #let Mark(fill) = table.cell(
    align: center + horizon,
    inset: (x: 5pt, y: 5pt),
  )[
    #rect(
      width: 25pt,
      height: 25pt,
      radius: 10pt,
      fill: fill,
      stroke: none,
    )
  ]

  #let X = Mark(neutral)
  #let XG = Mark(green)
  #let XB = Mark(blue)
  #let XR = Mark(red)
  #let E = table.cell(align: center + horizon)[]

  #let Pill(fill, body) = rect(
    fill: fill,
    stroke: none,
    radius: 3pt,
    inset: (x: 5.5pt, y: 3.8pt),
  )[
    #text(fill: white, size: 30pt, weight: "bold")[#body]
  ]

  #let KG(body) = Pill(green, body)
  #let KB(body) = Pill(blue, body)
  #let KR(body) = Pill(red, body)

  #let C(body, bg: none) = table.cell(
    align: center + horizon,
    fill: bg,
    inset: (x: 2pt, y: 3.6pt),
  )[
    #text(size: 25pt, weight: "bold")[#body]
  ]

  #let L(body, bg: none) = table.cell(
    align: left + horizon,
    fill: bg,
    inset: (x: 5pt, y: 5pt),
  )[
    #text(size: 25pt)[#body]
  ]

  #let VH(body, bg: head-gray) = table.cell(
    align: center + horizon,
    fill: bg,
    inset: (x: 0.5pt, y: 4.8pt),
  )[
    #rotate(-90deg, reflow: true)[
      #text(size: 25pt)[#body]
    ]
  ]

  #let SEC(rows, body) = table.cell(
    rowspan: rows,
    align: center + horizon,
    fill: head-gray,
    inset: (x: 1pt, y: 4pt),
  )[
    #rotate(-90deg, reflow: true)[
      #text(size: 32pt, weight: "bold")[#body]
    ]
  ]

  #let MGL(rows, body, bg: none) = table.cell(
    rowspan: rows,
    align: center + horizon,
    fill: bg,
    inset: (x: 2pt, y: 3.6pt),
  )[
    #text(size: 25pt, weight: "bold")[#body]
  ]

  #let MGO(body, bg: none) = table.cell(
    align: left + horizon,
    fill: bg,
    inset: (x: 2.5pt, y: 3.6pt),
  )[
    #text(size: 25pt)[#body]
  ]

  #block(width: 100%)[
    #set text(size: 25pt)

    #table(
      columns: (0.62fr, 2.5fr, 3.45fr) + (0.72fr,) * 6,
      stroke: grid,
      align: center + horizon,
      inset: (x: 1.2pt, y: 3.2pt),
      fill: (x, y) => if x == 8 and y > 0 {
        omni-row
      } else if y == 0 and x >= 3 {
        head-gray
      } else if x == 0 and y > 0 {
        head-gray
      } else {
        none
      },

      table.cell(
        colspan: 3,
        inset: (x: 0pt, y: 0pt),
        stroke: none,
        align: left + top,
      )[
        #KG[existing] #KR[heavily modified] #KB[re-implemented]
      ],

      VH([*4DGS-Nat.*]),
      VH([*1000FPS*]),
      VH([*Instant4D*]),
      VH([*MobileGS*]),
      VH([*Usplat4D*]),
      VH([*Omni-4DGS*], bg: omni-row),

      SEC(9, [*Gaussians*]),

      MGL(2, [Gaussians], bg: head-blue),
      MGO([4D], bg: head-blue),
      X, X, X, E, X, XG,

      MGO([3D], bg: head-blue),
      E, E, E, X, E, E,

      MGL(2, [Rotation], bg: head-orange),
      MGO([Quaternion], bg: head-orange),
      X, X, X, E, X, XG,

      MGO([Rotation Matrix], bg: head-orange),
      E, E, E, X, E, E,

      MGL(2, [Shape], bg: head-blue),
      MGO([Isotropic], bg: head-blue),
      E, E, X, E, E, XG,

      MGO([Anisotropic], bg: head-blue),
      X, X, E, X, X, XG,

      MGL(3, [Color \ Basis], bg: head-orange),
      MGO([RGB], bg: head-orange),
      E, E, X, E, E, XG,

      MGO([SH(1)], bg: head-orange),
      E, E, E, X, E, XG,

      MGO([SH(3)], bg: head-orange),
      X, X, E, X, X, XG,

      SEC(2, [*Init*]),

      MGL(2, [Point \ Cloud], bg: head-blue),
      MGO([Random], bg: head-blue),
      X, X, E, E, E, XG,

      MGO([MegaSAM], bg: head-blue),
      E, E, X, E, E, E,

      SEC(3, [*Train*]),

      C([Weighting], bg: head-blue),
      L([Uncertainty], bg: head-blue),
      E, E, E, E, X, XR,

      C([Sampling], bg: head-orange),
      L([Batch in Time], bg: head-orange),
      X, X, E, E, E, XG,

      C([Grid \ Reliance], bg: head-blue),
      L([Voxelization], bg: head-blue),
      E, E, X, X, X, XR,

      SEC(9, [*Prune*]),

      MGL(2, [Criterion], bg: head-orange),
      L([Contribution], bg: head-orange),
      X, E, E, X, E, XG,

      L([Gradient Loss], bg: head-orange),
      X, E, E, E, E, XG,

      MGL(2, [Quantile Filter], bg: head-blue),
      L([Spatio-Temporal], bg: head-blue),
      E, E, X, X, E, XB,

      L([Opacity], bg: head-blue),
      E, E, X, E, E, XG,

      MGL(2, [Strategy], bg: head-orange),
      L([One-shot], bg: head-orange),
      E, X, X, E, X, XG,

      L([Scheduled], bg: head-orange),
      E, E, E, E, E, XB,

      MGL(2, [Increase], bg: head-blue),
      L([Gradient Based], bg: head-blue),
      X, E, E, E, E, XG,

      L([Edge-guided Split], bg: head-blue),
      E, E, E, E, E, XB,

      C([Dropout], bg: head-orange),
      L([Dropout], bg: head-orange),
      E, E, E, E, E, XB,

      SEC(3, [*Render*]),

      C([Loading], bg: head-blue),
      L([Visibility Mask], bg: head-blue),
      E, X, E, E, E, XB,

      MGL(2, [Raster], bg: head-orange),
      MGO([Sort-based], bg: head-orange),
      X, X, X, E, X, XG,

      MGO([Sort-free], bg: head-orange),
      E, E, E, X, E, XR,

      // Header separation.
      table.hline(y: 1, start: 0, end: 9, stroke: group),

      // Main section boundaries.
      table.hline(y: 10, start: 0, end: 9, stroke: group),
      table.hline(y: 12, start: 0, end: 9, stroke: group),
      table.hline(y: 15, start: 0, end: 9, stroke: group),
      table.hline(y: 24, start: 0, end: 9, stroke: group),
      table.hline(y: 27, start: 0, end: 9, stroke: group),

      table.vline(x: 1, start: 1, end: 27, stroke: group),
      table.vline(x: 3, start: 0, end: 27, stroke: group),
      table.vline(x: 9, start: 0, end: 27, stroke: group),

      // Subgroup boundaries.
      table.hline(y: 3, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 5, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 7, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 23, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 25, start: 1, end: 9, stroke: 0.75pt + luma(175)),

      // Strong focus around proposed method.
      table.vline(x: 8, start: 0, end: 27, stroke: emph),
      table.vline(x: 9, start: 0, end: 27, stroke: emph),
      table.hline(y: 0, start: 8, end: 9, stroke: emph),
      table.hline(y: 27, start: 8, end: 9, stroke: emph),
    )
  ]
]

#let appendix = [

  #colbreak()

  = Ablation Results <ablations>

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
  #let row-encoding = [*Per-gaussian \ Variables*]
  #let row-training = [*Initialization, \ Training*]
  #let row-changes = [*Changes  \ to the number \ of gaussians*]
  #let row-rendering = [*Rendering*]

  #let summary-4dgs = [Train 4D gaussians directly]
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

    *Isotropic gaussians* (in space).

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

    Inherited from base model (SoM / MoSca):

    $mu_(x y u, t) in RR^3$ — 3D position at time t. This is what changes frame by frame as the Gaussian moves. It is the output of the base model's motion parameterization evaluated at each timestamp
    $q$ — quaternion encoding rotation
    $alpha in [0,1]$ — opacity
    $c in RR^(N^k)$ — color coefficients (spherical harmonics)

    Computed by USplat4D:
    Scalar uncertainty at frame t:

    $sigma_(i,t)^2 = 1 / sum_(p in P_(i,t)) (T_(i,t) dot alpha_i)^2$

    The more pixels strongly observe this Gaussian (high transmittance T, high opacity $alpha$), the bigger the denominator, the smaller the uncertainty. A Gaussian buried behind others or nearly transparent has a tiny denominator → huge uncertainty.
    Convergence check:

    $I_t = product_(h in Omega_(i,t)) l_t(h)$

    Equals 1 only if every pixel this Gaussian covers has color error below threshold ηc = 0.5. During early training the uncertainty estimate is biased and could be zero, so $II$ forces high uncertainty on unconverged Gaussians, preventing them from becoming anchors.
    Scalar uncertainty with convergence guard:

    $u_(i,t) = sigma_(i,t)^2$ if $bb(I)_i = 1$, else $K >> 1$

    Directional (anisotropic) uncertainty matrix:

    $U_(i,t) = R_(w c) dot "diag"(r_x u_(i,t), r_y u_(i,t), r_z u_(i,t)) dot R_(W C)^T$

    With [rx, ry, rz] = [1, 1, 0.01]. Monocular depth (z axis) is 100x less reliable than image-plane directions (x, y). This matrix appears as $U^(-1)$ in all loss terms — it down-weights gradient updates along uncertain directions and up-weights them along reliable ones.

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

    *Neural vector quantization* of gaussian attributes with multiple codebooks.
    Huffman coding is applied to the discrete codes at the end of training.
  ]
  #let training-usplat = [
    Stage 1 — Pretrain base model:
    Fully train SoM or MoSca to convergence. This produces the initial Gaussian positions p° and motion parameterization. USplat4D does not start from scratch — it always begins from a converged base model.
    Stage 2 — Graph construction (one time, before extra training):

    Deduplicate by voxels — remove spatially redundant Gaussians
    Key node selection — rank all Gaussians by uncertainty. Keep top 2% with lowest uncertainty AND observed confidently for >5 consecutive frames (significant period). These 1000 Gaussians become key nodes Vk — the reliable anchors
    Key-key edges (UA-kNN) — for each key node i, find k nearest neighbors among other key nodes using Mahalanobis distance weighted by uncertainty. Evaluated at t̂ = argmin_t uᵢ,ₜ (most reliable frame for each node). Unreliable nodes feel "farther away" — cross-object edges naturally don't form
    Non-key assignment — assign each non-key Gaussian to its single closest key node across the entire sequence: $j = op("argmin")_(l in V_k) sum_t norm(p_(i,t) - p_(l,t))_U$. Summing over all frames finds the key node that stays consistently close, not just close at one moment

    Stage 2 — Extra training iterations:
    Density control disabled for first 10% and last 20% of iterations to protect graph structure from new unassigned Gaussians being spawned.
    Total loss:

    $L = L_"RGB" + lambda_"key" L_"key" + lambda_"not-key" L_"not-key"$
    Key node loss:

    $L_"key" = sum_t sum_(i in V_K) norm(p_(i,t) - p_i^star)_(U^(-1)_(w,t,i))^2 + L_"motion,key"$


    Two parts: pull key nodes toward their pretrained positions p° (don't drift), weighted by U⁻¹ so depth direction gets weaker correction than image-plane directions. Plus motion locality constraints — isometry (distances between neighbors stay constant), rigidity (neighbors move together as rigid body), rotation smoothness (no sudden flips), low acceleration (velocity changes slowly).
    Non-key node loss:

    $L_"not-key" = sum_t sum_(i in.not V_K) norm(p_(i,t) - p_i^"DQB")_(U^(-1)_(w,t,i))^2 + L_"motion,not-key"$

    Three parts: weak pull toward pretrained position p° (weak because non-key uncertainty is high), pull toward DQB-interpolated position from key node anchor (main driver of motion), plus same motion locality constraints.
    DQB interpolation:

    $(p^"DQB"_(i,t), q^"DQB"_(i,t)) = "DQB"((w_(i,j), T_(j,t))_(j in E_i))$

    Blends the rigid SE(3) transformations of neighboring key nodes weighted by distance. p^DQB is NOT the final position — it is a soft target in L_non-key. The actual position pᵢ,ₜ remains a free variable and can deviate from p^DQB if the photometric loss strongly disagrees. This is why non-rigid motion is handled.
    The three roles of uncertainty in this framework:

    Re-weighting key node deviations — U⁻¹ controls how strongly key nodes are corrected per direction
    Guiding non-key interpolation — determines who becomes a key node, which determines who non-key nodes follow
    Balancing gradient updates — uncertain Gaussians get softer updates, reliable ones get stronger updates

    Complexity:

    O(N log N + NT + Nk)


    N log N → KD-tree for nearest neighbor search
    NT → assigning each non-key to closest key across T frames
    Nk → per-item optimization each iteration
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
    During early train, the certainty estimator is biased, so it could be zero. The $bb(I)_i$ term forces high uncertainty, so the points do not become anchors.

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
    view-dependent $phi_i$ depth, and gaussian scale.
  ]
  #let rendering-usplat = [
    Standard native 4DGS rendering — inherited entirely from the base model. No changes to the rendering pipeline.
    At render time: condition the 4D Gaussian to 3D at timestamp t:

    $mu_(x y u | t) = mu_(1:3) + Sigma_(1:3,4) dot Sigma_(4,4)^(-1) dot (t - mu_4)$

    Then project and alpha-blend as standard 3DGS:

    $C_t^p = sum_i T_i^p dot alpha_i dot c_i$



    with L2 loss over image → leads to the σ²ᵢ,ₜ formula for uncertainty estimation.
    USplat4D only changes how the Gaussians are trained, not how they are rendered or represented.
  ]


#show figure: set block(breakable: true)

#figure(
  caption: [bouncingballs],
  [
    #set text(size: 6.5pt)

    #let dark = rgb("#2c3e50")
    #let section = rgb("#f2f2f2")
    #let border = rgb("#cccccc")
    #let bad = rgb("#ffd6d6")
    #let bad-text = rgb("#b00020")

    #table(
      columns: (0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr),
      stroke: border,
      inset: 3pt,
      align: center + horizon,

      table.header(
        repeat: true,
        table.cell(colspan: 5, fill: section)[#text(weight: "bold")[ABLATIONS]],
        table.cell(colspan: 9, fill: section)[#text(weight: "bold")[VISUAL]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Isotropy]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Color]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Prune]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Drop]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[ESS]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Iter  ×10³]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[PSNR ↑]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[SSIM ↑]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[LPIPS ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[FPS ↑]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[VRAM ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[MB ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Gauss k ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Train m ↓]],
      ),

        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [11], [30.61], [0.971], [0.033], [889], [624], [4.5], table.cell(fill: rgb("#ffe0b2"))[54.5], table.cell(fill: rgb("#eaf6e9"))[34m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [8], [30.51], [0.972], [0.036], [516], [683], [3.2], table.cell(fill: rgb("#eaf6e9"))[38.8], table.cell(fill: rgb("#d9efdf"))[26m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [11], [29.82], [0.972], [0.031], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[400]], [544], [4.7], table.cell(fill: rgb("#ffe0b2"))[57.8], table.cell(fill: rgb("#fff4bf"))[51m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [12], [29.87], [0.970], [0.033], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[395]], [579], [6.1], table.cell(fill: rgb("#ffd6d6"))[74.6], table.cell(fill: rgb("#ffe0b2"))[62m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [11], [31.04], [0.967], [0.037], [662], [666], [4.6], table.cell(fill: rgb("#ffe0b2"))[55.9], table.cell(fill: rgb("#eaf6e9"))[39m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [8], [30.53], [0.966], [0.038], [757], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[718]], [3.3], table.cell(fill: rgb("#eaf6e9"))[39.7], table.cell(fill: rgb("#eaf6e9"))[27m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [9], [29.94], [0.967], [0.030], [522], [558], [3.7], table.cell(fill: rgb("#fff4bf"))[44.7], table.cell(fill: rgb("#fff4bf"))[47m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [11], [29.70], [0.965], [0.035], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[162]], [684], [5.8], table.cell(fill: rgb("#ffd6d6"))[70.7], table.cell(fill: rgb("#ffe0b2"))[58m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [8], [30.87], [0.969], [0.036], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[343]], [614], [3.0], table.cell(fill: rgb("#eaf6e9"))[36.3], table.cell(fill: rgb("#d9efdf"))[26m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [12], [30.55], [0.964], [0.038], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[406]], [672], [5.9], table.cell(fill: rgb("#ffd6d6"))[72.3], table.cell(fill: rgb("#fff4bf"))[42m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [9], [29.77], [0.966], [0.032], [789], [558], [3.7], table.cell(fill: rgb("#fff4bf"))[45.1], table.cell(fill: rgb("#fff4bf"))[46m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [9], [29.61], [0.965], [0.032], [600], [579], [4.1], table.cell(fill: rgb("#fff4bf"))[50.4], table.cell(fill: rgb("#fff4bf"))[47m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [9], [31.41], [0.974], [0.028], [2233], [608], [2.4], table.cell(fill: rgb("#d9efdf"))[29.0], table.cell(fill: rgb("#d9efdf"))[26m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [14], [31.38], [0.973], [0.031], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[369]], [656], [4.1], table.cell(fill: rgb("#ffe0b2"))[50.6], table.cell(fill: rgb("#fff4bf"))[45m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [11], [30.34], [0.974], [0.029], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[411]], [374], [2.8], table.cell(fill: rgb("#eaf6e9"))[34.0], table.cell(fill: rgb("#ffe0b2"))[63m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [12], [30.23], [0.973], [0.030], [560], [398], [4.0], table.cell(fill: rgb("#fff4bf"))[48.3], table.cell(fill: rgb("#ffd6d6"))[65m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [11], [33.98], [0.984], [0.022], [1098], [675], [24.4], table.cell(fill: rgb("#eaf6e9"))[39.3], table.cell(fill: rgb("#eaf6e9"))[37m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [10], [33.70], [0.983], [0.023], [626], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[741]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[26.0]], table.cell(fill: rgb("#fff4bf"))[41.9], table.cell(fill: rgb("#eaf6e9"))[36m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [12], [33.88], [0.984], [0.032], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[190]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[827]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[31.5]], table.cell(fill: rgb("#ffe0b2"))[50.8], table.cell(fill: rgb("#ffd6d6"))[69m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [10], [33.77], [0.984], [0.018], [444], [653], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[31.3]], table.cell(fill: rgb("#fff4bf"))[50.5], table.cell(fill: rgb("#ffe0b2"))[62m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [11], [34.05], [0.980], [0.021], [818], [676], [24.8], table.cell(fill: rgb("#eaf6e9"))[40.0], table.cell(fill: rgb("#eaf6e9"))[35m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [27], [34.11], [0.983], [0.017], [1196], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[698]], [15.5], table.cell(fill: rgb("#d9efdf"))[25.1], table.cell(fill: rgb("#ffd6d6"))[97m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [30], [33.90], [0.983], [0.018], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[132]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[773]], [15.8], table.cell(fill: rgb("#d9efdf"))[25.4], table.cell(fill: rgb("#ffd6d6"))[156m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [30], [33.86], [0.983], [0.016], [669], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[761]], [20.3], table.cell(fill: rgb("#d9efdf"))[32.7], table.cell(fill: rgb("#ffd6d6"))[167m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [10], [34.08], [0.979], [0.021], [544], [670], [23.0], table.cell(fill: rgb("#eaf6e9"))[37.1], table.cell(fill: rgb("#eaf6e9"))[32m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [10], [33.72], [0.979], [0.022], [682], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[733]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[27.1]], table.cell(fill: rgb("#fff4bf"))[43.7], table.cell(fill: rgb("#eaf6e9"))[34m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [11], [33.93], [0.979], [0.017], [416], [608], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[30.6]], table.cell(fill: rgb("#fff4bf"))[49.4], table.cell(fill: rgb("#ffe0b2"))[55m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [10], [33.82], [0.979], [0.018], [470], [618], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[32.5]], table.cell(fill: rgb("#ffe0b2"))[52.4], table.cell(fill: rgb("#fff4bf"))[53m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [12], [34.51], table.cell(fill: rgb("#d9efdf"))[0.986], table.cell(fill: rgb("#d9efdf"))[0.015], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[390]], [597], [17.2], table.cell(fill: rgb("#d9efdf"))[27.7], table.cell(fill: rgb("#eaf6e9"))[33m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [11], [34.42], [0.985], [0.016], [460], [621], [18.1], table.cell(fill: rgb("#d9efdf"))[29.2], table.cell(fill: rgb("#eaf6e9"))[35m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [15], [34.46], [0.986], [0.016], [417], [374], [24.4], table.cell(fill: rgb("#eaf6e9"))[39.3], table.cell(fill: rgb("#ffd6d6"))[83m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [12], table.cell(fill: rgb("#d9efdf"))[34.52], [0.986], [0.027], [429], [403], [25.0], table.cell(fill: rgb("#eaf6e9"))[40.4], table.cell(fill: rgb("#ffe0b2"))[62m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [6], [29.97], [0.971], [0.057], [2468], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[755]], [1.3], table.cell(fill: rgb("#d9efdf"))[28.5], table.cell(fill: rgb("#d9efdf"))[19m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [8], [30.12], [0.969], [0.052], [1222], [658], [2.0], table.cell(fill: rgb("#fff4bf"))[45.2], table.cell(fill: rgb("#d9efdf"))[26m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [8], [29.74], [0.972], [0.049], [503], [553], [2.1], table.cell(fill: rgb("#fff4bf"))[47.6], table.cell(fill: rgb("#fff4bf"))[41m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [11], [29.68], [0.970], [0.047], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[388]], [576], [3.0], table.cell(fill: rgb("#ffd6d6"))[68.8], table.cell(fill: rgb("#ffe0b2"))[63m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [6], [29.96], [0.964], [0.057], [2403], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[712]], [1.3], table.cell(fill: rgb("#d9efdf"))[29.9], table.cell(fill: rgb("#d9efdf"))[18m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [6], [29.92], [0.964], [0.058], [1384], [661], [1.4], table.cell(fill: rgb("#d9efdf"))[31.3], table.cell(fill: rgb("#d9efdf"))[20m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [9], [30.23], [0.967], [0.046], [464], [538], [2.5], table.cell(fill: rgb("#ffe0b2"))[56.2], table.cell(fill: rgb("#fff4bf"))[48m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [11], [29.51], [0.967], [0.048], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[395]], [571], [3.1], table.cell(fill: rgb("#ffd6d6"))[71.6], table.cell(fill: rgb("#ffd6d6"))[64m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [12], [29.92], [0.962], [0.054], [605], [666], [3.0], table.cell(fill: rgb("#ffd6d6"))[68.0], table.cell(fill: rgb("#fff4bf"))[43m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [10], [28.96], [0.965], [0.053], [1680], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[1581]], [2.6], table.cell(fill: rgb("#ffe0b2"))[59.1], table.cell(fill: rgb("#d9efdf"))[4m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [6], [29.46], [0.966], [0.056], [448], [535], [1.4], table.cell(fill: rgb("#d9efdf"))[32.6], table.cell(fill: rgb("#d9efdf"))[27m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [12], [29.75], [0.965], [0.047], [452], [572], [3.4], table.cell(fill: rgb("#ffd6d6"))[78.3], table.cell(fill: rgb("#ffe0b2"))[62m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [6], [30.02], [0.969], [0.058], [1878], [602], table.cell(fill: rgb("#d9efdf"))[1.0], table.cell(fill: rgb("#d9efdf"))[23.5], table.cell(fill: rgb("#d9efdf"))[18m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [9], [30.14], [0.967], [0.056], table.cell(fill: rgb("#d9efdf"))[2509], [665], [1.6], table.cell(fill: rgb("#eaf6e9"))[36.4], table.cell(fill: rgb("#eaf6e9"))[29m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [24], [29.89], [0.968], [0.055], [560], [345], [2.4], table.cell(fill: rgb("#ffe0b2"))[54.1], table.cell(fill: rgb("#ffd6d6"))[114m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [9], [30.10], [0.971], [0.058], [578], [370], [1.8], table.cell(fill: rgb("#eaf6e9"))[40.6], table.cell(fill: rgb("#fff4bf"))[45m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [8], [31.99], [0.974], [0.046], [626], [673], [22.9], table.cell(fill: rgb("#eaf6e9"))[39.4], table.cell(fill: rgb("#d9efdf"))[24m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [19], [32.07], [0.973], [0.043], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[273]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[802]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[46.2]], table.cell(fill: rgb("#ffd6d6"))[79.5], table.cell(fill: rgb("#ffe0b2"))[63m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [11], [32.23], [0.976], [0.040], [547], [636], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[33.1]], table.cell(fill: rgb("#ffe0b2"))[56.9], table.cell(fill: rgb("#ffe0b2"))[53m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [11], [32.21], [0.976], [0.040], [429], [653], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[33.6]], table.cell(fill: rgb("#ffe0b2"))[57.7], table.cell(fill: rgb("#ffe0b2"))[60m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [12], [31.98], [0.970], [0.045], [699], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[701]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[33.1]], table.cell(fill: rgb("#ffe0b2"))[56.9], table.cell(fill: rgb("#eaf6e9"))[40m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [17], [31.91], [0.968], [0.046], [817], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[809]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[44.9]], table.cell(fill: rgb("#ffd6d6"))[77.3], table.cell(fill: rgb("#ffe0b2"))[54m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [12], [32.04], [0.972], [0.040], [453], [639], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[35.9]], table.cell(fill: rgb("#ffd6d6"))[61.7], table.cell(fill: rgb("#ffd6d6"))[66m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [17], [32.06], [0.970], [0.041], [525], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[704]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[44.8]], table.cell(fill: rgb("#ffd6d6"))[77.1], table.cell(fill: rgb("#ffd6d6"))[94m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [8], [31.90], [0.970], [0.047], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[381]], [673], [23.8], table.cell(fill: rgb("#fff4bf"))[40.8], table.cell(fill: rgb("#eaf6e9"))[27m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [19], [32.00], [0.968], [0.046], [466], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[820]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[47.5]], table.cell(fill: rgb("#ffd6d6"))[81.7], table.cell(fill: rgb("#ffd6d6"))[67m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [11], [31.87], [0.971], [0.041], [552], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[912]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[34.1]], table.cell(fill: rgb("#ffe0b2"))[58.7], table.cell(fill: rgb("#ffe0b2"))[57m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [15], [31.99], [0.970], [0.042], [421], [687], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[41.1]], table.cell(fill: rgb("#ffd6d6"))[70.6], table.cell(fill: rgb("#ffd6d6"))[85m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [9], [31.95], [0.974], [0.045], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[412]], [584], [17.9], table.cell(fill: rgb("#d9efdf"))[30.8], table.cell(fill: rgb("#d9efdf"))[26m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [9], [31.99], [0.974], [0.045], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[341]], [617], [18.1], table.cell(fill: rgb("#d9efdf"))[31.1], table.cell(fill: rgb("#d9efdf"))[25m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [15], [31.99], [0.975], [0.042], [565], [313], [24.2], table.cell(fill: rgb("#fff4bf"))[41.6], table.cell(fill: rgb("#ffd6d6"))[74m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [10], [31.92], [0.975], [0.045], [578], table.cell(fill: rgb("#d9efdf"))[308], [21.5], table.cell(fill: rgb("#eaf6e9"))[36.9], table.cell(fill: rgb("#fff4bf"))[52m],
    )
  ],
)

#figure(
  caption: [trex],
  [
    #set text(size: 6.5pt)

    #let dark = rgb("#2c3e50")
    #let section = rgb("#f2f2f2")
    #let border = rgb("#cccccc")
    #let bad = rgb("#ffd6d6")
    #let bad-text = rgb("#b00020")

    #table(
      columns: (0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr),
      stroke: border,
      inset: 3pt,
      align: center + horizon,

      table.header(
        repeat: true,
        table.cell(colspan: 5, fill: section)[#text(weight: "bold")[ABLATIONS]],
        table.cell(colspan: 9, fill: section)[#text(weight: "bold")[VISUAL]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Isotropy]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Color]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Prune]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Drop]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[ESS]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Iter  ×10³]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[PSNR ↑]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[SSIM ↑]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[LPIPS ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[FPS ↑]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[VRAM ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[MB ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Gauss k ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Train m ↓]],
      ),

        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [8], [30.88], [0.982], [0.018], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[336]], [639], [6.4], table.cell(fill: rgb("#d9efdf"))[78.3], table.cell(fill: rgb("#eaf6e9"))[36m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [8], [30.82], [0.983], [0.017], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[379]], [705], [8.0], table.cell(fill: rgb("#fff4bf"))[97.8], table.cell(fill: rgb("#d9efdf"))[30m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [9], [30.99], [0.982], [0.018], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[314]], [626], [8.4], table.cell(fill: rgb("#fff4bf"))[102.6], table.cell(fill: rgb("#ffd6d6"))[118m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [8], [30.83], [0.982], [0.018], [594], [690], [8.6], table.cell(fill: rgb("#ffe0b2"))[105.3], table.cell(fill: rgb("#ffe0b2"))[102m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [9], [30.97], [0.983], [0.018], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[282]], [632], [7.4], table.cell(fill: rgb("#eaf6e9"))[90.9], table.cell(fill: rgb("#fff4bf"))[45m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [8], [30.86], [0.983], [0.017], table.cell(fill: rgb("#d9efdf"))[1474], [678], [8.0], table.cell(fill: rgb("#fff4bf"))[97.5], table.cell(fill: rgb("#d9efdf"))[31m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [8], [30.94], [0.982], [0.019], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[304]], [621], [6.7], table.cell(fill: rgb("#eaf6e9"))[82.2], table.cell(fill: rgb("#ffe0b2"))[109m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [8], [30.97], [0.982], [0.017], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[315]], [659], [8.8], table.cell(fill: rgb("#ffe0b2"))[107.4], table.cell(fill: rgb("#ffe0b2"))[96m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [9], [30.90], [0.983], [0.018], [395], [632], [7.5], table.cell(fill: rgb("#eaf6e9"))[91.7], table.cell(fill: rgb("#fff4bf"))[48m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [8], [30.82], [0.983], [0.017], [425], [680], [8.1], table.cell(fill: rgb("#fff4bf"))[99.2], table.cell(fill: rgb("#d9efdf"))[32m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [9], [31.03], [0.983], [0.018], [1140], [631], [8.2], table.cell(fill: rgb("#fff4bf"))[100.4], table.cell(fill: rgb("#ffd6d6"))[109m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [8], [31.01], [0.983], [0.017], [531], [674], [8.8], table.cell(fill: rgb("#ffe0b2"))[107.6], table.cell(fill: rgb("#ffe0b2"))[92m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [11], [31.56], [0.985], [0.016], [1005], [620], [6.2], table.cell(fill: rgb("#d9efdf"))[75.4], table.cell(fill: rgb("#fff4bf"))[52m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [10], [31.35], [0.984], [0.017], [709], [669], [7.5], table.cell(fill: rgb("#eaf6e9"))[91.3], table.cell(fill: rgb("#fff4bf"))[48m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [13], [31.51], [0.985], [0.017], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[359]], table.cell(fill: rgb("#d9efdf"))[365], [9.5], table.cell(fill: rgb("#ffe0b2"))[117.0], table.cell(fill: rgb("#ffd6d6"))[169m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [10], [31.36], [0.984], [0.017], [390], [370], [8.6], table.cell(fill: rgb("#fff4bf"))[104.9], table.cell(fill: rgb("#ffd6d6"))[127m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [8], [31.53], [0.983], [0.018], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[358]], [742], [44.6], table.cell(fill: rgb("#d9efdf"))[72.0], table.cell(fill: rgb("#eaf6e9"))[38m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [8], [31.57], [0.984], table.cell(fill: rgb("#d9efdf"))[0.015], [452], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[846]], [55.2], table.cell(fill: rgb("#eaf6e9"))[89.2], table.cell(fill: rgb("#eaf6e9"))[37m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [10], [31.55], [0.984], [0.016], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[254]], [742], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[72.0]], table.cell(fill: rgb("#ffe0b2"))[116.2], table.cell(fill: rgb("#ffd6d6"))[136m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [10], [31.76], [0.984], [0.016], [397], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[815]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[88.9]], table.cell(fill: rgb("#ffd6d6"))[143.5], table.cell(fill: rgb("#ffd6d6"))[133m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [8], [31.45], [0.983], [0.018], [404], [738], [43.4], table.cell(fill: rgb("#d9efdf"))[70.0], table.cell(fill: rgb("#eaf6e9"))[37m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [8], [31.36], [0.983], [0.017], [1046], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[834]], [55.7], table.cell(fill: rgb("#eaf6e9"))[90.0], table.cell(fill: rgb("#eaf6e9"))[37m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [10], [31.56], [0.984], [0.016], [410], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[797]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[71.5]], table.cell(fill: rgb("#ffe0b2"))[115.4], table.cell(fill: rgb("#ffd6d6"))[157m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [10], [31.55], [0.984], [0.016], [432], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[1161]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[90.3]], table.cell(fill: rgb("#ffd6d6"))[145.7], table.cell(fill: rgb("#ffd6d6"))[158m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [8], [31.43], [0.983], [0.017], [1387], [739], [43.7], table.cell(fill: rgb("#d9efdf"))[70.4], table.cell(fill: rgb("#eaf6e9"))[36m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [8], [31.53], [0.983], [0.016], [799], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[833]], [55.2], table.cell(fill: rgb("#eaf6e9"))[89.1], table.cell(fill: rgb("#eaf6e9"))[36m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [10], [31.77], [0.984], [0.016], [413], [765], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[71.5]], table.cell(fill: rgb("#ffe0b2"))[115.4], table.cell(fill: rgb("#ffd6d6"))[131m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [8], [31.66], [0.984], [0.017], [508], [735], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[62.4]], table.cell(fill: rgb("#fff4bf"))[100.8], table.cell(fill: rgb("#ffe0b2"))[103m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [13], [31.83], [0.984], [0.016], [422], [710], [50.1], table.cell(fill: rgb("#d9efdf"))[80.8], table.cell(fill: rgb("#fff4bf"))[57m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [10], [31.89], [0.984], [0.016], [386], [731], [50.2], table.cell(fill: rgb("#d9efdf"))[81.0], table.cell(fill: rgb("#eaf6e9"))[44m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [10], [31.84], [0.984], [0.020], [398], [587], [49.0], table.cell(fill: rgb("#d9efdf"))[79.1], table.cell(fill: rgb("#ffd6d6"))[142m],
        table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [10], table.cell(fill: rgb("#d9efdf"))[32.08], table.cell(fill: rgb("#d9efdf"))[0.985], [0.017], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[200]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[1246]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[58.3]], table.cell(fill: rgb("#eaf6e9"))[94.0], table.cell(fill: rgb("#ffd6d6"))[137m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [7], [27.88], [0.972], [0.037], [694], [668], [4.8], table.cell(fill: rgb("#ffe0b2"))[110.4], table.cell(fill: rgb("#eaf6e9"))[38m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [6], [27.79], [0.971], [0.039], [940], [720], [3.4], table.cell(fill: rgb("#d9efdf"))[78.0], table.cell(fill: rgb("#d9efdf"))[23m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [7], [28.12], [0.972], [0.038], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[289]], [782], [4.2], table.cell(fill: rgb("#fff4bf"))[96.7], table.cell(fill: rgb("#fff4bf"))[88m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [7], [27.90], [0.972], [0.036], [1421], [647], [6.1], table.cell(fill: rgb("#ffd6d6"))[141.1], table.cell(fill: rgb("#fff4bf"))[76m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [6], [27.81], [0.971], [0.042], [1114], [665], table.cell(fill: rgb("#d9efdf"))[2.5], table.cell(fill: rgb("#d9efdf"))[58.2], table.cell(fill: rgb("#d9efdf"))[28m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [6], [27.77], [0.971], [0.038], [1105], [761], [3.3], table.cell(fill: rgb("#d9efdf"))[76.7], table.cell(fill: rgb("#d9efdf"))[23m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [8], [28.05], [0.973], [0.034], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[305]], [615], [5.6], table.cell(fill: rgb("#ffd6d6"))[129.6], table.cell(fill: rgb("#ffe0b2"))[100m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [7], [27.99], [0.972], [0.035], [570], [638], [6.1], table.cell(fill: rgb("#ffd6d6"))[140.2], table.cell(fill: rgb("#fff4bf"))[78m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [7], [27.84], [0.972], [0.036], [453], [635], [4.7], table.cell(fill: rgb("#ffe0b2"))[107.7], table.cell(fill: rgb("#eaf6e9"))[34m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [7], [27.75], [0.972], [0.035], [985], [733], [6.3], table.cell(fill: rgb("#ffd6d6"))[144.9], table.cell(fill: rgb("#d9efdf"))[27m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [7], [28.10], [0.972], [0.038], [728], [581], [4.1], table.cell(fill: rgb("#eaf6e9"))[94.2], table.cell(fill: rgb("#fff4bf"))[80m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [6], [27.95], [0.972], [0.041], [402], [671], [3.3], table.cell(fill: rgb("#d9efdf"))[76.7], table.cell(fill: rgb("#fff4bf"))[67m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [11], [28.07], [0.974], [0.031], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[354]], [626], [7.5], table.cell(fill: rgb("#ffd6d6"))[173.3], table.cell(fill: rgb("#fff4bf"))[55m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [9], [27.89], [0.973], [0.032], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[298]], [685], [7.1], table.cell(fill: rgb("#ffd6d6"))[163.1], table.cell(fill: rgb("#eaf6e9"))[44m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [11], [28.14], [0.974], [0.030], [474], [455], [7.4], table.cell(fill: rgb("#ffd6d6"))[170.2], table.cell(fill: rgb("#ffd6d6"))[143m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [8], [28.13], [0.973], [0.033], [475], [794], [5.0], table.cell(fill: rgb("#ffe0b2"))[115.0], table.cell(fill: rgb("#fff4bf"))[90m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [7], [28.78], [0.974], [0.035], [397], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[807]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[59.0]], table.cell(fill: rgb("#fff4bf"))[101.5], table.cell(fill: rgb("#d9efdf"))[28m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [7], [28.76], [0.974], [0.033], [864], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[934]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[77.7]], table.cell(fill: rgb("#ffd6d6"))[133.7], table.cell(fill: rgb("#d9efdf"))[30m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [7], [28.92], [0.974], [0.036], [964], [715], [52.3], table.cell(fill: rgb("#eaf6e9"))[89.9], table.cell(fill: rgb("#ffe0b2"))[102m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[early], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [7], [28.91], [0.974], [0.034], [423], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[820]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[74.6]], table.cell(fill: rgb("#ffe0b2"))[128.4], table.cell(fill: rgb("#ffe0b2"))[102m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [7], [28.71], [0.973], [0.035], [490], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[1785]], [57.4], table.cell(fill: rgb("#fff4bf"))[98.7], table.cell(fill: rgb("#d9efdf"))[29m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [7], [28.79], [0.974], [0.033], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[206]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[942]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[77.0]], table.cell(fill: rgb("#ffd6d6"))[132.4], table.cell(fill: rgb("#d9efdf"))[29m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [7], [28.84], [0.974], [0.035], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[211]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[1561]], [51.4], table.cell(fill: rgb("#eaf6e9"))[88.4], table.cell(fill: rgb("#ffe0b2"))[102m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffffff"))[final], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [7], [29.07], [0.974], [0.033], [1366], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[2045]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[74.5]], table.cell(fill: rgb("#ffe0b2"))[128.1], table.cell(fill: rgb("#ffe0b2"))[107m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [7], [28.79], [0.973], [0.035], [567], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[1514]], [57.7], table.cell(fill: rgb("#fff4bf"))[99.3], table.cell(fill: rgb("#d9efdf"))[32m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [7], [28.85], [0.974], [0.033], [940], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[941]], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[77.4]], table.cell(fill: rgb("#ffd6d6"))[133.1], table.cell(fill: rgb("#eaf6e9"))[33m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [7], [28.85], [0.973], [0.036], [541], [681], [51.3], table.cell(fill: rgb("#eaf6e9"))[88.2], table.cell(fill: rgb("#ffe0b2"))[102m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [7], [28.95], [0.974], [0.034], [707], [796], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[75.0]], table.cell(fill: rgb("#ffd6d6"))[129.1], table.cell(fill: rgb("#ffe0b2"))[101m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [8], [28.70], [0.973], [0.034], [438], [672], [44.1], table.cell(fill: rgb("#d9efdf"))[75.9], table.cell(fill: rgb("#eaf6e9"))[37m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [7], [28.83], [0.973], [0.034], [394], [736], [51.6], table.cell(fill: rgb("#eaf6e9"))[88.7], table.cell(fill: rgb("#d9efdf"))[31m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [10], [28.64], [0.973], [0.032], [488], [588], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[72.5]], table.cell(fill: rgb("#ffe0b2"))[124.7], table.cell(fill: rgb("#ffd6d6"))[116m],
        table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [8], [28.91], [0.974], [0.033], [415], [710], table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[58.7]], table.cell(fill: rgb("#fff4bf"))[101.0], table.cell(fill: rgb("#ffe0b2"))[98m],
    )
  ],
)
]