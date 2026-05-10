#let contrib-table = [
  #let grid = 0.42pt + luma(220)
  #let group = 1.05pt + luma(135)
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
    #text(size: 10pt)[#body]
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
      columns: (0.62fr, 3fr, 3.45fr) + (0.72fr,) * 6,
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
        inset: (x: 3.5pt, y: 3.2pt),
        stroke: none,
        align: left + top,
      )[
        #stack(dir: ttb, spacing: 0pt)[
          Implementations in each architecture, and our codebase:
          #KG[existing] #KR[heavily modified] #KB[re-implemented]
        ]
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
      X, X, E, X, X, XR,

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

      SEC(3, [*Compress*]),

      C([SH], bg: head-orange),
      L([MLP Distillation], bg: head-orange),
      E, E, E, X, E, XR,

      table.cell(rowspan: 2, align: center + horizon, fill: head-orange)[Codebook],
      L([K-means ], bg: head-orange),
      E, E, E, X, E, XR,

      L([Spatial GPCC], bg: head-orange),
      E, E, E, X, E, XR,

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

      table.cell(rowspan: 2, align: center + horizon, fill: head-orange)[Criterion],
      L([Contribution], bg: head-orange),
      X, E, E, X, E, XG,

      L([Gradient Loss], bg: head-orange),
      X, E, E, E, E, XG,

      table.cell(rowspan: 2, align: center + horizon, fill: head-blue)[Quantile Filter],
      L([Spatio-Temporal], bg: head-blue),
      E, E, X, X, E, XB,

      L([Opacity], bg: head-blue),
      E, E, X, E, E, XG,

      table.cell(rowspan: 2, align: center + horizon, fill: head-orange)[Strategy],
      L([One-shot], bg: head-orange),
      E, X, X, E, X, XG,

      L([Scheduled], bg: head-orange),
      E, E, E, E, E, XB,

      table.cell(rowspan: 2, align: center + horizon, fill: head-blue)[Increase],
      L([Densify], bg: head-blue),
      X, E, E, E, E, XG,

      L([Edge-guided], bg: head-blue),
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
      table.hline(y: 18, start: 0, end: 9, stroke: group),
      table.hline(y: 27, start: 0, end: 9, stroke: group),
      table.hline(y: 30, start: 0, end: 9, stroke: group),

      table.vline(x: 1, start: 1, end: 30, stroke: group),
      table.vline(x: 3, start: 0, end: 30, stroke: group),
      table.vline(x: 9, start: 0, end: 30, stroke: group),

      // Subgroup boundaries.
      table.hline(y: 3, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 5, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 7, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 26, start: 1, end: 9, stroke: 0.75pt + luma(175)),
      table.hline(y: 28, start: 1, end: 9, stroke: 0.75pt + luma(175)),

      // Strong focus around proposed method.
      table.vline(x: 8, start: 0, end: 30, stroke: emph),
      table.vline(x: 9, start: 0, end: 30, stroke: emph),
      table.hline(y: 0, start: 8, end: 9, stroke: emph),
      table.hline(y: 30, start: 8, end: 9, stroke: emph),
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
      X, X, E, X, X, XR,

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
      L([Densify], bg: head-blue),
      X, E, E, E, E, XG,

      L([Edge-guided], bg: head-blue),
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

  = Ablation Results

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
    block(width: 100%)[
      #set text(size: 6.5pt)

      #let dark = rgb("#2c3e50")
      #let section = rgb("#f2f2f2")
      #let border = rgb("#cccccc")

      #table(
        columns: (
          0.7fr,
          0.7fr,
          0.7fr,
          0.7fr,
          0.7fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
        ),
        stroke: border,
        inset: 3pt,
        align: center + horizon,

        table.cell(colspan: 5, fill: section)[#text(weight: "bold")[ABLATIONS]],
        table.cell(colspan: 9, fill: section)[#text(weight: "bold")[VISUAL]],

        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Color]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Prune]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Drop]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[ESS]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[USplat]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Iter  ×10³]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[PSNR ↑]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[SSIM ↑]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[LPIPS ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[FPS ↑]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[VRAM ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[MB ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Gauss k ↓]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Train m ↓]],

        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [26.60],
        [0.958],
        [0.042],
        [1649],
        [53],
        [6.3],
        [25.3],
        [1m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [26.72],
        [0.958],
        [0.039],
        [1682],
        [54],
        [6.7],
        [27.1],
        [29m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [26.54],
        [0.958],
        [0.044],
        [1692],
        [53],
        table.cell(fill: rgb("#d9efdf"))[5.8],
        [23.6],
        [1m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [26.64],
        [0.959],
        [0.041],
        [1718],
        [53],
        [6.2],
        [25.0],
        [27m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [27.57],
        [0.962],
        [0.035],
        [1672],
        [53],
        [6.5],
        [26.2],
        [1m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [27.51],
        [0.960],
        [0.037],
        [1865],
        [54],
        [6.9],
        [28.1],
        [28m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [27.41],
        [0.961],
        [0.036],
        table.cell(fill: rgb("#d9efdf"))[1916],
        [53],
        [6.0],
        [24.3],
        [1m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [27.61],
        [0.961],
        [0.036],
        [1901],
        [53],
        [6.2],
        [25.0],
        [26m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [26.81],
        [0.959],
        [0.041],
        [1632],
        [53],
        [6.3],
        [25.7],
        [1m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [26.68],
        [0.958],
        [0.042],
        [1629],
        [54],
        [6.7],
        [27.2],
        [32m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [26.57],
        [0.958],
        [0.041],
        [1648],
        [53],
        [5.9],
        [23.9],
        [1m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [26.69],
        [0.959],
        [0.043],
        [1661],
        [53],
        [6.3],
        [25.6],
        [20m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [27.52],
        [0.961],
        [0.037],
        [1830],
        [53],
        [6.4],
        [25.8],
        [1m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [27.46],
        [0.961],
        [0.039],
        [1831],
        [54],
        [6.8],
        [27.6],
        [30m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [27.54],
        [0.962],
        [0.037],
        [1865],
        table.cell(fill: rgb("#d9efdf"))[53],
        [6.0],
        [24.1],
        [1m],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [27.40],
        [0.961],
        [0.037],
        [1630],
        [53],
        [6.3],
        [25.4],
        [19m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [28.98],
        [0.965],
        [0.035],
        [1302],
        [103],
        [50.0],
        [26.9],
        [2m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [28.82],
        [0.963],
        [0.035],
        [1281],
        [109],
        [52.6],
        [28.3],
        [30m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [28.77],
        [0.963],
        [0.038],
        [1380],
        [100],
        [46.0],
        [24.7],
        [2m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [28.90],
        [0.964],
        [0.037],
        [1370],
        [101],
        [48.1],
        [25.8],
        [28m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [30.17],
        table.cell(fill: rgb("#d9efdf"))[0.971],
        [0.028],
        [1448],
        [100],
        [47.7],
        [25.6],
        [1m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [30.19],
        [0.971],
        [0.028],
        [1428],
        [103],
        [50.0],
        [26.9],
        [28m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [30.02],
        [0.970],
        [0.028],
        [1535],
        [96],
        [44.2],
        [23.8],
        [1m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [30.03],
        [0.970],
        [0.028],
        [1543],
        [100],
        [45.6],
        [24.5],
        [26m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [28.91],
        [0.965],
        [0.034],
        [1300],
        [102],
        [49.2],
        [26.4],
        [2m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [28.88],
        [0.964],
        [0.036],
        [1271],
        [109],
        [51.9],
        [27.9],
        [30m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [28.81],
        [0.964],
        [0.039],
        [1357],
        [100],
        [45.7],
        [24.5],
        [2m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [28.97],
        [0.965],
        [0.035],
        [1334],
        [101],
        [48.9],
        [26.3],
        [21m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [30.11],
        [0.970],
        [0.029],
        [1449],
        [100],
        [46.6],
        [25.1],
        [1m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        [30.16],
        [0.971],
        table.cell(fill: rgb("#d9efdf"))[0.027],
        [1419],
        [102],
        [49.3],
        [26.5],
        [29m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [7],
        [30.03],
        [0.971],
        [0.028],
        [1415],
        [94],
        [43.5],
        [23.4],
        [1m],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [7],
        table.cell(fill: rgb("#d9efdf"))[30.23],
        [0.971],
        [0.028],
        [1512],
        [100],
        [45.3],
        [24.3],
        [19m],
      )
    ],
    caption: [ Ablations: bouncingballs, 7k iter, sort],
  )

  #figure(
    block(width: 100%)[
      #set text(size: 6.5pt)

      #let dark = rgb("#2c3e50")
      #let section = rgb("#f2f2f2")
      #let border = rgb("#cccccc")
      #let bad = rgb("#ffd6d6")
      #let bad-text = rgb("#b00020")

      #table(
        columns: (
          0.7fr,
          0.7fr,
          0.7fr,
          0.7fr,
          0.7fr,
          0.7fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
          0.65fr,
        ),
        stroke: border,
        inset: 3pt,
        align: center + horizon,

        table.cell(colspan: 6, fill: section)[#text(weight: "bold")[ABLATIONS]],
        table.cell(colspan: 9, fill: section)[#text(weight: "bold")[VISUAL]],

        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Isotropy]],
        table.cell(fill: dark)[#text(fill: white, weight: "bold")[Sorting]],
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

        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [28.30],
        [0.976],
        [0.032],
        [657],
        [101],
        [43.0],
        [175.7],
        [4m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [28.76],
        [0.978],
        [0.029],
        [651],
        [96],
        [40.7],
        [166.2],
        [4m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [28.92],
        [0.978],
        [0.029],
        [730],
        [98],
        [41.9],
        [171.3],
        [2m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.00],
        [0.978],
        [0.027],
        table.cell(fill: rgb("#d9efdf"))[732],
        [94],
        [39.2],
        [160.2],
        [2m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [29.72],
        [0.980],
        [0.026],
        [593],
        [95],
        [40.5],
        [165.4],
        [4m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.75],
        [0.980],
        [0.026],
        [588],
        [92],
        [38.2],
        [156.3],
        [4m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [29.73],
        [0.981],
        [0.024],
        [635],
        [95],
        [40.7],
        [166.4],
        [2m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [30.01],
        [0.981],
        [0.023],
        [623],
        [92],
        [38.4],
        [156.8],
        [2m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [29.22],
        [0.978],
        [0.031],
        [333],
        [477],
        [325.6],
        [175.2],
        [9m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.51],
        [0.979],
        [0.029],
        [359],
        [430],
        [288.2],
        [155.1],
        [8m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [28.98],
        [0.976],
        [0.031],
        [357],
        [476],
        [324.9],
        [174.8],
        [5m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.52],
        [0.979],
        [0.028],
        [373],
        [440],
        [297.6],
        [160.1],
        [5m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [30.24],
        [0.981],
        [0.025],
        [315],
        [467],
        [316.5],
        [170.3],
        [9m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [30.51],
        [0.982],
        [0.026],
        [328],
        [431],
        [291.3],
        [156.8],
        [9m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        table.cell(fill: rgb("#d9efdf"))[30.63],
        table.cell(fill: rgb("#d9efdf"))[0.983],
        table.cell(fill: rgb("#d9efdf"))[0.023],
        [341],
        [448],
        [301.4],
        [162.2],
        [5m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [30.54],
        [0.982],
        [0.023],
        [361],
        [415],
        [279.4],
        [150.3],
        [5m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [223],
        [105],
        [7.9],
        [30.7],
        [13m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [272],
        [105],
        [7.9],
        [30.7],
        [11m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [285],
        [105],
        [7.9],
        [30.7],
        [6m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [285],
        [105],
        [7.9],
        [30.7],
        [6m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [187],
        [152],
        [12.5],
        [50.0],
        [15m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [192],
        [152],
        [12.5],
        [50.0],
        [15m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [201],
        [152],
        [12.5],
        [50.0],
        [8m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [199],
        [152],
        [12.5],
        [50.0],
        [9m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [30.21],
        [0.978],
        [0.031],
        [68],
        [634],
        [252.9],
        [136.1],
        [76m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [30.28],
        [0.979],
        [0.031],
        [66],
        [648],
        [262.7],
        [141.4],
        [80m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [30.19],
        [0.978],
        [0.030],
        [70],
        [566],
        [238.7],
        [128.5],
        [40m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [30.09],
        [0.979],
        [0.031],
        [75],
        [560],
        [230.0],
        [123.8],
        [36m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [30.11],
        [0.978],
        [0.031],
        [59],
        [748],
        [287.4],
        [154.8],
        [91m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.97],
        [0.977],
        [0.031],
        [61],
        [730],
        [280.2],
        [150.9],
        [87m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [30.01],
        [0.978],
        [0.031],
        [57],
        [725],
        [285.8],
        [153.9],
        [45m],
        table.cell(fill: rgb("#fff4bf"))[aniso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [30.21],
        [0.977],
        [0.031],
        [62],
        [709],
        [278.6],
        [150.0],
        [44m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [28.28],
        [0.974],
        [0.035],
        [638],
        [80],
        [22.5],
        [173.0],
        [4m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [28.24],
        [0.974],
        [0.034],
        [621],
        [78],
        [21.8],
        [167.6],
        [4m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [28.54],
        [0.975],
        [0.033],
        [679],
        [80],
        [22.6],
        [173.5],
        [2m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [28.49],
        [0.975],
        [0.033],
        [681],
        [76],
        [20.9],
        [160.2],
        [2m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [28.38],
        [0.973],
        [0.040],
        [596],
        [77],
        [21.2],
        [163.0],
        [4m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [28.44],
        [0.974],
        [0.038],
        [591],
        [75],
        [19.9],
        [152.5],
        [4m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [28.37],
        [0.974],
        [0.036],
        [580],
        [78],
        [22.1],
        [169.8],
        [2m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [28.67],
        [0.975],
        [0.035],
        [600],
        table.cell(fill: rgb("#d9efdf"))[74],
        [19.6],
        [150.0],
        [2m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [29.11],
        [0.975],
        [0.034],
        [339],
        [449],
        [299.3],
        [171.6],
        [8m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.34],
        [0.976],
        [0.033],
        [340],
        [431],
        [287.2],
        [164.7],
        [8m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [29.15],
        [0.975],
        [0.034],
        [338],
        [456],
        [305.2],
        [175.0],
        [5m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.19],
        [0.976],
        [0.034],
        [354],
        [421],
        [279.2],
        [160.1],
        [5m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [29.34],
        [0.976],
        [0.034],
        [321],
        [448],
        [298.9],
        [171.4],
        [8m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.41],
        [0.976],
        [0.038],
        [333],
        [414],
        [274.4],
        [157.4],
        [8m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [29.40],
        [0.977],
        [0.032],
        [318],
        [447],
        [297.0],
        [170.3],
        [5m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.33],
        [0.977],
        [0.032],
        [329],
        [413],
        [273.9],
        [157.1],
        [5m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [276],
        [110],
        [4.4],
        [30.7],
        [10m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [269],
        [110],
        [4.4],
        [30.7],
        [10m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [280],
        [110],
        table.cell(fill: rgb("#d9efdf"))[4.4],
        [30.7],
        [6m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [287],
        [110],
        table.cell(fill: rgb("#d9efdf"))[4.4],
        [30.7],
        [6m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [191],
        [155],
        [6.7],
        [50.0],
        [14m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [178],
        [155],
        [6.7],
        [50.0],
        [13m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [197],
        [155],
        [6.7],
        [50.0],
        [7m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#fce4ec"))[RGB],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[15.60]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.843]],
        table.cell(fill: bad)[#text(fill: bad-text, weight: "bold")[0.523]],
        [194],
        [155],
        [6.7],
        [50.0],
        [7m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [28.78],
        [0.972],
        [0.040],
        [59],
        [682],
        [279.6],
        [160.5],
        [82m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.32],
        [0.974],
        [0.036],
        [60],
        [663],
        [263.7],
        [151.3],
        [80m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [28.89],
        [0.973],
        [0.040],
        [60],
        [628],
        [290.2],
        [166.6],
        [42m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#ffe0b2"))[prune+dense],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.22],
        [0.974],
        [0.036],
        [73],
        [535],
        [269.8],
        [154.9],
        [40m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [28.88],
        [0.972],
        [0.039],
        [57],
        [713],
        [269.9],
        [154.9],
        [83m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#d9efdf"))[yes],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.02],
        [0.973],
        [0.038],
        [61],
        [701],
        [263.4],
        [151.2],
        [81m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#d9efdf"))[yes],
        [20],
        [29.17],
        [0.974],
        [0.036],
        [62],
        [672],
        [261.9],
        [150.3],
        [42m],
        table.cell(fill: rgb("#e3f2fd"))[iso],
        table.cell(fill: rgb("#ffffff"))[sort_free],
        table.cell(fill: rgb("#ede7f6"))[SH3],
        table.cell(fill: rgb("#f2f2f2"))[none],
        table.cell(fill: rgb("#f2f2f2"))[no],
        table.cell(fill: rgb("#f2f2f2"))[no],
        [20],
        [29.27],
        [0.974],
        [0.034],
        [62],
        [679],
        [261.6],
        [150.1],
        [42m],
      )
    ],
    caption: [trex ablation results.],
  )
]
