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

  = List of Architecture Choices

  #set math.equation(numbering: none)

  #set text(10pt)
  #set par(justify: false)

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

    table.cell(rowspan: 7)[#rotate(-90deg, reflow: true)[*1. gaussians*]],

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

  #colbreak()

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

    table.cell(rowspan: 7)[#rotate(-90deg, reflow: true)[*1. gaussians*]],

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
    GS([*Sort*: aggregate color back to front], (top: thick)),
    GS([Standard technique], (top: thick)),
    GS([Runtime Bottleneck], (top: thick)),

    YS([*Sort-free*: weighted sum biased by MLP], (bottom: thick)),
    YS([Pipeline Change + time feature], (bottom: thick)),
    YS([Also train tiny MLPs], (bottom: thick)),

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

  #table(
    columns: (0.6fr, 0.6fr, 1.5fr, 1.7fr, 1fr, 1.2fr),
    fill: (x, y) => if x == 0 or y == 0 { luma(230) },

    [], [#row-summary], [#row-encoding], [#row-training], [#row-changes], [#row-rendering],

    [#h1], [#summary-4dgs], [#encoding-4dgs], [#training-4dgs], [#changes-4dgs], [#rendering-4dgs],

    [#h2], [#summary-1000], [#encoding-1000], [#training-1000], [#changes-1000], [#rendering-instant],

    [#h3], [#summary-instant], [#encoding-instant], [#training-instant], [#changes-instant], [#rendering-1000],

    [#h4], [#summary-mobile], [#encoding-mobile], [#training-mobile], [#changes-mobile], [#rendering-mobile],

    [#h5], [#summary-usplat], [#encoding-usplat], [#training-usplat], [#changes-usplat], [#rendering-usplat],
  )

  #let dark = rgb("#2c3e50")
  #let green = rgb("#d4edda")
  #let green-text = rgb("#1a5e30")
  #let section-bg = rgb("#f2f2f2")
  #let border = rgb("#cccccc")

  #let H(body) = table.cell(fill: dark)[#text(fill: white, weight: "bold", size: 10pt)[#body]]
  #let G(body) = table.cell(fill: green)[#text(fill: green-text, weight: "bold", size: 10pt)[#body]]
  #let N(body) = table.cell()[#text(size: 10pt)[#body]]
  #let S(body) = table.cell(fill: section-bg, colspan: 6)[#text(weight: "bold", size: 9pt)[ #body]]

  #let dark = rgb("#2c3e50")
  #let green = rgb("#d4edda")
  #let green-text = rgb("#1a5e30")
  #let section-bg = rgb("#f2f2f2")
  #let border = rgb("#cccccc")

  #let H(body) = table.cell(fill: dark)[#text(fill: white, weight: "bold", size: 10pt)[#body]]
  #let G(body) = table.cell(fill: green)[#text(fill: green-text, weight: "bold", size: 10pt)[#body]]
  #let N(body) = table.cell()[#text(size: 10pt)[#body]]
  #let S(body) = table.cell(fill: section-bg, colspan: 6)[#text(weight: "bold", size: 9pt)[ #body]]

  #figure(
    table(
      columns: (2fr, 1.5fr, 1.5fr, 1.3fr, 1.3fr, 1.5fr),
      stroke: border,
      inset: 6pt,
      align: center + horizon,

      H[Metric], H[Ours (quality)], H[Ours (efficient)], H[4DGS], H[1000FPS], H[1000FPS-PP],

      S[Reconstruction quality], N[PSNR ↑], G[30.63], N[30.01], N[29.85], N[30.47],
      N[30.61], N[SSIM ↑], G[0.983], N[0.981], N[0.9795], N[0.9811],
      N[0.9821], N[LPIPS ↓], N[0.023], N[0.023], N[0.019], N[0.018],
      N[0.017],

      S[Efficiency], N[FPS ↑], G[341], G[624], N[202], N[1361],
      N[1361], N[Storage (MB) ↓], G[301], G[38], N[792], N[118],
      N[16], N[Gaussians (k) ↓], G[162], G[157], N[1 265], N[189],
      N[189],

      S[Training], N[Train time ↓], G[4.9 min], G[2.0 min], N[~8 min], N[~30 min†],
      N[~30 min†],
    ),
    caption: [Comparison table],
  )

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
        [5.8],
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
        [1916],
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
        [53],
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
        [0.971],
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
        [0.027],
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
        [30.23],
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
]
