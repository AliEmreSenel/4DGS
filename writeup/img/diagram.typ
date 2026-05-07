

#let timeline = ```dot-render
digraph gs_evolution {
  graph [
    rankdir=TB,
    bgcolor="white",
    pad=0.18,
    nodesep=0.4,
    ranksep=0.5,
    splines=ortho,
    outputorder=edgesfirst,
    fontname="Inter",
    newrank=true
  ];

  node [
    shape=box,
    style="rounded,filled",
    fillcolor="#dbeafe",
    color="#60a5fa",
    penwidth=1.5,
    fontname="Inter",
    fontsize=14,
    fixedsize=false,
    width=1.35,
    height=0.7,
    margin="0.08,0.04",
    labelloc=c
  ];

  edge [
    color="#64748b",
    penwidth=1.35,
    arrowsize=0.6,
    fontname="Inter",
    tailclip=true,
    headclip=true
  ];

  // ---------------- Papers ----------------

  gs_origin [group=g1991, label="Gaussian Splatting\nWestover"];
  gs3d      [group=g2023, label="3DGS\nKerbl et al."];

  native4d  [group=g2024, label="4DGS-Native\nYang et al."];
  sortfree  [group=g2024, label="Sort-free GS\nHou et al."];

  fps1k     [group=g2025, label="1000+ FPS\nYuan et al."];
  instant4d [group=g2025, label="Instant4D\nLuo et al."];
  dropout   [group=g2025, label="DropoutGS\nXu et al."];

  usplat    [group=g2026, label="USplat4D\nGuo et al."];
  mobile    [group=g2026, label="Mobile-GS\nDu et al."];

  // ---------------- Explicit vertical rows ----------------
  // y = 0 is top

  edge [style=invis, weight=100, constraint=true];

  {
    rank=same;
    native4d -> instant4d;
  }

  {
    rank=same;
    usplat;
  }

  {
    rank=same;
    gs_origin -> gs3d -> dropout;
  }

  {
    rank=same;
    sortfree -> mobile;
  }

  // Force row order
  native4d -> usplat;
  usplat -> dropout;
  dropout -> mobile;

  // Keep same-column vertical ordering stable
  native4d -> sortfree;
  fps1k -> instant4d -> dropout;
  usplat -> mobile;

  // ---------------- Evolution links ----------------

  edge [
    style=solid,
    constraint=false,
    color="#64748b",
    penwidth=1.35,
    arrowsize=0.6,
    tailclip=true,
    headclip=true
  ];

  gs_origin:e -> gs3d:w;

  gs3d:e -> native4d:w;

  gs3d:e -> sortfree:w;
  gs3d:e -> dropout:w;
  gs3d:e -> mobile:w;

  native4d:e -> fps1k:w;
  native4d:e -> instant4d:w;
  native4d:e -> usplat:w;
  sortfree:e -> mobile:w;

  // ---------------- Timeline ----------------

  node [
    shape=box,
    style="rounded,filled",
    fillcolor="#f8fafc",
    color="#cbd5e1",
    fontcolor="#334155",
    fontsize=15,
    fixedsize=true,
    width=0.72,
    height=0.34,
    margin="0.06,0.03",
    labelloc=c
  ];

  t1991 [group=g1991, label="1991"];
  t2023 [group=g2023, label="2023"];
  t2024 [group=g2024, label="2024"];
  t2025 [group=g2025, label="2025"];
  t2026 [group=g2026, label="2026"];

  edge [
    constraint=true,
    color="#334155",
    penwidth=2.0,
    arrowsize=0.75
  ];

  {
    rank=same;
    t1991 -> t2023 -> t2024 -> t2025 -> t2026;
  }

  // ---------------- Vertical sync lines ----------------

  edge [
    color="#94a3b8",
    penwidth=1.0,
    arrowsize=0,
    dir=none,
    constraint=true,
    weight=90
  ];

  gs_origin:s -> t1991:n;
  gs3d:s -> t2023:n;
  sortfree:s -> t2024:n;
  dropout:s -> t2025:n;
  mobile:s -> t2026:n;
}

```


#let ablations = [
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
  #let row-head = luma(248)
  #let row-alt = luma(252)
  #let omni-row = rgb("#eef5ef")

  #let Mark(fill) = table.cell(
    align: center + horizon,
    inset: (x: 1pt, y: 6.2pt),
  )[
    #rect(
      width: 10pt,
      height: 10pt,
      radius: 2.4pt,
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
    radius: 4pt,
    inset: (x: 8pt, y: 6pt),
  )[
    #text(fill: white, size: 12pt, weight: "bold")[#body]
  ]

  #let KG(body) = Pill(green, body)
  #let KB(body) = Pill(blue, body)
  #let KR(body) = Pill(red, body)

  #let Group(body, span) = table.cell(
    colspan: span,
    align: center + horizon,
    fill: head-gray,
    inset: (x: 3pt, y: 7pt),
  )[
    #text(size: 16.2pt, weight: "bold")[#body]
  ]

  #let VH(body, bg: none) = table.cell(
    align: center + horizon,
    fill: bg,
    inset: (x: 1pt, y: 8pt),
  )[
    #rotate(-90deg, reflow: true)[
      #text(size: 13.8pt)[#body]
    ]
  ]

  #let Method(body, bg: row-head) = table.cell(
    align: left + horizon,
    fill: bg,
    inset: (x: 4pt, y: 8pt),
  )[
    #text(size: 14.2pt, weight: "bold")[#body]
  ]

  #block(width: 100%, height: 100%)[
    #table(
      columns: (5fr,) + (0.91fr,) * 28,
      stroke: grid,
      align: center + horizon,
      inset: (x: 2pt, y: 6.4pt),

      table.cell(
        rowspan: 2,
        fill: none,
        stroke: none,
        inset: (x: 5pt, y: 5pt),
        align: left + top,
      )[
        #stack(dir: ttb, spacing: 0pt)[
          #text(size: 15.4pt, weight: "bold")[Legend]
          #KG[existing] #KR[modified] #KB[rewritten]
        ]
      ],

      Group([Gaussians], 9),
      Group([Init], 2),
      Group([Compress], 3),
      Group([Train], 3),
      Group([Prune], 8),
      Group([Render], 3),

      VH([4D], bg: head-blue),
      VH([3D], bg: head-blue),
      VH([Quaternion], bg: head-orange),
      VH([Rotation Matrix], bg: head-orange),
      VH([Isotropic], bg: head-blue),
      VH([Anisotropic], bg: head-blue),
      VH([RGB], bg: head-orange),
      VH([SH(1)], bg: head-orange),
      VH([SH(3)], bg: head-orange),

      VH([Random], bg: head-blue),
      VH([MegaSAM], bg: head-blue),

      VH([MLP Distillation], bg: head-blue),
      VH([K-means], bg: head-orange),
      VH([Spatial GPCC], bg: head-orange),

      VH([Uncertainty], bg: head-blue),
      VH([Batch in Time], bg: head-orange),
      VH([Voxelization], bg: head-blue),

      VH([Contribution], bg: head-orange),
      VH([Gradient Loss], bg: head-orange),
      VH([Spatio-Temporal], bg: head-blue),
      VH([Opacity], bg: head-blue),
      VH([One-shot], bg: head-orange),
      VH([Scheduled], bg: head-orange),
      VH([Densify], bg: head-blue),
      VH([Dropout], bg: head-orange),

      VH([Visibility Mask], bg: head-blue),
      VH([Sort-based], bg: head-orange),
      VH([Sort-free], bg: head-orange),

      Method([4DGS-Nat.]),
      X, E, X, E, E, X, E, E, X, X, E, E, E, E, E, X, E, X, X, E, X, E, E, X, E, E, X, E,

      Method([1000FPS], bg: row-alt),
      X, E, X, E, E, X, E, E, X, X, E, E, E, E, E, X, E, E, E, X, E, X, E, E, E, X, X, E,

      Method([Instant4D]),
      X, E, X, E, X, E, X, E, E, E, X, E, E, E, E, E, X, X, E, E, X, X, E, E, E, E, X, E,

      Method([MobileGS], bg: row-alt),
      E, X, E, X, E, X, E, X, X, E, E, X, X, X, E, E, X, X, E, E, E, E, E, E, E, E, E, X,

      Method([Usplat4D]),
      X, E, X, E, E, X, E, E, X, E, E, E, E, E, X, E, X, E, E, E, E, X, E, E, E, E, X, E,

      Method([Omni-4DGS], bg: omni-row),
      XG, E, XG, E, XR, XG, XG, XG, XG, XG, E, XR, XR, XR, XR, XG, XR, E, XG, XB, XG, XG, XB, XG, XB, XB, XG, XR,

      // Main section boundaries.
      table.vline(x: 1, start: 0, end: 8, stroke: group),
      table.vline(x: 10, start: 0, end: 8, stroke: group),
      table.vline(x: 12, start: 0, end: 8, stroke: group),
      table.vline(x: 15, start: 0, end: 8, stroke: group),
      table.vline(x: 18, start: 0, end: 8, stroke: group),
      table.vline(x: 26, start: 0, end: 8, stroke: group),
      table.vline(x: 29, start: 0, end: 8, stroke: group),

      // Header separation.
      table.hline(y: 1, start: 1, end: 29, stroke: 0.75pt + luma(175)),
      table.hline(y: 2, start: 0, end: 29, stroke: group),

      // Strong focus around proposed method.
      table.hline(y: 7, start: 0, end: 29, stroke: emph),
      table.hline(y: 8, start: 0, end: 29, stroke: emph),
      table.vline(x: 0, start: 7, end: 8, stroke: emph),
      table.vline(x: 29, start: 7, end: 8, stroke: emph),
    )
  ]
]
