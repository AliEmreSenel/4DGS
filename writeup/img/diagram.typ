

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
  #let thick = 1.8pt
  #let base = 0.6pt

  #let soft-thick = 0.8pt + black
  #let light-blue = rgb("#eaf3ff")
  #let light-orange = rgb("#fff1e6")

  #let C(body, bg: none) = table.cell(align: center + horizon, fill: bg)[#body]
  #let L(body, bg: none) = table.cell(align: left + horizon, fill: bg)[#body]

  #let X = table.cell(align: center + horizon, fill: rgb("#b6b8b6"))[
    #text(fill: black, weight: "bold")[ ]
  ]
  #let XG = table.cell(align: center + horizon, fill: rgb("#56aa68"))[
    #text(fill: white, weight: "bold")[]
  ]
  #let XB = table.cell(align: center + horizon, fill: rgb("#5698df"))[
    #text(fill: white, weight: "bold")[]
  ]
  #let XR = table.cell(align: center + horizon, fill: rgb("#df6f6f"))[
    #text(fill: white, weight: "bold")[]
  ]
  #let E = table.cell(align: center + horizon)[]

  #let KG(body) = box(
    fill: rgb("#56aa68"),
    inset: (x: 3pt, y: 3pt),
    radius: 4pt,
  )[#text(fill: white)[#body]]

  #let KB(body) = box(
    fill: rgb("#5698df"),
    inset: (x: 3pt, y: 3pt),
    radius: 4pt,
  )[#text(fill: white)[#body]]

  #let KR(body) = box(
    fill: rgb("#df6f6f"),
    inset: (x: 3pt, y: 3pt),
    radius: 4pt,
  )[#text(fill: white)[#body]]

  #let VH(body, bg: none) = table.cell(align: center + horizon, fill: bg)[
    #rotate(-90deg, reflow: true)[#body]
  ]

  #table(
    columns: (120pt,) + (auto,) * 28,
    stroke: base,
    align: center + horizon,
    inset: (x: 3pt, y: 3pt),
    fill: (x, y) => if (x == 0 and y >= 2) or (y == 0 and x > 0) {
      luma(230)
    } else {
      none
    },

    table.cell(
      rowspan: 2,
      inset: (x: 5pt, y: 3pt),
      stroke: none,
      align: left + horizon,
    )[
      #par(justify: false)[
        #KG[existing] #KR[modified] #KB[rewritten]
      ]
    ],

    table.cell(colspan: 9, align: center + horizon)[*Gaussians*],
    table.cell(colspan: 2, align: center + horizon)[*Init*],
    table.cell(colspan: 3, align: center + horizon)[*Compr*],
    table.cell(colspan: 3, align: center + horizon)[*Train*],
    table.cell(colspan: 8, align: center + horizon)[*Prune*],
    table.cell(colspan: 3, align: center + horizon)[*Render*],

    VH([4D], bg: light-blue),
    VH([3D], bg: light-blue),
    VH([Quaternion], bg: light-orange),
    VH([Rotation Matrix], bg: light-orange),
    VH([Isotropic], bg: light-blue),
    VH([Anisotropic], bg: light-blue),
    VH([RGB], bg: light-orange),
    VH([SH(1)], bg: light-orange),
    VH([SH(3)], bg: light-orange),

    VH([Random], bg: light-blue),
    VH([MegaSAM], bg: light-blue),

    VH([MLP Distillation], bg: light-blue),
    VH([K-means], bg: light-orange),
    VH([Spatial GPCC], bg: light-orange),

    VH([Uncertainty], bg: light-blue),
    VH([Batch in Time], bg: light-orange),
    VH([Voxelization], bg: light-blue),

    VH([Contribution], bg: light-orange),
    VH([Gradient Loss], bg: light-orange),
    VH([Spatio-Temporal], bg: light-blue),
    VH([Opacity], bg: light-blue),
    VH([One-shot], bg: light-orange),
    VH([Scheduled], bg: light-orange),
    VH([Densify], bg: light-blue),
    VH([Dropout], bg: light-orange),

    VH([Visibility Mask], bg: light-blue),
    VH([Sort-based], bg: light-orange),
    VH([Sort-free], bg: light-orange),

    L([*4DGS-Nat.*], bg: luma(230)),
    X, E, X, E, E, X, E, E, X, X, E, E, E, E, E, X, E, X, X, E, E, E, E, X, E, E, X, E,

    L([*1000FPS*], bg: luma(230)),
    X, E, X, E, E, X, E, E, X, X, E, E, E, E, E, X, E, E, E, E, E, X, E, E, E, X, X, E,

    L([*Instant4D*], bg: luma(230)),
    X, E, X, E, X, E, X, E, E, E, X, E, E, E, E, E, X, E, E, X, X, X, E, E, E, E, X, E,

    L([*MobileGS*], bg: luma(230)),
    E, X, E, X, E, X, E, X, X, E, E, X, X, X, E, E, X, X, E, X, E, E, E, E, E, E, E, X,

    L([*Usplat4D*], bg: luma(230)),
    X, E, X, E, E, X, E, E, X, E, E, E, E, E, X, E, X, E, E, E, E, X, E, E, E, E, X, E,

    L([*Omni-4DGS*], bg: luma(230)),
    XG, E, XG, E, XG, XR, XG, XG, XG, XG, E, XR, XR, XR, XR, XG, XR, E, XG, XB, XG, XG, XB, XG, XB, XB, XG, XR,

    // Inner-table outline.
    table.vline(x: 1, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 29, start: 1, end: 8, stroke: soft-thick),
    table.hline(y: 1, start: 1, end: 29, stroke: soft-thick),
    table.hline(y: 8, start: 1, end: 29, stroke: soft-thick),

    // Extended mutually-exclusive group boundaries.
    table.vline(x: 3, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 5, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 7, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 10, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 12, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 25, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 27, start: 1, end: 8, stroke: soft-thick),

    // Outline around bottom row.
    table.hline(y: 7, start: 0, end: 29, stroke: luma(100) + thick),
    table.hline(y: 8, start: 0, end: 29, stroke: luma(100) + thick),
    table.vline(x: 0, start: 7, end: 8, stroke: luma(100) + thick),
    table.vline(x: 29, start: 7, end: 8, stroke: luma(100) + thick),

    // Thick black separators before/after bottom-row color bands.
    table.vline(x: 1, start: 1, end: 8, stroke: black + thick),

    table.vline(x: 3, start: 1, end: 8, stroke: black + thick),

    table.vline(x: 5, start: 1, end: 8, stroke: black + thick),

    table.vline(x: 7, start: 1, end: 8, stroke: black + thick),

    table.vline(x: 10, start: 1, end: 8, stroke: black + thick),

    table.vline(x: 13, start: 1, end: 8, stroke: black + thick),
    table.vline(x: 15, start: 1, end: 8, stroke: black + thick),

    table.vline(x: 16, start: 1, end: 8, stroke: black + thick),
    table.vline(x: 17, start: 1, end: 8, stroke: black + thick),
    table.vline(x: 18, start: 1, end: 8, stroke: black + thick),

    table.vline(x: 20, start: 1, end: 8, stroke: black + thick),
    table.vline(x: 22, start: 1, end: 8, stroke: black + thick),

    table.vline(x: 24, start: 1, end: 8, stroke: black + thick),
    table.vline(x: 25, start: 1, end: 8, stroke: black + thick),
    table.vline(x: 26, start: 1, end: 8, stroke: black + thick),

    table.vline(x: 27, start: 1, end: 8, stroke: black + thick),
    table.vline(x: 29, start: 1, end: 8, stroke: black + thick),
  )
]
