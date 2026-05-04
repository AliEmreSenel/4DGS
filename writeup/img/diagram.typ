

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
  #let soft-green = rgb("#5f9f6e")
  #let light-blue = rgb("#eaf3ff")
  #let light-orange = rgb("#fff1e6")

  #let C(body, bg: none) = table.cell(align: center + horizon, fill: bg)[#body]
  #let L(body, bg: none) = table.cell(align: left + horizon, fill: bg)[#body]
  #let X = table.cell(align: center + horizon, fill: rgb("#dff3df"))[
    #text(fill: black, weight: "bold")[×]
  ]
  #let E = table.cell(align: center + horizon)[]

  #let VH(body) = table.cell(align: center + horizon)[
    #rotate(-90deg, reflow: true)[#body]
  ]

  #let SEC(rows, body) = table.cell(
    rowspan: rows,
    align: center + horizon,
  )[#rotate(-90deg, reflow: true)[#body]]

  #let MGL(rows, body, bg: none) = table.cell(
    rowspan: rows,
    align: center + horizon,
    stroke: soft-thick,
    fill: bg,
  )[#body]

  #let MGO(body, bg: none) = table.cell(align: left + horizon, fill: bg)[#body]

  #table(
    columns: (120pt,) + (auto,) * 29,
    stroke: base,
    align: center + horizon,
    inset: (x: 3pt, y: 3pt),
    fill: (x, y) => if (x == 0 and y >= 3) or (y == 0 and x > 0) { luma(230) } else { none },

    table.cell(
      rowspan: 2,
      inset: (x: 5pt, y: 3pt),
      stroke: none,
      align: left + horizon,
    )[],

    table.cell(colspan: 9, align: center + horizon)[*Gaussians*],
    table.cell(colspan: 3, align: center + horizon)[*Init*],
    table.cell(colspan: 3, align: center + horizon)[*Compress*],
    table.cell(colspan: 3, align: center + horizon)[*Train*],
    table.cell(colspan: 8, align: center + horizon)[*Prune*],
    table.cell(colspan: 3, align: center + horizon)[*Render*],

    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[4D]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[3D]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[Quaternion]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[Rotation Matrix]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[Isotropic]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[Anisotropic]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[RGB]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[SH(1)]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[SH(3)]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[Random]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[MegaSAM]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[Uncertainty]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[MLP Distillation]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[K-means ]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[Spatial GPCC]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[Uncertainty]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[Batch in Time]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[Voxelization]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[Contribution]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[Gradient Loss]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[Spatio-Temporal]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[Opacity]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[One-shot]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[Scheduled]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[Densify]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[Dropout]],
    table.cell(align: center + horizon, fill: light-blue)[#rotate(-90deg, reflow: true)[Visibility Mask]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[Sort-based]],
    table.cell(align: center + horizon, fill: light-orange)[#rotate(-90deg, reflow: true)[Sort-free]],

    L([*4DGS-Nat.*], bg: luma(230)),
    X, E, X, E, E, X, E, E, X, X, E, E, E, E, E, E, X, E, X, X, E, E, E, E, X, E, E, X, E,

    L([*1000FPS*], bg: luma(230)),
    X, E, X, E, E, X, E, E, X, X, E, E, E, E, E, E, X, E, E, E, E, E, X, E, E, E, X, X, E,

    L([*Instant4D*], bg: luma(230)),
    X, E, X, E, X, E, X, E, E, E, X, E, E, E, E, E, E, X, E, E, X, X, X, E, E, E, E, X, E,

    L([*MobileGS*], bg: luma(230)),
    E, X, E, X, E, X, E, X, X, E, E, E, X, X, X, E, E, X, X, E, X, E, E, E, E, E, E, E, X,

    L([*Usplat4D*], bg: luma(230)),
    X, E, X, E, E, X, E, E, X, E, E, X, E, E, E, X, E, X, E, E, E, E, X, E, E, E, E, X, E,

    L([*Omni-4DGS*], bg: luma(230)),
    X, E, X, E, X, X, X, X, X, X, E, X, X, X, X, X, X, X, E, X, X, X, X, X, X, X, X, X, X,

    // Inner-table outline.
    table.vline(x: 1, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 29, start: 1, end: 8, stroke: soft-thick),
    table.hline(y: 1, start: 1, end: 29, stroke: soft-thick),
    table.hline(y: 8, start: 1, end: 29, stroke: soft-thick),

    // Group boundaries.
    table.vline(x: 3, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 5, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 7, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 10, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 12, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 26, start: 1, end: 8, stroke: soft-thick),
    table.vline(x: 28, start: 1, end: 8, stroke: soft-thick),

    // Outline around bottom row.
    table.hline(y: 7, start: 0, end: 30, stroke: luma(100) + thick),
    table.hline(y: 8, start: 0, end: 30, stroke: luma(100) + thick),
    table.vline(x: 0, start: 7, end: 8, stroke: luma(100) + thick),
    table.vline(x: 30, start: 7, end: 8, stroke: luma(100) + thick),
  )
]
