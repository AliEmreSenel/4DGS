

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
