#import "@preview/peace-of-posters:0.6.0" as pop

#set page("a0", margin: 1cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.uni-fr)

#let navy = rgb("#0b1026")
#let ink = rgb("#182033")
#let muted = rgb("#667085")
#let blue = rgb("#3466ff")
#let cyan = rgb("#18c7d9")
#let violet = rgb("#7c3aed")
#let green = rgb("#22a06b")
#let orange = rgb("#f97316")
#let red = rgb("#ef4444")
#let surface = rgb("#f7f9ff")
#let line = rgb("#d7def2")

#set text(size: pop.layout-a0.at("body-size"), fill: ink)
#set par(justify: false, leading: 0.62em)
#let box-spacing = 1.05em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)

// --- visual helpers ---------------------------------------------------------
#let tag(body, fill: surface, text-fill: blue) = box(
  fill: fill,
  stroke: 0.4pt + line,
  radius: 7pt,
  inset: (x: 7pt, y: 3.2pt),
)[#text(size: 0.72em, weight: "bold", fill: text-fill)[#body]]

#let callout(title, body, fill: surface, stroke: blue) = box(
  width: 100%,
  fill: fill,
  stroke: 1pt + stroke,
  radius: 12pt,
  inset: 13pt,
)[
  #text(weight: "bold", fill: stroke, size: 0.95em)[#title]
  #linebreak()
  #text(size: 0.82em)[#body]
]

#let metric-pill(name, direction, fill: surface, text-fill: ink) = box(
  fill: fill,
  radius: 7pt,
  inset: (x: 7pt, y: 3pt),
  stroke: 0.4pt + line,
)[#text(size: 0.7em, weight: "bold", fill: text-fill)[#name #direction]]

#let plot-placeholder(title, subtitle, height: 9.5cm, accent: blue) = box(
  width: 100%,
  height: height,
  fill: gradient.linear(surface, white, angle: 25deg),
  stroke: (paint: accent, thickness: 1.2pt, dash: "dashed"),
  radius: 14pt,
  inset: 12pt,
)[
  #align(center + horizon)[
    #text(size: 1.05em, weight: "bold", fill: accent)[#title]
    #linebreak()
    #text(size: 0.72em, fill: muted)[#subtitle]
    #linebreak()
    #v(0.6em)
    #text(size: 0.62em, fill: muted)[Replace this placeholder with the final high-resolution plot.]
  ]
]

#let mini-step(num, title, body, accent: blue) = grid(
  columns: (32pt, 1fr),
  gutter: 8pt,
  align: horizon,
  [#circle(fill: accent, inset: 5pt)[#text(fill: white, weight: "bold", size: 0.68em)[#num]]],
  [#text(weight: "bold", size: 0.86em)[#title] #linebreak() #text(size: 0.74em, fill: muted)[#body]],
)

#let check(fill) = box(width: 13pt, height: 13pt, fill: fill, radius: 3pt)
#let empty = box(width: 13pt, height: 13pt, stroke: 0.5pt + line, radius: 3pt)

#pop.title-box(
  [OMNI-4DGS: Fast, Light and Precise Video-to-Model Reconstruction],
  authors: [Ali Emre Senel¹ · Tebe Nigrelli¹ · Stefana Chiriac¹],
  institutes: [¹Bocconi University, Milan, Italy],
  keywords: [4D Gaussian Splatting · Dynamic Scene Reconstruction · Ablations · Mobile Rendering],
  logo: rect(width: 4.5cm, height: 4.5cm, radius: 50%, fill: gradient.radial(cyan, blue, violet), stroke: 1.4pt + white)[
    #align(center + horizon)[#text(fill: white, weight: "bold", size: 1.15em)[4DGS]]
  ],
)

#columns(3, [
  // ========================= LEFT THIRD: WRITEUP / REPORT ==================
  #pop.column-box(heading: "Motivation")[
    #callout(
      [Goal],
      [Build a single lightweight 4D Gaussian Splatting pipeline that keeps the quality of native 4DGS while reducing render cost, VRAM use, checkpoint size and Gaussian count.],
      fill: rgb("#eef5ff"),
      stroke: blue,
    )

    #v(0.4em)
    Dynamic 4DGS reconstructs a scene from video by learning Gaussian primitives over space and time. It enables novel-view and novel-time rendering, but current systems are still too heavy for practical low-memory or mobile inference.

    #v(0.35em)
    #tag([Quality], fill: rgb("#edf5ff"), text-fill: blue)
    #h(0.4em)
    #tag([Speed], fill: rgb("#effcf6"), text-fill: green)
    #h(0.4em)
    #tag([Memory], fill: rgb("#fff7ed"), text-fill: orange)
    #h(0.4em)
    #tag([Stability], fill: rgb("#f5f3ff"), text-fill: violet)
  ]

  #pop.column-box(heading: "4D Gaussian representation")[
    Each Gaussian is centered in space-time and carries opacity, shape, rotation and color. Native 4DGS uses a 4D covariance; compact variants reduce expressiveness to save memory and training cost.

    #v(0.25em)
    $
      G_i = (mu_i, Sigma_i, o_i, r_i, arrow("SH")_i),
      quad mu_i in RR^4,
      quad Sigma_i in RR^(4 times 4)
    $

    #v(0.2em)
    #table(
      columns: (0.9fr, 1.8fr),
      inset: 5pt,
      stroke: (x, y) => if y > 0 { (top: 0.35pt + line) },
      [#text(weight: "bold")[Component]], [#text(weight: "bold")[Poster-level message]],
      [4D mean], [where the primitive exists in space-time],
      [Covariance], [anisotropic detail vs. isotropic compactness],
      [Opacity], [visibility and pruning signal],
      [Color], [RGB for speed; SH(3) for fidelity],
    )
  ]

  #pop.column-box(heading: "Rendering and training objective")[
    Rendering first conditions the 4D Gaussian at time $t$, projects it to the camera plane, then alpha-composites all visible splats into pixels.

    #v(0.15em)
    $ C_p(t, v) = sum_i T_i(p,t) alpha_i(p,t)c_i(v,t) + T_(N+1)c_("bg") $

    #v(0.35em)
    #mini-step([1], [Photometric fit], [$cal(L)_"rgb" = (1-lambda)cal(L)_1 + lambda cal(L)_"SSIM"$], accent: blue)
    #mini-step([2], [Opacity masking], [discourages opaque background/sky Gaussians], accent: orange)
    #mini-step([3], [Motion regularization], [rigidity and global velocity terms suppress implausible dynamics], accent: violet)
  ]

  #pop.column-box(heading: "What we combine")[
    The poster should emphasize the *chimera* nature of the system: each prior codebase contributes a useful idea, but interactions matter more than isolated wins.

    #v(0.2em)
    #table(
      columns: (1fr, 1.35fr),
      inset: 4.5pt,
      stroke: (x, y) => if y > 0 { (top: 0.35pt + line) },
      [#text(weight: "bold")[Source]], [#text(weight: "bold")[Borrowed principle]],
      [4DGS-Native], [dynamic backbone + sort-based renderer],
      [Instant4D], [isotropic variants + spatio-temporal pruning],
      [MobileGS], [sort-free rendering + MLP compression],
      [1000FPS], [visibility masks + pruning schedule],
      [USplat4D], [uncertainty-aware motion loss],
    )
  ]

  #colbreak()

  // ========================= MIDDLE: MODEL / IMPLEMENTATION ===============
  #pop.column-box(heading: "Unified architecture")[
    #box(width: 100%, fill: gradient.linear(rgb("#eef5ff"), rgb("#f7f2ff"), angle: 18deg), radius: 16pt, inset: 13pt, stroke: 0.6pt + line)[
      #grid(
        columns: (1fr, 1fr, 1fr),
        gutter: 8pt,
        [#align(center)[#tag([Input video], fill: white, text-fill: blue)]],
        [#align(center)[#tag([4D Gaussian field], fill: white, text-fill: violet)]],
        [#align(center)[#tag([Rendered frames], fill: white, text-fill: green)]],
      )
      #v(0.35em)
      #align(center)[#text(size: 0.72em, fill: muted)[Initialization → densify/prune schedule → render mode → loss stack → compression]]
    ]

    #v(0.45em)
    *Ablated design axes:* shape, color basis, render mode, pruning, dropout, uncertainty loss and compression.
  ]

  #pop.column-box(heading: "Ablation matrix")[
    #table(
      columns: (1.1fr, 0.7fr, 0.7fr, 0.7fr, 0.75fr),
      inset: 4.2pt,
      stroke: (x, y) => if y == 0 { none } else { (top: 0.35pt + line) },
      align: center + horizon,
      [#text(weight: "bold")[Axis]], [Native], [Instant], [Mobile], [OMNI],
      [4D Gaussians], [#check(green)], [#check(green)], [#empty], [#check(blue)],
      [Isotropic shape], [#empty], [#check(green)], [#empty], [#check(violet)],
      [RGB / SH(3)], [#check(green)], [#check(green)], [#check(green)], [#check(blue)],
      [Sort-free render], [#empty], [#empty], [#check(green)], [#check(red)],
      [Spatio-temporal prune], [#empty], [#check(green)], [#check(green)], [#check(blue)],
      [Uncertainty weighting], [#empty], [#empty], [#empty], [#check(red)],
      [NVQ / GPCC compression], [#empty], [#empty], [#check(green)], [#check(red)],
    )

    #v(0.2em)
    #text(size: 0.65em, fill: muted)[Green = reused; blue = reimplemented/scheduled; red = heavy modification or constrained integration.]
  ]

  #pop.column-box(heading: "Implementation issues")[
    #grid(
      columns: (1fr, 1fr),
      gutter: 0.65em,
      [#callout([Sort-free in 4D], [Adding time to MobileGS-style MLP made inference CPU/MLP dominated; the speed promise did not transfer cleanly from 3D to 4D.], fill: rgb("#fff7ed"), stroke: orange)],
      [#callout([Prune × densify], [Spatio-temporal pruning had to be written from scratch and scheduled against densification to avoid deleting useful dynamic detail.], fill: rgb("#eef5ff"), stroke: blue)],
      [#callout([USplat loss], [Uncertainty-weighting is graph-heavy and GPU constrained; runs grew from roughly 30 min to about 5 h, limiting full ablation coverage.], fill: rgb("#f5f3ff"), stroke: violet)],
      [#callout([Compression], [Vector codebook compression is attractive post-train, but CPU-bound utilities and format conversions make it separate from the main training loop.], fill: rgb("#effcf6"), stroke: green)],
    )
  ]

  #pop.column-box(heading: "Evaluation protocol")[
    #grid(
      columns: (1.1fr, 1fr),
      gutter: 0.8em,
      [
        *Dataset:* D-NeRF examples, primarily *T-Rex* and *Bouncing Balls*.
        #linebreak()
        *Training:* 10k iterations for SOTA-style comparison; 7k for USplat-style runs due to compute constraints.
      ],
      [
        #metric-pill([FPS], [↑], fill: rgb("#effcf6"), text-fill: green)
        #metric-pill([PSNR], [↑], fill: rgb("#edf5ff"), text-fill: blue)
        #metric-pill([SSIM], [↑], fill: rgb("#edf5ff"), text-fill: blue)
        #metric-pill([LPIPS], [↓], fill: rgb("#fff7ed"), text-fill: orange)
        #metric-pill([Time], [↓], fill: rgb("#f5f3ff"), text-fill: violet)
        #metric-pill([VRAM], [↓], fill: rgb("#fff7ed"), text-fill: orange)
        #metric-pill([Size], [↓], fill: rgb("#f8fafc"), text-fill: navy)
        #metric-pill([\#Gauss.], [↓], fill: rgb("#f8fafc"), text-fill: navy)
      ],
    )
  ]

  #pop.column-box(heading: "Technical conclusion before results")[
    #callout(
      [Keep sort-based rendering as the reliable baseline],
      [The presentation results show that sorting, RGB appearance and interleaved pruning give the strongest practical trade-off. SH(3) remains useful for pure visual quality, but is not the default lightweight choice.],
      fill: rgb("#eef5ff"),
      stroke: blue,
    )
  ]

  #colbreak()

  // ========================= RIGHT: RESULTS FROM PRESENTATION =============
  #pop.column-box(heading: "Results overview")[
    #table(
      columns: (0.92fr, 1.6fr),
      inset: 5pt,
      stroke: (x, y) => if y > 0 { (top: 0.35pt + line) },
      [#text(weight: "bold")[Target]], [#text(weight: "bold")[Best observed configuration]],
      [Pure quality], [`anisotropic · SH3 · sort · no pruning · no dropout`],
      [Practical choice], [`anisotropic · RGB · sort · interleaved · no dropout`],
      [Highest FPS], [`sort · RGB`],
      [Lowest VRAM], [`RGB · interleaved`],
      [Lowest \#Gauss.], [`anisotropic · sort · interleaved · RGB`],
    )
  ]

  #pop.column-box(heading: "Ablation plots")[
    #grid(
      columns: (1fr, 1fr),
      gutter: 0.75em,
      [#plot-placeholder([Quality], [PSNR / SSIM / LPIPS across sort, sort-free and dropout], height: 8.1cm, accent: blue)],
      [#plot-placeholder([Storage + VRAM], [checkpoint size and peak training VRAM], height: 8.1cm, accent: orange)],
      [#plot-placeholder([Gaussian count], [effect of pruning, color basis and representation], height: 8.1cm, accent: violet)],
      [#plot-placeholder([Render speed], [FPS comparison across render mode and color basis], height: 8.1cm, accent: green)],
    )

    // Suggested replacements once the img/ folder is available:
    // #image("img/trex_quality_ablations.png", width: 100%)
    // #image("img/trex_storage_ablations.png", width: 100%)
    // #image("img/trex_gaussians_ablations.png", width: 100%)
    // #image("img/trex_fps_ablations.png", width: 100%)
  ]

  #pop.column-box(heading: "Result interpretation")[
    #callout([Visual quality], [Sorting is the dominant quality factor. Dropout damages quality when sorting is enabled, so it should not be part of the best visual model.], fill: rgb("#eef5ff"), stroke: blue)
    #v(0.25em)
    #callout([Memory], [SH appearance increases storage. Interleaved pruning is the cleanest VRAM reducer; RGB is the lighter practical appearance basis.], fill: rgb("#fff7ed"), stroke: orange)
    #v(0.25em)
    #callout([Speed], [RGB is faster than SH3. In these 4D ablations, sort-free rendering is not the fastest path because the MLP becomes the bottleneck.], fill: rgb("#effcf6"), stroke: green)
  ]

  #pop.column-box(heading: "Final model choice")[
    #box(width: 100%, fill: gradient.linear(rgb("#0b1026"), rgb("#1e3a8a"), angle: 20deg), radius: 18pt, inset: 16pt)[
      #text(fill: white, weight: "bold", size: 1.1em)[Practical best trade-off]
      #linebreak()
      #text(fill: white, size: 0.82em)[`anisotropic · RGB · sort-based · interleaved pruning · no dropout`]
      #v(0.35em)
      #text(fill: rgb("#dbeafe"), size: 0.72em)[Chosen because it preserves reliable reconstruction while reducing memory pressure and keeping inference fast.]
    ]
  ]

  #pop.column-box(heading: "Results discussion — leave open", stretch-to-next: true)[
    #box(width: 100%, height: 12.8cm, fill: white, stroke: (paint: line, thickness: 1pt, dash: "dashed"), radius: 16pt, inset: 14pt)[
      #text(size: 0.8em, fill: muted)[Reserved for the final discussion once plots are polished and exact numeric values are frozen.]
      #v(0.6em)
      #text(size: 0.72em, fill: muted)[Recommended content: Pareto frontier, failure cases, qualitative frame crops, and one sentence explaining why sort-free failed to dominate after the 4D extension.]
    ]
  ]
])

#pop.bottom-box()[
  #grid(
    columns: (1.1fr, 1fr, 1fr),
    gutter: 1em,
    [#text(weight: "bold")[OMNI-4DGS] · CV project poster],
    [Code and final results: add repository / QR code],
    [Contact: add email · Bocconi University],
  )
]
