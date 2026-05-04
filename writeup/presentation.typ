#import "@preview/touying:0.5.5": *
#import "@preview/clean-math-presentation:0.1.1": *
#import "@preview/diagraph:0.3.7": *
#import "img/diagram.typ": *

#show: clean-math-presentation-theme.with(
  config-info(
    title: [OMNI-4DGS: fast, light and precise \ Video-to-Model Reconstruction],
    short-title: [OMNI-4DGS: Chimera Model],
    authors: (
      (name: "Ali Emre Senel", affiliation-id: 1),
      (name: "Tebe Nigrelli", affiliation-id: 1),
      (name: "Stefana Chiriac", affiliation-id: 1),
    ),
    affiliations: (
      (id: 1, name: "Bocconi University, Milan, Italy"),
    ),
    date: datetime(year: 2026, month: 5, day: 3),
  ),
  config-common(slide-level: 2),
  progress-bar: false,
)

#title-slide()

#let block = tblock.with()

= Overview

#slide(title: "Lightweight 4D Gaussian Splatting")[
  \

  *Context:*

  - 3DGS represents scenes as learnable Gaussians.
  - 4DGS extends this to dynamic scenes over time.
  - Existing methods improve speed, training cost, compression, or motion modeling.
  - OMNI-4DGS combines these ideas into a lightweight dynamic reconstruction pipeline.
  \

  #block()[
    Compact 4DGS architecture validated through ablations and best-model evaluation.
  ]
]

= Implementation

#slide(title: "Modern Picture")[
  #grid(
    columns: (auto, auto),
    gutter: 1em,

    [
      \ \
      *Improvements*
      - Reduced Storage Size
      - Faster Inference Time
      - More Stable Training
      - Motion Regularization

      No single combination to date...

      \

      *Strategy*:

      Divide and Conquer
    ],

    align(right)[

      #box(width: 100%, height: 100%)[
        #raw-render(
          timeline,
          height: 100%,
          width: auto,
          stretch: false,
        )
      ]
    ],
  )
]

#slide(title: "Contributions")[
  \ 
  
  Each codebase contributes with features and clashes.

  #table(
    columns: (1.1fr, 1fr, 1fr, 1fr, 1fr),
    stroke: none,

    // vertical lines between columns only
    table.vline(x: 1, stroke: black),
    table.vline(x: 2, stroke: black),
    table.vline(x: 3, stroke: black),
    table.vline(x: 4, stroke: black),

    // horizontal line below header row
    table.hline(y: 1, stroke: black),
    table.hline(y: 2, stroke: gray),
    table.hline(y: 3, stroke: black),

    [*4DGS-Native*], [*Instant4D*], [*MobileGS*], [*1000FPS*], [*USPLAT*],

    [Training Logic], [Isotropic Gaussians], [Sort-Free Rendering], [Spatio-Temporal Pruning], [Uncertainty-aware Loss],

    [SH Ablations], [MegaSAM Init], [MLP Compress], [Visibility Masks], [Motion Loss],

    [Backbone], [Render], [Memory], [Pruning], [Movement],
  )
 
  \ 
  #block(title: "Additions", "Dropout, Densification Schedule")
 
]

#slide(title: "Contributions & Challenges")[
  \
  #ablations
]

#slide(title: "Comparison")[
  .
]

#slide(title: "Evaluation, Future Work")[
  .
]
