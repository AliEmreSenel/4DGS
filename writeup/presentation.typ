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

#let goal = tblock.with(blocktitle: "Goal")

= Overview

#slide(title: "Lightweight 4D Gaussian Splatting")[
  \

  *Context:*

  - 3DGS represents scenes as learnable Gaussians.
  - 4DGS extends this to dynamic scenes over time.
  - Existing methods improve speed, training cost, compression, or motion modeling.
  - OMNI-4DGS combines these ideas into a lightweight dynamic reconstruction pipeline.
  \

  #goal(title: "Final Model")[
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
  .
]

#slide(title: "Contributions & Challenges")[
  .
]


= Results

#slide(title: "Ablations")[
  .
]

#slide(title: "Comparison")[
  .
]

#slide(title: "Evaluation, Future Work")[
  .
]
