#import "@preview/touying:0.5.5": *
#import "@preview/clean-math-presentation:0.1.1": *
#import "@preview/diagraph:0.3.7": *
#import "@preview/fontawesome:0.6.0": *
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

= Problem Formulation

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

= Proposed Solution

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

= Performance Evaluation

#let metric(name, icon) = box(width: 100%)[
  #strong(name)
  #h(1fr)
  #fa-icon(icon)
]
#let th(body) = text(size: 1.08em, weight: "bold", body)

#slide(title: "Metrics for Comparison")[
  \ \

  #grid(
    columns: (2fr, 1fr),
    gutter: 2em,
    // left section

    table(
      columns: (0.8fr, 2fr),
      stroke: none,

      table.vline(x: 1, stroke: black),

      table.hline(y: 1, stroke: black),
      table.hline(y: 2, stroke: gray),
      table.hline(y: 3, stroke: gray),
      table.hline(y: 4, stroke: black),
      table.hline(y: 5, stroke: gray),
      table.hline(y: 6, stroke: gray),
      table.hline(y: 7, stroke: gray),
      table.hline(y: 8, stroke: gray),

      [#th[Metric]], [#th[Short description]],

      [*FPS*], [Render speed, in frames per second #h(1fr) #fa-icon("angle-up")],
      [*PSNR*], [Signal fidelity #h(1fr) #fa-icon("angle-up")],
      [*SSIM*], [Structural dissimilarity #h(1fr) #fa-icon("angle-up")],
      [*LPIPS*], [Perceptual difference #h(1fr) #fa-icon("angle-down")],
      [*TIME*], [Total runtime or processing time #h(1fr) #fa-icon("angle-down")],
      [*\#Gauss.*], [Number of Gaussian primitives used #h(1fr) #fa-icon("angle-down")],
      [*Memory*], [Storage use over train #h(1fr) #fa-icon("angle-down")],
      [*VRAM*], [Peak vram use over train #h(1fr) #fa-icon("angle-down")],
    ),

    // right section
    [
      #block(title: "Testing Datasets", [
        *DNerf*
        - Bouncing Balls
        - Trex

        *Train*
        - 10k SOTA comparison.
        - 7k for USPLAT
      ])
    ],
  )
]

#slide(title: "Dataset Example")[
  #grid(
    columns: (1fr, 0.8fr),
    gutter: 1em,

    // left section
    [
      #figure(
        rect(
          fill: black,
          inset: 0pt,
          width: 80%,
          image("img/r_017_true.png", width: 100%),
        ),
      )
        #align(center)[
        #text(size: 0.8em)[Still image from validation dataset]
      ]
    ],

    [
      #image("img/r_017_recon.png", width: 100%)
      #align(center)[
        #text(size: 0.8em)[Reconstruction from view angle and time]
      ]
    ],
  )
]

= Final Result

\

#slide(title: "Evaluation")[
  #grid(
    columns: (1fr, 0.8fr),
    gutter: 1em,
    // left section
    [
      \

      *Visual Quality*

      1. `sort - no_dropout`
      2. `sort - yes_dropout`
      3. `sort_free - no_dropout`
      4. `sort_free - yes_dropout`

      // Interpretation:
      // - Sorting is the dominant visual-quality factor.
      // From 3D to 4D requires a new hidden layer but makes data sparse
      // Profiling with Pytorch, 95% of CPU time gets spent on MLP
      // as opposed to 65% before
      // - Dropout hurts quality when sorting is enabled.
    ],
    [
      #move(dx: -3em)[
        #image("img/trex_quality_ablations.png", width: 100%)]
    ],
  )
]

#slide(title: "Evaluation")[
  #grid(
    columns: (1fr, 0.8fr),
    gutter: 1em,
    // left section
    [
      \

      *Storage Use*

      // Main ablations to explain range:
      // - `appearance × pruning` for VRAM
      // - `sorting × appearance` for checkpoint size - memory

      Peak VRAM over training:
      1. `rgb - interleaved`
      2. `rgb - no_pruning`, `sh3 - interleaved`
      3. `sh3 - no_pruning`

      Memory use in storage:
      1. `sort_free - rgb`
      2. `sort_free - sh3`, `sort - rgb`
      3. `sort - sh3`

      // Interpretation:
      // - SH appearance increases storage/memory.
      // - Pruning is the clearest VRAM reducer.
      // - Sorting and isotropy increase Gaussian count.

    ],
    [
      \

      #image("img/trex_storage_ablations.png", width: 110%)
    ],
  )
]


#slide(title: "Evaluation")[
  #grid(
    columns: (1fr, 0.8fr),
    gutter: 1em,
    // left section
    [
      \

      *Storage Use*

      Number of Gaussians
      1. `sort_free - anisotropic`
      2. `sort_free - isotropic`
      3. `sort - anisotropic`
      4. `sort - isotropic`

      // Interpretation:
      // - SH appearance increases storage/memory.
      // - Pruning is the clearest VRAM reducer.
      // - Sorting and isotropy increase Gaussian count.
      // Spikes in training come from instability of Droupout
    ],
    [
      \
      #move(dx: -5em)[
        #image("img/trex_gaussians_ablations.png", width: 140%)
      ]
    ],
  )
]


#slide(title: "Evaluation")[
  #grid(
    columns: (1fr, 0.8fr),
    gutter: 1em,
    // left section
    [
      \

      *Render Speed*

      // Main ablation to explain range:
      // - `sorting × appearance`

      1. `sort - rgb`
      2. `sort - sh3`
      3. `sort_free - rgb`
      4. `sort_free - sh3`

      // Interpretation:
      // - Avoiding sort-free strongly improves FPS.
      // - RGB is faster than SH3.
      // - Best speed comes from combining sorting with RGB appearance.
      // Sort-free spends 95% of CPU MLP for inference as opposed to 65%, roughly 10x longer for running
    ],
    [
      \
      #move(dx: -5em)[
        #image("img/trex_fps_ablations.png", width: 140%)
      ]
    ],
  )
]

#slide(title: "Best Model Combination")[

  \

  *Our results*: 
  
  #table(
    columns: (0.3fr, 1.2fr),
    stroke: none,

    table.vline(x: 1, stroke: black),

    table.hline(y: 1, stroke: gray),
    table.hline(y: 2, stroke: gray),
    table.hline(y: 3, stroke: gray),
    table.hline(y: 4, stroke: gray),

    [Pure quality], [`anisotropic - sh3 - sort - no_pruning - no_dropout`],
    [Practical choice], [`anisotropic - rgb - sort - interleaved - no_dropout`],
    [Highest FPS], [`sort - rgb`],
    [Lowest VRAM], [`rgb - interleaved`],
    [Most compact], [`anisotropic`],
  )

  #block(title: "Uncertainty-Weighting", [
    GPU-constrained. Ablations are impractical: $quad $30min $->$ 5h / run.
  ])


  // - Sorting should be kept because it improves quality and FPS, though it can increase memory/model size.
  // - Dropout should be disabled for quality; it improves speed/memory only by severely degrading quality.
  // - SH3 gives the best pure quality, but RGB is the best practical appearance choice because it is much faster and lighter.
  // - Interleaved pruning gives the best VRAM/model-size tradeoff, with a small quality cost.
  // - Anisotropic Gaussians keep the representation more compact and faster than isotropic in this run.

]
