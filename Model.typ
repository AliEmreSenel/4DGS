
= Reflections
Ablations are going to be very important in evaluating the performance of the model, since each architecture layer introduces errors. 


Ideally, architecture steps should be modular and exchangeable, so we can compare the effects of each combination of tools

- Important Metrics: we need stable metrics to measure performance in each architecture.
  - Loss evolution

There is a tradeoff between _reconstruction accuracy_ and _speed of training_ with _memory footprint_. 

Fewer parameters reduces train time and memory, at the cost of accuracy.

For sure the model will not train on a phone, but it needs to be able to render comfortably on a phone.

= Ongoing Questions

- How to upgrade Mobile-3GS pipeline to 4DGS?
- How does implementation differ between quaternions for rotation and a rotation matrix?
- What is the effect of evaluation of MegaSAM vs Other Initialization

#pagebreak()

= List of Architecture Choices

#let thick = 1.8pt
#let base = 0.6pt

#let g = rgb("#78f082")
#let r = rgb("#fb8686")

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

#table(
  columns: (auto, auto, 1fr, auto, auto, auto, auto, auto),
  stroke: base,
  inset: 6pt,
  fill: (x, y) => if y == 0 { luma(230) } else { none },

  [*Section*], [*Specialisation*], [*Element*], [*Easy \ Code*], [*Good \ Quality*], [*Low \ Memory*], [*Fast \ Train*], [*Fast \ Render*],

  table.cell(rowspan: 1)[#rotate(-90deg, reflow: true)[*0. Iter*]],
  [Schedule], [High Number \ of Iterations], N([]), G([]), N([]), R([]), N([]),

  table.cell(rowspan: 7)[#rotate(-90deg, reflow: true)[*1. Gaussians*]],

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

  [SH(3)],
  G([]),
  G([]),
  R([]),
  R([]),
  R([]),

  table.cell(stroke: (bottom: thick))[SH(1)],
  G([], bottom: true),
  N([], bottom: true),
  G([], bottom: true),
  G([], bottom: true),
  G([], bottom: true),

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
  [Deduplication], [Voxel Dedup\
Spatio-Temporal], R([]), G([]), G([]), G([]), G([]),
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
  [Loading], [Visibility Mask Loading], G([]), N([]), G([]), [], G([]),
)

#pagebreak()

== Ideal Combination

Pick the memory optimizations (fewer variables + codebook). One-pass is better, so do not use multiple

1. Gaussian Representation: 
  - Quaternions
  - Isotropic
  - RGB
 
2. Initialization:
  - MegaSAM-like initialization
  - Voxel Size
  - Uncertainty

3. Compression
  - Spherical Harmonic Distillation
  - Neural Vector Quantization

4. Training:
  - Contribution-based pruning
  - Uncertainty weighing
  - Batch Sampling in Time

5. Pruning Phase before Finetuning
  - Voxel Deduplication, Spatio-Temporal Score: they address different things, so they should be able to 
  - One-shot: less loops make training faster.

6. Rendering
  - Sort-free
  - Opacity Threshold
  - Visibility Mask Loading
