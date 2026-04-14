#set page(
  paper: "a4",
  margin: (
    left: 2cm,
    right: 2cm,
    top: 2cm,
    bottom: 2cm,
  ),
)

= Reflections
Ablations are going to be very important in evaluating the performance of the model, since each architecture layer introduces errors.


Ideally, architecture steps should be modular and exchangeable, so we can compare the effects of each combination of tools

- Important Metrics: we need stable metrics to measure performance in each architecture.
  - Loss evolution

There is a tradeoff between _reconstruction accuracy_ and _speed of training_ with _memory footprint_.

Fewer parameters reduces train time and memory, at the cost of accuracy.

For sure the model will not train on a phone, but it needs to be able to render comfortably on a phone.

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

#pagebreak()

#table(
  columns: (0.5fr, 1fr, 1fr, 1fr, 1fr),
  stroke: base,
  inset: 6pt,
  fill: (x, y) => if y == 0 { luma(230) } else { none },

  [], [*Specialisation*], [*Structure*], [*Compatibility*], [*Implementation*],

  table.cell(rowspan: 1)[#rotate(-90deg, reflow: true)[]],
  [*Schedule*], [High Iterations], [Much Testing], [Hyperparam],

  table.cell(rowspan: 7)[#rotate(-90deg, reflow: true)[*1. Gaussians*]],

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Rotation*: can be skipped but would need more gauss.],
  table.cell(stroke: (top: thick))[Quaternion],
  table.cell(stroke: (top: thick))[No problem, common choice],
  table.cell(stroke: (top: thick))[Fewer params, faster training],

  [Rotation Matrix],
  table.cell(stroke: (bottom: thick))[No problem],
  table.cell(stroke: (bottom: thick))[Probably easier],

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Shape*: remain in the computational graph and it's fine],
  table.cell(stroke: (top: thick))[Isotropic (but then no rotation either)],
  table.cell(stroke: (top: thick))[Simplification],
  table.cell(stroke: (top: thick))[Fewer Params, faster training],

  [Anisotropic],
  table.cell(stroke: (bottom: thick))[Default Choice],
  table.cell(stroke: (bottom: thick))[Slower training],

  table.cell(
    rowspan: 3,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Color Basis*: initialize from early point cloud ],
  table.cell(stroke: (top: thick))[RGB],
  table.cell(stroke: (top: thick))[Simplest],
  table.cell(stroke: (top: thick))[Fewest Params],

  [SH(1)],
  [Simple], [Intermediate],

  table.cell(stroke: (bottom: thick))[SH(3)],
  table.cell(stroke: (bottom: thick))[More Precise],
  table.cell(stroke: (bottom: thick))[More params],

  table.hline(y: 9, start: 3, end: 5, stroke: thick),

  table.cell(rowspan: 3)[#rotate(-90deg, reflow: true)[*2. Init*]],
  [*Segmentation*:  use model to init gaussians],
  [MegaSAM initialization],
  [Initialization (color + position)],
  [Recent models might be better],
  [*Grid*: decide the size of grid], [Voxel Size], [Needs testing], [Hyperparam],
  [*Size*: usually isotropic], [Uncertainty], [Initialization from model information], [Use Voxel Size initialization],

  table.cell(rowspan: 2)[#rotate(-90deg, reflow: true)[*3. Compress*]],
  [*Spherical Harmonics*: compression],
  [Spherical Harmonic Distillation],
  [Pipeline Restructure],
  [Additional loss component],
  [*Quantization*: at rest compression to run faster],
  [Neural Vector Quantization],
  [Pipeline Restructure],
  [After training, for compression as Huffman Encoding],

  table.cell(rowspan: 2)[#rotate(-90deg, reflow: true)[*4. Train*: \ No Densify, out \ of performance]],
  [*Uncertainty weighing*], [Usplat Algorithm], [High: Loss reweighting], [Many added loss terms],
  [*Strategies*: Adam + Batch + slow increase of \#SH],
  [Batch Sampling in Time],
  [Easy],
  [Batch training, general choice of optimizer],

  table.cell(rowspan: 3)[#rotate(-90deg, reflow: true)[*5. Prune*]],

  table.cell(
    rowspan: 2,
    stroke: (left: thick + black, top: thick + black, bottom: thick + black),
  )[*Criterion*: many possibilities.],
  table.cell(stroke: (top: thick + black))[Contribution-based pruning],
  table.cell(stroke: (top: thick + black))[Threshold tuning at render],
  table.cell(stroke: (top: thick + black, right: thick + black))[Hyperparam],

  table.cell(stroke: (bottom: thick + black))[Voxel Dedup\
    Spatio-Temporal],
  table.cell(stroke: (bottom: thick + black))[One-shot Pruning],
  table.cell(stroke: (bottom: thick + black, right: thick + black))[Write formula from clear instructions],

  [Strategy], [One-shot], [Faster train, but more finetune], [Very straightforward],

  table.cell(rowspan: 4)[#rotate(-90deg, reflow: true)[*6. Render*]],

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Rasterisation*: project 3D to 2D],
  table.cell(stroke: (top: thick))[*Sort*: aggregate color back to front],
  table.cell(stroke: (top: thick))[Standard technique],
  table.cell(stroke: (top: thick))[Runtime Bottleneck],

  table.cell(stroke: (bottom: thick))[*Sort-free*: weighted sum biased by MLP],
  table.cell(stroke: (bottom: thick))[Pipeline Restructure],
  table.cell(stroke: (bottom: thick))[Altro train tiny MLPs],

  table.hline(y: 21, start: 3, end: 5, stroke: thick),

  [Thresholding], [Opacity Threshold], [Highly], [Hyperparam],
  [Loading], [Visibility Mask Loading], [Highly], [Additional Layer],
)
