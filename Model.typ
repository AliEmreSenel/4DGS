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
Ablations are going to be very important in evaluating the performance of the model, since each architecture layer introduces errors. As a result, the architecture should be modular and exchangeable, so we can compare the effects of each technique.

- *Important Metrics*: measure performance in each architecture. This cannot be plain loss, since it varies between architectures.
  - Color Reconstruction Error
  - Memory Footprint
  - Training Time
  - Other SOTA?

There is a tradeoff between _reconstruction accuracy_ and _speed of training_ with _memory footprint_. If we don't care about training time, we can distill more to have renders that can run on phones.

Fewer parameters reduces train time and memory, at the cost of accuracy.

#pagebreak()

= List of Architecture Choices

#let thick = 1.8pt
#let base = 0.6pt

#let g = rgb("#78f082")
#let r = rgb("#fb8686")
#let y = rgb("#fff4bf")

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
#let Y(body, top: false, bottom: false) = M(body, fill: y, top: top, bottom: bottom)

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

#let g = rgb("#78f082")
#let r = rgb("#fb8686")
#let y = rgb("#fff4bf")

#let G(body) = table.cell(fill: g)[#body]
#let Y(body) = table.cell(fill: y)[#body]
#let R(body) = table.cell(fill: r)[#body]
#let GS(body, stroke) = table.cell(fill: g, stroke: stroke)[#body]
#let YS(body, stroke) = table.cell(fill: y, stroke: stroke)[#body]
#let RS(body, stroke) = table.cell(fill: r, stroke: stroke)[#body]

#table(
  columns: (0.5fr, 1fr, 1fr, 0.9fr, 1.1fr),
  stroke: base,
  inset: 6pt,
  fill: (x, y) => if y == 0 { luma(230) } else { none },

  [], [*Specialisation*], [*Structure*], [*Compatibility*], [*Implementation*],

  table.cell(rowspan: 1)[#rotate(-90deg, reflow: true)[]],
  [*Schedule*], G([High Iterations]), G([Much Testing]), G([Hyperparameters]),

  table.cell(rowspan: 7)[#rotate(-90deg, reflow: true)[*1. Gaussians*]],

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Rotation*: can be skipped but would need more gauss.],
  GS([Quaternion], (top: thick)),
  GS([No problem, common choice], (top: thick)),
  GS([Fewer params, faster training], (top: thick)),

  [Rotation Matrix],
  table.cell(stroke: (bottom: thick))[No problem],
  table.cell(stroke: (bottom: thick))[Probably easier],

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Shape*: remain in the computational graph and it's fine],
  YS([Isotropic (but then no rotation either)], (top: thick)),
  YS([Simplification], (top: thick)),
  YS([Fewer Params, faster training], (top: thick)),

  G([Anisotropic]),
  GS([Default Choice], (bottom: thick)),
  GS([Slower training], (bottom: thick)),

  table.cell(
    rowspan: 3,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Color Basis*: initialize from early point cloud ],
  YS([RGB], (top: thick)),
  YS([Simplest], (top: thick)),
  YS([Fewest Params], (top: thick)),

  Y([SH(1)]),
  Y([Simple]), Y([Intermediate]),

  GS([SH(3)], (bottom: thick)),
  GS([More Precise], (bottom: thick)),
  GS([More params], (bottom: thick)),

  table.hline(y: 9, start: 3, end: 5, stroke: thick),

  table.cell(rowspan: 1)[#rotate(-90deg, reflow: true)[*2. Init*]],
  [*Segmentation*:  use model to init gaussians],
  G([MegaSAM initialization]),
  G([Initialization (color + position)]),
  G([Recent models might be better]),

  table.cell(rowspan: 2)[#rotate(-90deg, reflow: true)[*3. Compress*]],
  [*Spherical Harmonics*: compression],
  Y([Spherical Harmonic Distillation]),
  Y([Pipeline Restructure]),
  Y([Additional loss component in the dictionary compression+]),
  [*Quantization*: at rest compression to run faster],
  R([Neural Vector Quantization]),
  R([Pipeline Restructure]),
  R([Train latent features + Time vector, use Huffman Encoding]),

  table.cell(rowspan: 2)[#rotate(-90deg, reflow: true)[*4. Train*: \
    No Densify, out \
    of performance]],
  [*Uncertainty weighing*],
  R([USPLAT Algorithm]),
  R([High: Loss reweighting]),
  R([Many added loss terms, algorithm steps]),
  [*Strategies*: Adam + Batch + slow increase of \#SH],
  G([Batch Sampling in Time]),
  G([Easy]),
  G([Batch training, general choice of optimizer]),

  table.cell(rowspan: 3)[#rotate(-90deg, reflow: true)[*5. Prune*]],

  table.cell(
    rowspan: 2,
    stroke: (left: thick + black, top: thick + black, bottom: thick + black),
  )[*Criterion*: many possibilities. Does not have to be one alone.],
  YS([Contribution-based pruning], (top: thick + black)),
  YS([Threshold tuning at render], (top: thick + black)),
  YS([Hyperparameters], (top: thick + black, right: thick + black)),

  GS(
    [Voxel Dedup\
      Spatio-Temporal],
    (bottom: thick + black),
  ),
  GS([One-shot Pruning], (bottom: thick + black)),
  GS([Write formula from clear instructions], (bottom: thick + black, right: thick + black)),

  [Strategy], G([One-shot]), G([Faster train, but more finetune]), G([Very straightforward]),

  table.cell(rowspan: 4)[#rotate(-90deg, reflow: true)[*6. Render*]],

  table.cell(
    rowspan: 2,
    stroke: (left: thick, top: thick, bottom: thick),
  )[*Rasterisation*: project 3D to 2D],
  YS([*Sort*: aggregate color back to front], (top: thick)),
  YS([Standard technique], (top: thick)),
  YS([Runtime Bottleneck], (top: thick)),

  GS([*Sort-free*: weighted sum biased by MLP], (bottom: thick)),
  GS([Pipeline Change + time feature], (bottom: thick)),
  GS([Altro train tiny MLPs], (bottom: thick)),

  table.hline(y: 21, start: 3, end: 5, stroke: thick),

  [Thresholding],
  G([Opacity Threshold]),
  G([Highly]),
  G([Hyperparameters]),

  [Loading],
  R([Visibility Mask Loading]),
  R([Highly]),
  R([Additional Layer]),
)
