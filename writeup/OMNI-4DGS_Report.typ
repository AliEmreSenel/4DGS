#import "template.typ": cvpr2025
#import "appendix.typ": appendix

#let authors = (
  (
    name: "Ali Emre Senel",
    affl: (0,),
  ),
  (
    name: "Tebe Nigrelli",
    affl: (0,),
  ),
  (
    name: "Stefana Chiriac",
    affl: (0,),
  ),
)

#let affls = (
  (
    institution: [Bocconi University],
    location: [Milan],
    country: [Italy],
  ),
)

#let abstract = [
  3DGS represents scenes as learnable gaussians, and 4DGS extends this to dynamic scenes over time. While these models enable high-quality rendering, they are still too heavy for mobile use. Recent work addresses existing limitations from different angles: from the native 4DGS baseline formulation for dynamic reconstruction, 1000FPS speeds up rendering through pruning and visibility masks, Instant4D cuts training cost with a simpler representation and better initialization, MobileGS makes 3D gaussian rendering compact enough for phones, and Usplat4D improves dynamic modeling with uncertainty-aware motion weighting. We build on these contributions to design a lightweight 4DGS pipeline. We validate our results through extensive ablations, and we describe and evaluate our best architecture.
]

#show: cvpr2025.with(
  title: [OMNI-4DGS: Chimera model for fast, light and precise  \ Video-to-Model Reconstruction],
  authors: (authors, affls),
  keywords: (),
  abstract: abstract,
  bibliography: bibliography("bibliography.bib"),
  accepted: true,
  id: none,
  appendix: appendix,
)

#show link: underline

// TODO add link to github

// TODO add references to why each technique is picked per-technique (ie. memory vs compute vs accuracy constraints)

= Introduction

Gaussian Splatting has been used extensively in the task of scene reconstruction from videos, thanks to its relative small training requirements. In the 4D setting (4DGS), frames from a video are used to train a machine learning model to "learn" a scene in space. This representation can then be used to produce novel camera views as a function of time, position, orientation, and space. \

In a 4DGS model, the building blocks of the scene are a vast number of Gaussian Distributions in space, and are being parametrized by a position vector $mu$, which places their center in space, a covariance matrix $Sigma = Sigma^T$ allowing for diagonal deformation, and a rotation matrix $R$, further orienting the distribution in space. Color is treated as a vector field over the surface of the gaussian, encoded from a fixed number of Spherical Harmonic (SH) coefficients, which can approximate uniformly any color distribution. Compared to having a constant color, SH coefficients allow smooth coloring over the surface, where expressiveness depends on the number of coefficients. Both the use of Covariance+Rotation and SH reduce the number of gaussians needed, since they allow for greater range of behavior, at the cost of a higher number of variables. Training involves the iterated reconstruction of ground truth images from the dataset by rendering the gaussians. Multiple similarity metrics are used to steer each gaussian features in the correct direction, balancing color fidelity with a known depth map, or a structural similarity metric. The final output consists of the list of gaussians, which can be projected to a camera plane and rasterized to obtain a reconstructed image. During inference, camera position is fixed, and a pixel color is obtained by integrating the scene through the view-pixel ray: each gaussian color contribution is added, while accounting for opacity, distance, and leftover un-occluded light, crossing the gaussians in the correct order. Naturally, this allows for the generation of new camera angles. \ \

Extending the task from static scenes to dynamic scenes, where the objects move or change shape and color, the state of the art (SOTA) architecture is Native 4D Gaussian Splatting (4DGS-Native) @yang2024_4dgs. This model extends the gaussian features by adding a time scalar to position, centering the gaussians in time, and time covariance components, allowing some movement of each gaussian. With this extension, rendering must first condition the gaussians with respect to time, before being able to render them as in the 3D setting. All mentions of 4D Gussian Splatting refer to the native representation. \ \

Although the field of dynamic scene reconstruction has attracted a lot of interest and produced impressive results, many problems persist with the current models: scenes are affected by a high number of low-importance Gaussians, which drastically increase training time; rendering techniques are limited by costly sorting algorithms; and initialization or training strategies are non-obvious. In this paper, we combine several 4DGS-Native improvements into a unified architecture, verifying their performance through ablations.

== Gaussian Representation

The parametrization of the gaussians in a Native 4DGS is a tradeoff between using few variables and being as expressive as possible, so fewer gaussians are necessary to describe a scene. In 4DGS-Native, each gaussian is characterized by a position vector $(mu_x, mu_y, mu_z, mu_t) = mu in RR^4$ representing the center of the gaussian, that is, the mean of the distribution, a covariance matrix $Sigma in RR^(4 times 4)$, which distorts the gaussian in time and space, a rotation, represented using two quaternions $r_1, r_2 in HH$. For color, the gaussians are equipped with opacity $o in [0,1]$, and a series of SH coefficients, which can are used to approximate uniformly any color field on a sphere surface: SH(m) uses $2*"m" + 1$ coefficients for color channels, so SH(3) requires 21 scalars in total, per Gaussian, to encode its color. It should be noted that SH(0) corresponds to having plain RGB colors.

$
  G_i = (mu_i, Sigma_i, o_i, (r_1, r_2), arrow("SH")(3)))
$

Variations of this representation have also been developed to reduce the number of variables, which reduces training time and storage size, but also expressiveness. Isotropic Gaussians are spherical, but retain the time covariance, allowing them to shrink or grow in time. This is encoded in a matrix with a $3 times 3$ constant-values submatrix, and a time-varying time vector. Here, $bold(1)$ represents the matrix with $1$ in each entry.

$
  Sigma =
  mat(
    Sigma_"xyz", 0;
    0, sigma_t,
  )               &&   "with"
                       Sigma_"xyz" = S_"xyz" bold(1)_(3 x 3) \
  Sigma = Sigma^T && => Sigma_(x y z,t) = Sigma_(t, x y z)^T
$

== Projection and Rendering

Images may be reproduced from 4DGS-Native gaussians by first conditioning the distributions in time, which produces a 3D normal distribution. This colored "cloud" is mapped to a 2D camera plane using a world-to-camera projection matrix, reproducing a pixel image. During training, the operation enables the calculation of loss, since the reprojected image can be directly compared to one of the reference images. Generalization loss can also be calculated from pictures the model was not trained on. Image reconstruction is obtained by first fixing camera position and orientation, then evaluating per-pixel color as a function of all the visible gaussians. More precisely, each pixel induces a ray that marches outwards, intersecting all gaussians in the way between the focal point of the camera end the background, passing through the pixel point. The color of the pixel is obtained through the integration of each color contribution per-gaussian. The formula for the final color is traditionally sort-dependent, but we also investigate sort-free rendering, as it has been shown to greatly reduce the render bottleneck in related work, such as for @du2026_mobilegs.

*Sort-based rendering* consists of filtering view to only consider relevant gaussians, and processing them one at a time, integrating their colors in order to obtain the resulting color of a pixel. For rendering, overall opacity is calculated sequentially in order to obtain "leftover" light level (Transmittance) $T_i (p, t)$, which modulates the color contribution $c_i (v,t)$ for each successive gaussian. Finally, leftover light $T_(N+1)$ is attributed to background color $c_("bg")$. It is important to stress that in the most general formulation, color is view-direction dependent, as well as time dependent. Moreover, the formula offers limited parallelization, as the sorting operation is a bottleneck.

$
  cases(
    alpha_i (p, t) & = o_i G_i^(2D)(p | t) G_i^t (t),
    T_i (p, t) & = product_(j = 1)^(i - 1) (1 - alpha_j (p, t)),
  )
$

$
  C_p(t, v) = sum_(i=1)^N T_i (p, t) & alpha_i (p, t) c_i (v, t) \
                                     & + T_(N+1)(p, t) c_("bg")
$

*Sort-Free Rendering*, first proposed in @Hou2024SortFreeGS, computes color directly through a sum, where the weighing of each color contribution is computed by multiple small Multilayer Perceptrons (MLP). The MLP "compresses" information from the gaussians, resulting in smaller storage requirements and faster inference. Moreover, transmittance is computed as an unsorted product, while weights $w_i$ depend on viewing angle, camera position and distance. We picked @du2026_mobilegs as our reference paper for its impressive inference speed on mobile devices. However, since it is based on 3DGS, we extended the MLP architecture by adding a time term $t$ to the MLP inputs. For clarity, we provide the original formula from @du2026_mobilegs, with computing pixel color and gaussian MLP weights. In the formulas, $Delta x_i$ is the screen-space offset between the pixel and the projected Gaussian center; $Sigma_i$ is the projected 2D covariance matrix of the Gaussian footprint; $d_i$ is the Gaussian depth in camera coordinates; $s_"max"$ is the maximum component of the Gaussian scale in camera coordinates; and $s_i$ and $r_i$ are the Gaussian scale and rotation parameters.

$
  C_"pix" = (1 - T)
  frac(
    sum_(i = 1)^N c_i alpha_i w_i,
    sum_(i = 1)^N alpha_i w_i,
  )
  + T c_"bg"
$
$
  T = product_(j = 1)^N (1 - alpha_j)
$

$
  alpha_i = o_i exp(-1 / 2 Delta x_i^T Sigma_i^(-1) Delta x_i)
$

$
  w_i =
  underbrace(phi_i^2, "MLP override")
  + underbrace(frac(phi_i, d_i^2), "proximity")
  + underbrace(exp(frac(s_"max", d_i)), "distance effects")
$

$
  P_i = frac(mu_i - k_v, norm(mu_i - k_v))
  quad #stack(dir: ttb, spacing: 5pt, align(left)[camera to gaussian], align(left)[unit direction])
$

$
  cases(
    F = "MLP"_f (P_i, s_i, r_i, arrow("SH")(3)_i),
    phi_i = "ReLU"("MLP"_phi (F)),
    o_i = sigma("MLP"_o (F)),
  )
$

Other optimizations have also been developed for the rendering (inference) operation: following @du2026_mobilegs, gaussians with opacity smaller than a threshold are dropped. Visibility masks are also an option for selectively loading gaussians at render time: @yuan2025_4dgs1k proposes binary labellings of the gaussians every 5 frames, so the rendering step loads only gaussians that are visible just before or just after. Both of these techniques result in faster inference time, since they drastically reduce memory loading and reducing the problem size.

*MLP Compression of SH* encodes the view-dependent color of the Gaussians into latent vectors: #box[$(h_d, h_v) = ("view", "diffuse")$], which isolate the information between gaussian-specific and perspective specific @du2026_mobilegs. To learn the split, while training "view", the diffuse component is fixed but camera angle changes. The opposite holds for "diffuse", where the view component is fixed. During inference, the MLPs are used to recover color.

*Teacher-Student SH Compression* works by training an MLP to compress existing high-order harmonic colors to lower order through a faithful mapping. The operation identifies the set of mappings that minimizes the color loss between the original representation and the lower-order one.

== Training

_Standard Backpropagation_ is used in the model to train the best parameters for scene fidelity, while loss can be measured using image similarity metrics. To reduce training instability, _Adam Optimizer_ and _Batch training_ are used @yang2024_4dgs, while loss choice depends on the architectural components.

Since memory usage is proportional to the number of Gaussians, one seeks to minimize their number by pruning less relevant ones. *Opacity Pruning* drops gaussians with opacity value smaller than a chosen threshold @yang2024_4dgs @du2026_mobilegs. *Contribution Pruning* accumulates a contribution value over training steps, and drops gaussians that remain not relevant enough for enough iterations. *Spatio-Temporal Pruning* drops the bottom $~ 90%$ quantile, ranking higher more persistent and longer-lasting gaussians @luo2025_instant4d. Finally, *Grid Pruning* de-duplicates gaussians by position, velocity, temporal scale and position @luo2025_instant4d. Since pruning rules are correlated, we only re-implement Spatio-Temporal pruning, which had to be written from scratch, and maintain Opacity Pruning from 4DGS-Native. *Densification* is the opposite strategy, splitting gaussians with high loss gradient: although the reference papers skipped it for a faster training, we reintroduce it as a regularization technique.

Custom CUDA kernels are used to compute pixel operations on the GPU directly, though it limits compatibility across codebases. For this reason, we opted for a mixed pruning-densify schedule, as opposed to MegaSAM @luo2025_instant4d.

== Loss

The primary objective is photometric fidelity, measured as a weighted
combination of pixel-wise L1 error and structural similarity:

$
  cal(L)_"rgb" = (1 - lambda_"dssim") cal(L)_1 + lambda_"dssim" cal(L)_"SSIM"
$

This alone is insufficient for dynamic outdoor scenes, where the background
is transparent but the model may place opaque Gaussians there. An opacity
mask loss discourages this by penalising opacity in sky regions identified
from the ground-truth alpha channel $m_"gt"$:

$
  cal(L)_"opa" = -1 / abs(Omega) sum_(p in Omega) (1 - m_"gt"(p)) dot log(1 - alpha(p))
$

Dynamic Gaussians additionally require motion regularization to prevent
physically implausible trajectories. A rigidity loss enforces that spatially
proximate Gaussians move with coherent velocities, weighted by squared
distance with $k = 20$ neighbours:

$
  cal(L)_"rigid" =
  1 / (k dot G)
  sum_(i=1)^G
  sum_(j in cal(N)(i))
  e^(-100 dot d_(i j)) dot
  norm(dot(mu)_i - dot(mu)_j)_2
$

A global motion loss further suppresses high-velocity Gaussians that are
unlikely to correspond to real scene motion:

$
  cal(L)_"motion" = 1 / G sum_(i=1)^G norm(dot(mu)_i)_2
$

where $dot(mu)_i$ is the temporal velocity by finite difference. These four
terms are active throughout training.

== Uncertainty-Aware Loss

The baseline losses treat all Gaussians equally, but poorly-observed Gaussians, occluded, nearly transparent, or rarely visible, are underconstrained and
tend to drift, producing artifacts at novel viewpoints. To address this, at
$N_"start" = 15000$ (coinciding with the end of densification, when the
Gaussian count is stable), we activate the uncertainty-aware graph losses of
USplat4D:

$
  cal(L) =
  cal(L)_"rgb"
  + lambda_"key" cal(L)_"key"
  + lambda_"non-key" cal(L)_"non-key"
$

with $lambda_"key" = lambda_"non-key" = 1.0$.


The first step is identifying which Gaussians are reliable. Uncertainty is
estimated per Gaussian per frame from alpha-blending weights. This shows
us that well-observed
Gaussians contribute strongly to many pixels and thus have low uncertainty:

$
  sigma^2_(i,t) =
  1 / (sum_(h in P_(i,t)) (T^h_(i,t) dot alpha_i)^2),
  quad
  u_(i,t) = cases(
    sigma^2_(i,t) & "if " bb(1)_(i,t) = 1,
    phi & "otherwise"
  )
$

with $phi = 10^6$. The convergence indicator $bb(1)_(i,t)$ forces $u = phi$
if any pixel in the Gaussian footprint exceeds color residual $eta_c = 0.5$,
preventing unconverged Gaussians from becoming anchors. The top $2%$ by
confidence with significant period $>= 5$ frames become key nodes $V_k$;
UA-kNN with $k = 8$ builds key-key edges; each non-key node is assigned to
its closest key over all frames.

*Key node loss.* Reliable Gaussians should not drift from their well-trained
positions, and their neighbourhoods should move geometrically consistently.
This is enforced by anchoring them to their pretrained positions $bold(p)^circle$
and applying motion locality constraints:

$
  cal(L)_"key" =
  sum_t sum_(i in V_k)
  norm(bold(p)_(i,t) - bold(p)^circle_(i,t))^2_(U^(-1)_(w,t,i))
  + cal(L)_"motion,key"
$

The uncertainty matrix $U_(i,t) = R_"wc" op("diag")(u, u, 0.01 u) R_"wc"^T$
down-weights depth corrections by $100 times$, reflecting monocular depth
unreliability. $cal(L)_"motion,key"$ combines isometry, rigidity, rotation,
velocity and acceleration constraints ($lambda_"iso" = lambda_"rigid" = 1.0$,
$lambda_"rot" = lambda_"vel" = lambda_"acc" = 0.01$).

*Non-key node loss.* Uncertain Gaussians have no reliable photometric signal
to constrain their motion. Rather than leaving them free, we pull them toward
positions predicted by interpolating the motion of their key neighbours via
DQB, which correctly handles rotation
and translation jointly:

$
  cal(L)_"non-key" =
  sum_t sum_(i in.not V_k)
  norm(bold(p)_(i,t) - bold(p)^circle_(i,t))^2_(U^(-1)_(w,i))
  + sum_t sum_(i in.not V_k)
  norm(bold(p)_(i,t) - bold(p)^"DQB"_(i,t))^2_(U^(-1)_(w,i))
  + cal(L)_"motion,non-key"
$

As Gaussians are isotropic, DQB interpolation reduces to a weighted blend of key
node positions in our isotropic setting. The DQB target is soft;
$bold(p)_(i,t)$ remains free and may deviate when the photometric loss
provides a stronger signal, preserving non-rigid deformation. Density control
is disabled in the first $10%$ and last $20%$ of USplat iterations to protect
graph index integrity.

== At-Rest Compression

The complete model is stored as the combination of compression MLPs, and lossy codebook approximations @du2026_mobilegs.

*Codebook Compression* is applied to gaussian features by splitting the complete vector of features into sub-vectors, which are replaced with the K-means nearest centroids to obtain a lossy compression of similar vectors. The corresponding index for each gaussian is stored using Huffman Encoding. This technique is referred to as Neural Vector Quantization (NVQ).

*GPCC Compression* is used to encode gaussian positions by first voxelizing the space, then sorting them by Morton Order and storing them in PLY format. The algorithm is implemented in C++ as a single-threaded utility.

== Points of Improvement

Several techniques have been developed to improve the known limitations of GS, but any addition may negate the contribution of another. For example, Isotropic Gaussians allow for faster training and smaller memory footprint, but reduce the model's capacity. We propose a shared parametrization model to test each architectural component in relation to the others, in a unified system.

= Model Parametrization

We combine the papers into a single architecture, which can be studied through ablations. We start from the 4DGS-Native Architecture, adding features from available implementations @yang2024_4dgs @luo2025_instant4d @du2026_mobilegs, and re-implementing the missing structures from the others @yuan2025_4dgs1k @guo2026uncertaintymattersdynamicgaussian.

*Initialization* of gaussian position can be random, or be provided from a point cloud model such as MegaSAM in @luo2025_instant4d. Good initializations massively reducing training time and produce better results. We experiment with the addition of a pruning-densify schedule to improve the robustness of our results over a reduced training time.

*Isotropy* involves choosing between Isotropic gaussians @luo2025_instant4d and Anisotropic Gaussians @yang2024_4dgs, corresponding to a tradeoff between faster training, stemming from a reduced number of variables, and more expressive Gaussians.

*Rotation* encoding relies on two quaternions per-gaussian to learn and encode rotation. Quaternions are preferred to rotation matrices because of their simpler implementation smaller number of parameters, which avoids and reduces the number of variables

*Spherical Harmonics* are trained with a predefined precision. Following the 4DGS-Native implementation, higher order harmonics are "unlocked" for learning over training.

*Loss* depends on the active components of the architecture. The "only" choice when using a combined model is the exact weighing of the different components.

*Pruning* is usually treated as a penultimate step in training the model: train from inizialization, prune once, and finetune at the end. Different, yet similar, strategies were proposed for pruning: Opacity Pruning @du2026_mobilegs, which discards transparent gaussians, Contribution pruning @du2026_mobilegs, dropping gaussians that affect the loss minimally, Grid Pruning @luo2025_instant4d. We focus only on Spatio-Temporal Contribution @yuan2025_4dgs1k, as it captures both aspects of prevalence in time and in space.

*Densification* consists of duplicating single Gaussians to give more expressivity where needed. In our reference models, the step is generally discarded to reduce training time, but we reintroduce it as it provides better than random Gaussian initialization than MegaSAM.

*Rendering* changes primarily whether Sort-Based @yang2024_4dgs or Sort-Free @du2026_mobilegs are used. We extend the latter with a time component, to also capture time-dependent behavior.

*Distillation* can be used to effectively reduce the storage size of the SH coefficients, by training a MLP to effectively compress color through a teacher-student model.

*Storage* reduction techniques consist of Neural Vector Quantization (NVQ), used in conjunction with MLP compression @du2026_mobilegs.

#place(
  top + right,
  float: true,
  clearance: 0.8em,
)[
  #let thick = 1.8pt
  #let base = 0.6pt

  #let soft-thick = 0.8pt + black
  #let soft-green = rgb("#5f9f6e")
  #let light-blue = rgb("#eaf3ff")
  #let light-orange = rgb("#fff1e6")

  #let C(body, bg: none) = table.cell(align: center + horizon, fill: bg)[#body]
  #let L(body, bg: none) = table.cell(align: left + horizon, fill: bg)[#body]
  #let X = table.cell(align: center + horizon, fill: rgb("#b4b6b4"))[
    #text(fill: black, weight: "bold")[]
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
  #let KG(body) = box(
    fill: rgb("#56aa68"),
    inset: (x: 2pt, y: 1pt),
    radius: 2pt,
  )[
    #text(fill: white, weight: "bold")[#body]
  ]

  #let KB(body) = box(
    fill: rgb("#5698df"),
    inset: (x: 2pt, y: 1pt),
    radius: 2pt,
  )[
    #text(fill: white, weight: "bold")[#body]
  ]

  #let KR(body) = box(
    fill: rgb("#df6f6f"),
    inset: (x: 2pt, y: 1pt),
    radius: 2pt,
  )[
    #text(fill: white, weight: "bold")[#body]
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
    columns: (12pt, 52pt, 74pt, 11.1pt, 11.1pt, 11.1pt, 11.1pt, 11.1pt, 11.1pt),
    stroke: base,
    align: center + horizon,
    inset: (x: 3pt, y: 3pt),
    fill: (x, y) => if (y == 0 and x >= 3) or (x == 0 and y > 0) { luma(230) } else { none },

    table.cell(
      colspan: 3,
      inset: (x: 5pt, y: 3pt),
      stroke: none,
      align: left + horizon,
    )[
      #par(justify: false)[
        Implementations in each architecture, and our codebase:
        #KG[existing], #KR[heavily modified] and #KB[re-implemented].
      ]
    ],

    VH([*4DGS-Nat.*]),
    VH([*1000FPS*]),
    VH([*Instant4D*]),
    VH([*MobileGS*]),
    VH([*Usplat4D*]),
    VH([*Omni-4DGS*]),

    SEC(9, [*Gaussians*]),

    MGL(2, [Gaussians], bg: light-blue),
    MGO([4D], bg: light-blue),
    X, X, X, E, X, XG,

    MGO([3D], bg: light-blue),
    E, E, E, X, E, E,

    MGL(2, [Rotation], bg: light-orange),
    MGO([Quaternion], bg: light-orange),
    X, X, X, E, X, XG,

    MGO([Rotation Matrix], bg: light-orange),
    E, E, E, X, E, E,

    MGL(2, [Shape], bg: light-blue),
    MGO([Isotropic], bg: light-blue),
    E, E, X, E, E, XG,

    MGO([Anisotropic], bg: light-blue),
    X, X, E, X, X, XR,

    MGL(3, [Color \ Basis], bg: light-orange),
    MGO([RGB], bg: light-orange),
    E, E, X, E, E, XG,

    MGO([SH(1)], bg: light-orange),
    E, E, E, X, E, XG,

    MGO([SH(3)], bg: light-orange),
    X, X, E, X, X, XG,

    SEC(2, [*Init*]),

    MGL(2, [Point \ Cloud], bg: light-blue),
    MGO([Random], bg: light-blue),
    X, X, E, E, E, XG,

    MGO([MegaSAM], bg: light-blue),
    E, E, X, E, E, E,

    SEC(3, [*Compress*]),

    C([SH], bg: light-blue),
    L([MLP Distillation], bg: light-blue),
    E, E, E, X, E, XR,

    table.cell(rowspan: 2, align: center + horizon, fill: light-orange)[Quantize],
    L([K-means ], bg: light-orange),
    E, E, E, X, E, XR,

    L([Spatial GPCC], bg: light-orange),
    E, E, E, X, E, XR,

    SEC(3, [*Train*]),
    C([Weighting], bg: light-blue),
    L([Uncertainty], bg: light-blue),
    E, E, E, E, X, XR,

    C([Sampling], bg: light-orange),
    L([Batch in Time], bg: light-orange),
    X, X, E, E, E, XG,

    C([Grid \ Reliance], bg: light-blue),
    L([Voxelization], bg: light-blue),
    E, E, X, X, X, XR,

    SEC(8, [*Prune*]),

    table.cell(rowspan: 2, align: center + horizon, fill: light-orange)[Criterion],
    L([Contribution], bg: light-orange),
    X, E, E, X, E, E,

    L([Gradient Loss], bg: light-orange),
    X, E, E, E, E, XG,

    table.cell(rowspan: 2, align: center + horizon, fill: light-blue)[Quantile Filter],
    L([Spatio-Temporal], bg: light-blue),
    E, E, X, X, E, XB,

    L([Opacity], bg: light-blue),
    E, E, X, E, E, XG,

    table.cell(rowspan: 2, align: center + horizon, fill: light-orange)[Strategy],
    L([One-shot], bg: light-orange),
    E, X, X, E, X, XG,

    L([Scheduled], bg: light-orange),
    E, E, E, E, E, XB,

    C([Increase], bg: light-blue),
    L([Densify], bg: light-blue),
    X, E, E, E, E, XG,

    C([Dropout], bg: light-orange),
    L([Dropout], bg: light-orange),
    E, E, E, E, E, XB,

    SEC(3, [*Render*]),

    C([Loading], bg: light-blue),
    L([Visibility Mask], bg: light-blue),
    E, X, E, E, E, XB,

    MGL(2, [Raster], bg: light-orange),
    MGO([Sort-based], bg: light-orange),
    X, X, X, E, X, XG,

    MGO([Sort-free], bg: light-orange),
    E, E, E, X, E, XR,

    // Inner-table outline.
    table.hline(y: 1, start: 1, end: 9, stroke: soft-thick),
    table.hline(y: 28, start: 1, end: 9, stroke: soft-thick),
    table.vline(x: 1, start: 1, end: 28, stroke: soft-thick),
    table.vline(x: 9, start: 1, end: 28, stroke: soft-thick),

    // Extended mutually-exclusive group boundaries.
    table.hline(y: 3, start: 1, end: 9, stroke: soft-thick),
    table.hline(y: 5, start: 1, end: 9, stroke: soft-thick),
    table.hline(y: 7, start: 1, end: 9, stroke: soft-thick),
    table.hline(y: 10, start: 1, end: 9, stroke: soft-thick),
    table.hline(y: 12, start: 1, end: 9, stroke: soft-thick),
    table.hline(y: 25, start: 1, end: 9, stroke: soft-thick),
    table.hline(y: 27, start: 1, end: 9, stroke: soft-thick),

    // Thin red outline around the rightmost column.
    table.vline(x: 8, start: 0, end: 29, stroke: luma(100) + thick),
    table.vline(x: 9, start: 0, end: 29, stroke: luma(100) + thick),
    table.hline(y: 0, start: 8, end: 9, stroke: luma(100) + thick),
    table.hline(y: 29, start: 8, end: 9, stroke: luma(100) + thick),
  )
]
== Loss

As all our reference papers except USPLAT rely on the _L1_ and _DSSIM_ metrics to train, with a $0.80 - 0.20$ weighing, we start from this prior and introduce the USPLAT loss components.

= Ablations

We train variations of the same architecture on the same example, with fixed computational resources. We record time for convergence, final reconstruction loss, Number of Gaussians, Memory Footprint, Compression, and inference time. The complete table of experiments is provided in the appendix. // TODO add table of results

= Discussion

= Conclusion
