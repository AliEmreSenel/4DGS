== Papers

- *4DGS* (*4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*) uses a *hybrid explicit representation* built from one *canonical set of 3D Gaussians* together with a *4D neural voxel field*, instead of storing separate Gaussians for each frame. This is the main compactness choice. A *Gaussian deformation field network* then maps the canonical Gaussians at time *t* into their deformed state for that timestamp.

  The encoder is a *spatial-temporal structure encoder* built from *6 decomposed multi-resolution planes*: *(x,y)*, *(x,z)*, *(y,z)*, *(x,t)*, *(y,t)*, and *(z,t)*. This is inspired by *HexPlane / K-Planes*, and the queried features are fused with a tiny *MLP*. The decoder is a *tiny multi-head network* with separate *MLP heads* for *position*, *rotation*, and *scale* deformation of each Gaussian.

  After deformation, the model uses standard *differentiable Gaussian splatting* for rendering. The pipeline is: *canonical 3D Gaussians + time -> plane features -> fused feature -> per-Gaussian deformation -> deformed 3D Gaussians -> splatting*. During training, the method first *warms up with static 3D Gaussian optimization* for about *3000 iterations*, then switches on the *4D deformation model*. In one line: *4D-GS = canonical 3D Gaussians + compact space-time feature planes + tiny deformation heads + Gaussian splatting renderer*.

- *Instant4D* uses a compact dynamic-scene design built around a *canonical set of 3D Gaussians* together with a *4D neural voxel field*, instead of storing separate Gaussians for each frame. For camera intrinsics it uses *MegaSAM*. Its Gaussian representation is made *isotropic* to improve training stability by reducing degrees of freedom, with the orientation fixed so *R = I*. It also simplifies appearance by using a single *RGB color* rather than spherical harmonics, and applies *grid pruning* as a voxel-based compression method to reduce activation size.

  The deformation model maps canonical Gaussians at time *t* into their deformed state for that timestamp. The encoder is a *spatial-temporal structure encoder* built from *six decomposed multi-resolution planes*: *(x,y)*, *(x,z)*, *(y,z)*, *(x,t)*, *(y,t)*, and *(z,t)*, inspired by *HexPlane / K-Planes*, followed by a tiny *MLP* for feature fusion. A small multi-head decoder then predicts *position*, *rotation*, and *scale* deformations per Gaussian. After deformation, rendering is done with standard *differentiable Gaussian splatting*.

  The training procedure first warms up with *static 3D Gaussian optimization* for about *3000 iterations*, then activates the 4D deformation model. In one line: *Instant4D / 4D-GS = canonical 3D Gaussians + compact space-time feature planes + tiny deformation heads + Gaussian splatting renderer*.

- *4DGS-1K* (*1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering*) starts from standard *4D Gaussian Splatting*, where the scene is represented by *4D Gaussians* over *(x,y,z,t)* with a *4D covariance*. At each timestamp, every 4D Gaussian is conditioned into a *3D spatial Gaussian* and a *1D temporal Gaussian*. The temporal component controls visibility and opacity, and the visible 3D Gaussians are then alpha-composited.

  The main issue the paper targets is the accumulation of many *short-lived Gaussians*, especially around moving object boundaries. These are wasteful and unstable, partly because vanilla 4DGS does not distinguish well enough between spatial and temporal behavior. The paper measures how many Gaussians are actually useful through quantities such as *active ratio* and *Activation-Intersection-Over-Union*, then introduces a *Spatio-Temporal Variation Score* to rank Gaussians globally. This score combines a *spatial score*, reflecting contribution to rendered pixels, with a *temporal score*, reflecting persistence over time. Low-score, flickering Gaussians are pruned.

  To accelerate inference further, the method stores *active-Gaussian masks* on sparse *keyframes*, and at test time reuses the *union of the two nearest keyframe masks* so inactive Gaussians can be skipped. Nearby Gaussians can also share a temporal mask, making them behave more like larger coherent objects. The overall pipeline is: *train vanilla 4DGS -> prune globally -> precompute active masks -> fine-tune the remaining Gaussians*. One caveat is that the *pruning ratio* is a sensitive hyperparameter and can hurt quality if chosen poorly.

- *USplat4D* is best understood as an *uncertainty-aware refinement layer* on top of a pretrained *dynamic Gaussian Splatting* model, rather than a completely new renderer. The key idea is that not all Gaussians should be treated equally: Gaussians that are consistently observed across time are more reliable and should act as *anchors*, while noisier Gaussians should be guided by them.

  The method assigns each Gaussian a *time-varying uncertainty score* based on rendering evidence, with uncertainty made *depth-aware and anisotropic* in 3D so that poorly constrained directions, especially depth, are handled more carefully. It then divides Gaussians into *key nodes* and *non-key nodes*. Key nodes are the stable, low-uncertainty anchors, selected through *voxelized spatial sampling* and *temporal stability filtering*.

  From there, USplat4D builds an *uncertainty-aware spatio-temporal graph*. Reliable key nodes connect to each other through uncertainty-aware *kNN*, while non-key nodes attach to nearby key nodes over time. Motion is handled in two stages: *key nodes* are optimized directly with uncertainty-weighted motion losses that keep them close to pretrained trajectories, while *non-key nodes* are moved through *dual-quaternion blending (DQB)* from neighboring key nodes and then softly regularized. The final objective combines *RGB reconstruction loss*, *key-node loss*, and *non-key-node loss*.

  In compact form, the pipeline is: *initialize with vanilla dynamic GS, estimate per-Gaussian uncertainty, select confident anchors, build an uncertainty-weighted graph, optimize anchors directly, propagate motion to uncertain Gaussians, and render the refined scene*. Its main downside is that graph construction is expensive and does not scale easily.

- *Mobile-GS* keeps *3D Gaussians* as the underlying scene representation, but redesigns the pipeline so it can run efficiently on mobile hardware. The central change is a *sort-free renderer*: instead of standard sorted alpha blending, it uses *depth-aware order-independent rendering*, where Gaussians are blended in parallel using learned depth- and scale-based weights. This removes the expensive depth-sorting step that usually makes 3DGS hard to deploy on mobile devices.

  Because sort-free blending can introduce transparency and overlap artifacts, the method adds a small *view-dependent correction MLP*. This network takes the *camera-to-Gaussian direction*, along with *scale*, *rotation*, and *spherical-harmonic appearance features*, and predicts corrections to opacity or blending weights. Appearance is compressed further through *SH distillation*, which reduces higher-order spherical harmonics down to *first-order SH* using a teacher model and a *scale-invariant depth distillation loss*.

  The model is also compressed through *neural vector quantization*. Gaussian attributes are split into subvectors, quantized with *multiple codebooks*, and entropy-compressed. Instead of storing SH features densely for every Gaussian, small *MLP decoders* reconstruct them when needed. Finally, *contribution-based pruning* removes Gaussians that consistently have low importance according to a joint *low-opacity + low-scale* criterion.

  Overall, *compressed 3D Gaussians + pruning* produce a small model, *decoded SH features + the correction MLP* restore appearance quality, and the final image is rendered with a *parallel sort-free depth-aware renderer* instead of standard sorted blending.

#pagebreak()
#let speed(body) = text(fill: red, weight: "bold")[#body]
#let memory(body) = text(fill: orange, weight: "bold")[#body]
#let quality(body) = text(fill: green, weight: "bold")[#body]
#let both(body) = text(fill: blue, weight: "bold")[#body]

#set text(size: 8pt)

#table(
  columns: (1.1fr, 1.5fr, 3.6fr, 1.3fr, 1.5fr),
  inset: 5pt,
  align: left + horizon,
  table.header([*Paper*], [*Main focus*], [*(Problem -> solution)*], [*Datasets*], [*Metrics*]),

  [4DGS],
  [Real-time dynamic scene rendering with compact canonical representation],
  [
    per-frame cost -> #both[canonical 3D Gaussians + 4D voxel field + deformation field]
    #linebreak()
    weak local ST coherence -> #quality[6 decomposed ST planes + tiny fusion MLP]
    #linebreak()
    heavy full 4D voxel -> #memory[plane decomposition]
    #linebreak()
    slow deformation decode -> #speed[tiny multi-head decoder (pos/rot/scale)]
    #linebreak()
    unstable joint training -> #quality[3DGS warm-up (~3000 iters)]
  ],
  [D-NeRF; HyperNeRF; Neu3D],
  [PSNR, SSIM / MS-SSIM / D-SSIM, LPIPS, FPS, train time, MB],

  [Instant4D],
  [Compact dynamic GS with simpler/stabler Gaussian parameterization],
  [
    per-frame cost -> #both[canonical 3D Gaussians + 4D voxel field + deformation field]
    #linebreak()
    unstable / heavy Gaussians -> #speed[isotropic Gaussians, fixed rotation]
    #linebreak()
    heavy appearance -> #both[RGB only (no SH)]
    #linebreak()
    large activation volume -> #memory[grid pruning]
  ],
  [same 4DGS-style dynamic benchmarks],
  [quality metrics + FPS / train time / storage],

  [4DGS-1K],
  [Compress and speed up 4DGS by removing temporal redundancy],
  [
    many short-lived / flickering Gaussians -> #both[ST variation score pruning]
    #linebreak()
    many inactive Gaussians still processed -> #speed[keyframe active-mask reuse]
    #linebreak()
    large remaining model -> #memory[VQ + bit-compressed masks (Ours-PP)]
    #linebreak()
    pruning hurts detail -> #quality[fine-tuning after pruning/filtering]
  ],
  [N3V; D-NeRF],
  [PSNR, SSIM, LPIPS, MB, FPS, Raster FPS, Gaussians],

  [USplat4D],
  [Improve monocular dynamic GS under occlusion / hard views via uncertainty],
  [
    all Gaussians treated equally -> #quality[per-Gaussian time-varying uncertainty]
    #linebreak()
    depth ambiguity -> #quality[anisotropic depth-aware uncertainty]
    #linebreak()
    weak global consistency -> #quality[uncertainty-aware ST graph]
    #linebreak()
    unstable motion in uncertain regions -> #quality[key nodes + UA-kNN + DQB propagation]
  ],
  [DyCheck; DAVIS; Objaverse; NVIDIA Dynamic Scenes; HyperNeRF],
  [mPSNR / PSNR, mSSIM / SSIM, mLPIPS / LPIPS; tracking: PCK, EPE],

  [Mobile-GS],
  [Real-time 3DGS on mobile with small storage and fast rendering],
  [
    depth sorting bottleneck -> #speed[depth-aware order-independent renderer]
    #linebreak()
    sort-free artifacts -> #quality[view-dependent correction MLP]
    #linebreak()
    heavy SH appearance -> #both[1st-order SH distillation]
    #linebreak()
    large attribute memory -> #memory[neural vector quantization + feature decoders]
    #linebreak()
    redundant Gaussians -> #both[contribution-based pruning]
  ],
  [Mip-NeRF 360; Tanks&Temples; Deep Blending; mobile eval on Snapdragon 8 Gen 3],
  [PSNR, SSIM, LPIPS, FPS, MB, train time; user study, power],
)

#set text(size: 9pt)

#text(size: 9pt)[
  Legend:
  #speed[speed],
  #memory[memory],
  #quality[render quality],
  #both[both].
]

= Mathematical Formulas and Implementation Details

== USplat4D

The public repo is still effectively empty ("Coming Soon"), so the paper + appendix is the real spec right now. USplat4D is presented as a model-agnostic add-on over an existing dynamic 4DGS pipeline: uncertainty estimation, uncertainty-guided graph construction, and uncertainty-weighted optimization on top of a pretrained dynamic Gaussian model. (#link("https://github.com/TAMU-Visual-AI/usplat4d")[GitHub])

From an existing 4DGS implementation, the minimum math you need is this.

1. *Base 4D Gaussian state*

  $
    G_i^t = (p_(i,t), q_(i,t), s_i, alpha_i, c_i)
  $

  with $p_(i,t) in RR^3$, $q_(i,t) in RR^4$, $s_i in RR^3$, $alpha_i in RR$, and $c_i in RR^(N_c)$.
  This is inherited from the base 4DGS model. (Main paper Eq. 1)

2. *Rendered color and photometric loss*

  For pixel $h$ at frame $t$:

  $
    C_t^h = sum_(i=1)^(N_g) T_(i,t)^h alpha_i c_i
  $

  $
    L_(2,t) = sum_(h in Omega) ||bar(C)_t^h - C_t^h||_2^2
  $

  where $T_(i,t)^h$ is transmittance and $Omega$ is the pixel set.
  This is also inherited, but USplat4D uses it to derive uncertainty. (Eq. 2 / App. S1--S2)

3. *Per-Gaussian scalar uncertainty*

  For Gaussian $i$ at frame $t$:

  $
    sigma_(i,t)^2 = (sum_(h in Omega_(i,t)) (T_(i,t)^h alpha_i)^2)^(-1)
  $

  $
    u_(i,t) := sigma_(i,t)^2
  $

  where $Omega_(i,t) subset Omega$ are pixels influenced by Gaussian $i$.
  This is the core new derivation. (Eq. 3 / App. S3--S6)

4. *Convergence gating for uncertainty*

  Per-pixel convergence:

  $
    bold(1)_t(h) =
    cases(
      1 & "if" abs(bar(C)_t^h - C_t^h)_1 < eta_c \
      0 & "otherwise"
    )
  $

  Per-Gaussian aggregate indicator:

  $
    bold(1)_(i,t) = product_(h in Omega_(i,t)) bold(1)_t(h)
  $

  Final scalar uncertainty:

  $
    u_(i,t) = bold(1)_(i,t) sigma_(i,t)^2 + (1 - bold(1)_(i,t)) phi
  $

  where $phi$ is a large constant. (Eq. 4--5)

5. *Depth-aware anisotropic uncertainty matrix*

  $
    U_(i,t) = R_(w c) U_c R_(w c)^T
  $

  $
    U_c = text("diag")(r_x u_(i,t), r_y u_(i,t), r_z u_(i,t))
  $

  This is what they actually use for graph distances and weighted losses, not just the scalar $u_(i,t)$. (Eq. 6)

6. *Key/non-key partition*

  You need the graph $G = (V, E)$ with:

  - key nodes $V_k$: low-uncertainty Gaussians
  - non-key nodes $V_n$: the rest

  The paper's selection rule is:

  - per-frame voxelization in 3D
  - discard voxels containing only high-uncertainty Gaussians
  - sample one low-uncertainty Gaussian per remaining voxel
  - keep only candidates whose *significant period* is at least $5$ frames, where significant period = number of frames with uncertainty below threshold
  - practical default: top $2%$ most confident Gaussians as keys (#link("https://arxiv.org/html/2510.12768")[arXiv])

7. *UA-kNN for key graph*

  For each key node $i$, choose its best frame:

  $
    hat(t)_i = arg min_t u_(i,t)
  $

  Then connect key neighbors using uncertainty-weighted Mahalanobis distance:

  $
    E_i = text("kNN")_(j in V_k, j != i) ||p_(i,hat(t)_i) - p_(j,hat(t)_i)||_(U_(i,hat(t)_i) + U_(j,hat(t)_i))
  $

  The paper does not explicitly define whether $||x||_A$ means $(x^T A x)^(1/2)$ or $x^T A x$; implement one Mahalanobis convention consistently. (Eq. 7)

8. *Non-key assignment to key graph*

  For each non-key node $i$, attach it to the closest key node across time:

  $
    j^*(i) = arg min_(l in V_k) sum_(t=0)^(T-1) ||p_(i,t) - p_(l,t)||_(U_(i,t) + U_(l,t))
  $

  Then:

  $
    E_i = E_(j^*(i)) union {j^*(i)}
  $

  (Eq. 8)

9. *Key-node objective*

  Let $p_(i,t)^o$ be the pretrained/base-model position. Then:

  $
    L_"key" =
    sum_(t=0)^(T-1) sum_(i in V_k)
    ||p_(i,t) - p_(i,t)^o||_(U_(-1)^(w,t,i))
    + L_"motion,key"
  $

  Interpret the weighted norm as your Mahalanobis penalty with inverse uncertainty. (Eq. 9)

10. *DQB interpolation for non-key nodes*

  For non-key node $i$:

  $
    (p_(i,t)^("DQB"), q_(i,t)^("DQB")) =
    ("DQB")({(w_(i j), T_(j,t))}_(j in E_i))
  $

  where $w_(i j)$ are normalized blending weights and $T_(j,t) in ("SE")(3)$ comes from key-node motion. (Eq. 10)

11. *Non-key objective*

  $
    L_"non-key" =
    sum_(t=0)^(T-1) sum_(i in V_n)
    ||p_(i,t) - p_(i,t)^o||_(U_(-1)^(w,i,t))
    +
    sum_(t=0)^(T-1) sum_(i in V_n)
    ||p_(i,t) - p_(i,t)^("DQB")||_(U_(-1)^(w,i,t))
    +
    L_"motion,non-key"
  $

  The paper notation around the $U_(-1)^(w,i)$ index is a bit inconsistent; in practice this should be the same per-time uncertainty weighting idea as for key nodes. (Eq. 11)

12. *Final objective*

  $
    L_"total" = L_"rgb" + L_"key" + L_"non-key"
  $

  (Eq. 12 / App. S7)

13. *Inherited motion-locality loss*

  USplat4D reuses the standard motion regularizers from prior dynamic 4DGS:

  $
    L_"motion" =
    lambda_"iso" L_"iso"
    + lambda_"rigid" L_"rigid"
    + lambda_"rot" L_"rot"
    + lambda_"vel" L_"vel"
    + lambda_"acc" L_"acc"
  $

  with appendix formulas (S8)--(S12) for isometry, rigidity, relative rotation, velocity, acceleration, and defaults:

  $
    lambda_"iso" = lambda_"rigid" = 1, quad
    lambda_"rot" = lambda_"vel" = lambda_"acc" = 0.01
  $

  If your base 4DGS already has these losses, reuse them exactly rather than re-deriving from OCR. (#link("https://arxiv.org/html/2510.12768")[arXiv])

14. *Perception loss*

  $
    L_"rgb" = "base model perception loss"
  $

  They explicitly say this is the inherited combo of RGB loss + SSIM, plus the base model's 2D priors such as mask, depth, depth-gradient, and tracking. It is not a new USplat4D contribution. (#link("https://arxiv.org/html/2510.12768")[arXiv])

Paper-default knobs worth copying:

- key ratio: top $2%$
- significant period threshold: $5$
- color threshold: $eta_c = 0.5$
- uncertainty scale: from the PDF appendix, they use $[r_x, r_y, r_z] = [1, 1, 0.01]$
- motion-loss weights: $(1, 1, 0.01, 0.01, 0.01)$ (#link("https://arxiv.org/html/2510.12768")[arXiv])

What is *not* fully specified in the paper, so you'll have to borrow from the base implementation or choose yourself:

- exact $k$ in UA-kNN
- exact voxel size / grid resolution for key sampling
- exact large constant $phi$
- exact normalized edge-weight formula $w_(i j)$ for DQB
- exact convention for $||x||_A$
- exact normalization they mention when adapting across SoM/MoSca scales (#link("https://arxiv.org/html/2510.12768")[arXiv])

So the clean implementation split is:

- *reuse from existing 4DGS*: Gaussian state, renderer, $L_"rgb"$, motion-locality losses, DQB primitive if present
- *add for USplat4D*: $u_(i,t)$, $U_(i,t)$, key/non-key partition, UA-kNN graph, uncertainty-weighted key/non-key penalties, final combined loss

== 1000FPS

Using the uploaded paper, this is the minimal math you need to add on top of an existing 4DGS implementation.

*1. Baseline 4DGS math they assume*

Each 4D Gaussian is parameterized by

$
  mu = (mu_x, mu_y, mu_z, mu_t) in RR^4, quad Sigma in RR^(4 times 4).
$

At time $t$, the 4D Gaussian is conditioned into a 3D Gaussian:

$
  mu_(("xyz") | t) = mu_(1:3) + Sigma_(1:3,4) Sigma_(4,4)^(-1) (t - mu_t)
$

$
  Sigma_(("xyz") | t) = Sigma_(1:3,1:3) - Sigma_(1:3,4) Sigma_(4,4)^(-1) Sigma_(4,1:3).
$

Rendering is standard front-to-back alpha compositing:

$
  I(u,v,t) = sum_(i=1)^N c_i(d) alpha_i product_(j=1)^(i-1) (1 - alpha_j)
$

with

$
  alpha_i = p_i(t) p_t(u,v | t) sigma_i, quad
  p_i(t) ~ cal(N)(t; mu_t, Sigma_(4,4)).
$

They then denote $Sigma_t := Sigma_(4,4)$. (#link("https://ar5iv.labs.arxiv.org/html/2503.16422")[ar5iv])

*2. New pruning score: Spatial-Temporal Variation Score*

They rank each Gaussian $g_i$ by a product of a spatial importance term and a temporal persistence term.

Spatial score:

$
  S_i^S = sum_(k=1)^(N H W) alpha_i product_(j=1)^(i-1) (1 - alpha_j)
$

where $k$ indexes all pixels/rays over the rendered training images. This is just the accumulated alpha-blending contribution of Gaussian $i$. (#link("https://ar5iv.labs.arxiv.org/html/2503.16422")[ar5iv])

Temporal score starts from the temporal Gaussian $p_i(t)$. They use its second derivative:

$
  p_i^(2)(t) = ((t - mu_t)^2 / Sigma_t^2 - 1 / Sigma_t) p_i(t).
$

Then they define an opacity-variation term:

$
  S_i^("TV") = sum_(t=0)^T 1 / (0.5 dot tanh(p_i^(2)(t)) + 0.5).
$

They also multiply by a normalized 4D volume term:

$
  gamma(S_i^(4D)) = "Norm"(V(S_i^(4D))),
  quad
  S_i^T = S_i^("TV") gamma(S_i^(4D)).
$

Finally, their combined score is written as

$
  S_i = sum_(t=0)^T S_i^T S_i^S.
$

In practice, that last summation is not important for ranking if $S_i^T$ and $S_i^S$ are already scalar per-Gaussian values; it is just a constant factor $(T + 1)$. The implementation-relevant quantity is effectively

$
  S_i prop S_i^T S_i^S.
$

Gaussians with lower $S_i$ are pruned. (#link("https://ar5iv.labs.arxiv.org/html/2503.16422")[ar5iv])

*3. New rendering-time filter*

They keep the rasterizer itself, but restrict which Gaussians are even sent into it.

Pick sparse keyframe times

$
  {t_i}_(i=0)^T
$

with interval $Delta t$.

For each keyframe $t_i$ and each training camera $j$, render once and collect a visibility mask

$
  m_(i,j)
$

from the standard 4DGS rendering process. They form a per-keyframe active set by unioning masks over views:

$
  M_i = union.big_(j=1)^N m_(i,j).
$

At test time for timestamp $t_"test"$, find the two nearest keyframes $(t_l, t_r)$, and only render Gaussians in

$
  M(t_"test") = M_l union M_r
  = union.big_(i in {l, r}) union.big_(j=1)^N m_(i,j).
$

So the only rendering change is:

- before 4D-to-3D conditioning and rasterization, select Gaussians by the union of the two nearest keyframe masks
- ignore all others for that frame (#link("https://ar5iv.labs.arxiv.org/html/2503.16422")[ar5iv])

*4. What this means in code*

From an existing 4DGS codebase, the method is:

1. Train vanilla 4DGS.
2. For every Gaussian, accumulate $S_i^S$ from rendered training pixels.
3. For every Gaussian, compute $S_i^T$ from its temporal Gaussian $p_i(t)$.
4. Rank by $S_i prop S_i^T S_i^S$.
5. Prune the lowest-scoring Gaussians.
6. Build keyframe visibility masks $M_i$.
7. At render time, use only $M_l union M_r$ for timestamp $t_"test"$.
8. Fine-tune the pruned/filtered model. The paper describes this as a post-processing stage on top of trained 4DGS, followed by 5,000 fine-tuning iterations with clone/split disabled. (#link("https://arxiv.org/html/2503.16422v1")[arXiv])

*5. Paper ambiguities you will have to resolve*

These are the parts not fully specified mathematically:

- *Exact visibility-mask criterion.* They say $m_(i,j)$ is obtained from Eq. 2, but they do not formalize the threshold. In practice, use the rasterizer's per-frame visible/touched Gaussian IDs.
- *Exact 4D volume* $V(S_i^(4D))$. They only say it is normalized “following LightGaussian”. For a 4DGS implementation, the natural choice is the 4D scale product from the Gaussian scales, then normalize across Gaussians.
- *Eq. (7) notation is inconsistent.* The written $sum_t S_i^T S_i^S$ is redundant if $S_i^T$ already includes the temporal sum from Eq. (6). Ranking is unchanged if you drop that extra sum.
- *Time discretization.* Their $t = 0, dots, T$ is over discrete training timestamps / frames, not continuous time. (#link("https://ar5iv.labs.arxiv.org/html/2503.16422")[ar5iv])

*6. Optional, not required for reimplementation*

These are analysis/debug metrics, not core method:

$
  "ActiveRatio"(t) = frac(#"active Gaussians at " t, #"all Gaussians")
$

$
  "ActivationIoU"(t) = frac(|M_0 inter M_t|, |M_0 union M_t|).
$

They use these to justify the filter, not as training objectives. (#link("https://ar5iv.labs.arxiv.org/html/2503.16422")[ar5iv])

If you want, I can also turn this one into a Typst version with stricter compile-safe notation, since a couple of source expressions here are paper-faithful but may need small syntax adjustments depending on your Typst setup.

= Evaluation Combining the Models

There's a naming trap in your list. Your base row matches *Wu et al. 4DGS*: canonical 3D Gaussians, a 4D HexPlane/K-Planes-style encoder, and a deformation decoder that outputs per-time 3D Gaussians before standard splatting. *Instant4D* and *4DGS-1K* are *not* that formulation; they operate in the *native 4D Gaussian* family, where a 4D primitive is conditioned into a 3D Gaussian at render time. *USplat4D* is a model-agnostic uncertainty/graph layer added on top of motion-parameterized monocular dynamic GS, and *Mobile-GS* is a static 3DGS mobile rendering/compression stack. So from a Wu-4DGS codebase, some are patches and some are representation swaps.

My implementation read, if you are starting from *Wu-4DGS*, is: *Mobile-GS = easiest transfer*, *4DGS-1K temporal filtering = next*, *USplat4D = invasive but plausible*, *Instant4D = mostly a re-architecture rather than a patch*.

1) Base model: Wu et al. 4DGS

The Wu et al. base keeps a *canonical 3D Gaussian set* and learns a *Gaussian deformation field*. For a query time `t`, it encodes canonical Gaussian position plus time with a *6-plane multi-resolution spatio-temporal encoder* plus tiny fusion MLP, then uses *separate heads for position, rotation, and scale* to produce deformed 3D Gaussians, which are rendered with ordinary differentiable 3DGS splatting. Training uses a *3000-iteration 3DGS warm-up* before enabling deformation, and the paper reports *L1 image loss + grid TV regularization*. Official repos expose train/render/eval scripts and dataset configs.

Implementation-wise, the clean module split is: `Gaussian state` = canonical 3DGS attributes, `time encoder` = six pairwise spatio-temporal planes + tiny MLP, `decoder` = three tiny heads for `Δx, Δrot, Δscale`, `renderer` = unchanged 3DGS rasterizer after deformation, `trainer` = 3DGS warm-up then joint optimize. That is the baseline everything below should be compared against.

2) Instant4D

A *faithful Instant4D implementation* does *not* start from Wu's deformation-field representation. It starts from a *monocular SLAM/depth front-end*: MegaSAM gives camera/intrinsics/depth initialization, depths are refined for temporal consistency, pixels are back-projected to a dense colored point cloud, motion probabilities are thresholded with *Otsu* to get static/dynamic masks, and pseudo-frames are added at sequence ends to stabilize motion segmentation. Then it *grid-prunes* the dense back-projected cloud by voxel hashing, keeping centroids and averaging attributes per occupied voxel.

The actual 4D representation in Instant4D is a *native 4D Gaussian*, not Wu's canonical-3D-plus-deformation model. The paper explicitly says it uses a *native 4D representation*, conditions a 4D Gaussian to 3D at render time, replaces high-order temporal appearance with *plain RGB*, and makes the covariance *isotropic* by fixing orientation to identity and using only one spatial scale plus one temporal scale. It also makes temporal scale *motion-aware*: static regions get large temporal support, while dynamic regions get smaller temporal support so far-away times are naturally suppressed. On DyCheck it uses *5000 iterations*, lowers the position LR to `1e-5`, and on NVIDIA shortens optimization to *1500 iterations*; the repo pipeline is essentially `reconstruct -> prune -> optimize`.

From a *Wu-4DGS codebase*, the parts that transfer cleanly are the *SLAM/backprojection initializer*, *voxel/grid pruning*, *RGB-only appearance*, and optionally *isotropic Gaussians*. The part that does *not* transfer cleanly is the published representation itself, because Instant4D's core paper result is tied to *native 4D primitives conditioned to 3D*, not to Wu's HexPlane deformation field. So a faithful reproduction should use their repo or a native-4D base; a Wu-based port would be a *hybrid inspired by Instant4D*, not the paper's exact model.

3) 4DGS-1K

4DGS-1K is also built on the *native 4D Gaussian* family: each primitive has a 4D mean/covariance and is decomposed into conditional 3D Gaussians plus a 1D temporal marginal at render time. The paper's first step is *global pruning* with a *Spatial-Temporal Variation Score*. The *spatial term* aggregates each Gaussian's rendering contribution over pixels/views; the *temporal term* uses the *second derivative of the temporal opacity function*, maps that stability measure into a bounded score, and combines it with *normalized 4D volume* so short-lived flickering Gaussians get low scores and are pruned.

Its second step is the part that transfers best: *keyframe-based temporal filtering*. They render training views at sparse keyframe times, union visibility across views to get an *active mask* per keyframe, and at arbitrary time `t` they rasterize only the Gaussians activated by the *two nearest keyframes' masks*. After pruning/filtering, they *fine-tune 5000 iterations* with clone/split disabled. The optional *Ours-PP* stage adds *vector quantization on SH* plus *bit-compressed masks*. On paper this is reported on *N3V and D-NeRF* with PSNR/SSIM/LPIPS plus storage, FPS, raster FPS, and \#Gaussians.

In a *Wu-4DGS* codebase, the *exact* ST-variation formula does *not* transfer verbatim, because Wu's model has no native 4D covariance/temporal-opacity primitive to differentiate in the same way. But the *temporal active-mask idea transfers very well*: Wu still produces deformed 3D Gaussians for each queried time, so you can precompute keyframe visibility masks and filter rasterization the same way. A practical Wu-style analogue of their pruning score is: sample timesteps, accumulate each Gaussian's alpha contribution over training views, estimate its lifespan from visibility/opacity over time, then prune low-contribution short-lifespan Gaussians before mask-based rendering. That last sentence is my implementation inference, not the exact 4DGS-1K paper recipe.

4) USplat4D

USplat4D is best thought of as an *uncertainty-aware optimization wrapper* over an existing monocular dynamic Gaussian model. The paper estimates *per-Gaussian, per-frame scalar uncertainty* from the photometric objective using a closed-form variance approximation; if the Gaussian's covered pixels are not converged, it assigns a *large fallback uncertainty*. It then converts that scalar into an *anisotropic 3D uncertainty matrix* by propagating image-space error into 3D with camera rotation, specifically to avoid being over-confident in depth.

On top of that, it builds an *uncertainty-encoded graph*. Key nodes are selected from the most reliable Gaussians using voxelized coverage and a *significant-period* rule; the paper says it keeps about the *top 2%* as key nodes and requires at least *5 reliable frames*. Key-to-key edges use *uncertainty-aware nearest neighbors* with a Mahalanobis distance that down-weights uncertain directions; non-key nodes attach to nearby reliable key nodes over time. Optimization then splits into *key-node anchoring losses* and *non-key Dual Quaternion Blending (DQB) losses*, with motion regularizers such as isometry, rigidity, rotation, velocity, and acceleration, plus photometric loss and the inherited 2D priors from the base model. The appendix training schedule continues *SoM for 400 epochs* or *MoSca for 1600 steps*, batch size 8, and disables density control / opacity reset during the first 10% and last 20% of training.

For *Wu-4DGS*, USplat4D is plausible but invasive. The paper is explicit that it is compatible with dynamic GS methods that provide initial motion parameters, but their implementation section is only for *SoM and MoSca*, and the public repo currently just says *“Coming Soon.”* So on Wu-4DGS you would need to sample each canonical Gaussian's deformed pose across time from the deformation network, estimate uncertainties from rendered views, build the graph on canonical Gaussian IDs, and backprop the uncertainty-weighted graph losses into the deformation MLP. That is a reasonable research implementation, but it is not an officially validated port.

5) Mobile-GS

Mobile-GS is the cleanest engineering import into Wu-4DGS because it mostly changes the *renderer/compression/pruning stack*, not the dynamic representation itself. The core renderer swap is *depth-aware order-independent rendering*: instead of sorted alpha blending, it accumulates weighted Gaussian colors with a global transmittance term and *inverse-depth-aware weighting*, removing the expensive depth sort. Because that introduces transparency artifacts in overlapping geometry, Mobile-GS adds a *view-dependent opacity/enhancement MLP* that takes the camera-to-Gaussian vector plus Gaussian scale, rotation, and SH features to predict a corrective view-conditioned modulation.

For memory, it distills a teacher to *first-order SH*, uses *scale-invariant depth distillation*, then applies *neural vector quantization*: split each Gaussian attribute vector into sub-vectors, run *k-means per subspace*, Huffman-code the indices, and reconstruct SH from two small *16-bit MLPs* at inference. For redundancy, it uses *contribution-based pruning* driven jointly by opacity and max scale, with a vote-accumulation rule; the paper's ablation picks *0.2* as the best pruning threshold. Training is *60k iterations*, NVQ is turned on at *35k*, the view-dependent branch is initialized to output *1*, and the released repo uses a *30k pretrain + checkpoint fine-tune* workflow. The paper evaluates on *Mip-NeRF 360, Tanks & Temples, Deep Blending*, and reports mobile results on *Snapdragon 8 Gen 3*, including FPS and power.

In *Wu-4DGS*, this is almost a drop-in at the back end: Wu already converts canonical Gaussians into *time-conditioned 3D Gaussians before rendering*, so you can replace the final sorted 3DGS rasterizer with Mobile-GS's sort-free renderer and then add its opacity-MLP, pruning, and quantization around that stage. That is my implementation inference, but it is a very natural one because both methods operate on a set of 3D Gaussians at render time.

If your actual starting point is *Wu-4DGS*, the most practical roadmap is: keep the Wu deformation field, add *Mobile-GS* for rendering/compression, add *4DGS-1K-style active masks* for speed, then decide whether you want a heavier research branch for *USplat4D*, and treat *Instant4D* as either a monocular-initialization hybrid or a full switch to a native-4D codebase. Official code is public for *Wu-4DGS*, *Instant4D*, and *Mobile-GS*; *USplat4D*'s repo is only a placeholder right now, and for *4DGS-1K* I found the project page but not a released training repo as of today.

= Links
video
- #link("https://github.com/KAIR-BAIR/dycheck/blob/main/docs/DATASETS.md")[dycheck - monocular clips]
- #link(
    "https://drive.google.com/drive/folders/1qRLBwb5qU5yCS1gb06TQieC4_sMHNeTN?usp=drive_link",
  )[Youtube VOS - monocular video for object segmentation]
- #link("https://davischallenge.org/")[Davis Dataset - video + annotation of objects]
- #link("https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0")[Neural_3D_Video - high quality video]
- #link(
    "https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=1&dl=0",
  )[D-Nerf Dataset - monocular synthetic videos]


image
- #link("https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html")[Image Depth]
