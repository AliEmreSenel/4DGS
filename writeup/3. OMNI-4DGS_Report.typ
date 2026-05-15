#import "template.typ": cvpr2025
#import "appendix.typ": appendix, contrib-table

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
  Dynamic scene reconstruction converts videos into compact, renderable 4D models. The dominant approach, Native 4D Gaussian Splatting is fast and effective but often suffers from Gaussian overgrowth, high VRAM use, large checkpoints, slow rendering, and fragile pruning or densification choices. OMNI-4DGS studies these quality-efficiency tradeoffs by jointly evaluating representation, rendering, and training decisions. We ablate covariance type, RGB versus 4DSH, rendering strategy, pruning schedules, ESS, dropout, and motion regularization across quality and efficiency metrics. Our best tested quality-compact preset improves visual quality while keeping model size compact, reaching 34.42 PSNR/29k Gaussians on _bouncingballs_ and 31.89 PSNR/81k on _trex_, improving the quality-compactness tradeoff for practical 4DGS deployment.
]

#show: cvpr2025.with(
  title: [OMNI-4DGS: \ Chimera Model for Fast, Lightweight and Precise  \ Video-to-Model Reconstruction],
  authors: (authors, affls),
  keywords: (),
  abstract: abstract,
  bibliography: bibliography("bibliography.bib"),
  accepted: true,
  id: none,
  appendix: appendix,
)

#show link: underline

#link("https://github.com/AliEmreSenel/4DGS")[github.com/AliEmreSenel/4DGS]

= Introduction

Gaussian Splatting has been used extensively for scene reconstruction from videos, thanks to its relatively small training requirements. In the 4D setting (4DGS), frames from a video are used to train a model that represents a scene in space and time. This representation can then be rendered from novel camera views as a function of time, position, orientation, and space. \

In a 4DGS model, the scene is represented by many Gaussian primitives, parametrized by a position vector $mu$, which places their center in space, and a covariance matrix $Sigma = Sigma^T$ that allows anisotropic deformation. In practice, this covariance is parameterized through scale and rotation rather than by storing an arbitrary matrix directly. Color is associated with each Gaussian and encoded by basis coefficients such as Spherical Harmonics (SH) or, in Native 4DGS, 4D Spherindrical Harmonics (4DSH), which approximate view-dependent and time-dependent radiance. Compared to constant color, these coefficients allow smoother appearance at the cost of more variables. Both anisotropic covariance and richer color bases can reduce the number of Gaussians needed by increasing per-primitive expressiveness.

Training iteratively reconstructs ground-truth images by rendering the Gaussians and optimizing image-similarity losses. The final output is a list of Gaussians, which can be projected to a camera plane and rasterized to obtain a reconstructed image. During inference, each pixel color is obtained by integrating Gaussian contributions along the view ray, while accounting for opacity, distance, and remaining transmittance. \

Extending the task from static scenes to dynamic scenes, where objects move or change shape and color, Native 4D Gaussian Splatting (4DGS-Native) @yang2024_4dgs adds a time coordinate and time covariance components to each Gaussian. Rendering must therefore first condition the Gaussians with respect to time before rendering them as in the 3D setting. All mentions of 4DGS refer to the native representation. \

Although dynamic scene reconstruction has produced impressive results, many problems persist: scenes contain many low-importance Gaussians, which increase training time; rendering is limited by costly sorting algorithms; and initialization or training strategies are non-obvious. In this paper, we combine several improvements into a unified implementation, evaluating performance through ablations.

== Gaussian Representation

To understand where these tradeoffs arise, we first describe the Gaussian representation itself. The parametrization of Gaussians in Native 4DGS is a tradeoff between using few variables and being expressive enough that fewer Gaussians are necessary to describe a scene. In 4DGS-Native, each Gaussian is characterized by a position vector $(mu_x, mu_y, mu_z, mu_t) = mu in RR^4$ representing the center of the Gaussian, that is, the mean of the distribution; scale parameters and a 4D rotation, represented using two quaternions \ $r_1, r_2 in HH$, which together define the covariance $Sigma in RR^(4 times 4)$ and distort the Gaussian in time and space. For color, the native Gaussians are equipped with opacity $o in [0,1]$ and 4D Spherindrical Harmonic (4DSH) coefficients, which approximate view- and time-dependent color. In our ablations, the rows labeled 4DSH use spatial degree 3 with a temporal basis, i.e. $(m + 1)^2 (m_t + 1)$ coefficients per color channel; for $m = 3$ and $m_t = 1$, this gives 32 coefficients per channel, or 96 RGB scalars per Gaussian. SH(0) corresponds to plain RGB colors, while the spatial-only SH(3) setting is an outside-the-paper control.

$
  G_i = (mu_i, S_i, (r_1, r_2), o_i, arrow("4DSH")_i)
$

Variations of this representation have also been developed to reduce the number of variables, which reduces training time and storage size but also lowers expressiveness. Isotropic Gaussians are spherical in the spatial dimensions and keep the temporal dimension independent, allowing them to shrink or grow over time while using fewer scalars. This is encoded with a $3 times 3$ spatial covariance proportional to the identity matrix and a separate temporal variance. Here, $I_(3 times 3)$ represents the $3 times 3$ identity matrix.

$
  Sigma =
  mat(
    Sigma_"xyz", 0;
    0, sigma_t,
  )               && "with"
                     Sigma_"xyz" = s_"xyz"^2 I_(3 times 3) \
  Sigma = Sigma^T &&                => Sigma_(x y z,t) = 0
$

== Projection and Rendering

Images may be reproduced from 4DGS-Native Gaussians by first conditioning the distributions in time, which produces a 3D normal distribution. This colored "cloud" is mapped to a 2D camera plane using a world-to-camera projection matrix, reproducing a pixel image. During training, the operation enables the calculation of loss, since the reprojected image can be directly compared to one of the reference images. Generalization loss can also be calculated with images the model was not trained on. Image reconstruction is obtained by first fixing camera position and orientation, projecting the visible Gaussians to screen space, and alpha-blending their projected 2D footprints in depth order. The color of each pixel is obtained by accumulating per-Gaussian color contributions according to opacity and remaining transmittance. The formula for the final color is traditionally sort-dependent, but we also investigate sort-free rendering, as it was shown to reduce render time in related work @du2026_mobilegs.

*Sort-based rendering* consists of filtering the view to only consider relevant Gaussians, and processing them one at a time, integrating their colors in order, to obtain the resulting color of a pixel. For rendering, overall opacity is calculated sequentially to obtain Transmittance $T_i (p, t)$, that is, the "leftover" light level, which modulates the color contribution $c_i (v,t)$ for each successive Gaussian. Remaining light $T_(N+1)$ is attributed to background color $c_("bg")$. In the most general formulation, color is view-direction dependent, as well as time dependent. Moreover, the formula offers limited parallelization, as the sorting operation is a bottleneck.

$
  cases(
    alpha_i (p, t) & = o_i G_i^(2D)(p | t) G_i^t (t),
    T_i (p, t) & = product_(j = 1)^(i - 1) (1 - alpha_j (p, t)),
  )
$

$
  C_p (t, v) = sum_(i=1)^N T_i (p, t) & alpha_i (p, t) c_i (v, t) \
                                      & + T_(N+1)(p, t) c_("bg")
$

*Sort-free rendering*, first proposed in @Hou2024SortFreeGS, removes the depth-sorting step by computing color through an unordered weighted sum. We used @du2026_mobilegs as our reference implementation because it combines sort-free rendering with compact, mobile-oriented 3DGS components, including MLP-predicted weights and opacity. In this formulation, transmittance is computed as an unsorted product, while weights $w_i$ depend on viewing angle, camera position and distance. Since @du2026_mobilegs is based on 3DGS, we extended the MLP architecture by concatenating three time features, $Delta t / T$, $t_"norm"$, and $m_t(t)$, to the MLP inputs. For clarity, we provide the original formula from @du2026_mobilegs, for computing pixel color and Gaussian MLP weights. In the formulas, $Delta x_i$ is the screen-space offset between the pixel and the projected Gaussian center; $Sigma_i$ is the projected 2D covariance matrix of the Gaussian footprint; $d_i$ is the Gaussian depth in camera coordinates; $s_"max"$ is the maximum component of the Gaussian scale in camera coordinates; and $s_i$ and $r_i$ are the Gaussian scale and rotation parameters.

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
  underbrace(phi_i^2, #[MLP \ override])
  + underbrace(frac(phi_i, d_i^2), #[proximity \ effects])
  + underbrace(exp(frac(s_"max", d_i)), #[distance\ effects])
$

$
  P_i = frac(mu_i - k_v, norm(mu_i - k_v))
  quad #stack(dir: ttb, spacing: 5pt, align(left)[camera to Gaussian], align(left)[unit direction])
$

$
  cases(
    tau_i = (Delta t / T, t_"norm", m_t(t)),
    F = "MLP"_f (P_i, s_i, r_i, arrow("4DSH")_i, tau_i),
    phi_i = "ReLU"("MLP"_phi (F)),
    o_i = sigma("MLP"_o (F)),
  )
$

Other optimizations have also been developed for the rendering (inference) operation: following @du2026_mobilegs, contribution-based pruning removes Gaussians using accumulated importance votes based on opacity and scale statistics. Visibility masks are also an option for selectively loading Gaussians at render time: @yuan2025_4dgs1k proposes key-frame temporal filtering with binary labels at sparse key frames, so the rendering step loads only Gaussians visible in the nearby temporal window. Both techniques result in faster inference, by drastically reducing memory loading and problem size.

== Training

Backpropagation is used for training, minimizing image-similarity metrics, with additional loss components as they are introduced by the architecture. We use the Adam optimizer and batch training to improve stability @yang2024_4dgs.

As memory usage is proportional to the number of Gaussians, one seeks to minimize this number by pruning less relevant ones. Opacity pruning drops Gaussians with opacity values below a chosen threshold @yang2024_4dgs. Contribution pruning accumulates a contribution value over training steps and drops Gaussians that remain insufficiently relevant for enough iterations @du2026_mobilegs. Spatio-Temporal pruning ranks Gaussians by visibility and temporal persistence @yuan2025_4dgs1k; in our D-NeRF ablations it removes 15% per pass, applied repeatedly between iterations 2000 and 7500 at interval 2000, while the `_prune` variants use 85%. Finally, Grid pruning de-duplicates Gaussians by position, velocity, temporal scale, and time placement @luo2025_instant4d. since most pruning rules tend to be correlated, in trying to achieve the same goal, we implement Spatio-Temporal pruning, and keep opacity and size-based pruning from 4DGS-Native. Related to pruning, we apply densification to increase the number of Gaussians where necessary. We also implement Edge-Guided densification, adding Gaussians near image edges @Xu_2025_CVPR, and Gradient-Based densification, splitting Gaussians with high loss gradients @sun2024highfidelityslam.

Rasterization is performed through custom `CUDA` kernels for speed. However, compatibility issues arise because the codebases use different versions. For this reason, we opted for a mixed pruning-densification schedule, aiming to simulate a better-than-random initialization, as opposed to MegaSAM, at the cost of worse initialization @luo2025_instant4d.

== Loss

The core training objective is photometric fidelity, while additional regularizers are optional and depend on the architecture. The reconstruction loss is expressed as a weighted combination of the pixel-wise L1 distance between the original and reconstructed images and an SSIM-based structural term, which accounts for luminance, contrast, and texture:

$
  cal(L)_"rgb" = (1 - lambda_"dssim") cal(L)_1 + lambda_"dssim" (1 - "SSIM")
$

For dynamic outdoor scenes, photometric supervision alone is not sufficient: transparent sky regions may still be filled with foreground Gaussians. In our code, the optional sky-mask term is implemented as a ground-truth-alpha opacity loss on sky pixels:

$
  cal(L)_"sky" = -1 / abs(Omega) sum_(p in Omega) m_"sky" (p) log(1 - alpha(p))
$

Dynamic Gaussians also require temporal regularization to avoid physically implausible motion. The rigidity loss encourages nearby Gaussians to move with similar velocities. It is weighted by spatial distance and averaged over
$k = 20$ neighbours:

$
  cal(L)_"rigid" =
  1 / (k dot G)
  sum_(i=1)^G
  sum_(j in cal(N)(i))
  e^(-100 dot d (i,j)) dot
  norm(dot(mu)_i - dot(mu)_j)_2
$

A global motion loss further suppresses high-velocity Gaussians that are unlikely to correspond to real scene motion:

$
  cal(L)_"motion" = 1 / G sum_(i=1)^G norm(dot(mu)_i)_2
$

where $dot(mu)_i$ is the temporal velocity estimated by finite difference. Only $cal(L)_"rgb"$ is active in the `trex` and `bouncingballs` ablations; sky, rigidity, motion, and depth losses are described for completeness, but are disabled in those configs.

Reconstruction quality is then reported using three complementary image metrics:

- Peak Signal-to-Noise Ratio (PSNR): measures pixel-level fidelity.
- Structural Similarity Index Measure (SSIM): captures structural similarity in luminance, contrast, and texture.
- Learned Perceptual Image Patch Similarity (LPIPS): estimates similarity from neural feature activations.

== Uncertainty-Aware Loss

When optimization is applied equally to all Gaussians, low-quality Gaussians remain unconstrained, producing visual inconsistencies. Uncertainty Awareness addresses this problem, activating after base-model convergence @chien2026usplat4d. First, uncertainty $u_(i,t)$ is estimated per Gaussian per frame from alpha-blending weights: well-observed Gaussians contribute strongly to many pixels, thus receiving low uncertainty:

$
  sigma^2_(i,t) =
  1 / (sum_(h in P_(i,t)) (T^h_(i,t) alpha_i)^2),
  quad
  u_(i,t) = cases(
    sigma^2_(i,t) & " " bb(I)_(i,t),
    phi &
  )
$

with $phi = 10^6$. The term $bb(I)_(i,t)$ detects gaussian convergence in color, which forces $u = phi$ unless every pixel in the Gaussian footprint has color residual below $eta_c = 0.5$. This mechanism prevents unconverged Gaussians from being attributed high certainty. Next, the top $2%$ highest-confidence Gaussians over a significant period ($>= 5$ frames) become key nodes $V_k$. From the point cloud of key nodes, kNN is used to add edge connections between key nodes. Each non-key Gaussian is assigned to the closest key node over the full sequence.

*Key-node loss* ensures that reliable Gaussians do not drift from their well-trained and certain positions, while their neighbourhoods move consistently. This is enforced by anchoring them to their pretrained positions $bold(mu)^circle$ and applying motion locality constraints:

$
  cal(L)_"key" =
  sum_t sum_(i in V_k)
  norm(bold(mu)_(i,t) - bold(mu)^circle_(i,t))^2_(U^(-1)_(w,t,i))
  + cal(L)_"motion,key"
$

The uncertainty matrix $U_(i,t) = R_"wc" op("diag")(u, u, 100 u) R_"wc"^T$
encodes the special treatment of the depth direction under monocular depth
unreliability by downweighting depth deviations in the Mahalanobis loss. $cal(L)_"motion,key"$ combines isometry, rigidity, rotation,
velocity and acceleration constraints ($lambda_"iso" = lambda_"rigid" = 1.0$,
$lambda_"rot" = lambda_"vel" = lambda_"acc" = 0.01$).

*Non-key node loss* pulls uncertain Gaussians toward positions predicted by interpolating the motion of their key neighbours. It uses Dual Quaternion Blending (DQB) to interpolate multiple rotations:

$
  cal(L)_"non-key" =
  cal(L)_"pretr,non-key" + cal(L)_"DQB,non-key"
  + cal(L)_"mot,non-key"
$

$
  cal(L)_"pretr,non-key" =
  sum_t sum_(i in.not V_k)
  norm(bold(mu)_(i,t) - bold(mu)^circle_(i,t))^2_(U^(-1)_(w,t,i))
$

$
  cal(L)_"DQB,non-key" =
  sum_t sum_(i in.not V_k)
  norm(bold(mu)_(i,t) - bold(mu)^"DQB"_(i,t))^2_(U^(-1)_(w,t,i))
$

DQB is a soft interpolation for rotations, where $bold(mu)_(i,t)$ remains free and may deviate when the photometric loss provides a stronger signal, preserving non-rigid deformation. Density control is disabled in the first $10%$ and last $20%$ of USPLAT iterations to maintain optimization stability.

#place(
  top + right,
  float: true,
  clearance: 0.8em,
)[
  #figure(
    contrib-table,
    caption: [Contributions of Each Architecture],
  ) <tab:contributions>
]

== Gaussian Regularization

Taking inspiration from @Xu_2025_CVPR, we implement *Random Dropout Regularization*, extending it from 3D to 4D, to ensure that remaining Gaussians are not overly dependent on any particular subset of primitives. At each training iteration, a random mask disables a fraction of Gaussians before rendering, while the full model remains the reference target. This encourages neighboring primitives to share responsibility for explaining observed pixels, reducing sparse-view overfitting, floating artifacts, and hollow artifacts. During inference, all Gaussians are restored, effectively aggregating many low-complexity submodels. The intended effect is smoother geometry and improved generalization, while later refinement can recover high-frequency details.

We also implement an approximate *Edge-guided Splitting Strategy* (ESS) inspired by @Xu_2025_CVPR. ESS increases Gaussian density near image discontinuities instead of distributing new primitives uniformly. Our implementation is not a faithful DropoutGS port: it uses Sobel edges on the ground-truth image rather than a rendering-error map, and it performs a periodic mask-split pass rather than integrating edge probabilities into the standard densification rule. This complements dropout: dropout regularizes the existing primitives, while ESS determines where extra primitives should be allocated.

= OMNI Architecture

Having described the individual components, we now explain how they are integrated into our unified OMNI-4DGS pipeline. We combine components from different codebases into a single unified architecture to test the effectiveness of each component. We summarize our codebase in @tab:contributions, which also highlights the efforts required to combine them.

We train ablations on the `trex` dataset with inference of $400 times 400$, allowing results to be compared with SOTA @yang2024_4dgs. We record PSNR, LPIPS, peak RAM and VRAM usage, memory footprint, number of Gaussians, and render-time FPS. For completeness, we tested the `bouncingballs` dataset. We also run USPLAT ablations on limited 7k-step runs, due to the massive time overhead introduced by our reference USPLAT implementation (Appendix @ablations, while @ablation-coverage explains Ablations Coverage). We run each ablation once, reporting the best checkpoint over training rather than necessarily the final 30k-step checkpoint. Although runs were allowed to continue up to 30k iterations, performance generally peaked earlier, at around 10k--15k iterations, after which mild degradation started. Each ablation was run once, so standardized marginal differences should be interpreted descriptively rather than as estimates of run-to-run statistical significance. Baseline values are taken from the corresponding papers rather than rerun under identical hardware. Moreover, FPS depends strongly on resources and renderers:  comparisons are indicative rather than hardware-normalized.

= Ablation Experiments

We use this unified implementation to isolate the effect of each design choice. From analyzing the `trex` dataset, component effects are not independent: initialization, Gaussian parametrization, color representation, pruning, and rendering all interact through the same budget of Gaussians, parameters, and execution time. To compare effects across metrics with different units, we report relative changes and direction-corrected standardized marginal differences across non-target axes, where positive indicates improvement and negative indicates degradation. Since each ablation was run once, these values are descriptive rankings rather than estimates of run-to-run statistical significance.

== Visual Quality

Visual quality improves most from anisotropic Gaussians and 4DSH color, with the best pruning schedule stemming from interleaved pruning and densification. Anisotropic outperforms isotropic with a higher PSNR ($31.78$ vs. $29.66$, $+7.2%$, $"SMD"=1.42$), higher SSIM ($0.9797$ vs. $0.9713$, $+0.9%$, $"SMD"=1.63$), and lower LPIPS ($0.0218$ vs. $0.0416$, $-47.6%$, $"SMD"=2.55$), showing that directional covariance is important for representing geometry and motion-dependent appearance. 4DSH also strongly improves quality over RGB, raising PSNR from $29.81$ to $31.63$, raising SSIM from $0.9729$ to $0.9780$, and reducing LPIPS from $0.0349$ to $0.0285$. Interleaved pruning and densification gives the best mean PSNR, SSIM, and LPIPS among pruning schedules, while early initialization pruning is not competitive with the strongest quality-compact configurations. Pruning is therefore most useful when paired with densification and evaluated jointly with the Gaussian representation.

== Memory Usage

This quality gain, however, comes with a memory cost. Memory usage is mainly determined by color representation, pruning strategy, and Gaussian count. RGB is the most memory-efficient color model, reducing checkpoint size from $45.1$ to $4.8$ MB ($-89.3%$, $"SMD"=2.78$) and evaluation VRAM from $784.8$ to $631.9$ MB ($-19.5%$, $"SMD"=0.64$). However, this comes with a large quality loss, so RGB is best suited to compact deployment rather than high-fidelity reconstruction. 4DSH gives much better quality, but increases memory. Pruning is beneficial when it reduces active primitives during training: interleaved pruning and densification reduces evaluation VRAM from $752.8$ to $575.0$ MB ($-23.6%$, $"SMD"=0.75$) and gives the smallest mean checkpoint size among pruning schedules. No pruning is less compact than the interleaved schedule, but remains competitive only when speed is prioritized over memory.

== Gaussian Count

The same trend is visible when looking at Gaussian count. Gaussian count is highly scene-dependent, but it qualitatively measures how efficiently the Gaussians are used. Interleaved pruning and densification gives the lowest mean count among pruning schedules, lowering the average to $72.8$k Gaussians while also giving the lowest VRAM and checkpoint size. Anisotropic Gaussians also reduce the required primitives from $82.6$k to $70.8$k while improving quality, meaning they increase representational efficiency rather than merely adding complexity. 4DSH and no-dropout also reduce counts, likely because better appearance modeling and more stable optimization reduce densification pressure. In contrast, ESS increases the mean count from $71.2$k to $82.2$k, so its edge allocation must be balanced against compactness.

== Inference FPS

Speed shows a slightly different pattern. Inference FPS benefits most from removing dropout and from reducing active primitive cost. No-dropout improves FPS from $493.7$ to $791.4$ ($+60.3%$, $"SMD"=0.69$), likely because it produces fewer Gaussians and a simpler final model. Final pruning gives the highest mean FPS among pruning schedules at $690.4$ FPS, while interleaved pruning reduces count and VRAM but reaches $621.7$ FPS on average. This suggests that throughput depends on primitive distribution, memory layout, and renderer implementation, not only the total number of Gaussians.

== Training Time

Training time is most affected by dropout, pruning, and color representation. Removing dropout gives the largest improvement, reducing training time from $89$ to $36$ minutes ($-60.3%$, $"SMD"=2.01$). Dropout therefore adds optimization cost without clear quality or compactness gains. Interleaved pruning and densification remains attractive for VRAM, serialized model size, and quality, but it increases training time relative to the no-pruning schedule because pruning and densification add control overhead. RGB trains faster than 4DSH, but the quality loss makes it less suitable for high-fidelity reconstruction. Dropout is the most harmful choice for training efficiency because it substantially slows convergence.

== Role of each Component

Taken together, these results suggest that no component should be evaluated in isolation. Anisotropic Gaussians are one of the most beneficial components: they improve PSNR, SSIM, LPIPS, and reduce Gaussian count. Isotropic Gaussians use fewer variables per primitive but lose too much representational power. RGB reduces memory and checkpoint size but hurts quality, while 4DSH improves visual fidelity and reduces primitive count at the cost of higher VRAM and checkpoint size. Interleaved pruning best balances quality, count, memory, and VRAM; final pruning is strongest for FPS; early initialization pruning is not selected by the strongest quality-compact configurations. Dropout gives no clear benefit, and removing it improves time, FPS, count, and mean PSNR. Sort-free rendering was not tested as extensively as the other components because pilot runs were noticeably slower than the sorted renderer, while also giving worse overall performance. ESS and USPLAT likewise provided limited gains or excessive overhead in this setting.

We did not identify a clear winner among the ablations, beyond individual tendencies: the best combination must be considered on a case-by-case basis. However, in our experiments the best quality-compact configuration improves visual quality over the reported 4DGS-Native @yang2024_4dgs and 4DGS-1K @yuan2025_4dgs1k baselines, while remaining more compact than 4DGS-Native but slower than 4DGS-1K: \ \

#block(breakable: false)[
  #[*aniso · 4DSH · sort · ESS · interleaved prune · dropout*]

  #v(3pt)

  #table(
    columns: (0.5fr, 0.75fr, 0.42fr, 0.42fr, 0.42fr, 0.35fr, 0.38fr, 0.42fr),
    inset: 4.5pt,
    align: (x, y) => if x > 1 { center } else { left },
    stroke: (x, y) => if y > 0 { (top: 0.35pt + black) },

    text(size: 0.8em)[],
    text(size: 0.9em)[*Method*],
    text(size: 0.8em)[*PSNR↑*],
    text(size: 0.8em)[*SSIM↑*],
    text(size: 0.8em)[*LPIPS↓*],
    text(size: 0.8em)[*FPS↑*],
    text(size: 0.8em)[*MB↓*],
    text(size: 0.8em)[*Gauss↓*],

    table.vline(x: 2, stroke: black + 1pt),

    [BBalls], [4DGS], [33.35], [0.982], [0.025], [462], [84], [134k],
    [], [4DGS-1K], [33.45], [0.983], [0.025], [1509], [13], [20k],
    [], [*Ours*], [*34.42*], [*0.985*], [*0.016*], [460], [18], [29k],

    table.hline(stroke: black + 1pt),

    [TRex], [4DGS], [29.85], [0.980], [0.019], [202], [792], [1265k],
    [], [4DGS-1K], [30.47], [0.981], [0.018], [1361], [118], [189k],
    [], [*Ours*], [*31.89*], [*0.984*], [*0.016*], [386], [50], [*81k*],
  )

  #v(3pt)
  #text(
    size: 0.72em,
  )[MB figures include optimizer state and should be divided by approximately 3 for true size, while base MB and FPS come from the cited papers, on their hardware.]
]

== Discussion and Limitations

These results highlight the central theme of our study: 4DGS performance is governed by coupled tradeoffs rather than independent improvements. 4DGS design is a coupled resource tradeoff: each component affects Gaussian count, per-primitive parameters, memory bandwidth, CUDA time, and stability. Anisotropic covariance improves PSNR, SSIM, and LPIPS despite higher per-Gaussian cost by reducing required primitives. 4DSH similarly increases appearance capacity and lowers densification pressure, while RGB is insufficient for high fidelity. Pruning must follow the target deployment constraint: interleaved pruning balances quality, serialized compactness, and VRAM, while final pruning aids FPS. Best quality-compact results use anisotropy, 4DSH, sort rendering, ESS, interleaved pruning, and no dropout. Limitations include narrow datasets, noisy MOG capture, approximate USPLAT comparison, baselines taken from papers, hardware-dependent FPS, CUDA adaptations, and weak initialization causing floating Gaussians.

= Conclusion

We presented *OMNI-4DGS*, a unified framework for fast, compact dynamic reconstruction built on Native 4D Gaussian Splatting, combining architectural choices from recent work into a single ablation study. Efficient 4DGS requires increasing expressiveness per Gaussian while controlling primitive growth: anisotropic covariance and SH(3) improve reconstruction quality; interleaved pruning and densification reduce Gaussian count, serialized model size, and VRAM, and a final one-shot prune further improves FPS at deployment. Dropout and the adapted sort-free renderer were not consistently beneficial across scenes. The *aniso · SH(3) · sort · ESS · interleaved prune · dropout* configuration achieved the best tested quality-compactness tradeoff on both `trex` and `bouncingballs`.

#colbreak()
