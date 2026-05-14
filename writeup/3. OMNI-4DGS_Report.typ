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
Dynamic scene reconstruction converts videos into compact, renderable 4D models. The dominant approach, Native 4D Gaussian Splatting, is fast and effective but often suffers from Gaussian overgrowth, high VRAM use, large checkpoints, slow rendering, and fragile pruning or densification choices. OMNI-4DGS addresses these quality-efficiency tradeoffs by jointly evaluating representation, rendering, and training decisions. We ablate covariance type, RGB versus SH(3), rendering strategy, pruning schedules, ESS, dropout, and motion regularization across quality and efficiency metrics. The best preset improves visual quality while reducing model size, reaching 34.38 PSNR/37k Gaussians on `bouncingballs` and 32.05 PSNR/97k on `trex`, making 4DGS more deployment-ready.
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

#link("https://github.com/AliEmreSenel/4DGS")[github.com/AliEmreSenel/4DGS]

// TODO add references to why each technique is picked per-technique (ie. memory vs compute vs accuracy constraints)

= Introduction

Gaussian Splatting has been used extensively in the task of scene reconstruction from videos, thanks to its relative small training requirements. In the 4D setting (4DGS), frames from a video are used to train a machine learning model to "learn" a scene in space. This representation can then be used to produce novel camera views as a function of time, position, orientation, and space. \

In a 4DGS model, the building blocks of the scene are a vast number of Gaussian Distributions in space, and are being parametrized by a position vector $mu$, which places their center in space, a covariance matrix $Sigma = Sigma^T$ allowing for diagonal deformation, and a rotation matrix $R$, further orienting the distribution in space. Color is treated as a vector field over the surface of the gaussian, encoded from a fixed number of Spherical Harmonic (SH) coefficients, which can approximate uniformly any color distribution. Compared to having a constant color, SH coefficients allow smooth coloring over the surface, where expressiveness depends on the number of coefficients. Both the use of Covariance+Rotation and SH reduce the number of gaussians needed, since they allow for greater range of behavior, at the cost of a higher number of variables. Training involves the iterated reconstruction of ground truth images from the dataset by rendering the gaussians. Multiple similarity metrics are used to steer each gaussian features in the correct direction, balancing color fidelity with a known depth map, or a structural similarity metric. The final output consists of the list of gaussians, which can be projected to a camera plane and rasterized to obtain a reconstructed image. During inference, camera position is fixed, and a pixel color is obtained by integrating the scene through the view-pixel ray: each gaussian color contribution is added, while accounting for opacity, distance, and leftover un-occluded light, crossing the gaussians in the correct order. Naturally, this allows for the generation of new camera angles. \

Extending the task from static scenes to dynamic scenes, where the objects move or change shape and color, the state of the art (SOTA) architecture is Native 4D Gaussian Splatting (4DGS-Native) @yang2024_4dgs. This model extends the gaussian features by adding a time scalar to position, centering the gaussians in time, and time covariance components, allowing some movement of each gaussian. With this extension, rendering must first condition the gaussians with respect to time, before being able to render them as in the 3D setting. All mentions of 4D Gussian Splatting refer to the native representation. \

Although the field of dynamic scene reconstruction has attracted a lot of interest and produced impressive results, many problems persist with the current models: scenes are affected by a high number of low-importance Gaussians, which drastically increase training time; rendering techniques are limited by costly sorting algorithms; and initialization or training strategies are non-obvious. In this paper, we combine several 4DGS-Native improvements into a unified architecture, verifying their performance through ablations.

== Gaussian Representation

The parametrization of the gaussians in a Native 4DGS is a tradeoff between using few variables and being as expressive as possible, so fewer gaussians are necessary to describe a scene. In 4DGS-Native, each gaussian is characterized by a position vector $(mu_x, mu_y, mu_z, mu_t) = mu in RR^4$ representing the center of the gaussian, that is, the mean of the distribution, a covariance matrix $Sigma in RR^(4 times 4)$, which distorts the gaussian in time and space, a rotation, represented using two quaternions $r_1, r_2 in HH$. For color, the gaussians are equipped with opacity $o in [0,1]$, and a series of SH coefficients, which are used to approximate uniformly any color field on a sphere surface: SH(m) uses $(m + 1)^2$ coefficients for color channels, so SH(3) requires 21 scalars in total, per Gaussian, to encode its color. It should be noted that SH(0) corresponds to having plain RGB colors.

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

*Sort-free rendering*, first proposed in @Hou2024SortFreeGS, computes color directly through a sum, where the weighing of each color contribution is computed by multiple small Multilayer Perceptrons (MLP). The MLP "compresses" information from the gaussians, resulting in smaller storage requirements and faster inference. Moreover, transmittance is computed as an unsorted product, while weights $w_i$ depend on viewing angle, camera position and distance. We picked @du2026_mobilegs as our reference paper for its impressive inference speed on mobile devices. However, since it is based on 3DGS, we extended the MLP architecture by adding a time term $t$ to the MLP inputs. For clarity, we provide the original formula from @du2026_mobilegs, with computing pixel color and gaussian MLP weights. In the formulas, $Delta x_i$ is the screen-space offset between the pixel and the projected Gaussian center; $Sigma_i$ is the projected 2D covariance matrix of the Gaussian footprint; $d_i$ is the Gaussian depth in camera coordinates; $s_"max"$ is the maximum component of the Gaussian scale in camera coordinates; and $s_i$ and $r_i$ are the Gaussian scale and rotation parameters.

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

== Training

We use backpropagation for training, minimizing image-similarity metrics, with additional loss components introduced by the architecture. We use the Adam optimizer and batch training to improve stability @yang2024_4dgs.

Memory usage is proportional to the number of Gaussians, so one seeks to minimize this number by pruning less relevant ones. Opacity pruning drops Gaussians with opacity values below a chosen threshold @yang2024_4dgs @du2026_mobilegs. Contribution pruning accumulates a contribution value over training steps and drops Gaussians that remain insufficiently relevant for enough iterations. Spatio-Temporal pruning drops the bottom $~ 90%$ quantile, ranking higher more persistent and longer-lasting Gaussians @luo2025_instant4d. Finally, Grid pruning de-duplicates Gaussians by position, velocity, temporal scale, and time placement @luo2025_instant4d. Since most pruning rules are correlated, we only implemented Spatio-Temporal pruning, as it accounts for both spatial and temporal contribution, and kept Opacity Pruning from 4DGS-Native. Related to pruning, we apply densification to increase the number of Gaussians where necessary. We implement Edge-Guided densification, which adds Gaussians near image edges @Xu_2025_CVPR, and Gradient-Based densification, which splits Gaussians with high gradients @sun2024highfidelityslam.

Rasterization is performed through custom `CUDA` kernels for speed. However, compatibility issues arise because the codebases use different CUDA versions. For this reason, we opted for a mixed pruning-densification schedule, aiming to simulate a better-than-random initialization, as opposed to MegaSAM, at the cost of worse initalization @luo2025_instant4d.

== Loss

In its full form, training is driven by four loss terms: a photometric reconstruction loss, an opacity regularizer, and two motion regularizers for dynamic Gaussians. The main objective is photometric fidelity, expressed as a weighted combination of the pixel-wise L1 distance between the original and reconstructed images and an SSIM-based structural term, which accounts for luminance, contrast, and texture:

$
  cal(L)_"rgb" = (1 - lambda_"dssim") cal(L)_1 + lambda_"dssim" cal(L)_"SSIM"
$

For dynamic outdoor scenes, photometric supervision alone is not sufficient: transparent background regions may still be filled with opaque Gaussians. To discourage this, an opacity mask loss penalises opacity in sky regions, using the ground-truth alpha channel $m_"gt"$:

$
  cal(L)_"opa" = -1 / abs(Omega) sum_(p in Omega) (1 - m_"gt"(p)) dot log(1 - alpha(p))
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

where $dot(mu)_i$ is the temporal velocity estimated by finite difference. These four terms remain active throughout training.

Reconstruction quality is then reported using three complementary image metrics:

- Peak Signal-to-Noise Ratio (PSNR): measures pixel-level fidelity.
- Structural Similarity Index Measure (SSIM): captures structural similarity in luminance, contrast, and texture.
- Learned Perceptual Image Patch Similarity (LPIPS): estimates perceptual similarity from neural feature activations.

== Uncertainty-Aware Loss

When optimization is applied equally to all Gaussians, low-quality Gaussians remain unconstrained, producing visual inconsistencies. Uncertainty Awareness addresses this problem, activating after base-model convergence. First, uncertainty $u_(i,t)$ is estimated per Gaussian per frame from alpha-blending weights: well-observed Gaussians contribute strongly to many pixels, thus receiving a low uncertainty score:

$
  sigma^2_(i,t) =
  1 / (sum_(h in P_(i,t)) (T^h_(i,t) dot alpha_i)^2),
  quad
  u_(i,t) = cases(
    sigma^2_(i,t) & " " bb(I)_(i,t),
    phi &
  )
$

with $phi = 10^6$. The term $bb(I)_(i,t)$ detects gaussian convergence in color, which forces $u = phi$ unless every pixel in the Gaussian footprint has color residual below $eta_c = 0.5$. This mechanism prevents unconverged Gaussians from being attributed high certainty. Next, the top $2%$ highest confidence gaussians over a significant period ($> 5$ frames) become key nodes $V_k$. From the point cloud of key nodes,
The kNN algorithm is used with $k = 8$ to add edge connections between key nodes. Each non-key Gaussian is assigned to the closest key node over the full sequence.

*Key-node loss* ensures that reliable Gaussians do not drift from their well-trained and certain positions, while their neighbourhoods move consistently. This is enforced by anchoring them to their pretrained positions $bold(mu)^circle$ and applying motion locality constraints:

$
  cal(L)_"key" =
  sum_t sum_(i in V_k)
  norm(bold(mu)_(i,t) - bold(mu)^circle_(i,t))^2_(U^(-1)_(w,t,i))
  + cal(L)_"motion,key"
$

The uncertainty matrix $U_(i,t) = R_"wc" op("diag")(u, u, 0.01 u) R_"wc"^T$
encodes the special treatment of the depth direction under monocular depth
unreliability. $cal(L)_"motion,key"$ combines isometry, rigidity, rotation,
velocity and acceleration constraints ($lambda_"iso" = lambda_"rigid" = 1.0$,
$lambda_"rot" = lambda_"vel" = lambda_"acc" = 0.01$).

*Non-key node loss* pulls uncertain Gaussians toward positions predicted by interpolating the motion of their key neighbours. It uses Dual Quaternion Blending (DQB) to interpolate multiple rotations:

$
  cal(L)_"non-key" =
  cal(L)_"pretrain,non-key" + cal(L)_"DQB,non-key"
  + cal(L)_"motion,non-key"
$

$
  cal(L)_"pretrain,non-key" =
  sum_t sum_(i in.not V_k)
  norm(bold(mu)_(i,t) - bold(mu)^circle_(i,t))^2_(U^(-1)_(w,t,i))
$

$
  cal(L)_"DQB,non-key" =
  sum_t sum_(i in.not V_k)
  norm(bold(mu)_(i,t) - bold(mu)^"DQB"_(i,t))^2_(U^(-1)_(w,t,i))
$

DQB is a soft interpolation, where $bold(mu)_(i,t)$ remains free and may deviate when the photometric loss provides a stronger signal, preserving non-rigid deformation. Density control is disabled in the first $10%$ and last $20%$ of USPLAT iterations to protect graph index integrity.

#place(
  bottom + right,
  float: true,
  clearance: 0.8em,
)[
  #figure(
    contrib-table,
    caption: [Contributions of Each Architecture],
  ) <tab:contributions>
]

== Gaussian Regularization

Taking inspiration from @Xu_2025_CVPR, we implement *Random Dropout Regularization* to ensure that remaining gaussians are not overly dependent on any particular subset of primitives. At each training iteration, a random mask disables a fraction of Gaussians before rendering, while the full model remains the reference target. This encourages neighboring primitives to share responsibility for explaining observed pixels, reducing sparse-view overfitting, floating artefacts, and hollow artifacts. During inference, all Gaussians are restored, effectively aggregating many low-complexity submodels. The result is smoother geometry and improved generalization, while later refinement can recover high-frequency details lost through dropout during training.

We also implement *Edge-guided Splitting Strategy* (ESS) from @Xu_2025_CVPR. ESS increases Gaussian density near image discontinuities instead of distributing new primitives uniformly. We compute edge cues from the training views and use them to identify projected regions where reconstruction error is likely to concentrate around silhouettes, thin structures, and texture transitions. During densification, Gaussians associated with these high-gradient regions are preferentially split or duplicated, giving the model additional capacity where small geometric or color errors are most visible. This complements dropout: dropout regularizes the existing primitives, while ESS determines where extra primitives should be allocated.

= OMNI Architecture

We combine components from different codebases into a single unified architecture, to test the effectiveness of single components singularly. We summarize the contribution of each codebase in the unified @tab:contributions, which also highlights the efforts required to combine them.

We train variations of the same architecture on the `trex` and `bouncingballs` datasets for 20k training steps, with inference of $400 times 400$, so our results are comparable @yang2024_4dgs. We record performance metrics over training, namely PSNR, LPIPS, Peak RAM and VRAM usage, Memory Footprint, Number of Gaussians and render-time FPS. We also run USPLAT ablations on limited 7k step runs, due to the massive time overhead introduced by the USPLAT implementation available to us. The full results are available in Appendix @ablations.

We also record the "MOG" real-life dataset, which consists of 5 videos, captured with handheld mobile devices from different angles. The dataset depicts a sitting male subject, who raises his eyebrows, removes his glasses and puts them back on. We synchronize the videos using clapping as auditory signals to identify beginning and end of the clip. The cameramen were instructed to move their hands in circular motions while recording, to provide all-round imaging with fewer floating artefacts.

= Ablation Experiments

The ablations show that component effects are not independent: initialization, Gaussian parametrization, color representation, pruning, and rendering interact through the same budget of Gaussians, parameters, and CUDA execution time. We therefore report the main trends. To compare effects across metrics with different units, we use Hedges' $g$, a standardized mean-difference coefficient. Since metrics have different improvement directions - higher is better for PSNR, SSIM, and FPS, while lower is better for LPIPS, memory, and Gaussian count - we report direction-corrected $g$ values, where positive indicates improvement and negative indicates degradation.

== Visual Quality

Visual quality improves most from anisotropic Gaussians, SH(3) color, and interleaved pruning. Anisotropic Gaussians outperform isotropic ones with higher PSNR ($+28.0%$, $g=1.40$), higher SSIM ($+6.2%$, $g=1.06$), and lower LPIPS ($-67.1%$, $g=1.15$), showing that directional covariance is important for representing geometry and motion-dependent appearance. SH(3) also strongly improves quality over RGB, raising PSNR from $23.82$ to $30.65$ and reducing LPIPS from $0.0925$ to $0.0280$. Interleaved pruning and densification further improves PSNR, SSIM, and LPIPS, while early initialization pruning degrades all quality metrics. Pruning is therefore useful only after meaningful Gaussian structure has formed.

== Memory Usage

Memory usage is mainly determined by color representation, pruning strategy, and Gaussian count. RGB is the most memory-efficient color model, reducing checkpoint size from $1183.5$ to $303.4$ MB ($-74.4%$, $g=1.66$) and evaluation VRAM from $2231.6$ to $1542.2$ MB ($-30.9%$, $g=0.71$). However, this comes with a large quality loss, so RGB is best suited to compact deployment rather than high-fidelity reconstruction. SH(3) gives much better quality, but increases memory. Pruning is consistently beneficial: final pruning reduces checkpoint size by $80.6%$, while interleaved pruning reduces VRAM by $28.6%$. No pruning is harmful across memory, size, time, and FPS.

== Gaussian Count

 Gaussian count additionally reflects how well each component controls model growth. Interleaved pruning and densification gives the strongest reduction, lowering the count from $2,077,623$ to $769,462$ ($-63.0%$, $g=0.92$). Anisotropic Gaussians also reduce the required primitives from $2,190,855$ to $970,152$ ($-55.7%$, $g=0.84$) while improving quality, meaning they increase representational efficiency rather than merely adding complexity. SH(3) and no-dropout also reduce counts substantially, likely because better appearance modeling and more stable optimization reduce densification pressure. In contrast, no pruning causes uncontrolled growth, while early initialization pruning removes useful structure too early and harms quality and compactness.

== Inference FPS

Inference FPS benefits most from pruning and from reducing active primitive cost. Final pruning is the strongest FPS-oriented component, increasing render speed from $199.7$ to $362.0$ FPS ($+81.2%$, $g=1.03$). No-dropout also improves FPS from $191.0$ to $317.6$ ($+66.2%$, $g=0.79$), likely because it produces fewer Gaussians and a simpler final model. No pruning lowers FPS due to larger Gaussian count and memory use. Interleaved pruning reduces count and VRAM, but gives weaker FPS gains than final pruning, suggesting that throughput also depends on primitive distribution, memory layout, and renderer implementation, not only the total number of Gaussians.

== Training Time

Training time is most affected by dropout, pruning, and color representation. Removing dropout gives the largest improvement, reducing training time from $19696.5$ to $4881.7$ seconds ($-75.2%$, $g=1.62$). Dropout therefore adds optimization cost without clear quality or compactness gains. Interleaved pruning and densification also reduces training time from $19196.7$ to $9607.7$ seconds ($-50.0%$, $g=0.91$), consistent with its lower Gaussian count and VRAM. SH(3) unexpectedly reduces time from $16476.3$ to $10677.6$ seconds, likely because it explains appearance variation with fewer primitives. No pruning and dropout are the most harmful choices because they increase model size or slow convergence.

== Role of each Component

Anisotropic Gaussians are one of the most beneficial components: they improve PSNR, SSIM, LPIPS, and reduce Gaussian count. Isotropic Gaussians use fewer variables per primitive but lose too much representational power. RGB reduces memory and checkpoint size but hurts quality, while SH(3) improves visual fidelity and reduces primitive count at the cost of higher VRAM and checkpoint size. Interleaved pruning best balances quality, count, memory, and training time; final pruning is strongest for FPS and deployment size; early initialization pruning is harmful. Dropout gives no clear benefit, and removing it improves time, FPS, count, and PSNR. Sort-free rendering, ESS, and USPLAT provided limited gains or excessive overhead in this setting.

== Choice of Model

We did not identify a clear winner among the ablations, but we propose some reasonable presets. However, with the following presets, we were able to surpass both 4DGS-Native @yang2024_4dgs and 1000FPS baselines @yuan2025_4dgs1k: \ \


*Anisotropic · SH(3) · sort · ESS · interleaved prune*

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
  [BBalls], [4DGS-1K], [33.45], [0.983], [0.025], [1509], [13], [20k],
  [BBalls], [*Ours*], [*34.38*], [*0.985*], [*0.017*], [1133], [23], [*37k*],
  
  table.hline(stroke: black + 1pt),
  
  [TRex], [4DGS], [29.85], [0.980], [0.019], [202], [792], [1265k],
  [TRex], [4DGS-1K], [30.47], [0.981], [0.018], [1361], [118], [189k],
  [TRex], [*Ours*], [*32.05*], [*0.985*], [*0.015*], [786], [60], [*97k*],
)

== Discussion and Limitations

4DGS design is a coupled resource tradeoff: each component affects Gaussian count, per-primitive parameters, memory bandwidth, CUDA time, and stability. Anisotropic covariance improves PSNR, SSIM, and LPIPS despite higher per-Gaussian cost by reducing required primitives. SH(3) similarly increases appearance capacity and lowers densification pressure, while RGB is insufficient for high fidelity. Pruning must follow structure formation: early pruning destabilizes training, interleaved pruning balances quality and compactness, and final pruning aids deployment. Best results use anisotropy, SH(3), sort rendering, ESS, and interleaved pruning. Limitations include narrow datasets, noisy MOG capture, approximate USPLAT comparison, CUDA adaptations, and weak initialization causing floating Gaussians.

= Conclusion

We have presented *OMNI-4DGS*, a unified framework for fast, compact dynamic reconstruction built on Native 4D Gaussian Splatting, combining architectural choices from recent work into a single ablation study. Efficient 4DGS requires increasing expressiveness per Gaussian while controlling primitive growth: anisotropic covariance and SH(3) improve reconstruction quality, interleaved pruning and densification reduce Gaussian count, memory, and training time, and a final one-shot prune further improves FPS and checkpoint size at deployment. Early pruning, dropout, and the adapted sort-free renderer were not consistently beneficial across scenes. The *anisotropic · SH(3) · sort · ESS · interleaved prune* configuration achieves the best tested quality-compactness tradeoff on both `trex` and `bouncingballs`.