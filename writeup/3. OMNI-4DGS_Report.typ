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
  Dynamic scene reconstruction from video is a core problem in computer vision, with broad applications in AR, robotics, and content creation. 4D Gaussian Splatting (4DGS) has emerged as the dominant approach, offering fast training and flexible scene representation. However, existing methods each optimize for a single axis of performance, leaving practitioners without clear guidance on how to combine improvements or navigate tradeoffs. We present OMNI-4DGS, a unified architecture that integrates recent advances in representation, rendering, and training into a single jointly-evaluated system. Through systematic ablations, we show that no single configuration dominates across all metrics: reconstruction fidelity, inference speed, memory footprint, and training time are in fundamental tension. Our results offer practical guidance for selecting configurations based on deployment priorities, and surface limitations of several recent techniques — including sort-free rendering and uncertainty-aware training — that do not transfer straightforwardly to the 4D setting. This positions OMNI-4DGS as a reference for practitioners designing 4DGS pipelines under real-world constraints.
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

In a 4DGS model, the building blocks of the scene are a vast number of Gaussian Distributions in space, and are being parametrized by a position vector $mu$, which places their center in space, a covariance matrix $Sigma = Sigma^T$ allowing for diagonal deformation, and a rotation matrix $R$, further orienting the distribution in space. Color is treated as a vector field over the surface of the gaussian, encoded from a fixed number of Spherical Harmonic (SH) coefficients, which can approximate uniformly any color distribution. Compared to having a constant color, SH coefficients allow smooth coloring over the surface, where expressiveness depends on the number of coefficients. Both the use of Covariance+Rotation and SH reduce the number of gaussians needed, since they allow for greater range of behavior, at the cost of a higher number of variables. Training involves the iterated reconstruction of ground truth images from the dataset by rendering the gaussians. Multiple similarity metrics are used to steer each gaussian features in the correct direction, balancing color fidelity with a known depth map, or a structural similarity metric. The final output consists of the list of gaussians, which can be projected to a camera plane and rasterized to obtain a reconstructed image. During inference, camera position is fixed, and a pixel color is obtained by integrating the scene through the view-pixel ray: each gaussian color contribution is added, while accounting for opacity, distance, and leftover un-occluded light, crossing the gaussians in the correct order. Naturally, this allows for the generation of new camera angles. \

Extending the task from static scenes to dynamic scenes, where the objects move or change shape and color, the state of the art (SOTA) architecture is Native 4D Gaussian Splatting (4DGS-Native) @yang2024_4dgs. This model extends the gaussian features by adding a time scalar to position, centering the gaussians in time, and time covariance components, allowing some movement of each gaussian. With this extension, rendering must first condition the gaussians with respect to time, before being able to render them as in the 3D setting. All mentions of 4D Gussian Splatting refer to the native representation. \

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

== Training

We use *Backpropagation* for training, minimizing image similarity metrics, but additional loss components are also introduced by the architecture,. We use *Adam Optimizer* and *Batch training* to improve instability @yang2024_4dgs.

Memory usage is proportional to the number of Gaussians, so one seeks to minimize this number by pruning less relevant ones. *Opacity Pruning* drops gaussians with opacity value smaller than a chosen threshold @yang2024_4dgs @du2026_mobilegs. *Contribution Pruning* accumulates a contribution value over training steps, and drops gaussians that remain not relevant enough for enough iterations. *Spatio-Temporal Pruning* drops the bottom $~ 90%$ quantile, ranking higher more persistent and longer-lasting gaussians @luo2025_instant4d. Finally, *Grid Pruning* de-duplicates gaussians by position, velocity, temporal scale and time placement @luo2025_instant4d. Pruning rules are correlated, so we only implement Spatio-Temporal pruning, which had to be written from scratch, and maintain Opacity Pruning from 4DGS-Native. *Densification* is the operation of increasing the number of gaussians: our references skipped it to speed up generation, but we reintroduce it. We use *Edge Guided*, which adds gaussians near image edges @Xu_2025_CVPR, and *Loss Guided*, which propagates per-pixel error back to the gaussians @sun2024highfidelityslam.

Custom `CUDA` kernels are used to compute pixel operations on the GPU directly, though it limits compatibility across codebases. For this reason, we opted for a mixed pruning-densify schedule, as opposed to MegaSAM @luo2025_instant4d.

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

Reconstruction quality is then reported using three complementary image metrics: *Peak Signal to Noise Ratio* (PSNR) measures pixel-level fidelity, *Structural Similarity Index Measure* (SSIM) captures structural similarity in luminance, contrast, and texture, and *Learned Perceptual Image Patch Similarity* (LPIPS) estimates perceptual similarity using the similarity between the activations of two image patches in a network.

== Uncertainty-Aware Loss

If loss minimization is applied to all Gaussians equally, low-importance splats are unconstrained, producing visual artifacts. Uncertainty Awareness addresses this problem, activating after base-model convergence. First, uncertainty $u_(i,t)$ is estimated per Gaussian per frame from alpha-blending weights: well-observed Gaussians contribute strongly to many pixels, thus receiving a low uncertainty score:

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

*Key-node loss* ensures that reliable Gaussians do not drift from their well-trained and certain positions, while their neighbourhoods move consistently.
This is enforced by anchoring them to their pretrained positions $bold(mu)^circle$
and applying motion locality constraints:

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

== Gaussian Regularization

Taking inspiration from @Xu_2025_CVPR, we implement *Random Dropout Regularization* to ensure that remaining gaussians are not overly dependent on any particular subset of primitives. At each training iteration, a random mask disables a fraction of Gaussians before rendering, while the full model remains the reference target. This encourages neighboring primitives to share responsibility for explaining observed pixels, reducing sparse-view overfitting, floaters, and hollow artifacts. During inference, all Gaussians are restored, effectively aggregating many low-complexity submodels. The result is smoother geometry and improved generalization, while later refinement can recover high-frequency details lost through dropout during training.

We also implement *Edge-guided Splitting Strategy* (ESS)

= Implementation Practices // TODO remove if not implemented

Implementation choices have been identified to reduce storage size and improve render speed. As they fall beyond the scope of this project, we only benchmark them in some settings.

== Training Compression

The training procedure can be altered to use less memory, such as by storing a compressed representation, which can be unpacked at inference. These optimizations generaly target Spherical Harmonics, which are stored in numerous scalars @du2026_mobilegs.

*MLP Compression of SH* encodes the view-dependent color of the Gaussians into dense latent vectors using an MLP to : #box[$(h_d, h_v) = ("view", "diffuse")$], which isolate the information between gaussian-specific and perspective specific @du2026_mobilegs. To learn the split, while training "view", the diffuse component is fixed but camera angle changes. The opposite holds for "diffuse", where the view component is fixed. During inference, the MLPs are used to recover color, effectively trading compute for

*Teacher-Student SH Compression* works by training an MLP to compress existing high-order harmonic colors to lower order through a faithful mapping. The operation identifies the set of mappings that minimizes the color loss between the original representation and the lower-order one.

We do not provide support for MLP compression of SH. This issue is also similar to the case of sort-free rendering extended to 4D, which is mentioned in the results section.

== At Rest Compression

The model can be stored as the combination of compression MLPs, and lossy "codebook" approximations @du2026_mobilegs.

*Codebook Compression* is applied to gaussian features by splitting the complete vector of features into sub-vectors, which are replaced with the K-means nearest centroids to obtain a lossy compression of comparatively close vectors. The corresponding index for each gaussian is stored using Huffman Encoding. This technique is also referred to as Neural Vector Quantization (NVQ) @du2026_mobilegs.

*GPCC Compression* is used to encode gaussian positions by first grouping them by position in a grid-based (voxels) discretization of space, then sorting them by Morton Order and storing them in PLY format. The algorithm is implemented in C++ as a single-threaded utility, which runs .

== Render / Inference Optimizations

Similarly to how pruning reduces the number of gaussians, *per-frame visibility masking* increases render speed by loading fewer gaussians from memory, based on contribution to the view @yuan2025_4dgs1k. The technique stores a boolean mask for the gaussians every n frames (n=5 in @yuan2025_4dgs1k), which records if the gaussian is actively contributing to the image. During inference, at each frame t, only the gaussians that are visible in the last computed mask before t and the nearest after t are loaded, which both reduces memory use and compute.

#place(
  bottom + right,
  float: true,
  clearance: 0.8em,
)[
  #contrib-table
]
== Points of Improvement

Several techniques have been developed to improve the known limitations of GS, but any addition may negate the contribution of another. For example, Isotropic Gaussians allow for faster training and smaller memory footprint, but reduce the model's capacity. We run a shared parametrization model to test each architectural component in relation to the others. We combine the papers into a single architecture, which can be studied through ablations. We start from the 4DGS-Native Architecture, adding features from available implementations @yang2024_4dgs @luo2025_instant4d @du2026_mobilegs, and re-implementing the missing structures from the others @yuan2025_4dgs1k @guo2026uncertaintymattersdynamicgaussian.

*Initialization* of gaussian position can be random, or be provided from a point cloud model such as MegaSAM in @luo2025_instant4d. Good initializations massively reducing training time and produce better results. We experiment with the addition of a pruning-densify schedule to improve the robustness of our results over a reduced training time.

*Isotropy* involves choosing between Isotropic gaussians @luo2025_instant4d and Anisotropic Gaussians @yang2024_4dgs, corresponding to a tradeoff between faster training, stemming from a reduced number of variables, and more expressive Gaussians.

*Rotation* encoding relies on two quaternions per-gaussian to learn and encode rotation. Quaternions are preferred to rotation matrices because of their simpler implementation smaller number of parameters, which avoids and reduces the number of variables

*Color* can be represented with Spherical Harmonics, whose precision can be tuned: SH(0) represents RGB colors, while SH(3) uses 48 coefficients to define a color gradient over the gaussian's isosurfaces. Following the 4DGS-Native implementation, training SH of higher order is incremental: higher order harmonics are "unlocked" for learning over training.

*Pruning* is usually treated as a penultimate step in training the model: train from inizialization, prune once, and finetune at the end. Different, yet similar, strategies were proposed for pruning: Opacity Pruning @du2026_mobilegs, which discards transparent gaussians, Contribution pruning @du2026_mobilegs, dropping gaussians that affect the loss minimally, Grid Pruning @luo2025_instant4d. We focus only on Spatio-Temporal Contribution @yuan2025_4dgs1k, as it captures both aspects of prevalence in time and in space.

*Densification* consists of duplicating single Gaussians to give more expressivity where needed. In our reference models, the step is generally discarded to reduce training time, but we reintroduce it as it provides better than random Gaussian initialization than MegaSAM.

*Rendering* changes primarily whether Sort-Based @yang2024_4dgs or Sort-Free @du2026_mobilegs are used. We extend the latter with a time component, to also capture time-dependent behavior.

*Distillation* can be used to effectively reduce the storage size of the SH coefficients, by training a MLP to effectively compress color through a teacher-student model.

*Storage* reduction techniques consist of Neural Vector Quantization (NVQ), used in conjunction with MLP compression @du2026_mobilegs.

*Loss*, as all our reference papers except USPLAT use only _L1_ and _DSSIM_ metrics to train, with a $0.80 - 0.20$ weighing. We start from this prior and introduce the USPLAT later.

= Ablation Results

We train variations of the same architecture on the `trex` and `bouncingballs` datasets for 20k training steps. We record performance metrics over training, namely PSNR, LPIPS, Peak RAM and VRAM usage, Memory Footprint, Number of Gaussians and render-time FPS. We also run USPLAT ablations on limited 7k step runs, due to the massive time overhead introduced by the USPLAT implementation available to us. We discuss our findings in the following section, and we provide the complete table of experiments is provided in the appendix. // TODO add table of results

*Visual Quality*

*Memory Usage*

*Gaussian Count*

*Inference FPS*

*Training time*

== Role of each Component

// TODO add numbers indicatively

*Initialization* plays the role of a strong prior on the voxels in 4DGS. Consequently, random initialization will allow "floaters" in front of the cameras cameras are still in front of the camera, which require either multiple moving camera to improve scene reconstruction or a better initalization. This is particularly apparent in the MOG dataset comparison, which recreates a scene with fixed cameras and moving cameras. // TODO add side by side comparison

// TODO add image

*Isotropy* affects number of gaussians and the reconstruction error. Anisotropic gaussians are more expressive, hence a smaller number is required, at the cost of more variables.

*Appearance* acts similar to isotropy: more RGB gaussians are required to encode color up to the same visual quality than SH(3) gaussians, but the memory use is drastically reduced.

*Sorting* primarily affects training and inference time: sort-free training is much slower, . Moreover, the weight MLP is not able to learn anough features to

*Pruning and Densification* affects the number of gaussians, with _no pruning_ allowing gaussians to grow unchecked, affecting memory but producing visually better results. On the other hand, _interleave_ pruning both adds and removes gaussians, with slightly better visual results.

*Dropout and ESS* both resulted in limited visual improvements. We attribute their minimal help to the fact that our datasets were not highly detailed or complex.

*USPLAT* could not be tested to the same extent as the other ablation due to its massive computation overhead. In fact, while a single run of the default model runs in ~20min, a single USPLAT ablation requires 5h. We ran all of the ablations for 7k iterations, but could not find conclusive differences beyond a very minor visual equality improvement. We were not able to reproduce the motion regularization improvements as they appear in the @guo2026uncertaintymattersdynamicgaussian. We attribute this to our reliance on an unofficial implementation @chien2026usplat4d. // TODO add reference
\

== Choice of Model

We did not identify a clear winner among the ablations. In fact, depending on the list of priorities, we show a sequence of decision diagrams to pick the best set of ablations.

= Discussion and Limitations



= Conclusion
