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
  3DGS represents scenes as learnable gaussians, and 4DGS extends this to dynamic scenes over time. While these models enable high-quality rendering, they are still too heavy for mobile use. Recent work addresses existing limitations from different angles: from the native 4DGS baseline formulation for dynamic reconstruction, 1000FPS speeds up rendering through pruning and visibility masks, Instant4D cuts training cost with a simpler representation and better initialization, MobileGS makes 3D gaussian rendering compact enough for phones, and Usplat4D improves dynamic modeling with uncertainty-aware motion weighting. We build on these contributions to design a lightweight 4DGS pipeline for mobile devices. We validate our results through extensive ablations, and we describe and evaluate the best architecture.
]

#show: cvpr2025.with(
  title: [OMNI-4DGS: Chimera model for fast, light and precise  \ Video-to-Model Reconstruction],
  authors: (authors, affls),
  keywords: (),
  abstract: abstract,
  bibliography: bibliography("bibliography.bib"),
  accepted: none,
  id: none,
  appendix: appendix,
)

#show link: underline

// TODO add references to why each technique is picked per-technique (ie. memory vs compute vs accuracy constraints)

= Introduction

The task of scene reconstruction from videos has been studied extensively in the last years, due to its far-reaching applications in computer vision. As a technique, Gaussian Splatting was originally developed in the late 90' [], but due to its computational demands, it only resurfaced more recently through the 3DGS Architecture. In this static setting, multiple images are used to train a machine learning model to "learn" a scene, obtaining a function that can be used to sample a pixel color from position and orientation of the camera in space. This is functionally similar to previous paradigms such as NERF [], which trained MLPs to reproduce a scene. Gaussian Splatting follows a different paradigm, in that the latent structure of a model is composed of numerous Gaussian Distributions placed 3D space. The Gaussians are projected to a 2D plane and rasterized to obtain a camera representation of the colors corresponding to each pixels of a reconstructed image. Learning is comparing the reconstructed image with the baseline truth from the dataset, but inference allows the syntesis of new camera angles with impressive accuracy. In the formulation, Gaussian Splats serve as building blocks for complex scenes, and are being parametrized by a position vector $mu$, which places their center in space, a covariance matrix $Sigma = Sigma^T$ allowing for diagonal deformation, and a rotation matrix $R$, further orienting the distribution in space. Color is treated as a vector field over the surface of the gaussian, encoded from a fixed number of Spherical Harmonic (SH) coefficients, which can approximate uniformly any color distribution. Compared to having a constant color, SH coefficients allow smooth coloring over the surface, where expressiveness depends on the number of coefficients. Both the use of Covariance+Rotation and SH reduce the number of gaussians needed, since they allow for greater range of behavior, at the cost of a higher number of variables.

During inference, camera position is fixed, and a pixel color is obtained by integrating the scene through the view-pixel ray: each gaussian color contribution is added, while accounting for opacity, distance, and leftover un-occluded light, crossing the gaussians in the correct order.

[3DGS Splatting Image]

Gaussian Splatting allows for novel view synthesis, while offering major improvements in terms of speed and accuracy, compared to previous methods such as NERF, _reaching ..._ []. In moving the task from static scenes to videos, that is, building a model that can learn an evolving scene and reproduce it from novel angles, two major directions have been identified: Wu et al [] used a static 3DGS model with an added decoder-encoder architecture for representing time bias through a latent self-consistent embedding; while 2024, Yang et al [] developed native 4D Gaussian Splatting, which learns the distribution directly using native 4D Gaussian primitives, as opposed to 3D Gaussians with an added time embedding. In this paper, all mentions of 4D Gussian Splatting refer to the native representation.

Although the field of dynamic scene reconstruction has attracted a lot of interest and produced impressive results, many problems persist with the current models. Particularly, they are affected by a high number of low-importance Gaussians, describing the scene inefficiently. Rendering techniques are often limited by costly sorting algorithms, and non-obvious training strategies. In this paper, we combine several 4DGS-Native improvements into a unified architecture, verifying their performance through ablations.

== Gaussian Representation

The parametrization of the gaussians in a Native 4DGS tries to be as compact as possible, while ensuring a high degree of freedom, so individual Guassians are expressive and their contribution to the scene is as high as possible, to effectively reduce their number. In 4DGS-Native, each gaussian is characterized by a position vector $mu in RR^4$ representing the center of the gaussian, that is the mean of the distribution, a covariance matrix $Sigma in RR^(4 times 4)$, which distorts the gaussian in time and space, a rotation, represented using two quaternions $r_1, r_2 in HH$. For color, the gaussians are equipped with an opacity scalar $o$, and a series of SH coefficients: SH(m) uses $2*"m" + 1$ coefficients for color channels, so SH(3) requires 21 scalars in total, per Gaussian, to encode its color. It should be noted that SH(0) corresponds to having plain RGB colors.

$
  G_i = (mu_i, Sigma_i, o_i, (r_1, r_2), arrow("SH")(3)))
$

Variations of the representation have been developed to reduce the number of variables, which reduces training time and storage size. Specifically, the Isotropic is rotationally invariant in space, while changing in time. This is encoded in a matrix with a $3 times 3$ constant-values submatrix, and a time-varying time vector. Here, $bold(1)$ represents the matrix with $1$ in each entry.

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

From a set of 4DGS-Native gaussian, images may be reproduced from known and unknown angles. Mathematically, the procedure is conducted by first conditioning the gaussians distributions in time, which produces a 3D normal distribution. This "colored cloud" is mapped to a 2D camera plane using a world-to-camera projection matrix. During training, the operation enables the calculation of loss, since the reprojected image can be directly compared to one of the reference images. Generalization loss can also be calculated, evaluating loss between pictures that the model has not seen, and reconstructed images. The image reconstruction is obtained traditionally by first fixing camera position and orientation, and evaluating per-pixel color as a function of all the visible gaussians, which contribute to the color profile. More precisely, each pixel induces a ray that marches outwards, intersecting all gaussians in the way between the focal point of the camera end the background, passing through the pixel point. The color of the pixel is obtained through the integration of each color contribution per-gaussian. The formula for the final color is traditionally sort-dependence, but we also investigate sort-free rendering, as it has been shown to greatly reduce the render bottleneck in related work, such as @du2026_mobilegs.

Sort-based rendering has been the traditional technique to compute color: at a high level, it consists of sorting the relevant gaussians for evaluating pixel color, processing their overall opacity one at a time to compute the "leftover" light term (Transmittance) $T_i (p, t)$, which modulates the color contribution $c_i (v,t)$ for each successive gaussian. All leftover light $T_(N+1)$ is simply attributed to the background color $c_("bg")$. It is important to stress that in the most general formulation, color is view-direction dependent, as well as time dependent. Moreover, the formula offers limited parallelization, containing a sorting operation, which induces a bottleneck.

$
  cases(
    alpha_i (p, t) & = o_i G_i^(2D)(p | t) G_i^t (t),
    T_i (p, t) & = product_(j = 1)^(i - 1) (1 - alpha_j (p, t)),
  )
$

$
  C_p(t, v) = sum_(i=1)^N T_i (p, t) &alpha_i (p, t) c_i (v, t) \
  &+ T_(N+1)(p, t) c_("bg")
$

First proposed in @Hou2024SortFreeGS, Sort-Free Rendering removes the bottleneck by computing color directly through a direct sum, where each color contribution is weighed through weights that are computed by multiple small Multilayer Perceptrons (MLP). In a sense, the presence of MLP "compresses" information from the gaussians, resulting in both smaller storage requirements and a faster inference time. Moreover, transmittance is computed as an unsorted product, while weights $w_i$ depend on viewing angle, camera position and distance. In our work, we picked @du2026_mobilegs as our reference paper for its impressive inference speed on mobile devices. However, since it is based on 3DGS, we extend the architecture trivially by adding a time term $t$ to the MLP inputs. For clarity, we provide the original formula from @du2026_mobilegs, with computing pixel color and gaussian MLP weights. In the formulas, $Delta x_i$ is the screen-space offset between the pixel and the projected Gaussian center; $Sigma_i$ is the projected 2D covariance matrix of the Gaussian footprint; $d_i$ is the Gaussian depth in camera coordinates; $s_"max"$ is the maximum component of the Gaussian scale in camera coordinates; and $s_i$ and $r_i$ are the Gaussian scale and rotation parameters.

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
  quad #stack(dir: ttb, spacing: 5pt, align(left)[camera-gaussian], align(left)[unit direction])
$

$
  cases(
    F = "MLP"_f (P_i, s_i, r_i, arrow("SH")(3)_i),
    phi_i = "ReLU"("MLP"_phi (F)),
    o_i = sigma("MLP"_o (F)),
  )
$

Other optimizations have also been developed for the rendering (inference) operation: following @du2026_mobilegs, gaussians with opacity smaller than a threshold are dropped. Visibility masks are also an option for selectively loading gaussians at render time: @yuan2025_4dgs1k proposes binary labellings of the gaussians every 5 frames, so the rendering step loads only gaussians that are visible just before or just after. Both of these techniques result in faster inference time, since they drastically reduce memory loading and reducing the problem size.

== Training Procedure

Standard backpropagation is used to learn the best parameters for scene fidelity, while loss can be measured using common image similarity metrics.

[SSIM metrics]

[Image Reconstruction L1 or L2 loss]

Specific Loss for 4DGS + Loss variations (see USPLAT)

Mention Custom CUDA kernel, difficulty in compatibility (relevant to later saying we did not implement all options)

[In-training Pruning]

== Compression

Pruning

Render-time MLPs

Codebook compression, K-means and GPCC

== Points of Improvement

Several techniques have been developed to improve the known limitations of GS: poor initialization, slow training, high-memory usage, inconsistent reconstruction, unstable training, expensive rendering.

[Explain each of them, citing the paper for each]

These architectural changes have produces incredible improvements, but they have not been tested individually in a unified system.

We compare reconstruction loss, training time, storage size, and render speed, to identify the best combination of techniques.

[Examples of orders of magnitude of each term]

[Mention difficulty with one thing giving an improvement at the cost of another aspect, which is why we do ablations]

= Model Parametrization



= Ablations

We use ablations extensively to evaluate the performance of the model, since each architecture layer introduces errors which may affect the other components negatively. Consequently, the architecture is developed to be modular and exchangeable, for better awareness of each part. We provide a diagram of the combined architecture, and perform repeated ablation experiments, comparing the results to the different techniques.

= Discussion

= Conclusion
