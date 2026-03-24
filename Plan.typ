
= Papers
- Instant4D:: 
  1. MegaSAM for camera intrisic estimation
  2. Gaussian Splatting, but isotropic, in order to make training more stable by reducing the degrees of freedom (orientation matrix is fixed so R = I)
  3. Color uses a single RGB instead of superimposition of spherical harmonics. 
  4. Grid Pruning: voxel-based compression, which reduces the size of the model activations. 

- 1000fps:
  1. Penalisation of Short-Lived Gaussians: by not having the temporal effect, gaussians normally get added at each frame, which is problematic and wasteful. The smallest gaussians happen the most at the edge of moving objects. This is because 4DGS treats space and distance directions the same.
  2. Inactive Gaussians: they use `active ratio` to quantify how many gaussians are actually rendered, since a lot of them are small, near and inactive. They also use `Activation-Intersection-Over-Union` to quantify role.
  3. Solution is to prune globally gaussians with low `Spatio-Temporal Variation Score`: Spatial Score (refers to contribution to the pixels of the video), Temporal Score (higher for longer-lasting - 2nd derivative of temporal opacity -> tanh). Scores are combined so Gaussians have to last and be visible at the same time. Then optimise the remaining gaussians.
  4. Nearby Gaussians are made to share their temporal mask, so they get treated as a larger objects.
_They note that pruning ratio is a hyperparameter that may result in poor performance_

- Usplat4D:
  1. Gaussians that are observed in multiple frames should be used as anchors.
  2. A graph is built, where nodes and edges are partitioned by confident / not. Edges are tempral/spatial consistency.
  3. Gaussians are split into: high-confidence (stable) and low-confidence (noisy). The high-confidence onces act as anchors for motion consistency.
  4. Not all Gaussians are equal -> treat the stable ones differently. This helps reduce unstable geometry and flickery.
  5. Graph construction is expensive. Hard to scale

= Metrics

- Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and
- Learned Perceptual Image Patch Similarity (LPIPS): using AlexNet [15] and VGGNet on the N3V dataset and the D-NeRF dataset,.

= Datasets:

= Important differences between the methods:
-Focus:
  Instant4D: fast reconstruction
  1000fps: rendering speed and pruning
  Usplat4D: stability and confidence
-Weaknesses:
  Instant4D: no semantic awarness
  1000fps: afressive pruning -> may lose detail
  Usplat4D: complex pipeline, which is harder to modify

= What none of them is doing:
  1.Object level control
  2.Interactive editing
  3.Segmentation integration
  -All operate at Gaussian-level only!

video
- #link("https://github.com/KAIR-BAIR/dycheck/blob/main/docs/DATASETS.md")[dycheck - monocular clips]
- #link("https://drive.google.com/drive/folders/1qRLBwb5qU5yCS1gb06TQieC4_sMHNeTN?usp=drive_link")[Youtube VOS - monocular video for object segmentation]
- #link("https://davischallenge.org/")[Davis Dataset - video + annotation of objects]
- #link("https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0")[Neural_3D_Video - high quality video]
- #link("https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=1&dl=0")[D-Nerf Dataset - monocular synthetic videos]


image
- #link("https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html")[Image Depth]

= Project Plan

(Tebe)
*Context*: what is 3DGS, 4DGS, current techniques.

Since 2023, 3D Gaussian Splatting (3DGS) has become an important model architecture for converting images into a 3D representation. This allows users to construct novel views and reconstruct scenes in space since the environment can be learnt with high visual quality. Scenes are encoded into models by performing training using a set of images, which produces a set of Gaussians distributions in space, with mean, covariance matrix representing spread, and encode color. During training, the Gaussians are projected into images, compared with observed frames, and iteratively updated, so views match the captured scene across space and time faithfully. At this point, rendering is immediate, by combining the color contributions of all distributions. Adding a time component, we obtain 4DGS, which is able to learn from a video to reconstruct the scene in 3D space, tracing how it evolves over time. Gaussian Splatting models can produce high-resolution results, but can be inefficient in the number and size of Gaussians, which is an issue.

(Stefana)
*Problem Formulation*: Clearly define the problem your project aims to address. Explain what gap, challenge, or opportunity you are focusing on.

== Problem Formulation

Current 4D Gaussian Splatting represents a breakthrough in the dynamic scene reconstruction, but its high computational and storage demands remain a critical limitation for mobile deployment. Standard models are GPU-intensive, requiring gigabytes of data for short videos and relying on a depth-sorting bottleneck that cripples mobile frame rates. Our project addresses these limitations by targeting the 'Gigabyte Problem' and the inefficiency of traditional alpha blending, which forces a per-frame depth-sorting process that consumes up to 50% of the GPU's rendering time. Without significant optimization, photorealistic 3D and 4D experiences will stay stuck on high-end computers. Our goal is to break that barrier so anyone with a phone can experience high-quality 4D rendering.


(Tebe)
*Importance and Relevance of the Problem*: Justify why this problem matters. Discuss its practical, societal, or academic significance, and explain who would benefit from solving it.

If one can make 4D Gaussian splatting models lightweight and efficient, it would move high-quality dynamic scene capture from specialized hardware to everyday devices. Real-world uses include mobile AR, telepresence, digital twins, gaming, filmmaking, e-commerce, robotics, and assistive technologies that need fast understanding of changing environments. Running such models on a phone would democratize 3D and 4D content creation, allowing users to scan, replay, and share immersive scenes anywhere without a powerful GPU. This would lower cost, improve accessibility, and expand adoption in education, healthcare, field inspection, tourism, and social media. We are talking about being able to take a video, and immediately obtain a 3D representation of the scene, over time, with spatial consistency.

(Ali)
*Data Sourcing Strategy*: Describe how you plan to obtain or generate the data required for your project.

Datasets already exists, we can use the same ones from the datasets.

(Stefana)
*Proposed Solution* (High-Level Overview): Provide an overview of your proposed approach or solution. Focus on the key idea and overall strategy rather than implementation details.

Combine the techniques and evaluate achievable results. 

We identified two papers that add inductive bias to make training faster and the final results smaller. 
- 1000FPS:
- Instant4D:
We would like to close the gap by being able to run the model on a phone: this means reducing compute, model size and output size.

We would use for compression the techniques in:
- Mobile-GS:

== Proposed solution
Our idea is to combine efficiency-oriented techniques from Instant4D and 1000FPS to design a lightweight 4D Gaussian Splatting pipeline suitable for mobile devices. Instant4D provides a simplified representation through isotropic Gaussians and reduced color complexity, while 1000FPS introduces aggressive pruning based on spatio-temporal importance to minimize redundant Gaussians. To further reduce memory footprint and computation, we incorporate compression strategies that we have discovered in other research papers. Our overall strategy is to balance quality and efficiency by adapting these methods into a unified pipeline, enabling near real-time 4D scene rendering on resource-constrained devices such as smartphones.



(Ali)
*Performance Evaluation Approach*: Explain how you plan to assess your solution's effectiveness. Specify the metrics, benchmarks, or evaluation criteria you intend to use and why they are appropriate for your problem.

