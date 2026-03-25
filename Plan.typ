
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

3D Gaussian Splatting (3DGS) has become an important model architecture for converting images into a 3D representation. This allows users to construct novel views and reconstruct scenes in space since the environment can be learnt with high visual quality. Scenes are encoded into models by performing training using a set of images, which produces a set of Gaussians distributions in space, with mean, covariance matrix representing spread, and encode color. During training, the Gaussians are projected into images, compared with observed frames, and iteratively updated, so views match the captured scene across space and time faithfully. At this point, rendering is immediate, by combining the color contributions of all distributions. Adding a time component, we obtain 4DGS, which is able to learn from a video to reconstruct the scene in 3D space, tracing how it evolves over time. Gaussian Splatting models can produce high-resolution results, but can be inefficient in the number and size of Gaussians, which is an issue. Recent developments have produced more efficient architectures, and we identified a paper that was able to run 3DGS on a mobile device.

(Stefana)
*Problem Formulation*: Clearly define the problem your project aims to address. Explain what gap, challenge, or opportunity you are focusing on.

Standard models are GPU-intensive, requiring tens of gigabytes of VRAM (GPU ram) for short videos. Since the technique has to run on a mobile device, we need to reduce memory and processing during training, as well as the final size of the output. Recent state-of-the-art results show that 4D model size can be reduced from 2.1 GB to 50 MB, a 41x compression, while efficient pipelines cut GPU memory from about 21 GB to 8 GB, or even 1.1 GB in lightweight modes, and reduce training from roughly 1.2 hours to just 2-7 minutes. On-device Gaussian rendering has also compressed final scene representations from 121 MB to 4.6 MB while sustaining 74-127 FPS on phone-class hardware. Without significant optimization, photorealistic 3D and 4D experiences will stay stuck on high-end computers. Our goal is to break that barrier, so anyone with a phone can experience high-quality 4D rendering.

(Tebe)
*Importance and Relevance of the Problem*: Justify why this problem matters. Discuss its practical, societal, or academic significance, and explain who would benefit from solving it.

Making 4D Gaussian splatting models lightweight and efficient would move high-quality dynamic scene capture from specialized hardware to everyday devices. Real-world uses include mobile AR, telepresence, digital twins, gaming, filmmaking, e-commerce, robotics, and assistive technologies that need fast understanding of changing environments. Running such models on a phone would democratize 3D and 4D content creation, allowing users to scan, replay, and share immersive scenes anywhere without a powerful GPU. This would lower cost, improve accessibility, and expand adoption in education, healthcare, field inspection, tourism, and social media. We are talking about being able to take a video, and immediately obtain a 3D representation of the scene, over time, with spatial consistency.

(Ali)
*Data Sourcing Strategy*: Describe how you plan to obtain or generate the data required for your project.

- *Neural 3D Video (N3V)* contains real dynamic scenes captured from multiple viewpoints and is commonly used to evaluate novel-view synthesis methods on realistic motion, appearance changes, and temporal consistency.
- *D-NeRF* is a synthetic dynamic-scene benchmark with monocular videos and known camera trajectories, making it useful for testing view synthesis quality under controlled motion and geometry.
- *NSFF* contains real dynamic scenes captured with forward-facing monocular videos, and is commonly used to evaluate methods for jointly modeling scene geometry, motion, and novel-view synthesis in casually captured dynamic videos.
- *Nerfies* is a real-world dynamic-scene benchmark focused on non-rigid deformations such as facial expressions and body motion, making it useful for testing view synthesis and deformation modeling under complex non-rigid motion.
- *Hyper-NeRF* extends the Nerfies setting with scenes that exhibit more complex topological changes and non-rigid motion, and is widely used to evaluate methods that need to model deformations beyond simple continuous warps.
- *Ego4DGS* contains egocentric dynamic scenes captured from wearable first-person cameras, and is useful for evaluating dynamic rendering and reconstruction methods under strong camera motion, frequent occlusions, and everyday real-world interactions.
- *NVIDIA Dynamic Scene* contains multi-view dynamic scenes and is widely used to evaluate 4D reconstruction and rendering methods on jointly modeling space, time, and viewpoint changes.
- *DyCheck iPhone* is a monocular dynamic-scene benchmark captured with handheld phones, and is useful for assessing robustness on real-world videos with challenging motion, parallax, and capture noise.

(Stefana)
*Proposed Solution* (High-Level Overview): Provide an overview of your proposed approach or solution. Focus on the key idea and overall strategy rather than implementation details. 
Our *proposed solution* is to combine efficiency-oriented techniques from Instant4D and 1000FPS and design a lightweight 4D Gaussian Splatting pipeline suitable for mobile devices. We use `Instant4D paper` for a faster pipeline, based on isotropic Gaussians and a reduced color complexity, and `1000FPS` to introduce aggressive pruning based on spatio-temporal importance to minimize redundant Gaussians, which speeds up rendering. To further reduce memory footprint and computation, we incorporate compression strategies from the Mobile-GS paper such as quantization and k-means codebook compression, but we extend their results to a 4D architecture. Our overall strategy is to balance quality and efficiency by adapting these methods into a unified pipeline, enabling near real-time 4D scene rendering on resource-constrained devices like smartphones.  

(Ali)
*Performance Evaluation Approach*: Explain how you plan to assess your solution's effectiveness. Specify the metrics, benchmarks, or evaluation criteria you intend to use and why they are appropriate for your problem.

We will evaluate the effectiveness of our solution using a combination of visual quality, compactness, and efficiency metrics, following the evaluation approach used across the uploaded papers. *Storage* measures the compressed model size or footprint, *training cost/time* reflects the optimization effort required to fit a scene, and *memory* measures GPU memory usage during training or inference. For image quality, we will use the following:

- Peak signal-to-noise ratio *PSNR*: measures pixel-level reconstruction accuracy
- Structural similarity index measure *SSIM*: measures structural similarity to the ground-truth image
- Learned Perceptual Image Patch Similarity *LPIPS*: measures perceptual similarity using deep features

We will also assess efficiency and practical usability using other minor metrics:
- *FPS*: frames per second during rendering, measuring runtime speed
- *\# Gaussians*: number of Gaussian primitives, reflecting scene representation complexity
- *Memory*: peak GPU memory usage, indicating hardware efficiency
- *Storage*: final model size, indicating compactness for deployment
- *Training cost/time*: total optimization time, indicating how quickly a scene can be learned
