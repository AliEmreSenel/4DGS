= Project Plan

(Tebe)
*Context*: what is 3DGS, 4DGS, current techniques.

3D Gaussian Splatting (3DGS) has become an important model architecture for converting images into a 3D representation. Scenes are encoded into models by performing training using a set of images, which produces a set of Gaussians distributions in space, with mean, covariance matrix representing spread, and encode color. During training, the Gaussians are projected into images, compared with observed frames, and iteratively updated, so views match the captured scene across space and time faithfully. Rendering occurs by combining the color contributions of all distributions. Adding a time component, we obtain 4DGS, which is able to learn from a video to reconstruct the scene in 3D space, tracing how it evolves over time. Gaussian Splatting models can produce high-resolution results, but can be inefficient in the number and size of Gaussians. Recent developments have produced more efficient architectures, and we even identified a paper that was able to run 3DGS on a mobile device.

(Stefana)
*Problem Formulation*: Clearly define the problem your project aims to address. Explain what gap, challenge, or opportunity you are focusing on.

Standard models are GPU-intensive, requiring tens of gigabytes of VRAM (GPU ram) for short videos. Since the technique has to run on a mobile device, we need to reduce memory and processing during training, as well as the final size of the output. Recent state-of-the-art results show that 4D model size can be reduced from 2.1 GB to 50 MB, a 41x compression, while efficient pipelines cut GPU memory from about 21 GB to 8 or even 1.1 GB in lightweight modes, and reduce training from roughly 1.2 hours to just 2-7 minutes. On-device Gaussian rendering has also compressed final scene representations from 121 MB to 4.6 MB while sustaining 74-127 frames per second on phone-class hardware. Without significant optimization, photorealistic 3D and 4D experiences will stay stuck on high-end computers. Our goal is to break that barrier, so anyone with a phone can benefit from high-quality 4D rendering.


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
Our *proposed solution* is to combine efficiency-oriented techniques from Instant4D and 1000FPS and design a lightweight 4D Gaussian Splatting pipeline suitable for mobile devices. We use the first paper for a faster pipeline, based on isotropic Gaussians and a reduced color complexity, and the second one to introduce aggressive pruning based on spatio-temporal importance to minimize redundant Gaussians, which speeds up rendering. To further reduce memory footprint and computation, we incorporate compression strategies from the Mobile-GS paper such as quantization and k-means codebook compression, but we extend their results to a 4D architecture. Our overall strategy is to balance quality and efficiency by adapting these methods into a unified pipeline, enabling near real-time 4D scene rendering on resource-constrained devices like smartphones.  

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
