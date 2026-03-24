
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

4DGS is a popular technique, but limited by the fact that the model must train on the video. This makes it GPU intensive.

(Stefana)
*Problem Formulation*: Clearly define the problem your project aims to address. Explain what gap, challenge, or opportunity you are focusing on.

(Tebe)
*Importance and Relevance of the Problem*: Justify why this problem matters. Discuss its practical, societal, or academic significance, and explain who would benefit from solving it.

Many applications in the real world:
- Scene Capture and Reconstruction
- Democratizes the tool, so anyone can do it without an powerful computer.

(Ali)
*Data Sourcing Strategy*: Describe how you plan to obtain or generate the data required for your project.

Datasets already exists, we can use the same ones from the datasets.

(Stefana)
*Proposed Solution* (High-Level Overview): Provide a overview of your proposed approach or solution. Focus on the key idea and overall strategy rather than implementation details.

Combine the techniques and evaluate achievable results. 

We identified two papers that add inductive bias to make training faster and the final results smaller. 
- 1000FPS:
- Instant4D:
We would like to close the gap by being able to run the model on a phone: this means reducing compute, model size and output size.

We would use for compression the techniques in:
- Mobile-GS:

(Ali)
*Performance Evaluation Approach*: Explain how you plan to assess your solution's effectiveness. Specify the metrics, benchmarks, or evaluation criteria you intend to use and why they are appropriate for your problem.

