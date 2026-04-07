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
  table.header(
    [*Paper*],
    [*Main focus*],
    [*(Problem -> solution)*],
    [*Datasets*],
    [*Metrics*],
  ),

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

= Links
video
- #link("https://github.com/KAIR-BAIR/dycheck/blob/main/docs/DATASETS.md")[dycheck - monocular clips]
- #link("https://drive.google.com/drive/folders/1qRLBwb5qU5yCS1gb06TQieC4_sMHNeTN?usp=drive_link")[Youtube VOS - monocular video for object segmentation]
- #link("https://davischallenge.org/")[Davis Dataset - video + annotation of objects]
- #link("https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0")[Neural_3D_Video - high quality video]
- #link("https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=1&dl=0")[D-Nerf Dataset - monocular synthetic videos]


image
- #link("https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html")[Image Depth]