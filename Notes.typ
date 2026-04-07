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
