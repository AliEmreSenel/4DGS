# Abstract

## Bouncing Balls ⚾

**Fixed:** No USplat/Prune/ESS/Dropout, Sort, 10k.

<table>
  <tr>
    <th></th>
    <th>Sort</th>
    <th>Sort-Free</th>
  </tr>
  <tr>
    <th>Ellipsoid</th>
    <td>
      <img src="./writeup/img/bouncingballs__anisotropic__no_usplat__sh3__sort__no_pruning__no_dropout__no_ess__10000.gif" width="100%">
    </td>
    <td>
      <img src="./writeup/img/bouncingballs__anisotropic__use_usplat__sh3__sort_free__no_pruning__no_dropout__no_ess__10000.gif" width="100%">
    </td>
  </tr>
  <tr>
    <th>Spherical</th>
    <td>
      <img src="./writeup/img/bouncingballs__isotropic__no_usplat__sh3__sort__no_pruning__no_dropout__no_ess__10000.gif" width="100%">
    </td>
    <td>
      <img src="./writeup/img/bouncingballs__isotropic__use_usplat__sh3__sort_free__no_pruning__no_dropout__no_ess__10000.gif" width="100%">
    </td>
  </tr>
</table>

## TRex 🐉

**Fixed:** No USplat/Prune/ESS/Dropout, Sort, 20k.

<table>
  <tr>
    <th></th>
    <th>SH(3)</th>
    <th>RGB</th>
  </tr>
  <tr>
    <th>Ellipsoid</th>
    <td>
      <img src="./writeup/img/trex__anisotropic__no_usplat__sh3__sort__no_pruning__no_dropout__no_ess__20000.gif" width="100%">
    </td>
    <td>
      <img src="./writeup/img/trex__anisotropic__no_usplat__rgb__sort__no_pruning__no_dropout__no_ess__20000.gif" width="100%">
    </td>
  </tr>
  <tr>
    <th>Spherical</th>
    <td>
      <img src="./writeup/img/trex__isotropic__no_usplat__sh3__sort__no_pruning__no_dropout__no_ess__20000.gif" width="100%">
    </td>
    <td>
      <img src="./writeup/img/trex__isotropic__no_usplat__rgb__sort__no_pruning__no_dropout__no_ess__20000.gif" width="100%">
    </td>
  </tr>
</table>

# MOG'D - Dataset

<table>
  <tr>
    <td align="center" width="50%">
      <img src="./writeup/img/mog_moving_cameras.png" width="90%">
      <br>
      <strong>5 x Moving Cameras</strong>
    </td>
    <td align="center" width="50%">
      <img src="./writeup/img/mog_still_cameras.png" width="90%">
      <br>
      <strong>5 x Still Cameras</strong>
    </td>
  </tr>
</table>

# Code Visualization

![](writeup/img/treemap.png)

# Conclusion


# Feature Matrix

| Feature                      | 1000FPS | Instant4D | MobileGS | Usplat4D | Code reference                                                                                                                                                                                                                                                                                                 |
| ---------------------------- | ------- | --------- | -------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gaussians <br> 4D              | ✓       | ✓         |          | ✓        | **Existing** <br> [`GaussianModel.__init__`](scene/gaussian_model.py#L198-L237), [`get_xyzt`](scene/gaussian_model.py#L422-L424), [`render`](gaussian_renderer/__init__.py#L159-L174)                                                                                                                            |
| Gaussians <br> 3D              |         |           | ✓        |          | Not implemented as a mode: [`4D-only guard`](scene/gaussian_model.py#L198-L219), [`--gaussian_dim choices=[4]`](train.py#L1924-L1931)                                                                                                                                                                          |
| Gaussians <br> Quaternion      | ✓       | ✓         |          | ✓        | **Existing** <br> [`get_rotation`](scene/gaussian_model.py#L372-L385), [`build_rotation_4d`](utils/general_utils.py#L113-L133)                                                                                                                                                                                   |
| Gaussians <br> Rotation Matrix |         |           | ✓        |          | Derived only, not optimized as parameters: [`build_rotation`](utils/general_utils.py#L79-L100), [`build_rotation_4d`](utils/general_utils.py#L113-L133)                                                                                                                                                        |
| Gaussians <br> Isotropic       |         | ✓         |          |          | **Existing** <br> [`isotropic_gaussians`](scene/gaussian_model.py#L198-L237), [`get_scaling`](scene/gaussian_model.py#L357-L362), [`_to_isotropic_scaling`](scene/gaussian_model.py#L392-L397)                                                                                                                   |
| Gaussians <br> Anisotropic     | ✓       |           | ✓        | ✓        | **Heavily modified** <br> [`get_scaling`](scene/gaussian_model.py#L357-L362), [`build_scaling_rotation`](utils/general_utils.py#L102-L111), [`build_scaling_rotation_4d`](utils/general_utils.py#L135-L145)                                                                                                      |
| Gaussians <br> RGB             |         | ✓         |          |          | **Existing** <br> [`RGB2SH`](utils/sh_utils.py#L225-L226), [`create_from_pcd`](scene/gaussian_model.py#L484-L555), [`override_color`](gaussian_renderer/__init__.py#L307-L334)                                                                                                                                   |
| Gaussians <br> SH(1)           |         |           | ✓        |          | **Existing** <br> [`_first_order_features`](utils/mobile_compression.py#L65-L84), [`eval_sh`](utils/sh_utils.py#L58-L113)                                                                                                                                                                                        |
| Gaussians <br> SH(3)           | ✓       |           | ✓        | ✓        | **Existing** <br> [`sh_degree = 3`](arguments/__init__.py#L54-L56), [`eval_shfs_4d`](utils/sh_utils.py#L115-L223), [`get_max_sh_channels`](scene/gaussian_model.py#L436-L441)                                                                                                                                    |
| Init <br> Random               | ✓       |           |          |          | **Existing** <br> [`readNerfSyntheticInfo`](scene/dataset_readers.py#L312-L337), [`create_from_pcd`](scene/gaussian_model.py#L484-L555)                                                                                                                                                                          |
| Init <br> MegaSAM              |         | ✓         |          |          | Not found in repo                                                                                                                                                                                                                                                                                              |
| Compress <br> MLP Distillation |         |           | ✓        |          | **Heavily modified** <br> [`MobileOpacityPhiNN`](scene/gaussian_model.py#L85-L128), [`_load_mobilegs_teacher`](train.py#L132-L141), [`MobileGS distillation setup`](train.py#L720-L729), [`distillation loss`](train.py#L941-L950)                                                                               |
| Compress <br> K-means          |         |           | ✓        |          | **Heavily modified** <br> [`_run_kmeans`](utils/mobile_compression.py#L131-L159), [`nvq_encode_tensor`](utils/mobile_compression.py#L162-L209)                                                                                                                                                                   |
| Compress <br> Spatial GPCC     |         |           | ✓        |          | **Heavily modified** <br> [`compress_gpcc`](utils/gpcc_utils.py#L250-L262), [`capture_mobile_payload`](utils/mobile_compression.py#L241-L345)                                                                                                                                                                    |
| Train <br> Uncertainty         |         |           |          | ✓        | **Heavily modified** <br> [`compute_uncertainty_all_frames`](utils/uncertainty.py#L150-L206), [`compute_uncertainty_single_frame`](utils/uncertainty.py#L76-L148), [`build_graph`](utils/graph.py#L105-L204)                                                                                                     |
| Train <br> Batch in Time       | ✓       |           |          |          | **Existing** <br> [`DataLoader batch_size`](train.py#L808-L833), [`batch render loop`](train.py#L902-L903)                                                                                                                                                                                                       |
| Train <br> Voxelization        |         | ✓         | ✓        | ✓        | **Heavily modified** <br> [`build_graph`](utils/graph.py#L145-L171), [`voxelize`](utils/gpcc_utils.py#L79-L88)                                                                                                                                                                                                   |
| Prune <br> Contribution        |         |           | ✓        |          | [`render gaussian_scores`](gaussian_renderer/__init__.py#L523-L541), [`compute_spatio_temporal_variation_score`](scene/gaussian_model.py#L785-L852)                                                                                                                                                            |
| Prune <br> Gradient Loss       |         |           |          |          | **Existing** <br> [`add_densification_stats`](scene/gaussian_model.py#L1337-L1364), [`densify_and_prune`](scene/gaussian_model.py#L1292-L1334)                                                                                                                                                                   |
| Prune <br> Spatio-Temporal     |         | ✓         | ✓        |          | **Re-implemented** <br> [`compute_spatio_temporal_variation_score`](scene/gaussian_model.py#L785-L852), [`prune_with_spatio_temporal_score`](scene/gaussian_model.py#L883-L964), [`scheduled pruning call`](train.py#L1508-L1541)                                                                                |
| Prune <br> Opacity             |         | ✓         |          |          | **Existing** <br> Threshold-based in code: [`densify_and_prune`](scene/gaussian_model.py#L1292-L1334), [`thresh_opa_prune`](arguments/__init__.py#L113-L114)                                                                                                                                                     |
| Prune <br> One-shot            | ✓       | ✓         |          | ✓        | **Existing** <br> [`final_prune_from_iter <br> final_prune_ratio`](arguments/__init__.py#L121-L122), [`final ST prune`](train.py#L1594-L1617)                                                                                                                                                                      |
| Prune <br> Scheduled           |         |           |          |          | **Re-implemented** <br> [`ST pruning args`](arguments/__init__.py#L151-L161), [`scheduled pruning loop`](train.py#L1508-L1541)                                                                                                                                                                                   |
| Prune <br> Densify             |         |           |          |          | **Existing** <br> [`densify_and_clone`](scene/gaussian_model.py#L1249-L1290), [`densify_and_split`](scene/gaussian_model.py#L1065-L1154), [`training call`](train.py#L1394-L1437)                                                                                                                                |
| Prune <br> Edge-guided         |         |           |          |          | **Re-implemented** <br> [`compute_edge_guided_split_mask`](train.py#L199-L294), [`split_points_by_mask`](scene/gaussian_model.py#L1156-L1247), [`ESS schedule`](train.py#L1551-L1578)                                                                                                                            |
| Prune <br> Dropout             |         |           |          |          | **Re-implemented** <br> [`render dropout mask`](gaussian_renderer/__init__.py#L360-L382), [`RDR dropout loss`](train.py#L957-L982), [`random_dropout_prob`](arguments/__init__.py#L78-L91)                                                                                                                       |
| Render <br> Visibility Mask    | ✓       |           |          |          | **Re-implemented** <br> [`build_temporal_visibility_filter`](utils/mobile_compression.py#L450-L535), [`attach_temporal_visibility_filter`](utils/mobile_compression.py#L538-L553), [`_select_temporal_active_mask`](gaussian_renderer/__init__.py#L85-L156)                                                      |
| Render <br> Sort-based         | ✓       | ✓         |          | ✓        | **Existing** <br> [`sorted render path`](gaussian_renderer/__init__.py#L231-L253), [`duplicateWithKeys`](diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu#L70-L113), [`SortPairs`](diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu#L184-L197)                                       |
| Render <br> Sort-free          |         |           | ✓        |          | **Heavily modified** <br> [`sort_free_render`](gaussian_renderer/__init__.py#L195-L213), [`duplicateWithTileKeys`](diff-gaussian-rasterization-ms-nosorting/cuda_rasterizer/rasterizer_impl.cu#L127-L166), [`OIT render`](diff-gaussian-rasterization-ms-nosorting/cuda_rasterizer/rasterizer_impl.cu#L460-L555) |

# References

- [4DGS Native / 4D Gaussian Splatting](https://github.com/fudan-zvg/4d-gaussian-splatting): base 4DGS training/rendering pipeline and dynamic-scene representation.

  - [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): original 3DGS utilities and base Gaussian-splatting components.
  - [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization): CUDA Gaussian rasterizer.
  - [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn): KNN CUDA extension used by Gaussian-splatting code.
  - [Stratified-Transformer / pointops2](https://github.com/JIA-Lab-research/Stratified-Transformer): point cloud CUDA utility ops.

- [Mobile-GS](https://github.com/xiaobiaodu/Mobile-GS): mobile-oriented Gaussian compression / pruning / rendering optimizations.

  * [mpeg-pcc-tmc13](https://github.com/MPEGGroup/mpeg-pcc-tmc13): MPEG GPCC point-cloud compression backend.

- [DropoutGS](https://github.com/xuyx55/DropoutGS): dropout-based Gaussian pruning/compression ideas.

- [Instant4D](https://github.com/Zhanpeng1202/Instant4D): lightweight 4DGS pruning / isotropic Gaussian / fast dynamic-scene optimization ideas.

- [4DGS-1K / 1000FPS 4DGS](https://github.com/4DGS-1K/4DGS-1K.github.io): 1000+ FPS 4DGS project reference and performance-oriented design ideas.

- [USplat4D](https://github.com/cywhitebear/usplat4d): unified/static-dynamic 4D Gaussian splatting reference implementation.

- [stb](https://github.com/nothings/stb): vendored image-writing utility, e.g. `stb_image_write.h`.
