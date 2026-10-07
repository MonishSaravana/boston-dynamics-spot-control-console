# Milestone 2: semantic objects in the 3D world

Status: implemented and validated in this workspace. The existing known-pose map and Spot console remain independent.

## Architecture

`RgbdFrame` → `ObjectDetector.detect()` → `ObjectObservation2D` → depth projection → `ObjectObservation3D` → episode-local entity fusion → `WorldEntity`.

The first detector uses exact synthetic raycast part IDs grouped into table, chair A, chair B, and backpack instances. Its masks prove projection and association without neural-model error. The 3D observation records mask provenance, valid-depth fraction, sampled world points, robust center, observed bounds, covariance, and timestamp. Bounds describe observed surfaces; hidden object extent is not inferred.

The real detector is a separate optional dependency. [TorchVision Mask R-CNN v2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html) provides COCO instance masks and scores. Its [BSD-3-Clause library license](https://github.com/pytorch/vision/blob/main/LICENSE) is more suitable for this MIT repository than [Ultralytics' AGPL package](https://docs.ultralytics.com/help/contributing/). Mask2Former was also considered, but the available instance checkpoint's model card lists an unclear `other` license. Model weights download from the official PyTorch host into the local user cache and remain outside Git. TorchVision cautions that pretrained weights may carry training-data terms; the repository does not redistribute them. Mask R-CNN's detection API is marked beta, so `torch==2.14.1` and `torchvision==0.29.1` are pinned for the optional detector.

## Decisions and limitations

- Known camera poses remain an input. This milestone does not estimate pose.
- Association scores class compatibility, normalized center distance, observed-extent overlap, and 3D point support. The gates and candidate scores are retained for inspection. A one-to-one frame assignment prevents two detections from updating the same entity at once. Synthetic truth IDs are evaluation metadata only and never enter fusion.
- Axis-aligned extents are used unless orientation can be supported by observations. Partial surfaces do not justify a full oriented box.
- Entity identity is limited to one episode. Cross-session persistence belongs to Milestone 3.
- Real-data detection quality depends on the COCO vocabulary and image domain. The TUM sample has no object ground-truth labels, so its detections are demonstrations, not accuracy claims.
- The displayed entity confidence is a heuristic combining fused class evidence, valid-depth fraction, and supporting view count. It is not calibrated as a probability of correctness.
- A 3D observation needs at least 12 valid depth samples and a valid-depth fraction of 25%. It rejects isolated extreme depth outliers. The covariance is an uncertainty proxy for the observed surface and does not quantify pose or calibration error.
- An incorrect but rigid camera pose can create a false split. The association audit records the gate/score; synthetic ground truth exposes the error. This is not a pose estimator.
- Geometry relations use distances and world coordinate axes. `left_of_world_x` is not a camera- or viewer-relative language interpretation.

## Reproduction and results

From the repository root with `.venv` active:

```sh
scope detect synthetic
scope project-objects synthetic
scope entities synthetic --output runs/room-entities
scope map synthetic --entities --output runs/room-semantic-map
scope inspect runs/room-entities chair_01
scope benchmark-entities --output runs/room-object-benchmark
python -m pip install -e '.[detector]'
scope entities tum /path/to/rgbd_dataset_freiburg1_xyz --frames 6 --frame-stride 10 --width 320 --classes chair,keyboard,cup,tv --output runs/tum-entities
scope map tum /path/to/rgbd_dataset_freiburg1_xyz --frames 6 --frame-stride 10 --width 320 --entities --classes chair,keyboard,cup,tv --output runs/tum-semantic-map
```

The first projection tests use deterministic synthetic RGB-D and exact masks; they verify points lie on the correct analytic object surfaces and that invalid/noisy depths are bounded. Ten synthetic views fused into four entities (two chairs, table, backpack). The 2D truth-mask precision, recall, and mean IoU were 1.0 by construction, while mean single-view center error was 0.100 m and mean final entity center error was 0.019 m. Point precision within 0.05 m, visible-point completeness within 0.10 m, association pair accuracy, and class accuracy were each 1.0; there were no false merges, false splits, or ID switches. These are **synthetic truth-pipeline** results, not neural detector scores.

Controlled synthetic cases use fixed random seed `20261003`, 10 frames at 128 × 96, and separate machine-readable metrics. Four-centimeter depth noise plus 10% holes raised mean entity center error from about 0.020 m to 0.036 m without a merge or split. A 0.2 m gap between two chairs produced two stable IDs. Half-mask chair occlusion and one conflicting class view did not split entities. With 80% missing depth, most observations were explicitly rejected; association accuracy is **N/A** when too few projected observations remain. A 1 m pose error in one frame produced four false splits and about 0.49 m mean entity center error. This is a measured failure, not a passed pose-robustness claim.

| Synthetic case | Projected / detected | Association pair accuracy | False merges / splits | Mean entity center error |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 36 / 36 | 1.000 | 0 / 0 | 0.0199 m |
| 4 cm depth noise, 10% holes | 36 / 36 | 1.000 | 0 / 0 | 0.0358 m |
| 80% depth holes | 1 / 36 | N/A | 0 / 0 | 0.0468 m on one surviving observation |
| Half-mask chair occlusion | 36 / 36 | 1.000 | 0 / 0 | 0.0201 m |
| One-frame 1 m pose shift | 36 / 36 | 0.949 | 0 / 4 | 0.4881 m |
| One-view class disagreement | 36 / 36 | 1.000 | 0 / 0 | 0.0199 m |
| Chairs with 0.2 m gap | 16 / 16 | 1.000 | 0 / 0 | 0.0191 m |

Association pair accuracy asks whether each pair of projected observations was correctly grouped together or kept apart. The point precision and completeness thresholds measure visible surfaces, not recovery of hidden object geometry. The class Brier value is saved for the synthetic fixture, but the small truth-only sample does not establish calibration on real imagery.

The real run on six frames of the public TUM `freiburg1_xyz` recording produced 25 model masks, 21 valid 3D object observations, and five episode-local entities using the class filter in the command above. Keyboard and cup each received support from six frames. One chair candidate has only one supporting view and 22% heuristic entity confidence; its identity is unverified. The combined known-pose map integrated all six frames, saved and reloaded successfully, and displayed the entities over 2,057 TSDF triangles. Their coordinates and labels are inspectable in Rerun and JSON, but no labeled TUM truth was available to score accuracy. RGB, depth, and poses are **recorded data**; masks and classes are **model predictions**. These checks ran on macOS in this workspace; other platforms are installation guidance only.

## Verification

```sh
QT_QPA_PLATFORM=offscreen python -m unittest -q \
  tests.test_scope_semantic_pipeline tests.test_scope_entities tests.test_scope_objects \
  tests.test_scope_mapping tests.test_spot_gesture tests.test_spot_gui_offline tests.test_spot_model_view
SCOPE_REAL_DATASET=/path/to/rgbd_dataset_freiburg1_xyz \
  QT_QPA_PLATFORM=offscreen python -m unittest -q \
  tests.test_scope_semantic_pipeline.RealDetectorIntegrationTests
```

The first command passed 36 tests with the optional real-dataset test skipped. The separate real-detector integration test passed with the local TUM recording. The synthetic and recorded-data combined views were opened and reviewed in Rerun; the synthetic view was also checked at large and normal window sizes. No live robot was used for visual verification.

## Checkpoints

- `99f4365 Project synthetic object masks into 3D`: typed masks, exact synthetic truth, and depth projection.
- `d82241b Fuse object views into stable entities`: episode-local identity fusion, association audit, and stress tests.
- `de0d34d Add optional instance segmentation`: pinned optional TorchVision detector and verified weight cache.
- `0c30d88 Show and evaluate semantic entities in the map`: stage-specific CLI, evidence, relations, Rerun overlays, synthetic metrics, and robustness tests.

## Failed approaches

PyTorch's standard Python weight downloader stalled before receiving bytes in this workspace; a `curl` request to the same official host completed normally. The detector now downloads into the user cache with `curl`, verifies the upstream SHA-256 filename prefix, and leaves weights out of Git. The first real-data viewer used Rerun's automatic eye position and made the 3D scene too small; an explicit eye target based on projected object centers improved the layout. The 80%-missing-depth benchmark originally printed a numeric association accuracy with no valid pairs; it now reports N/A.
