# Milestone 2: semantic objects in the 3D world

Status: in progress. The existing known-pose map and Spot console remain independent.

## Architecture

`RgbdFrame` → `ObjectDetector.detect()` → `ObjectObservation2D` → depth projection → `ObjectObservation3D` → episode-local entity fusion → `WorldEntity`.

The first detector uses exact synthetic raycast part IDs grouped into table, chair A, chair B, and backpack instances. Its masks prove projection and association without neural-model error. The 3D observation records mask provenance, valid-depth fraction, sampled world points, robust center, observed bounds, covariance, and timestamp. Bounds describe observed surfaces; hidden object extent is not inferred.

The real detector is a separate optional dependency. TorchVision Mask R-CNN v2 provides COCO instance masks and scores. Its BSD-3-Clause library license is more suitable for this MIT repository than Ultralytics' AGPL package. Model weights download into the local PyTorch cache and remain outside Git. TorchVision cautions that pretrained weights may carry training-data terms; the repository does not redistribute them. Mask R-CNN's detection API is marked beta, so versions will be pinned for reproducibility.

## Decisions and limitations

- Known camera poses remain an input. This milestone does not estimate pose.
- Association will score class compatibility and 3D geometric consistency with inspectable gates. Synthetic truth IDs are evaluation metadata only and never enter fusion.
- Axis-aligned extents are used unless orientation can be supported by observations. Partial surfaces do not justify a full oriented box.
- Entity identity is limited to one episode. Cross-session persistence belongs to Milestone 3.
- Real-data detection quality depends on the COCO vocabulary and image domain. The TUM sample has no object ground-truth labels, so its detections are demonstrations, not accuracy claims.

## Reproduction and results

Commands and benchmark numbers will be added as each capability passes its tests. The first projection tests use deterministic synthetic RGB-D and exact masks; they verify points lie on the correct analytic object surfaces and that invalid/noisy depths are bounded.

## Checkpoints

- Typed mask observations, synthetic truth masks, and world projection: implemented and passing relevant tests; commit pending.

## Failed approaches

None in Milestone 2 yet.
