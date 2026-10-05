# Milestone 4.5: real-world perception hardening

Work starts from tested `main` at `f823cec`, on `codex/m4.5-real-world-perception`. The separate local browser-console branch and untracked technical-guide PDF are preserved. This milestone does not change Spot control or implement spatial-language reasoning.

## Audit before changes

The checked host is an Apple M4 with 24 GB memory, macOS 26.6.2, Python 3.14.2, PyTorch 2.14.1, TorchVision 0.29.1. MPS is available; CUDA is not. M4 explicitly selects CPU for Mask R-CNN v2 (480-pixel shortest model edge) and Keypoint R-CNN (640-pixel shortest edge), with four Torch threads. RGB-D input was 320×240. Synchronous replay runs the model stages serially. M4 realtime workers decouple inference, but synchronous Rerun, entity association, and mapping remain on the core tick.

The unmodified baseline passed 63 tests without skips. Six consecutive public TUM sitting-static frames (indices 355–360) were replayed per configuration, after warming the models. Results are retained outside Git in `runs/m45-audit/baseline.json` and `baseline-tests.txt`.

| Baseline | Mean tick ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: |
| All stages | 1772.64 | 1750.89 | 1892.69 |
| Rerun disabled | 1836.75 | 1834.73 | 2009.98 |
| Mapping disabled | 1889.23 | 1891.18 | 2022.98 |
| Semantics disabled | 464.99 | 443.13 | 544.38 |
| Humans disabled | 1509.24 | 1492.83 | 1750.10 |
| Models, mapping, Rerun disabled | 0.51 | 0.51 | 0.53 |

The all-stage mean includes 1233.89 ms semantic detection, 397.60 ms pose, 92.03 ms entity association, 27.42 ms mapping, 4.66 ms 3D object projection, 2.11 ms pointing intersections, and 8.95 ms Rerun logging. The short sequential ablations have host variation: they do not establish a speedup from disabling mapping or Rerun. Inference is the clear bottleneck. Model-transform preprocessing is measured as a nested portion of inference; frame decode and JSON serialization are separate. Memory collection is measured, but final M3 persistence is outside these tick timings. No Spot or network performance is inferred.

Reproduce the audit with an empty output directory:

```sh
scope audit-perception runs/m4-public-pose/rgbd_dataset_freiburg3_sitting_static --start-frame 355 --frames 6 --output runs/perception-audit
```

## Local query implementation and development evidence

The primary proposal model is [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny). [OWLv2 Base](https://huggingface.co/google/owlv2-base-patch16-ensemble) provides independent spatial corroboration. Both use pinned upstream revisions through Transformers 4.57.6, safe tensor weights, and local inference. Default query input is 512 pixels; MPS is selected when available, with explicit CPU/CUDA overrides. There is no hosted inference or language planner.

The first comparison on the two development rooms had six annotated positive phrase cases and 42 annotation-level negatives. At a 0.4 proposal threshold, DINO localized all six positives but accepted 10 negatives. OWLv2 localized two positives with one negative; its other pillow/lamp boxes failed the 0.5 box-IoU evaluation gate despite finding the region. These failures are preserved in `runs/m45-development/cases.json`. Annotation-level absence can disagree with fine-grained everyday names; for example, NYU labels a visible book without establishing whether it is a notebook. The office keyboard/phone confusion and absent toolbox proposal are actual model limitations.

Defaults were chosen on development rooms only: a DINO score of at least 0.45, an OWLv2 proposal of at least 0.15, and box overlap of at least 0.3. Disagreement becomes `LOW_CONFIDENCE`; rejected proposals remain traceable and never initialize 3D evidence or tracks. This is an evidence gate, not calibrated certainty or a guarantee against correlated mistakes. Scene IDs and queries are frozen in `docs/reality-spec.json` before held-out evaluation.

[SAM 2.1 Tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) refines supported boxes during discovery/refresh. On the first development comparison, matched mask IoU averaged 0.859 with SAM 2 versus 0.511 with GrabCut. A separately run sofa query achieved mask IoU 0.882 and persisted one depth entity through M3. GrabCut remains an explicit lower-cost color-boundary alternative; boxes never masquerade as precise masks. Weak or unsupported masks do not produce 3D entities.

The first MPS query/segmentation calls took seconds; warm SAM 2 calls in the development probe took 302 and 279 ms. Startup and warm timings are kept separate. A native UI review was pending while the Mac was locked; no live camera or robot was used for these checks.

Grounding DINO, the selected OWLv2 bundle, and SAM 2 list Apache-2.0 licenses in their [upstream](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE) [sources](https://github.com/facebookresearch/sam2/blob/main/LICENSE) and model cards. [YOLO-World](https://github.com/AILab-CVC/YOLO-World) was reviewed as an alternative: its GPL-3.0 distribution and additional detection framework made integration less suitable here. It was not installed or given an invented local speed/accuracy result. No model bundles or datasets are committed.

The query console accepts raw phrases and shows their normalization/expansion, detector match, score, mask support, track state, measured depth location, latency, backend, and input age. Five small alias groups complement the detector's learned text encoding; arbitrary other phrases pass through unchanged. Motion tracking uses sparse forward/backward optical flow, affine/appearance gates, and bounded tracks. Verification timestamps are retained. Lost, ambiguous, stale, and unsupported results are explicit.

Discovery, background common objects, pose, acquisition, and Rerun use bounded workers where needed. The core owns mutable geometry and consumes results with their original frames. Memory retains bounded verified keyframes and imports them using the existing M3 path in an explicit world frame. NYU stills each use a separate camera-local frame, registered raw depth before inpainting, and the official pinhole calibration; still capture timing and room localization are unavailable.
