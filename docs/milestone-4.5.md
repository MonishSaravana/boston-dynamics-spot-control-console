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
