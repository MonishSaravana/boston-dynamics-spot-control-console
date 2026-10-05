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


| Original all-stage cost | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: |
| Input decode | 9.326 | 8.360 | 12.752 |
| semantic/front | 1233.895 | 1221.221 | 1313.858 |
| pose/front | 397.600 | 396.948 | 413.732 |
| entity_projection | 4.663 | 4.098 | 7.069 |
| world_entities | 92.033 | 101.452 | 144.323 |
| pointing | 0.612 | 0.393 | 1.450 |
| intersection | 2.112 | 2.058 | 2.480 |
| mapping | 27.417 | 27.555 | 28.582 |
| memory | 0.002 | 0.002 | 0.004 |
| rerun | 8.949 | 5.147 | 23.067 |
| objects preprocessing (nested) | 0.819 | 0.792 | 0.910 |
| humans preprocessing (nested) | 1.187 | 1.170 | 1.251 |
| JSON summary serialization | 1.112 | 1.099 | 1.224 |

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

## Frozen real-room comparison

The public [NYU Depth V2 labeled dataset](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) supplies real indoor RGB, instance labels, and registered raw depth. Development uses `office_0003` (index 2) and `living_room_0000` (49). Before viewing held-out imagery or running it through models, `docs/reality-spec.json` froze the first eligible frames from six other room types: living room 145, classroom 283, computer lab 332, bedroom 55, kitchen 0, and student lounge 638. All 24 phrases run in every room, including synonyms, equipment, small targets, `red toolbox`, and the nonsense phrase `flarblenox`. Labels enter scoring after inference. No threshold depends on room identity.

Development at the final default localized 6/6 annotated targets, with 0/42 annotation-level negatives accepted. SAM 2 mask IoU averaged 0.859 and five positive cases supported depth projection. This tiny development set is not an accuracy guarantee. The original raw DINO and first corroboration trial remain preserved, including the phone/keyboard and toolbox errors.

The held-out split contains 25 annotated positive phrase cases and 119 annotation-level negatives. Success requires an accepted or ambiguous candidate with box IoU ≥0.5 against any annotated instance. A case with several candidates can therefore succeed while containing a wrong extra region; candidate-level errors are also retained. A label missing from NYU is not exhaustive proof that an everyday phrase is absent. In particular, TV/monitor and book/notebook boundaries are imperfect. “Water bottle” scores against bottle labels; water content/brand is unverified. Upstream pretraining overlap with NYU is unknown: these rooms are held out from SCOPE tuning, not proven unseen by the foundation models.

| Held-out path | Localized /25 | Accepted negatives /119 | Matched mask IoU | Positive depth cases | Mean / median / p95 ms |
| --- | ---: | ---: | ---: | ---: | --- |
| closed | 6 | 1 | 0.797 | 6 | 1155.90 / 1137.60 / 1334.26 |
| grounding-dino | 18 | 19 | — | 0 | 415.63 / 388.78 / 438.23 |
| grounding-dino+grabcut | 18 | 19 | 0.722 | 13 | 488.12 / 394.79 / 964.08 |
| grounding-dino+sam2 | 18 | 19 | 0.816 | 17 | 490.64 / 394.79 / 687.73 |
| grounding-dino+verified+sam2 | 16 | 6 | 0.820 | 15 | 509.77 / 393.48 / 844.87 |
| owlv2 | 7 | 3 | — | 0 | 173.70 / 172.47 / 181.02 |

Cold first calls are included. The old Mask R-CNN runs once per image; its cost is repeated per phrase for comparison, not summed as 144 independent runs. Raw DINO and OWLv2 comparison thresholds are 0.4; the primary corroborated path requires 0.45. SAM/GrabCut variants reuse DINO boxes, so their timing is proposal cost plus the measured refinement, not a second detection. Detailed aggregates, model revisions, per-room scores, and failed candidate evidence are in [perception-results.json](perception-results.json). Full per-case outputs stay in ignored `runs/m45-heldout`.

The primary found five of eight non-COCO positive cases (two bookshelves, pillow, fire extinguisher, door), versus zero for the old detector. It found keyboard, chair, monitor/screen, water bottle, and trash can/garbage bin in several rooms. Common synonyms produced the same regions; aliases retain each original request. Geometry fusion canonicalizes those five alias groups, so “gray sofa” and “couch” can support the same entity rather than duplicating it. Modifiers remain in the query evidence; color/brand correctness is not independently verified.

Preserved failures include:

- All three cabinet positive cases failed the acceptance/localization test. A distant classroom chair proposal fell below the default gate; the computer-lab trash can was refused at 0.408. This is a recall cost of conservative defaults.
- Student-lounge sofa/couch were accepted on the wrong portion of furniture: box IoU 0.293, mask IoU 0.144. Corroboration did not fix the mistake.
- Six annotation-level negatives were accepted: monitor/screen in the living room, classroom whiteboard, bedroom door, and kitchen sofa/couch. Both models can agree incorrectly. Some naming boundaries also differ from dataset labels; none is silently relabeled as a success.
- A correctly segmented student-lounge bookshelf (mask IoU 0.940) could not produce sufficient in-range raw depth. No fake 3D center was supplied.
- Every held-out `red toolbox` and `flarblenox` case stayed `LOW_CONFIDENCE`, with no target forced into tracks or memory. The fresh demo's backpack returned `NOT_FOUND`.

This sample covers real clutter, furniture, equipment, distant/small and partly occluded targets. It is not a balanced evaluation of illumination, reflectivity, severe occlusion, hallway scenes, or people interacting with objects. Public TUM/IPO tests cover people separately; user capture/import tooling supports expanding the room benchmark.

## Backend and resolution measurements

Six fixed development queries (sofa, pillow, lamp, chair, door, absent red toolbox) were rerun after warming each model, with four Torch CPU threads. These are raw model comparisons; DINO still proposed the absent toolbox on every device/resolution. MPS and CPU localized the same five positives on this subset.

| Model / device / pixels | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: |
| grounding-dino / cpu / 512 | 661.58 | 661.24 | 663.68 |
| grounding-dino / mps / 512 | 384.15 | 381.37 | 394.54 |
| grounding-dino / mps / 800 | 842.80 | 828.06 | 904.70 |
| owlv2 / cpu / 512 | 284.15 | 282.92 | 288.94 |
| owlv2 / mps / 512 | 167.88 | 162.83 | 180.49 |
| owlv2 / mps / 768 | 445.67 | 443.92 | 452.48 |

MPS 512 reduced mean DINO wall time by 42% and OWLv2 by 41% versus CPU on this subset. OWLv2 at 768 improved pillow/lamp box localization (from IoU below 0.5 to about 0.70), but cost 446 ms versus 168 ms at 512. DINO 800 cost 843 ms versus 384 ms without improving positive recall here. Default 512 is a measured cost/quality tradeoff, not a claim that small targets are solved. Float32 is retained; Core ML/export, quantization, and reduced precision were not validated. Available MPS acceleration is used, while GPU utilization is explicitly unavailable. No CUDA or Neural Engine result is claimed.

The cheaper common detector is TorchVision SSDLite320 MobileNetV3 on CPU. On two development images its whole detector/GrabCut path averaged 571 ms (median 383, p95 1291), versus 1239/1232/1389 ms for Mask R-CNN. It found the living-room couch and missed the office objects; the old model returned more categories, including false ones. Mode A buys lower periodic cost with weaker coverage. It uses a color-boundary heuristic mask, not SAM-quality segmentation. Mode B spends more compute only when queried. Foreground DINO weights occupy about 689 MB and OWLv2 about 620 MB; runtime RSS and MPS allocations are reported separately because process RSS alone excludes some accelerator memory. Weights, datasets, captures, logs, and SDK files are excluded from Git.

## Responsiveness and module costs

Seven sequential, warmed cases replayed the same 180 public TUM sitting-static frames (355–534), paced at 30 host updates/s. Source timestamps/poses are retained. Frame decoding was measured separately (10.35/8.60/18.17 ms mean/median/p95) and preloaded for the core experiment. This isolates scheduling and compute; it does not claim physical camera acquisition at 30 Hz. Every enabled model produced consumed evidence. Core timings include geometry, association, tracking, result consumption, telemetry/snapshot construction, and worker submission; they exclude model-worker wall time and final persistence. New semantic/background work and rates differ from the serial baseline, so the core improvement is responsiveness, not an equivalent all-model throughput speedup.

| Runtime case | Mean core ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: |
| all | 14.98 | 8.16 | 44.81 |
| no_rerun | 12.47 | 7.17 | 39.05 |
| no_mapping | 8.56 | 7.26 | 12.63 |
| no_common | 11.36 | 5.32 | 42.93 |
| no_humans | 11.73 | 6.13 | 41.65 |
| no_query | 10.78 | 5.99 | 37.53 |
| core_only | 5.59 | 5.49 | 7.05 |

All cases sustained about 30 host input updates/s in this short paced replay. Disabling mapping lowered the mean core tick from 14.98 to 8.56 ms; disabling query inference lowered it to 10.78 ms. Disabling pose/common/Rerun also reduced measured mean cost. These short sequential differences include host variation. Turning a module off does not unload already shared model weights or imply lower process RSS.

The all-enabled run measured these effective host rates and worker/stage latencies:

| Stage | Effective Hz | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| mapping | 4.63 | 31.82 | 29.18 | 48.05 |
| pose/front | 2.04 | 537.87 | 547.07 | 724.89 |
| object_detector | 0.52 | 303.73 | 257.52 | 414.31 |
| open_vocabulary | 0.67 | 1005.10 | 949.48 | 1167.32 |
| segmentation | 0.73 | 361.23 | 280.74 | 506.05 |
| object_tracking | 30.15 | 1.96 | 1.70 | 3.93 |
| tracked_depth | 30.14 | 0.86 | 0.77 | 1.81 |
| world_entities | 1.00 | 2.61 | 2.57 | 4.33 |
| query_memory | 1.00 | 0.05 | 0.05 | 0.08 |
| viewer | 4.90 | 4.87 | 4.90 | 10.02 |
| lifting | 2.04 | 2.22 | 1.74 | 4.30 |
| fusion | 2.05 | 0.72 | 0.51 | 1.50 |
| tracking | 30.16 | 0.05 | 0.01 | 0.36 |
| pointing | 30.16 | 0.01 | 0.01 | 0.03 |
| intersection | 30.16 | 0.01 | 0.01 | 0.01 |

Pose delivered 11 outputs, foreground queries four, and background/common three. Mapping, tracks, and memory continued when the foreground detector was disabled. The all-enabled case retained four entities and eight track records, and persisted four global entities; no-query mode still persisted two background entities. Initial inference/some geometry can exceed a UI tick; threading removes model waits from that tick, but does not remove CPU/GIL contention. Mapping and association remain synchronous, so very large scenes can still make ticks slow.

Rerun writer calls in the original baseline averaged 8.95 ms (median 5.15, p95 23.07). In the warmed 320×240 asynchronous replay they averaged 4.87/4.90/10.02 ms at 4.90 Hz on a separate worker; no-Rerun mean core cost was 12.47 versus 14.98 ms. Native Rerun rendering, remote/network streaming, and full-resolution live-camera rendering are outside these measurements. Queue depth stayed at most one pending job per worker, with dropped/replaced inputs visible. The viewer receives detached snapshots rather than mutable core geometry.

Event memory collection averaged 0.051 ms. Final M3 keyframe/map/SQLite persistence cost 274 ms in the all-enabled replay and 144–268 ms in relevant ablations, measured outside the core tick. Serialized JSON writes are measured separately; PNG/mask files and full memory persistence are not hidden inside an inference timing. Live retention bounds are 24 target tracks, 32 query results, 60 recent frames/flow-history samples, 128 memory keyframes, 256 events, 128 episode entities, and 512 recent entity observations/association decisions. Aggregate fusion counts remain intact when history is trimmed. This is recent evidence retention, not an unlimited long-term video archive.

The cached per-entity KD-tree change did not establish a substantial association improvement on the original six-frame closed-detector audit: mean association stayed 90.39 ms versus 92.03 ms before changes. The new periodic/query path's 2.61 ms association mean comes from fewer observations and different cadence; it is not attributed entirely to caching.

## Tracking experiment

Twenty-four real public TUM `freiburg1_xyz` frames (stride 3) compare continuous DINO/OWL/SAM redetection against reuse. Continuous redetection averaged 825/785/904 ms and required 24 model calls, totaling 19.8 seconds of detector/segmentation compute. Detect-once retained an active track on 11/24 frames and eventually lost it as the camera moved. Three refreshes at frames 0/8/16 raised that to 19/24, totaling 3.09 seconds of detector compute, with optical-flow/initialization costs 1.49/1.59/3.10 ms. Detect-once's first OpenCV initialization outlier raised its mean to 8.85 ms (median 0.65, p95 7.92).

Mean active-box agreement with independent rerun detections was 0.932 with refreshes (0.941 without); this is detector agreement, not annotated tracking ground truth or a proof of physical accuracy. Motion/appearance gates can reject a useful track. Reacquisition after a lost track may create a new ID; long-lived identity through occlusion is not guaranteed. Synthetic tests separately exercise disappearance, flow disagreement, staleness, unique spatial reacquisition, and continuation after the detector is disabled. Current depth under a propagated mask is marked `ESTIMATED_MASK_MEASURED_DEPTH`; it does not create new semantic verification or memory evidence. Last verified projection and current estimated geometry remain separate.

## Capture, input, and reproducible commands

Install the optional dependencies:

```sh
source .venv/bin/activate
python -m pip install -e '.[perception,perception-gui]'
scope hardware
```

Public NYU preparation used the official mat and calibration toolbox (about 2.97 GB for the mat). These files remain ignored:

```sh
mkdir -p runs/m45-data
curl -L --fail -o runs/m45-data/nyu_depth_v2_labeled.mat https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
curl -L --fail -o runs/m45-data/nyu-toolbox.zip https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip
unzip runs/m45-data/nyu-toolbox.zip -d runs/m45-data/toolbox
scope prepare-reality --nyu-mat runs/m45-data/nyu_depth_v2_labeled.mat --output runs/new-reality-spec.json
scope benchmark-reality --nyu-mat runs/m45-data/nyu_depth_v2_labeled.mat --spec docs/reality-spec.json --split development --output runs/reality-development
scope benchmark-reality --nyu-mat runs/m45-data/nyu_depth_v2_labeled.mat --spec docs/reality-spec.json --split heldout --output runs/reality-heldout
python benchmarks/query_backends.py --nyu-mat runs/m45-data/nyu_depth_v2_labeled.mat --output runs/query-backends.json
python benchmarks/perception_runtime.py --tum runs/m4-public-pose/rgbd_dataset_freiburg3_sitting_static --output runs/runtime-comparison
python benchmarks/object_tracking.py --tum runs/m4-public-pose/rgbd_dataset_freiburg1_xyz --output runs/tracking-comparison.json
```

Use an empty/new output path for every run. TUM data preparation and the pre-existing M1–M4 demos are documented in their reports. The checked acceleration probes require this host's available MPS backend. NYU `rawDepths` are used before inpainting, with the official RGB pinhole intrinsics and MATLAB principal point converted to zero-based pixels. RGB lens distortion is not corrected; supplied pinhole projection is approximate. Each still has its own camera-local Z-up frame, no measured room localization, and no original acquisition clock/FPS. Independent stills are never fused as a common measured world.

For user RGB-D files, calibration JSON must contain the following measured quantities; substitute real calibration and pose values, not these illustrative numbers:

```json
{"intrinsics":{"width":640,"height":480,"fx":525,"fy":525,"cx":319.5,"cy":239.5},"depth_scale_m":0.001,"T_world_camera":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
```

```sh
scope query-object "power strip" --image room.png --depth depth.npy --calibration calibration.json --world-frame camera-A --output runs/rgbd-query
scope perceive --image room.jpg --interactive --output runs/rgb-query-console
scope perceive --video phone-video.mp4 --interactive --background --output runs/video-queries
scope capture-reality --video phone-video.mp4 --frames 100 --output runs/room-capture
scope capture-reality --webcam 0 --frames 100 --hz 10 --output runs/webcam-capture
scope benchmark-reality --manifest runs/room-capture/reality.json --split heldout --output runs/capture-evaluation
```

Capture/import writes RGB frames and an annotation template. Add `evaluation_cases` entries such as `{"frame":0,"room_id":"room-C","split":"heldout","query":"power strip","present":true,"boxes_xyxy":[[10,20,150,90]],"masks":[]}` using actual annotations. Negatives require explicit `present:false`; missing annotations are not negatives. Positive boxes score localization, and only actual annotated masks score mask IoU. A room appearing in both splits is rejected. A public TUM-derived video import was checked for five frames, then evaluated through the manifest path. No physical webcam or phone recording was captured here. Calibration/depth can be added per frame with relative file paths; RGB-only capture never invents 3D.

CLI controls include `--disable open_vocabulary|human_pose|mapping|memory|rerun`, `--rate rerun=1`, `--rate object_detector=0.25`, `--backend owlv2`, `--device cpu`, `--model-size 512`, `--segmentation grabcut|none`, and `--unverified` for raw DINO comparison. Independent module tests cover failures, superseded queries, disabling in-flight work, background memory without humans, RGB-only tracks, and detector/viewer isolation. `SemanticQuery` consumes the existing `TextCommand` and returns `TextResponse`; future language grounding can reuse it without changing this milestone into a parser/planner.

## Fresh demo and offline visual review

`docs/reality-demo-spec.json` selected the first additional living-room scene outside every prior evaluation room, index 150 (`living_room_0005`), before inspecting its image/query output. The scene was not staged. Thresholds remained unchanged. The requested “gray sofa” localized the sofa with a mask and depth; “couch” reused the same 2D track and M2/M3 entity. The sofa appears green under the captured lighting, illustrating that the color modifier is not independently verified. Lamp also had a supported mask/depth entity. Backpack was `NOT_FOUND`; power strip, red toolbox, and whiteboard eraser remained `LOW_CONFIDENCE`. The benchmark does not pretend that absent backpack is a successful backpack demonstration.

```sh
scope perceive --nyu-mat runs/m45-data/nyu_depth_v2_labeled.mat --nyu-index 150 --interactive --query "gray sofa" --output runs/fresh-room-demo
```

Type `backpack`, `power strip`, `couch`, `red toolbox`, and `lamp`. Toggle **Queries** off, then on and request `whiteboard eraser`. Existing masks and depth entities remain visible while the heavy detector is disabled; moving-frame tests establish independent tracking/map continuation. This public still demonstrates one observed camera frame, not live moving-room tracking. `perception.json`, `query-overlay.png`, mask arrays, Rerun recording, and the M3 memory episode/SQLite retain actual outputs outside Git.

The real Qt query console was rendered offscreen at normal 1120×780 and full 1440×1000 sizes using actual models/input, plus disable/re-enable controls. The first review exposed clutter from historical weak proposals and duplicate qualifier/synonym entities; both were corrected and retested. The image now shows current-query weak proposals and bounded retained tracks; prior query evidence stays inspectable. The native Mac was locked and automatic unlock failed. Offscreen software rendering satisfies the offline layout review, but native macOS input/window interaction and Rerun viewer rendering remain untested in this session. No live robot was used for UI checks.


The initial offscreen console was substantially worse than headless inference: seven query calls averaged 7355 ms, median 4304, p95 19896 (cold startup included). Reducing redraw/Rerun rates alone did not solve it: median query time remained about 4.1–4.7 seconds. Automatic table column resizing and repeated item creation were removed; cells are reused and updates batched. Rendering is capped at 5 Hz, with core polling independent and the interactive Rerun default capped at 1 Hz. The first corrected console run had query wall time 1615/641/5549 ms; the final layout review measured 1677/684/5840 ms, core 3.01/1.20/5.67 ms, and GUI rendering 5.28/4.84/13.19 ms (mean/median/p95). Its Rerun worker cost was 10.95/10.16/17.96 ms at about 1 Hz; initialization and resolution explain why this differs from the small warmed replay. All slow early runs remain in the aggregate report, rather than being replaced with the fast core-only numbers.

The console shows immutable-still evidence as `STATIC` instead of implying a lost physical camera feed. Loaded-image age and time since query-result consumption are separate; actual capture time remains unknown. Sequences/cameras still use source age and typed stale/lost gates. CPU/RSS and MPS allocation are shown; accelerator utilization is unavailable. First query startup can still take several seconds. Performance on a native macOS window, Windows/Linux, a physical webcam, or Spot must be measured separately.

## Regression and completion checks

The full suite passed **79 tests, zero failures, zero skips, in 37.150 seconds**, including the configured real TUM, IPO, pose, and hand-model checks. The original 63-test baseline passed before implementation. New tests cover phrase traceability, synonym entity identity, explicit absence/ambiguity/disagreement, measured/estimated depth separation, independent workers/toggles, bounded fusion history, RGB-only tracking, and room-disjoint recording annotations. Spot GUI, velocity/gesture, model-view and sensor safety checks passed; control files have no milestone diff.

```sh
SCOPE_REAL_DATASET=runs/m4-public-pose/rgbd_dataset_freiburg1_xyz \
SCOPE_HUMAN_EPISODE=runs/m4-estimated-episode \
SCOPE_IPO_DATASET=runs/m4-public-pose/ipo/Innsbruck_Pointing_Dataset \
SCOPE_HAND_MODEL=models/gesture_recognizer.task \
QT_QPA_PLATFORM=offscreen python -m unittest -v
scope map synthetic --entities --humans --frames 24 --width 192 --toggle-demo --no-viewer --output runs/m45-legacy-demo
```

No physical Spot/camera connection was opened. M4.5 does not establish safe robot-target selection from these detections: correlated false matches, unknown calibration/distortion, depth holes/range, motion blur, texture-dependent flow, occlusion, queue/result age, and native viewer load remain practical risks for class testing. Fresh SDK readiness, velocity expiry/zero-on-release/Stop/focus/failure/shutdown, and the separate class E-stop workflow are unchanged. Monday testing must measure real aligned camera/depth poses and source clocks, throughput under native visualization, person association/pointing, and tracking across motion before relying on perception output. This milestone implements no spatial-language parser, LLM planner, navigation, active perception, audio, or live-control changes.

The combined 24-frame legacy mapping/entity/human toggle demo completed successfully and saved its memory/Rerun outputs. A final real-room lamp query also exercised the updated Rerun entity labels and persisted one depth entity. The reviewed console persisted two global entities: the sofa shared by both qualified/synonym requests, and the lamp. Its final JSON snapshot write cost 0.51 ms; other files/memory persistence are separate.

Checkpoint subjects use ordinary history: `286bc48` measures the original perception stages, and `4f628cb` adds the local query/tracked-depth implementation. The final hardening/evaluation checkpoint follows them on `codex/m4.5-real-world-perception`; `git log --oneline f823cec..HEAD` and `git rev-parse HEAD` identify the exact integrated revision without rewriting prior history. The separate local browser-console branch, technical-guide PDF, and browser artifacts are preserved.
