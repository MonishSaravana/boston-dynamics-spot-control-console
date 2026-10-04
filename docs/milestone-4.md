# Milestone 4: human geometry and pointing

SCOPE now runs human perception alongside the M1 map, M2 entities, and M3 memory. It lifts body joints from calibrated RGB-D, associates compatible camera observations, maintains short-lived person tracks, samples uncertain pointing directions, and scores their intersections with current entities and occupied voxels. Each stage has its own configuration and failure boundary. The combined Rerun recording includes the geometry and measured runtime health.

This is offline perception. The Spot command path is unchanged. No language model, reasoning layer, navigation, audio, face recognition, or automatic robot commands were added. Real Spot human perception and real Spot multi-camera fusion have not been tested. SCOPE remains an independent project without Boston Dynamics endorsement.

## Architecture and reused code

```text
camera RGB-D + calibration + supplied pose
  ├─ M1 VoxelMap
  ├─ semantic detector → M2 projection → EntityStore → M3 episode import
  └─ body/optional hand model → 2D observations → depth lifting
       → conservative multi-view association → temporal person tracks
       → sampled pointing → voxel/semantic intersections

Runtime configurations and telemetry surround individual stages.
Rerun consumes snapshots; it does not issue actions.
```

The new code is in `scope/runtime.py`, `humans.py`, `pose_detector.py`, `hand_pose.py`, `person_tracking.py`, `pointing.py`, `human_pipeline.py`, `human_synthetic.py`, `human_eval.py`, `human_viewer.py`, `human_cli.py`, `ipo.py`, and `sensor_diagnostic.py`. `interaction.py` contains only the `TextCommand` and `TextResponse` records for a future transport boundary.

M4 reuses `RgbdFrame`, camera/world transforms, the M1 occupied/free/unknown voxel map, M2 object projection and entity fusion, and the unchanged M3 immutable-episode import/link path. `VoxelMap.integrate` now accepts a stream ID so synchronized cameras can have independent calibrations and increasing timestamps. Existing single-camera callers retain the strict original behavior. The M4 combined view uses occupied voxels for surface intersections; it does not add a multi-camera TSDF mesher. The existing M1 command still produces its TSDF and mesh.

Synthetic memory collects matched primary-camera frames and semantics, then persists an immutable episode and SQLite memory at the end. Human output is not a dependency. Recorded memory remains an explicit M3 import with a declared shared frame or uncertain rigid alignment; M4 does not infer registration or silently assign a recording to the synthetic room.

## Pose model choice

| Option | Practical fit and limits |
| --- | --- |
| [TorchVision Keypoint R-CNN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html) | Chosen for multi-person COCO body joints and compatibility with the existing TorchVision detector environment. It supplies shoulders, elbows, wrists, and torso anchors. CPU inference is expensive, and the 17-joint body model has no finger direction. |
| [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) | A practical body-landmark alternative with more landmarks and configurable pose count. No interchangeable speed/accuracy benchmark was performed here; normalized model coordinates alone are not measured map coordinates. |
| [Ultralytics pose models](https://docs.ultralytics.com/tasks/pose/) | Another multi-person body-pose option. Would require an additional model/package and a licensing decision. It was not installed or benchmarked for this milestone. |

The chosen body adapter uses `KeypointRCNN_ResNet50_FPN_Weights.COCO_V1`, CPU, model width 640, detection score threshold 0.7, and at most four Torch CPU threads. Checked environment: macOS Apple Silicon, Python 3.14.2, Torch 2.14.1, TorchVision 0.29.1, MediaPipe 0.10.35, Rerun SDK 0.38. Windows/Linux and GPU performance were not checked. Official TorchVision weights are hash-verified in the user cache, outside Git.

An optional [MediaPipe Gesture Recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer) adapter supplies wrist and index MCP/PIP/tip image landmarks. It associates hands with body wrists using a distance gate and ambiguity margin, then calculates a continuous extended-index/curled-other-fingers quality cue. Gesture categories do not dispatch commands. Hand failure falls back to body geometry and appears in its own telemetry. Camera workers serialize access to a shared model/hand instance.

The local hand bundle used for the check was `gesture_recognizer.task`, SHA-256 `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`. Model bundles, SDK files, datasets, recordings, logs, and databases are excluded from project history. Download bundles from their official model pages; this repository does not redistribute them.

## Coordinates, association, and uncertainty

`PersonObservation2D` binds finite keypoints to one frame, camera, and timestamp. TorchVision heatmap strength is only a heuristic quality gate; its output visibility field is not treated as a calibrated confidence. `lift_pose` samples a 5×5 aligned-depth patch, requires at least three valid samples in 0.15–6 m, checks the central pixel, rejects depth discontinuities, and uses median depth and robust spread. Pixel/depth covariance is transformed with the supplied camera pose into the map frame. Implausible arm segment lengths reject the distal observation. Missing coordinates remain `None` with `UNAVAILABLE`, never an exact invented position.

Multi-view observations must share calibrated world coordinates and be within 80 ms. Association uses shared shoulders/hips, a tight 0.10 m gate for one anchor or 0.30 m for several, a unique-match margin, and at most one observation per camera in a group. There is no biometric or appearance identity. Conflicting joints become unavailable; ambiguous candidate people remain unresolved and are excluded from pointing. These assumptions are proven on virtual cameras, not on Spot.

Fusion combines inverse covariance and adds cross-view disagreement. Tracks retain episode-local IDs for one second, record source observations and history, filter observed coordinates, and estimate joint velocity from source-time differences. A missing joint may be held for at most 0.20 s as `ESTIMATED`, with its original timestamp and expanded covariance. A person who leaves view is not used for a fresh pointing result.

Body pointing blends the forearm direction (75%) with shoulder-to-wrist direction (25%). Continuous arm extension, joint quality, and a resting-arm penalty determine whether geometry supports a hypothesis. Missing elbow uses a broader shoulder/wrist baseline; missing wrist, stale person geometry, folded/downward arms, or two plausible arms can cause abstention. This estimates geometry, not human intent.

Each hypothesis samples 256 origins/directions. Joint covariance, held-joint age, and a heuristic model-discrepancy floor contribute spread: nominally 5° for one camera, 3° for multiple cameras, 14° without elbow, and at least 16° with held joints. Reported angular uncertainty is the RMS angle of the actual samples, not a calibrated confidence interval. A reliable index cue can replace the body baseline, but its short measured baseline retains a 12° discrepancy floor and is always labeled `DEGRADED`.

Rays intersect M1 occupied voxels and M2 observed entity AABBs with bounded covariance padding. The immediate 0.20 m around the wrist/body is skipped; near-field targets inside that distance are unsupported. Surface occlusion limits entity hits, with a 0.20 m voxel tolerance. Entities older than 0.5 s are excluded and their ages are reported. Sample fractions, including `unknown`, sum to one. They are uncalibrated geometric scores, not probabilities of intended reference or correctness. The viewer shows a central arrow, representative samples, surface hit points, and joint uncertainty radii; those radii summarize covariance and are not anatomical volume or a calibrated coverage region.

## Module configuration and telemetry

Every runtime state exposes `OK`, `DEGRADED`, `STALE`, `UNAVAILABLE`, `FAILED`, or `DISABLED`, plus reason, source, last successful source timestamp, retained output, source-data/input age, mean/median/p95 duration, host/source update rates, failures/restarts, dropped inputs, queue depth, and recent failures. Unexpected exceptions at subsystem boundaries are logged with tracebacks. CPU measurements are calling-thread time, not total process utilization. GPU time and one-way network latency remain unavailable.

Synchronous replay is the default for deterministic evaluation. `--realtime` uses one running and one replaceable pending inference input per camera/model stream. Slow inference does not block map ticks. Completed results keep their original frames and must pass age checks; obsolete inputs/results are dropped. This worker design is not a guarantee of real-time performance. Core geometry, mapping, and viewer calls are synchronous. Shutdown waits a bounded interval for workers and avoids closing the shared hand model while a worker is still finishing.

Frequency gates use source timestamps, independently of host processing speed. A 10 Hz source replayed at 4 Hz on the host is not a 10 Hz sensor measurement. IPO uses ordered still indices, so its acquisition timestamp and source FPS are explicitly unavailable. Already resident episodes have no separately measured decode stage; no split is fabricated. Synthetic acquire time includes fixture rendering. Host input-to-perception duration ends after core perception, before logging and final persistence.

| Configuration name | Independent control |
| --- | --- |
| `camera/front`, `camera/left`, `camera/right` | Accept/drop each input stream |
| `depth/front`, `depth/left`, `depth/right` | Enable measured depth for that stream; RGB inference can continue without 3D lifting |
| `pose/front`, `semantic/front` (and other cameras) | Per-camera model rate/stride and enable flag |
| `mapping`, `semantic_detection`, `human_pose`, `hand_pose` | Major map/model stages |
| `lifting`, `fusion`, `tracking`, `pointing`, `intersection` | Human geometry and target stages |
| `entity_projection`, `world_entities`, `memory`, `memory_persist` | Entity updates, primary-camera evidence collection, completed episode save |
| `visualization`, `map_visualization`, `rerun` | Viewer calls, map drawing, recording/logging |
| `robot_visualization` | Reserved, disabled; enabling reports unavailable because this viewer has no robot model attached |

The dashboard abbreviates long display labels (for example, semantics, humans, entities, map view, memory save, and input); JSON and configuration retain the exact names above. The existing Spot GUI's robot model is unchanged. `--no-viewer` suppresses opening the app but still records; `--no-recording` disables Rerun. Configuration is a small mutable Python boundary suitable for later GUI controls, not a new live-robot GUI.

```json
{
  "human_pose": {"max_hz": 10, "stale_after_s": 0.5},
  "semantic_detection": {"max_hz": 5},
  "mapping": {"every_n": 2},
  "camera/left": {"enabled": false},
  "map_visualization": {"enabled": false}
}
```

These example rates are configuration examples, not measured Spot rates. `--disable`, `--rate`, and `--every` override matching JSON fields. A disabled camera broadens the surviving single-view hypothesis and degrades aggregate health; human inference disabled clears skeleton/rays while mapping, entities, and memory continue. A pose exception is `FAILED`, retained output has an explicit age, and pointing becomes unavailable. Stale semantics do not become fresh target evidence.

## Commands

Run from the repository root. Use a new/empty output directory; normal demos reject overwriting previous evidence.

```sh
source .venv/bin/activate
pip install -e .
# Optional estimated body/finger support:
pip install -e '.[pose,hands]'

scope humans synthetic
scope humans synthetic --camera front
scope humans synthetic --camera front --camera left
scope pointing synthetic
scope pointing synthetic --oracle-keypoints
scope pointing synthetic --estimated-keypoints --frames 3 --no-viewer
scope map synthetic --entities --humans --frames 24 --width 192 --toggle-demo --output runs/room-human-demo
scope telemetry synthetic --disable camera/left --rate semantic_detection=5 --every mapping=2
scope pointing synthetic --config runtime.json
scope pointing synthetic --disable map_visualization
scope pointing synthetic --no-recording --no-viewer
scope pointing synthetic --realtime --rate human_pose=5 --rate semantic_detection=2
scope pointing synthetic --scenario wrist_missing
scope pointing synthetic --scenario overlapping_people
scope benchmark-humans --frames 10 --output runs/human-benchmark

# Ordinary earlier milestones remain independently runnable:
scope map synthetic --frames 3 --no-viewer --output runs/map-only
scope entities synthetic --frames 3 --no-viewer --output runs/entities-only
scope memory-demo --output runs/memory-demo
```

Use `scope --help` and each command's `--help` for arguments. Body pose does not reliably detect the illustrative synthetic stick person; `--estimated-keypoints` there is a failure/isolation check, not neural accuracy evidence. Use actual recorded people for model evaluation:

```sh
scope humans tum runs/m4-public-pose/rgbd_dataset_freiburg3_sitting_static --start-frame 355 --frame-stride 1 --frames 12 --width 640 --estimated-keypoints --no-viewer --output runs/recorded-humans
scope pointing episode runs/m4-estimated-episode --estimated-keypoints --disable semantic_detection --no-viewer --output runs/recorded-pointing
scope benchmark-ipo runs/m4-public-pose/ipo/Innsbruck_Pointing_Dataset/00017 --calibration-folder runs/m4-public-pose/ipo/Innsbruck_Pointing_Dataset/00001 --hand-model models/gesture_recognizer.task --output runs/ipo-benchmark
scope pointing ipo runs/m4-public-pose/ipo/Innsbruck_Pointing_Dataset/00017 --calibration-folder runs/m4-public-pose/ipo/Innsbruck_Pointing_Dataset/00001 --hand-model models/gesture_recognizer.task --frames 4 --disable semantic_detection --output runs/ipo-viewer
rerun runs/m4-final-viewer/humans.rrd
```

The TUM consecutive-frame episode was also saved separately in `runs/m4-estimated-episode` for repeatable optional tests. Download/extract public sources under ignored `runs/`, never into tracked history:

```sh
mkdir -p runs/m4-public-pose
curl -fL https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz -o runs/m4-public-pose/sitting_static.tgz
tar -xzf runs/m4-public-pose/sitting_static.tgz -C runs/m4-public-pose
curl -fL https://iis.uibk.ac.at/public/datasets/ibkpointingdataset/ibkpointingdataset.zip -o runs/m4-public-pose/ipo.zip
unzip runs/m4-public-pose/ipo.zip -d runs/m4-public-pose/ipo
```

See [TUM's dataset page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) and [the Innsbruck pointing dataset](https://iis.uibk.ac.at/datasets/ipo) for source descriptions and terms. Images are public dataset material, not Spot captures.

To recreate the optional human-test episode in an empty location:

```sh
python - <<'PY'
from pathlib import Path
from scope.tum import TumRgbd
from scope.storage import save_episode
output = Path("runs/m4-estimated-episode")
if output.exists():
    raise SystemExit("Use a new directory; the existing episode is preserved")
source = TumRgbd(Path("runs/m4-public-pose/rgbd_dataset_freiburg3_sitting_static"),
                 max_frames=12, frame_stride=1, width=640)
source.selected = source.rgb[355:367]
save_episode(output, list(source), "public-tum-sitting-static-consecutive")
PY
```

## Evaluation results

The final synthetic benchmark is `runs/m4-release-benchmark/human_benchmark.json`, with `multi_camera_comparison.json` and `human_benchmark.png`. It covers 51 cases: 15 scenarios × front/left/fused cameras, three target distances (0.8/1.2/1.8 m), and one/two/three camera counts; each case has ten frames at 128×96. The fixture has exact visible 2D joints, known 3D joints/rays/target, camera visibility, simulated depth, and analytic occlusion. Truth is used after inference for scoring, not to associate people or choose targets.

| Oracle case | Available-joint 3D error | Angular error | RMS spread | Top-1 / top-3 | No emitted hypothesis |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete, front | 0.773 mm | 0.0088° | 5.50° | 1.00 / 1.00 | 0% |
| Complete, fused | 0.566 mm | 0.0115° | 3.66° | 1.00 / 1.00 | 0% |
| Complementary, front | 0.763 mm | unavailable | unavailable | unavailable | 100% |
| Complementary, fused | 0.717 mm | 0.0225° | 4.18° | 1.00 / 1.00 | 0% |
| Noisy depth, best single (left) | 4.895 mm | 0.865° | 8.35° | 1.00 / 1.00 | 10% |
| Noisy depth, fused | 1.540 mm | 0.181° | 4.05° | 1.00 / 1.00 | 0% |
| Elbow missing, fused | 0.568 mm | 0.0113° | 13.51° | 1.00 / 1.00 | 0% |
| Overlapping people, fused | unavailable | unavailable | unavailable | unavailable | 100% |

Top-k and angular error are conditional on a supported target hypothesis; they must be read beside the no-hypothesis rate. Overlap yields unresolved people, not a perfect angular score. Visible-joint detection and 2D error are 1.00 and zero for oracle input by construction, not neural model accuracy. Complete fused missing-joint rate is 7.69% of the fixture's labeled body joints; partial views and occlusion increase it. Three cameras reduce complete 3D error to 0.515 mm and spread to 3.49°. All three tested target distances retain top-1 1.00. Fusion improves noisy-depth error and availability; it does not improve every metric in every case—the complete front camera has slightly smaller angular error than fused input.

Explicit failure scenarios include complete arm, wrist/elbow missing, lower body outside, partial frame, hand outside, complementary views, 4 cm Gaussian depth noise plus holes, two people, overlapping people, resting arm, outside cone, no target, brief exit, and camera loss. Wrist/hand missing and resting-arm cases abstain. Elbow loss broadens spread. Two separated people retain two IDs; overlapping views remain unresolved. Brief exit preserves the episode ID while suppressing stale pointing. Outside-cone/no-target cases retain high unknown score. Unit tests separately cover delayed views, stale entity exclusion, failed inference, bounded queues, toggles, and independent memory persistence.

The estimated-pose check uses actual public RGB-D people. Twelve consecutive TUM sitting-static frames produce person tracks and conservative body hypotheses, but intended targets are unannotated, so this is integration evidence rather than pointing accuracy. No 2D body-joint ground truth was available for a neural joint-error claim.

The labeled IPO check uses natural pointing stills in folder `00017` and a separate reference folder `00001` to fit the camera pinhole from supplied object 2D/3D correspondences (0.217 px RMS residual). Depth PNGs use millimeters, checked against supplied optical-Z annotations. World coordinates are a declared camera-local Z-up axis conversion; there is no measured room localization or acquisition interval. Target/hand labels enter only after model output.

The model emits a direction in **7/10 target cases**, with **4.67° mean angular error** on those seven. All seven rank the correct supplied object center first: top-1/top-3 over all cases are **0.70**, and non-output rate is **0.30**. Two cases abstain; `00006d.png` is missing and counts as unavailable. Ranking is angular ranking of supplied object centers, not neural object detection accuracy or a real-source M2 semantic hit benchmark. The hand truth labels a hand center whereas the model origin is an index fingertip; the origin difference is reported separately. A small selected subset cannot establish general accuracy, calibrated uncertainty, or intent recognition. Body-only inference often lacks usable arm geometry in these cropped frames; fingers help without becoming a required dependency.

## Measured offline performance

Timings below are actual host durations, with no Spot/network measurements. `runs/m4-demo/telemetry.json` is a 24-frame 192-pixel, two-camera synchronous oracle demo with toggles. Its input-to-perception duration is **165.76 / 166.07 / 207.36 ms mean/median/p95**. Wall replay throughput, including recording and the final memory save, is **4.52 frames/s**. Mapping made 38 camera integrations; the run retained four M2 entities and persisted four M3 global entities.

| Stage | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: |
| Map integration (active cameras per tick) | 35.97 | 42.52 | 49.10 |
| 3D lifting (per camera batch) | 0.59 | 0.54 | 0.80 |
| Multi-view fusion | 0.75 | 0.71 | 1.20 |
| Temporal tracking | 0.15 | 0.15 | 0.36 |
| Pointing | 0.43 | 0.32 | 0.72 |
| Entity/surface intersection | 2.06 | 1.70 | 2.83 |
| Viewer logging | 6.90 | 5.02 | 14.66 |
| Completed memory save (one sample) | 1135.47 | 1135.47 | 1135.47 |

Input rendering, per-camera semantic timings, projection, each module's host/source rate, drops, and all distributions are in the JSON. Aggregate model health/rate is derived from camera stages; it has no fabricated aggregate duration. Oracle pose retrieval time is not neural inference time. The single memory-save duration is not a stable latency percentile estimate.

`runs/m4-release-ipo/ipo_metrics.json` measures nine present target stills with estimated body plus hands, on CPU: pose **695.87 / 635.58 / 1161.14 ms**, lifting **1.08 / 0.55 / 2.85 ms**, tracking **0.020 / 0.013 / 0.038 ms**, and pointing **0.43 / 0.31 / 1.32 ms**. Model initialization is outside these per-image inference durations. There is no real source FPS for unordered capture intervals, and no multi-camera fusion timing for that single-camera source.

Host load matters. An earlier complete room run in `runs/m4-accepted-demo` measured 1510 ms mean and 2178 ms p95 input-to-perception duration, with 0.55 frames/s wall replay. An earlier IPO run measured 2475 ms mean inference. These slower results are retained, not replaced with a claimed robot performance figure. None of these runs establishes a sustained Spot processing rate.

After the final display adjustments, `runs/m4-final-viewer/telemetry.json` measured 141.43 / 152.96 / 173.32 ms input-to-perception duration and 5.28 frames/s wall replay. The table above retains the separate `m4-demo` stage measurement, so different runs are not mixed into one timing distribution.

## Viewer and acceptance checks

`runs/m4-final-viewer/humans.rrd` is the final combined developer demo. Rerun was reviewed offline at full and normal window sizes. It displays the room's occupied voxels, chair/table/backpack labels, person skeleton, both camera frustums, RGB/depth, central/sample rays, surface hits, geometric scores, runtime rates/ages, and completed memory state. Long tables and failure details can be scrolled. A direct screenshot-at-start command can capture before loading completes; the native loaded recording was used for visual review.

For 24-frame `--toggle-demo`, frames 8–17 disable the last camera, frames 12–17 disable human inference too, and frame 18 resumes them. The single camera widens pointing uncertainty and reports degraded aggregate health. Human-off frames retain map/entity updates and memory collection while skeleton/rays disappear. In the complete fixture `chair_02` has the strongest score (1.000); other scores can be zero. The viewer does not manufacture weaker nonzero candidates for presentation.

The final automated command passed **63 tests, zero skips**, including optional M2 real-model projection, TUM human integration, IPO estimated pointing, and the existing Spot control/offline UI/model checks:

```sh
SCOPE_REAL_DATASET=runs/m4-public-pose/rgbd_dataset_freiburg1_xyz \
SCOPE_HUMAN_EPISODE=runs/m4-estimated-episode \
SCOPE_IPO_DATASET=runs/m4-public-pose/ipo/Innsbruck_Pointing_Dataset \
SCOPE_HAND_MODEL=models/gesture_recognizer.task \
QT_QPA_PLATFORM=offscreen python -m unittest -v
```

The M2 dataset was restored from its official [TUM archive](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz) after an old `/tmp` path had expired. The earlier run's only error was that missing external directory; the final run uses the persistent ignored path above. Without these optional datasets, normal `python -m unittest -v` skips the three real-data checks. Test output is retained locally in `runs/m4-verification/full-tests-final.txt`.

The acceptance evidence covers M1–M3 continuity, existing command safety, visible/measured health, independent toggles/failure isolation, world-frame skeletons, persistent short tracks, accurate oracle pointing, estimated pointing on labeled real RGB-D, uncertainty/abstention, virtual multi-view contributions, camera loss, M2 scoring, independent memory, stage timing, and the combined debugger. The only future interaction work is two text records; it has no LLM/audio/action implementation.

## Monday read-only sensor check

Confirm the robot hostname with the instructor, connect according to the class procedure, and run:

```sh
source .venv/bin/activate
scope spot-sensors --hostname 192.168.80.3 --samples 5 --output runs/monday-sensors.json
# Inspect a selected source after reading the inventory:
scope spot-sensors --hostname 192.168.80.3 --source frontleft_fisheye_image --samples 10 --output runs/frontleft-sensors.json
```

The diagnostic imports only SDK authentication/time-sync/image/frame-helper APIs. It obtains no lease, controls no E-stop, requests no power, and sends no robot command. It decodes images in memory and saves no captures. A mocked SDK test verifies the report and failed-source isolation; no real robot was contacted in this workspace session. The relevant SDK baseline is the official [get-image example](https://dev.bostondynamics.com/python/examples/get_image/readme).

The report includes advertised image sources, color/grayscale/depth formats, actual returned resolution, pinhole calibration/depth scale, robot acquisition timestamp, host monotonic/Unix receive times, decode duration, RPC round-trip duration, duplicate acquisitions, and transforms from body/odom/vision to each sensor. Observed acquisition rate is request-limited, not maximum camera FPS. One-way network latency is unavailable. The report leaves pixel alignment unverified even when names, dimensions, and transforms look compatible.

On the robot, verify RGB/depth pixel correspondence on stationary geometry, timestamp separation, depth scale, actual calibration model (including fisheye handling), and transform availability before building an M4 input adapter. Measure the chosen sources together, packet/RPC delays, decode and inference at the intended rates, overlap/occlusion, and multi-camera association. Confirm an explicit shared coordinate frame for memory. The diagnostic alone does not validate any of these perception assumptions.

## Known limits and git integration

This is a geometric baseline. It can reject valid poses at occlusion/depth discontinuities, miss cropped hands, associate close people ambiguously, retain dynamic body voxels in the static map, or emit body geometry that resembles pointing without intended reference. Short-baseline finger estimates remain broad. Model confidence, covariance floors, and scores are heuristic; no uncertainty calibration was established. Camera extrinsic/time uncertainty is assumed controlled in the synthetic fixture and must be measured for Spot. Real multi-view people, fisheye/depth alignment, platform portability, sustained throughput, and live-robot sensing remain untested.

Git ancestry was inspected before development. The original `main` (`bd50c6e`) was an ancestor of the complete M1–M3 branch (`e3b0068`), with no divergence or working-tree changes. After the baseline suite and diff review, `main` was fast-forwarded to `e3b0068` and pushed normally. M4 started from that stable state on `codex/milestone-4-human-perception`. Historical milestone branches remain available; no force push, rewrite, or branch deletion was used.

| Checkpoint | Change |
| --- | --- |
| `d9ab82d` | Expose module health and timing telemetry |
| `24ad377` | Track humans and estimate 3D pointing uncertainty |
| `368378f` | Add human perception viewer and sensor diagnostics, including recorded-pose evaluation and failure benchmarks |
| Documentation checkpoint | `Document human perception results and limits`; its hash appears in the repository log and completion report |

The final code and documentation checkpoints are committed and pushed only after their relevant checks. The milestone is accepted for its offline scope, and integration into `main` uses a normal fast-forward after the complete acceptance evidence and a fresh remote ancestry/status check. `main` and the retained `codex/milestone-4-human-perception` branch contain the same integrated milestone. Generated datasets, captures, recordings, telemetry, models, SDK bundles, and private information are not part of that commit history.
