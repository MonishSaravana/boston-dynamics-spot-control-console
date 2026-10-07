# Offline mapping and perception tools

These commands run on recorded or synthetic data and need no robot. They use the `scope` command from this repository. The browser console reads their outputs (World, History and Debug), and the advanced 3D views open in Rerun. Run outputs, recordings, databases and model bundles stay outside Git.

Contents: [3D mapping](#offline-3d-mapping), [semantic objects](#semantic-objects-in-the-map), [persistent memory](#persistent-memory-across-visits), [human geometry and pointing](#human-geometry-pointing-and-module-health), [object queries and benchmarks](#local-object-queries-and-reality-benchmarks).

## Offline 3D mapping

The `scope` command maps a supplied-pose RGB-D sequence into a colored point cloud, observed free/occupied/unknown voxels, and a TSDF surface. It opens a Rerun recording with RGB, depth in meters, camera trajectory and frustum, a growing 3D map, and a top-down occupancy view. Use the **frame** timeline to scrub or press Play to watch observations accumulate; drag in the 3D view to orbit. Orange means occupied, teal means observed free, and dark gray means unknown. A depth hole leaves space unknown rather than marking it free. Map voxel size defaults to **0.10 m**; depth integration accepts **0.15–5.0 m**.

This is a known-pose mapper, not a localization system. A wrong but geometrically valid camera pose can distort the map; the synthetic accuracy report exposes that error, but the overlap warning does not detect every bad pose. TSDF fusion uses a room-scale dense grid capped at two million voxels. Input RGB and depth are synchronized by the source adapter; frame validation rejects delayed depth, changing intrinsics, invalid transforms, missing valid depth, and nonincreasing timestamps. The existing Spot GUI is separate and continues to run as described above.

From this repository, install the mapping command in a Python 3.11+ environment. In this workspace, the existing `.venv` was used:

```sh
cd /path/to/boston-dynamics-spot
source .venv/bin/activate
python -m pip install -e .
scope map synthetic
```

The synthetic run needs no download or robot. Its deterministic room has a floor, four walls, a table, two chairs, and a backpack-sized box. It supplies rendered color/depth, calibrated pinhole intrinsics, exact camera poses, and analytic geometry truth. The command creates a time-stamped directory under `runs/` and opens the interactive viewer. For a named run or for use without a display:

```sh
scope map synthetic --output runs/room-demo
scope map synthetic --output runs/room-headless --no-viewer
scope view runs/room-headless
```

For recorded data, download and extract an RGB-D sequence from the [TUM RGB-D dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) that includes `rgb.txt`, `depth.txt`, and `groundtruth.txt`, then run:

```sh
scope map tum /path/to/rgbd_dataset_freiburg1_xyz --frames 20 --frame-stride 10 --width 160 --output runs/tum-xyz
```

The TUM adapter uses the recorded camera poses, associates nearby RGB/depth/pose timestamps, converts 16-bit depth using the dataset's 1/5000 m scale, and resizes its default pinhole calibration with the images. The 20-frame `freiburg1_xyz` command above was run in this workspace against a downloaded recording. Its geometry is sensitive to dataset calibration and pose accuracy; the included intrinsics are an approximation for this sample, not a calibration procedure for arbitrary cameras. No TUM images are committed here.

Each run saves `episode.npz` and a manifest (the synchronized input), `map.npz` (voxel evidence), `tsdf.npz` and `tsdf_mesh.ply` (fused surface when extractable), `reconstruction.rrd` (the scrubable viewer), `metrics.json`, and `diagnostics.json`. Synthetic runs also save `quality_report.png`. Its surface error uses analytic scene boxes. Occupied and free precision compare voxels with those boxes; completeness and unknown correctness compare the online ray sample with a denser ray sample of the same observations. Those latter values measure sampling coverage, not recovery of surfaces that no camera saw. Reload or remap a saved input with:

```sh
scope view runs/room-demo
scope map episode runs/room-demo --output runs/room-replay --no-viewer
```

Run mapping and existing offline tests without a robot:

```sh
QT_QPA_PLATFORM=offscreen python -m unittest -v tests.test_scope_mapping tests.test_spot_gesture tests.test_spot_gui_offline tests.test_spot_model_view
```

## Semantic objects in the map

The optional semantic pipeline converts instance masks and valid depth into world-space object observations, then fuses repeated views into episode-local entities. Each entity has a stable ID, class distribution, observed 3D bounds and center in meters, uncertainty, timestamps, and the frame observations that support it. The Rerun viewer overlays 2D boxes and masks, projected object points, and labeled 3D entity bounds. Select an entity's `evidence` path in Rerun or use `scope inspect` to see its observations and association scores. The displayed confidence combines class evidence, valid depth, and view count; it is a heuristic, not a calibrated probability of correctness. These are current-episode estimates; they do not establish long-term identity or hidden object shape.

Start with exact synthetic masks, which need no model download:

```sh
scope detect synthetic
scope project-objects synthetic
scope entities synthetic --output runs/room-entities
scope map synthetic --entities --output runs/room-semantic-map
scope inspect runs/room-entities chair_01
scope benchmark-entities --output runs/room-object-benchmark
```

`detect` can be inspected without depth projection or entity fusion; `project-objects` can be inspected without fusion. The ordinary `scope map synthetic` command still runs geometry alone. The synthetic fixture labels two separate chairs, a table, and a backpack. Its masks and poses are **exact simulated truth**, not the output of a neural detector. `semantic_metrics.json`, `semantic_quality.png`, and the benchmark's `semantic_robustness.json`/`.png` report projection, association, and controlled failure results.

For recorded TUM imagery, install the optional detector and run a short sample:

```sh
python -m pip install -e '.[detector]'
scope entities tum /path/to/rgbd_dataset_freiburg1_xyz --frames 6 --frame-stride 10 --width 320 --classes chair,keyboard,cup,tv --output runs/tum-entities
scope view runs/tum-entities
scope map tum /path/to/rgbd_dataset_freiburg1_xyz --frames 6 --frame-stride 10 --width 320 --entities --classes chair,keyboard,cup,tv --output runs/tum-semantic-map
```

The real adapter uses TorchVision Mask R-CNN v2 COCO instance masks. The first run uses `curl` to download its approximately 177 MB weights from the official PyTorch host into the user's PyTorch cache, checks the filename's SHA-256 prefix, and does not add weights to this repository. It runs on CPU and excludes `person` by default; `--classes` chooses exact COCO labels. Recorded TUM masks are model predictions, and their class scores are not measured accuracy for that scene. The depth and poses come from the recording, not Spot. The model uses the [TorchVision pretrained weights API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html); TorchVision warns that pretrained weights can have terms derived from their training data.

The entity JSON and association decisions are saved beside the original episode. `scope inspect runs/tum-entities keyboard_01` shows source detections, match evidence, and relations. Geometric relations in `relations.json` include distances, near, above/below, and ordering along the **world X axis**. World X ordering is a coordinate-frame measurement, not a viewer-relative or language-grounded notion of left and right. See [the Milestone 2 report](milestone-2.md) for benchmark definitions, limits, and commits.


## Persistent memory across visits

The offline memory layer links saved episode-local entities to global hypotheses in an explicitly shared coordinate frame. SQLite retains each episode, identity audit, belief revision, and change event; previous positive locations remain available after an object moves or becomes unobserved. `last-seen` returns the last positive timestamp, fused position in meters, supporting frame and observation IDs, map revision, current status, location support, and later negative checks.

Run the two-visit synthetic example after installing `scope` as above:

```sh
scope memory-demo moved_backpack --output runs/memory-backpack
scope last-seen global/backpack_0001 --db runs/memory-backpack/memory.sqlite
scope history global/backpack_0001 --db runs/memory-backpack/memory.sqlite
scope replay-memory --db runs/memory-backpack/memory.sqlite
```

In Rerun, select the **visit** timeline and switch between `#1` and `#2`. Both chairs keep their global IDs. The backpack moves about **1.35 m** onto the table; its old position is gray and labeled **HISTORY**, with a yellow movement vector. These are episode-end belief snapshots with a supporting RGB/depth frame. Select `world/current/global/backpack_0001/evidence` in the stream tree to inspect the identity scores and evidence. Colors distinguish positive, stale, missing, and uncertain beliefs. The dates and imagery are simulated.

Other scenarios expose removal, unobserved regions, similar-object ambiguity, occlusion, detector misses, noise, and alignment errors:

```sh
scope memory-demo removed --output runs/memory-removed
scope memory-demo unobserved --output runs/memory-unobserved
scope memory-demo ambiguous_chairs --output runs/memory-ambiguous
scope benchmark-memory --output runs/memory-benchmark
```

An object becomes `POSSIBLY_MISSING` only after two adequate, distinct views see depth rays pass through its old region. A region outside the view, blocked by another surface, or supported by poor depth does not establish absence. Missing detections alone do not establish absence either. Real-model absence reliability defaults to zero because it has not been measured.

Use `scope compare-episodes RUN_A RUN_B --shared-frame FRAME` to inspect identity candidates without adding to persistent memory. This flag asserts that both runs already share exact coordinates. For separately aligned maps, supply rigid-transform JSON with explicit translation/rotation uncertainty. Memory does not estimate map registration. Scores are heuristics; similar chairs can remain unresolved, and a new object of a previously seen class can also be left unresolved. The [Milestone 3 report](milestone-3.md) gives separate import/build commands, alignment examples, all benchmark results, and failure limits. Run outputs, recordings, and databases stay outside Git.

## Human geometry, pointing, and module health

The Milestone 4 runtime lifts visible shoulders, elbows, wrists, and other body joints through aligned depth, fuses compatible virtual-camera observations, and maintains short-lived person IDs. Pointing uses sampled origins and directions with an explicit angular spread. Missing joints can widen the distribution or produce `ABSTAIN`; ambiguous people remain unresolved. Candidate scores are geometric hit fractions against current M2 entity bounds and M1 occupied voxels, not calibrated probabilities of human intent.

Install the offline package and launch the combined room demo:

```sh
source .venv/bin/activate
pip install -e .
scope map synthetic --entities --humans --frames 24 --width 192 --toggle-demo --output runs/human-room
```

The Rerun recording shows the map, four entities, a 3D skeleton, front/left camera frustums, ray samples, candidate scores, module health, measured host update rates, and source-data ages. The person points toward chair B (`chair_02`). On the **frame** timeline, frames 8–17 disable the left camera, frames 12–17 also disable human inference, and frame 18 resumes both. Mapping and semantic entities continue; the completed primary-camera episode is then imported through M3 into the run's `memory.sqlite`. The synthetic RGB skeleton and visible keypoints are illustrative/oracle data, not a neural detector result.

Stages and expensive outputs can run separately:

```sh
scope humans synthetic --camera front
scope humans synthetic --camera front --camera left
scope pointing synthetic --oracle-keypoints
scope telemetry synthetic --disable camera/left --rate semantic_detection=5 --every mapping=2
scope pointing synthetic --disable map_visualization
scope pointing synthetic --no-recording --no-viewer
```

Use `--disable MODULE`, `--rate MODULE=HZ`, `--every MODULE=N`, or `--config CONFIG.json`; the same mutable runtime configuration is available to a future GUI. `--realtime` moves pose and semantic inference into bounded workers so a slow model can drop obsolete inputs while mapping continues. Replay defaults to synchronous processing for reproducible evaluation. Per-camera/depth/model settings use names such as `camera/front`, `depth/left`, `pose/front`, and `semantic/left`. The robot-model view is reserved and disabled in this offline viewer; the existing GUI still provides its own model view.

Estimated pose is optional and uses CPU TorchVision Keypoint R-CNN. Add the hand extra and a local Gesture Recognizer bundle to use reliable index-finger geometry:

```sh
pip install -e '.[pose,hands]'
scope pointing synthetic --estimated-keypoints --frames 3 --no-viewer
scope pointing episode /path/to/saved-rgbd-episode --estimated-keypoints --hand-model models/gesture_recognizer.task
scope benchmark-humans --frames 10 --output runs/human-benchmark
```

The model does not recognize the synthetic stick person reliably. Its real-source checks instead use public TUM RGB-D people and labeled Innsbruck pointing stills. The labeled subset produced a direction in 7 of 10 target cases, averaging 4.67° angular error on those seven; two cases abstained and one lacked depth. This small subset does not establish general pointing accuracy or real Spot support. CPU inference and the complete room demo are slow in this workspace. See [the Milestone 4 report](milestone-4.md) for source preparation, exact benchmark commands, mean/median/p95 timings, calibration assumptions, failures, and acceptance evidence.

A read-only sensor inventory is ready for the next robot session. Confirm the hostname with the instructor; credentials are prompted by the SDK:

```sh
scope spot-sensors --hostname 192.168.80.3 --samples 5 --output runs/monday-sensors.json
```

It reads image capabilities, timestamps, transforms, RPC durations, and in-memory decode durations. It does not acquire a lease or issue motion. Pixel-level RGB/depth alignment still needs a separate check on the robot; the report explicitly leaves it unverified.

## Local object queries and reality benchmarks

M4.5 adds two independent modes: periodic common-object discovery with SSDLite, and on-demand text queries with Grounding DINO Tiny plus OWLv2 corroboration. Supported query boxes receive SAM 2.1 Tiny masks during discovery/refresh. Sparse optical flow propagates masks; disagreement, expired verification, lost tracks, and ambiguity remain visible. Scores are uncalibrated evidence. Text modifiers such as color or brand are not independently verified, and two models can agree on the wrong object.

Install the optional local models and Qt query console, then select an input:

```sh
source .venv/bin/activate
python -m pip install -e '.[perception,perception-gui]'
scope hardware
scope query-object "gray sofa" --image /path/to/room.jpg --output runs/sofa-query
scope perceive --image /path/to/room.jpg --interactive --output runs/room-query-console
scope perceive --webcam 0 --interactive --background --output runs/camera-queries
```

Model weights download on first use into the local model cache. Startup can take seconds and requires network access until cached; subsequent inference is local. The camera opens only when `--webcam` is supplied. Video files use `--video /path/to/phone-video.mp4`. RGB inputs support masks and 2D tracks; measured 3D requires aligned depth, calibrated intrinsics, and a declared camera/world pose. Use `--image ... --depth ... --calibration ...` with the JSON format in the [M4.5 report](milestone-4.5.md), or replay `--episode`/`--tum` recordings with supplied poses. These commands have no robot control path.

Enter a phrase and press **Find object**. The console shows the raw phrase, expanded query, proposed region, mask, observed center in meters when supported, track state, evidence scores, backend/device, latency, and output age. Scroll the telemetry table for mean/median/p95, queues, drops, and model details. The Queries, Common objects, Human pose, Mapping, Memory, and Rerun checkboxes change modules independently. Retained tracks can outlive the query detector briefly; verification expires after 3 seconds on moving sources. A still image represents one immutable observation, with unknown original acquisition time.

Common objects and human pose are opt-in (`--background`, `--humans`). Defaults cap common detection at 0.5 Hz, pose at 3 Hz, mapping at 5 Hz, Rerun at 5 Hz (1 Hz in the interactive console), and query refresh at one per 1.5 seconds. Actual rates are measured and may be lower. On this M4 host, a warmed 180-frame public replay maintained about 30 input updates/s with a 15.0 ms mean core tick; query results averaged about 1 second during concurrent inference. This is a core responsiveness measurement, not a 30 Hz detector claim. Rerun recording runs in a bounded worker; native viewer rendering cost is outside those measurements.

```sh
scope perceive --video /path/to/room.mp4 --query "power strip" --background --duration 30 --no-rerun
scope perceive --episode /path/to/rgbd-episode --interactive --disable human_pose --rate rerun=1
scope capture-reality --video /path/to/phone-video.mp4 --frames 100 --output runs/room-capture
scope capture-reality --webcam 0 --frames 100 --hz 10 --output runs/webcam-capture
```

Capture/import writes RGB frames and `reality.json` outside Git. Add explicit per-frame presence/absence, boxes or mask files, room IDs, and development/heldout splits before evaluation. A room cannot appear in both splits. An empty annotation list is rejected rather than scored as negatives:

```sh
scope benchmark-reality --manifest runs/room-capture/reality.json --split heldout --output runs/room-evaluation
```

The [M4.5 report](milestone-4.5.md) records room selection, public dataset preparation, closed/open vocabulary comparisons, failures, tracking persistence, acceleration, exact reproducible probes, and limits for Spot testing. Run outputs and model bundles stay outside Git. This milestone adds no spatial-language parser, LLM planner, navigation, or audio.
