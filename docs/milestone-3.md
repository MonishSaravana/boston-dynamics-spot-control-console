# Milestone 3: persistent world model and memory

Completed on `codex/milestone-3-memory`, based on the completed Milestone 2 head `d418400`. This adds offline memory across supplied-pose episodes. The flagship example preserves four global identities while the backpack moves from beside a chair onto the table. Its old location, new location, evidence, and movement event can be inspected independently.

No pointing, language resolution, dialogue, active search, navigation, autonomous motion, or Spot live integration was added. Existing robot-control code remains separate.

## Architecture and stored evidence

```text
saved RGB-D + poses + masks
  → immutable episode-local entities and observation references
  → supplied episode-to-global map alignment
  → inspectable identity candidates and one-to-one links
  → appended global belief revisions and explicit events
  → history / last-seen / Rerun visit replay
```

| Layer | Representation |
| --- | --- |
| Raw observations | Saved RGB, depth, intrinsics, camera poses, instance masks, detections, and projected observations from M1/M2. |
| EpisodeEntity | Detached local ID, class distribution, observed bounds/center/covariance, up to 1,200 surface points, supporting observations and frames. |
| Episode | ID, start/end time, clock declaration, map file/revision, detector metadata, local association audit, alignment, absolute asset path and SHA-256 hashes. |
| GlobalEntity | A persistent hypothesis ID such as `global/backpack_0001`, connected through immutable link records. |
| Current belief | Latest appended revision: geometry, class distribution, lifecycle, location support/confidence, latest positive evidence, alignment, and unobserved-since time. |
| History | Earlier episode states, link audits, belief revisions, and timestamped event records. |

`memory_store.py` uses SQLite tables for episodes, links, beliefs, events, and processed episodes. SQL triggers reject UPDATE and DELETE on every historical table. Identical imports are idempotent; reusing an episode ID with different evidence fails. Builds process complete, nonoverlapping episodes chronologically and commit links, beliefs, events, and the processed marker in one transaction. Earlier beliefs remain queryable with `store.beliefs(episode_id)`.

Large RGB/depth/mask payloads stay in the saved run directories. Processing, retrieval, and replay verify asset hashes. Moving or changing files requires keeping the stored paths valid; portable archive relocation and schema migration are not implemented. `load_semantic_run` replays saved masks without running a neural detector and checks that they reproduce saved local entity summaries. Legacy M2 masks are supported when frame IDs contain a usable numeric index. New manifests retain original frame IDs and detector/projection/fusion versions; TorchVision runs also record the weights name and package versions. Files lacking model metadata are labeled legacy.

Main modules are `memory_store`, `memory_identity`, `memory_visibility`, `memory`, `repeat_visits`, `memory_eval`, `memory_viewer`, and `memory_cli`. Matching consumes detached local hypotheses, never synthetic truth IDs. Truth enters only the evaluator after decisions have been made.

## Alignment and clocks

Every import requires either a named exact shared frame (`--shared-frame`), or an externally supplied rigid `T_global_episode` with global frame name, provenance, translation sigma in meters, and rotation sigma in radians. External JSON must include both uncertainty fields.

The transform maps episode points into the global frame: `p_global = R p_episode + t`. Camera transforms compose as `T_global_camera = T_global_episode @ T_episode_camera`. Covariance is rotated and expanded by declared alignment uncertainty. Rotation uncertainty acts about the episode origin. Identity gates and negative visibility checks account for uncertainty from both the previous and current episode. Invalid nonrigid transforms, different global frame names, and incompatible clock declarations are rejected.

Example alignment JSON for an identity transform with externally justified uncertainty:

```json
{
  "world_frame": "my_room",
  "provenance": "external_rigid",
  "T_global_episode": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "translation_sigma_m": 0.01,
  "rotation_sigma_rad": 0.005
}
```

Replace the transform and uncertainty with the actual supplied alignment. Memory does not estimate registration, loop closure, or localization. A static-table displacement is a contradiction diagnostic, not registration. Declared uncertainty is trusted; a falsely confident transform can still corrupt identity when the diagnostic does not catch it.

`--clock source_seconds` is the default: ordered source seconds without a fabricated calendar date. `--time-offset-s` puts relative episode clocks on a common ordered axis. Use `--clock unix_utc` only when timestamps plus offsets are Unix epoch seconds. Demos use **simulated** dates of October 3 and 4, 2026. Freshness is age at the latest saved episode, not elapsed wall-clock time since capture.

## Identity evidence and uncertainty

The candidate score combines class-distribution affinity (weight 0.25), observed dimensions (0.22), centered visible-surface similarity from symmetric median nearest-point distances (0.18), location compatibility scaled by mobility and elapsed time (0.25), and nearest neighboring-class distances (0.10). Alignment uncertainty and static-anchor residual reduce the combined score.

Mobility priors are 0.10 for tables, 0.75 for chairs, 1.0 for backpacks, and 0.5 otherwise. Shape is compared after translation; this is not a full object-pose or rotation-invariant descriptor. Bounds describe observed surfaces, not hidden volume.

Gates require class affinity ≥0.25, dimension score ≥0.38, surface score ≥0.30, combined score ≥0.66, alignment sigma ≤0.20 m, static-anchor residual ≤0.50 m, and a mobility-dependent displacement bound. Links also require ≥0.085 separation from competing local/global assignments. Audits retain distances, elapsed time, every score component, gate reasons, and assignment margins.

Confident unique links are assigned first. Compatible unused identities with weak geometry or ambiguous scores remain `UNRESOLVED`, without replacement global IDs. A new entity is created when there is no remaining compatible prior hypothesis. This supports the new-box fixture but limits same-class novelty: a new chair can remain unresolved against an unobserved old chair. Appearance embeddings and identity correction/merging tools are not implemented.

Scores and confidence are heuristics, not calibrated probabilities. This finite fixture suite has no incorrect accepted links; that does not establish general accuracy or calibrate reliability on real scenes.

## Events, lifecycle, and negative evidence

Positive links append `OBSERVED` or `REOBSERVED`, `IDENTITY_LINK`, and, when applicable, `NEW_ENTITY`. Displacement greater than `max(0.20 m, 3 × alignment sigma)` records `MOVED` when link score is ≥0.72 and identity margin is adequate; otherwise it records `POSSIBLY_MOVED`. Movement records retain old/new positions, previous episode, evidence IDs, confidence, identity audit, time, and algorithm version. Old geometry is never overwritten.

The five lifecycle states are `VISIBLE`, `NOT_CURRENTLY_OBSERVED`, `POSSIBLY_MOVED`, `POSSIBLY_MISSING`, and `IDENTITY_UNCERTAIN`. Location support/confidence are separate. Unresolved identities retain the last positive position and mark it unsupported. A later positive reappearance restores supported belief without erasing an earlier missing hypothesis; a third-visit regression test covers this.

Unmatched prior entities receive `NOT_OBSERVED` and per-frame visibility diagnostics. Old surface samples are projected into each later view. A qualified absence check requires:

| Test | Threshold |
| --- | --- |
| Old surface coverage inside view | ≥55% |
| Unique projected pixels | ≥12 |
| Valid depth fraction | ≥80%, within 0.15–5.0 m |
| Nearer occluding-depth fraction | ≤20% |
| Rays passing beyond the old surface | ≥65%, by more than 0.12 m |
| Detector absence reliability | ≥80% |
| Combined alignment sigma | ≤0.10 m |

Each check retains viewpoint, expected region, frame/evidence IDs, coverage, depth quality, occlusion/free-ray fractions, detector capability, alignment uncertainty, and timestamp. M1 free-space classification is an additional diagnostic, not a replacement for current-frame visibility. Unqualified checks are `VISIBILITY_CHECK` with explicit reasons. Qualified `OBSERVED_ABSENT` results become `NOT_VISIBLE_FROM_VIEW` events.

`MISSING_HYPOTHESIS` requires two qualifying views separated by at least 0.10 m or 0.10 rad. It reduces current location confidence and sets `POSSIBLY_MISSING`; it never deletes the entity or concludes physical destruction. The removed fixture has two qualified negative checks; unobserved, occluded, and dropped-detection fixtures have none for the backpack.

Synthetic exact masks permit known detector capability. Real-model absence reliability defaults to zero because miss probability and image-condition reliability have not been measured. `--negative-reliability` accepts an externally justified override; this work provides no calibration for it. There is no separate blur/lighting model. Another surface at the old distance can prevent an absence claim even when the object is gone.

All events carry episode, time, global ID, evidence, and `scope-memory-v1`; score/confidence and state-change information are included where relevant. Historical queries sort by time and insertion order. Episode-end identity decisions can be later than the positive supporting frame.

## Commands and module isolation

Use the Python 3.11+ installation steps from [the offline tools page](offline-tools.md#offline-3d-mapping):

```sh
source .venv/bin/activate
python -m pip install -e .
scope memory-demo moved_backpack --output runs/memory-backpack
scope last-seen global/backpack_0001 --db runs/memory-backpack/memory.sqlite
scope history global/backpack_0001 --db runs/memory-backpack/memory.sqlite
scope replay-memory --db runs/memory-backpack/memory.sqlite
```

Use `--no-viewer` on demo/replay commands to save without opening a window. The flagship query returns simulated `2026-10-04T14:00:01.400000+00:00`, frame `episode_002-0007`, map revision 8, and position approximately `[-0.045, 0.951, 1.085] m`. The supporting image is a positive frame; the position is the whole episode's fused observed-bounds center, not a frame-only physical-center measurement.

Using that demo's saved episodes and a new database, each layer can run separately:

```sh
scope inspect runs/memory-backpack/episode_001 backpack_01
scope compare-episodes runs/memory-backpack/episode_001 runs/memory-backpack/episode_002 --shared-frame synthetic-room
scope memory add runs/memory-backpack/episode_001 --db runs/memory-manual/memory.sqlite --episode-id episode_001 --shared-frame synthetic-room --clock unix_utc
scope memory build --db runs/memory-manual/memory.sqlite
scope memory add runs/memory-backpack/episode_002 --db runs/memory-manual/memory.sqlite --episode-id episode_002 --shared-frame synthetic-room --clock unix_utc
scope memory episode episode_002 --db runs/memory-manual/memory.sqlite
scope link-entities episode_002 --db runs/memory-manual/memory.sqlite
scope memory build --db runs/memory-manual/memory.sqlite
scope memory list --db runs/memory-manual/memory.sqlite
scope history global/backpack_0001 --db runs/memory-manual/memory.sqlite
scope last-seen global/backpack_0001 --db runs/memory-manual/memory.sqlite
scope replay-memory --db runs/memory-manual/memory.sqlite
```

`memory add` imports without linking. `link-entities` previews an unprocessed episode; afterward it returns the persisted audit. `compare-episodes` uses a temporary store and leaves the source runs untouched. For external alignment, replace `--shared-frame` on imports with `--alignment /path/to/alignment.json`; comparison accepts `--alignment-a` and `--alignment-b`. Missing databases and modified assets produce explicit errors. Use a new output directory for benchmarks; demos also reject unrelated existing assets or different scenario settings.

These commands were exercised independently in `runs/m3-module-check`. At 192×144, four preview scores were approximately 0.929 (table), 0.947/0.951 (chairs), and 0.844 (backpack). Build and standalone comparison agreed. A unit test supplies a 90° map rotation plus translation and recovers four links with scores above 0.95.

## Synthetic scenarios and benchmark

```sh
scope memory-demo removed --output runs/memory-removed
scope memory-demo unobserved --output runs/memory-unobserved
scope memory-demo ambiguous_chairs --output runs/memory-ambiguous
scope benchmark-memory --output runs/memory-benchmark
```

The benchmark uses two visits of eight frames each at 128×96, exact poses/masks, analytic box geometry, and fixed random seed 20261003. Each case saves RGB-D/masks/map, database, and metrics. Combined results are `memory_metrics.json`; plots are `memory_robustness.png`, `alignment_sensitivity.png`, and `identity_score_error.png`. Outputs are ignored by Git. The final measured run is `runs/m3-benchmark-final`.

Identity precision counts correct accepted links; recall counts correct links among visit-two local entities whose truth existed in visit one. This is association recall conditional on detections, not end-to-end detector recall. Unresolved rate is among visit-two local entities. False merges count global IDs containing multiple truth objects; splits count extra IDs per truth object; switches count changed IDs across sightings. Change precision/recall compare final-visit events with analytic changes. Rates without a denominator are JSON `null` (N/A), not zero.

| Scenario | Identity precision / recall | Unresolved | Mean updated position error (m) | Changes / behavior |
| --- | --- | --- | --- | --- |
| unchanged | 1.00 / 1.00 | 0% | 0.0195 | Four identities retained. |
| moved_backpack | 1.00 / 1.00 | 0% | 0.0242 | One supported move; true displacement 1.35 m. |
| moved_chair | 1.00 / 1.00 | 0% | 0.0483 | One supported move; true displacement 0.79 m. |
| removed | 1.00 / 1.00 | 0% | 0.0233 | One missing hypothesis; positive history retained. |
| unobserved | N/A / 0.00 | 100% | N/A | Narrow left view; no backpack absence claim. |
| new_object | 1.00 / 1.00 | 0% | 0.0354 | One new box/global ID; five positive positions. |
| ambiguous_chairs | 1.00 / 0.50 | 50% | 0.0207 | Two nearby chairs remain unresolved. |
| occluded | N/A / 0.00 | 100% | N/A | Partial views abstain; no missing hypothesis. |
| missing_detections | 1.00 / 1.00 | 0% | 0.0233 | Backpack masks dropped; occupied depth prevents absence. |
| class_disagreement | 1.00 / 1.00 | 0% | 0.0195 | Chair/stool probabilities 0.55/0.45 retain identity. |
| depth_noise | 1.00 / 1.00 | 0% | 0.0365 | 0.04 m Gaussian noise and 10% depth holes. |
| alignment_10cm | 1.00 / 1.00 | 0% | 0.0993 | Lower scores, retained links, shifted positions. |
| alignment_25cm | N/A / 0.00 | 100% | N/A | Declared uncertainty blocks every link. |
| alignment_1m | N/A / 0.00 | 100% | N/A | Declared uncertainty blocks every link. |
| alignment_1m_undeclared | N/A / 0.00 | 100% | N/A | Static-table contradiction blocks links despite zero declared sigma. |

Declared-alignment fixtures apply an X translation error and declare the same value as sigma. They measure the combined effect of error and uncertainty, not an independently calibrated registration estimator. The undeclared 1 m case tests the contradiction diagnostic.

All 15 cases had zero false identity merges, splits, switches, and false missing claims. Movement precision/recall was 1.00/1.00 for both resolved move cases. New-object and missing-hypothesis precision/recall were each 1.00/1.00 in their respective single-object fixtures. Ambiguous-chair movement recall was **0.00**: abstention misses both real movements.

Position error is Euclidean distance from observed-bounds center to analytic whole-object center. Updated-position metrics include every visit-two positive update, including new objects. JSON also reports current-belief error for all physically present global hypotheses, including stale positions. In the ambiguous case, only table/backpack update: mean error 0.0207 m, while all four retained/current beliefs average **0.5493 m**. The chairs' old positions are unsupported. N/A updated errors in unresolved cases do not mean zero localization error.

Historical positive-position error is 0.0195 m unchanged, 0.0218 m moved backpack, 0.0339 m moved chair, and 0.0594 m under 10 cm alignment error. Last-seen timestamp correctness, location within 0.15 m of truth at the positive visit, and evidence retrieval correctness were 1.00 in every case. Retrieval checks inspect actual saved frame/observation IDs; mutation tests verify hashes. Score/error samples retain identity-correctness flags. The plot shows localization error versus accepted-link score, excluding rejected links; no probability calibration is claimed.

## Replay and review

The `visit` timeline records episode-end beliefs. Each tick shows that episode's aligned map, supporting RGB/mask image and depth, current bounds/IDs, lifecycle colors, freshness, and events. Earlier positive positions separated by at least 0.20 m are gray HISTORY boxes. Moves draw yellow vectors; qualified negative checks draw red viewpoint-to-region lines. Select `world/current/global/backpack_0001/evidence` for belief, support, and identity audit.

The memory timeline does not progressively rerun identity or display every raw support frame within a visit. Original indices, poses, masks, and observation references remain available through episode/evidence inspection. Historical snapshots stay on earlier ticks; current change/history layers are rebuilt on each tick.

The flagship was visually checked at approximately 2,750×1,806 and 2,106×1,456 screenshot pixels on macOS, switching visits and reviewing stable chair IDs, backpack HISTORY/movement, RGB/depth support, and freshness. The removed-object replay was also checked: red old bounds, POSSIBLY_MISSING status, and qualified negative rays. These are synthetic views; no live robot or private captures were used.

Local outputs to inspect:

- `runs/m3-flagship/memory.rrd` and `memory.sqlite`.
- `runs/m3-benchmark-final/removed/memory.rrd`.
- `runs/m3-benchmark-final/memory_metrics.json` and the three plots.

## Verification and limits

```sh
QT_QPA_PLATFORM=offscreen python -m unittest -v \
  tests.test_scope_memory tests.test_scope_semantic_pipeline tests.test_scope_entities \
  tests.test_scope_objects tests.test_scope_mapping tests.test_spot_gesture \
  tests.test_spot_gui_offline tests.test_spot_model_view
```

The final default suite collected **46 tests: 45 passed, one optional real-data test skipped**. The skipped test was enabled against the existing local TUM `freiburg1_xyz` dataset and passed separately:

```sh
SCOPE_REAL_DATASET=/path/to/rgbd_dataset_freiburg1_xyz \
QT_QPA_PLATFORM=offscreen python -m unittest -v \
  tests.test_scope_semantic_pipeline.RealDetectorIntegrationTests
```

Coverage includes immutability, evidence/replay, idempotence, explicit transforms, old/new alignment uncertainty, identity/change scenarios, depth/visibility gates, missing-then-returning objects, metrics/retrieval, M1/M2 pipelines, and offline Spot safety/UI/model tests. Geometry-only and semantic-map CLI demos were rerun successfully with five frames at width 96. A previously saved real M2 map was imported and built as one episode with five local/global entities: evidence interoperability, **not independent real-visit identity validation**. The real detector test checks positive projection/fusion. No live Spot evaluation was performed.

Checked environment: macOS 26.6.2, Apple Silicon, Python 3.14.2, Rerun 0.38.1. Windows/Linux were not checked. A scikit-image/NumPy deprecation warning did not fail mapping tests.

Known limits:

- Narrow views and occlusion can reject every match, sacrificing recall.
- Identical objects swapping the same positions cannot be distinguished. The ambiguity fixture moves chairs near one another; it does not solve exact symmetric swaps.
- Same-class novelty, orientation changes, hidden surfaces, large motion, M2 fusion errors, and wrongly declared alignment can defeat identity. A moved table can trigger room-wide abstention.
- Confidence is heuristic. Staleness marks support and reports age; no calibrated time-decay model is provided.
- Negative evidence needs adequate depth and justified detector capability. Real-image lighting/blur and independent real multi-visit performance remain unmeasured.
- Fifteen fixed room cases with few objects are not a real-world accuracy guarantee.

During development, partial views initially produced unwanted new IDs; abstaining against compatible unused priors fixed that failure. An early ambiguity layout intersected geometry and obscured other objects; final chairs have a small physical gap and remain distinct locally. Automatic Rerun depth reprojection initially obscured the map; the 3D view now excludes it while retaining the depth panel. No automatic registration was added to compensate for bad alignment.

## Git checkpoints

All implementation checkpoints were pushed to `origin/codex/milestone-3-memory` without rewriting history or force pushing:

| Commit | Change | Verification before commit | Pushed |
| --- | --- | --- | --- |
| `6e546f0` | Persist immutable semantic episodes | Storage/replay checks and then-current full suite. | Yes |
| `d96d966` | Link global entities and retain change evidence | Identity/lifecycle/visibility scenarios and then-current full suite. | Yes |
| `98c5831` | Replay persistent memory and measure repeat visits | Final 46-test default suite, optional real test, benchmark, modular CLI and offline visual checks. | Yes |

This report and README follow in a documentation commit; `git log -4 --oneline` lists the complete milestone history. Run data, databases, recordings, datasets, models, SDK files, and credentials are excluded from Git. Milestone 4 has not begun.
