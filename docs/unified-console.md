# SCOPE workstation

This pass joins the existing interfaces. It adds no model, fusion system or
planner. The supervised GO executor runs in Robot control and is
**READY FOR PHYSICAL VALIDATION**. No physical Spot was connected or
commanded during this pass.

## Launch

From the repository, using the existing environment:

```sh
.venv/bin/python scope_web.py --demo
```

For the sensor connection form, omit `--demo`. Observe is the default.
`--port 8766 --no-browser` prints a fixed local URL without opening a tab.
`--runs-dir PATH` is repeatable; the default is this repository's `runs/`.
The managed macOS launcher follows `main`, so use this checkout's command
for the development branch. Nothing in this pass was merged into `main`.

## Live

The rail opens Live, World, History and Debug. Camera / World / Split switch
the central canvas; the inspector changes from interaction to evidence to
destination. The inspectors resize and collapse; long evidence scrolls inside them.
Control and Inspect open bottom drawers. Desktop workspaces fill the window
without page scrolling. Existing URL hashes are retained.

Type `chair`, select a candidate, inspect the box and world evidence, then
**Confirm target**. Inspect destination x/y in meters, heading in degrees,
standoff, clearance and sampled route before **GO**. Confirmation alone
never dispatches. Open **Inspect → Stages & command telemetry** for the unrounded SE2 destination
(x/y in meters, yaw in radians and world frame). GO consumes its preview. Stop or a settings change clears
approval; preview again only after inspecting current evidence.
The existing ten-second preview limit disables GO. The inspector retains the
target and coordinates, shows Review destination / Refresh required, and explains
the expired preview beside GO. Refresh still performs the existing checks.

**Use pointing** produces the same TargetCandidate. It requires human pose
to be enabled when connecting and sufficient existing geometric evidence;
an unresolved ranking abstains. In demo it is explicitly **Simulate
pointing**, using a fixture rather than a detected person. **World → select
entity → Use as target** enters the same confirmation path.

Targets and destinations live in the world/odom frame. Camera source is
provenance, not an allowed travel direction. The facing arrow and numeric
heading show a turn toward targets behind the robot. Camera switching or
automatic reacquisition during navigation has not been added.

Manual controls remain under **Control** on the rail: Power On, Stand, held W/A/S/D or arrow
buttons, 0.05–0.35 m/s speed, requested height 0–10 cm and roll/pitch ±5°,
Apply, gesture mode, five fisheye views, split and approximate panorama.
Manual motion/posture/gesture requests are blocked during active GO. The
SDK model is read only: measured joints when available, clearly labeled
simulated reference in demo. Pose requests do not establish measured pose.

Observe authenticates only images/state, with no command lease. Dry run
records the proposed SE2 destination and sends no movement command. Robot
control requires an explicitly authorized connection and the separate
class E-stop. It reuses the manual SpotSession's SDK command client and
lease for manual drive, posture and the supervised GO executor. Changing modes stops motion;
an already acquired command lease remains until Disconnect. Reconnect in
Observe for a connection that has no command authority.

## Cameras and modules

**Live → Camera or Split → Sources** shows real discovered sources, independent
acquisition, processing, display and request rate (0.2–30 Hz), plus exactly one selected
visual/depth processing pair. Additional display feeds do not imply
multi-camera Spot mapping. Alignment verification is a physical operator
prerequisite, not a calibration function. Unverified/missing depth permits
image/query inspection but blocks map-based destinations.

Acquire and display the desired feeds for split view. The retained manual
session's panorama needs fresh front-left and front-right frames; stitching
and measured model telemetry currently use that command-connected session.
Read-only Observe connections still provide individual and split sources.
Human pose/model initialization is selected when connecting. Open
**Inspect** for module enable/rate controls and detailed stage timing. Processing
a source does not add another fusion input: only the chosen visual/depth pair
feeds the current pipeline. Changing any source policy clears target approval
and destination; source toggles cannot authorize a move.

## World, History and Debug

World provides current occupancy/free/unknown, entities, robot pose, selected
target and route, layer toggles, world units and an entity inspector. Its
isometric projection is a view of the same occupancy data, not a reconstructed
point cloud. **3D · Rerun** lists actual local recordings and launches native Rerun;
**History → choose .rrd → Open in Rerun** does the same.
Point clouds, meshes, skeletons, pointing rays, camera frustums, memory layers,
entity history and playback timelines retain their existing CLI/Rerun paths.

History discovers the latest 500 supported local artifacts in configured roots.
JSON inspection is limited to 2 MB. Memory SQLite databases are read only,
with up to 100 recent episode, belief and event payloads each. Saved maps
show metadata/counts and their viewer command; meshes/RGB-D episodes retain
CLI replay. Symbolic links and paths outside configured roots cannot open
private files. No cloud storage or automatic capture upload is introduced.
Use a narrower `--runs-dir` to browse older sessions beyond the index limit.

Debug offers existing perception, mapping, pointing/human, memory,
entity and test commands, results through History, and current module health.
It does not silently start benchmarks or download model bundles. Replace
shown dataset/path placeholders before copying a command.

The original entry points remain:

```sh
.venv/bin/python spot_control_gui.py --demo
.venv/bin/python -m scope.m5_console --demo
.venv/bin/python -m scope --help
.venv/bin/python -m scope query-live --help
.venv/bin/python -m scope view RUN_DIRECTORY
.venv/bin/python -m scope memory list --db MEMORY_DB
.venv/bin/python -m scope history ENTITY_ID --db MEMORY_DB
.venv/bin/python -m scope replay-memory --db MEMORY_DB
```

## Capability locations

| Existing capability | Access in this revision |
| --- | --- |
| M1 occupancy, free/unknown, robot pose, metric grid | World; Live → World or Split for the current target/route |
| M1 point clouds, TSDF/meshes, camera poses and frustums | World → 3D · Rerun; existing Rerun layers and timeline |
| M1 map creation/save/load and RGB-D replay | Debug → Mapping and Retained tools → Map reload / viewer; History → saved map metadata and viewer command |
| M2 detection, semantic entities, geometric evidence | World → entity inspector; Live → candidate processing frame and Evidence details; Inspect → module controls |
| M3 global entities, episodes, last seen, changes, supporting evidence | History → memory.sqlite read-only episode/belief/event payloads; Debug → Retained tools → Memory episode & history / Memory replay; Rerun memory layers |
| M4 human pose, skeletons, pointing cones/rays | Enable human pose in Connection; Live → Use pointing enters shared target flow; World → 3D · Rerun for full recorded geometry; Debug → Multi-camera humans |
| M4 telemetry | Inspect → module health, enable/rate controls and raw module/source/stage data, including ages, dropped inputs, host/source rates and backend |
| M4.5 open-vocabulary queries, segmentation, tracking, detector toggles | Live search; Inspect → existing runtime module switches/rates; Debug → Common perception for all retained CLI options |
| M5 typed, pointing and direct TargetCandidate | Live search / Use pointing; World → select entity → Use as target |
| M5 confirm/reject, destination, standoff, clearance, heading, route | Live contextual inspector and World canvas; Confirm does not dispatch |
| M5 Observe, Dry run, Robot control, separate GO | Footer mode selector; Connection → explicit Robot control authority; Live → GO |
| Spot power, stand, speed, held keyboard/button drive, requested posture | Control drawer; speed 0.05–0.35 m/s, height 0–10 cm, roll/pitch ±5° |
| Gesture control and measured/simulated robot model | Control → Robot model → Model & gesture details |
| Fisheye, split, approximate panorama and camera source policy | Live camera selector and Sources; acquire/process/display/rate plus processing pair |
| Evaluations and original Qt interfaces | Debug → Evaluation commands and Retained tools; copy commands, no automatic benchmark execution |

## Design review imagery and limits

`web/fixtures/lab-review.png` is a generated mock photograph used only when
the offline demo selects the lab preview. **Sensor fixture** switches back to
the existing procedural RGB-D view. The mock photograph never reaches the
sensor adapter or perception pipeline, has no fabricated detection overlay,
and is not evidence of Spot imagery or map accuracy. Live connections show
the selected actual stream. Candidate thumbnails still show the actual
processing frame, so a demo target foregrounds World rather than annotating
the unrelated lab photograph.

The map derives from `mapping.config` and `mapping.topdown`, plus current
entity AABBs, robot pose and destination/route. No walls or room geometry are
added for appearance. The demo map is therefore still a simple room. Native
Rerun is the chosen advanced 3D architecture: it preserves richer real outputs
and playback without building a competing renderer. It opens in a separate
window; there is no embedded point-cloud canvas or browser playback timeline.
The browser does not project live skeletons onto its 2D occupancy canvas.

Live and World inspectors resize or collapse; drawers are docked rather
than freely rearrangeable. History exposes bounded artifact and memory payloads,
not a new graphical event-history editor. On small screens, detailed inspector
content scrolls internally. These are current UI limits, not future capability
claims. README screenshots remain unchanged pending the user's review.
At 320px the map shrinks to keep captions clear; annotations are omitted
when they cannot fit safely, retaining world markers and inspector coordinates.

## Safety and verification limits

Stop advances the command epoch before waiting for perception. It clears
held inputs, gesture mode, navigation ownership and confirmation. Focus
loss, hidden tabs, failed connections and shutdown request zero velocity.
The browser heartbeat expires after 0.30 s and manual commands after 0.35 s.
Supervised trajectory commands expire after 0.75 s. GO performs the existing fresh target, geometry,
route and single-use checks.
The class E-stop remains a separate process; GUI Stop is a zero-velocity
request, not an E-stop.

Offline verification: 119 unittest cases, 116 passed and 3 existing optional
external-dataset checks skipped. Added integration checks cover the shared target
paths, virtual/dry/Observe authority, target revisions, behind-robot heading,
Stop during slow planning and a full queue, presence expiry, disconnect lock
ordering, failed navigation Stop fallback, manual/GO exclusion and local
artifact containment. Browser checks exercised typed confirmation, virtual
GO, map selection, artifact inspection and all four pages with no JavaScript
page errors. The design review covers normal 1280×720 and full 1920×1080
windows plus narrower layouts; emulated platform checks are not installation
or physical hardware validation. See `CRITIQUE.md` for visual evidence.

Still unvalidated on Spot: actual image-source compatibility, depth
registration/extrinsics/timestamps, odom geometry, lease and E-stop behavior
in this browser integration, live model latency, target persistence,
destination/route quality, measured movement and arrival, manual controls
and stop/focus-loss/network-failure response. Use the ordered
[milestone 5 checklist](milestone-5.md) when the robot is available. No
successful SDK acknowledgement is presented as measured arrival.

## Maintenance

`web/style.css` owns tokens and layouts. Add pages within the same shell,
using direct labels, units, source/age and honest empty/disabled states.
`web/workspace.js` binds cached telemetry and actions; `web/app.js` retains
manual input/focus safety. `scope/workspace.py` coordinates worker-owned
adapters and guarded dispatch. `scope/console_backend.py` contains the
existing shared demo/Spot adapters; `scope/m5_console.py` remains the Qt
presentation. Perception, target geometry and navigation implementations
stay in their original modules. Preserve strict loopback/token/origin/CSP
checks and keep SDK assets, model bundles, credentials and captures ignored.
