# Unified local console

This pass joins the existing interfaces on `codex/m5-live-spot-interaction`.
It adds no model, fusion system or planner. M5 remains **READY FOR PHYSICAL
VALIDATION**. No physical Spot was connected or commanded during this pass.

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

## Operate

Type `chair`, select a candidate, inspect the box and world evidence, then
**CONFIRM TARGET**. Inspect destination x/y in meters, heading in degrees,
standoff, clearance and sampled route before **GO**. Confirmation alone
never dispatches. Expand Developer telemetry for the unrounded SE2 destination
(x/y in meters, yaw in radians and world frame). GO consumes its preview. Stop or a settings change clears
approval; preview again only after inspecting current evidence.

**Use pointing** produces the same TargetCandidate. It requires human pose
to be enabled when connecting and sufficient existing geometric evidence;
an unresolved ranking abstains. In demo it is explicitly **Simulate
pointing**, using a fixture rather than a detected person. **Maps → select
entity → Use as target** enters the same confirmation path.

Targets and destinations live in the world/odom frame. Camera source is
provenance, not an allowed travel direction. The facing arrow and numeric
heading show a turn toward targets behind the robot. Camera switching or
automatic reacquisition during navigation has not been added.

Manual controls remain under Operate: Power On, Stand, held W/A/S/D or arrow
buttons, 0.05–0.35 m/s speed, requested height 0–10 cm and roll/pitch ±5°,
Apply, gesture mode, five fisheye views, split and approximate panorama.
Manual motion/posture/gesture requests are blocked during active GO. The
SDK model is read only: measured joints when available, clearly labeled
simulated reference in demo. Pose requests do not establish measured pose.

Observe authenticates only images/state, with no command lease. Dry run
records the proposed SE2 destination and sends no movement command. Robot
control requires an explicitly authorized connection and the separate
class E-stop. It reuses the manual SpotSession's SDK command client and
lease for the existing supervised executor. Changing modes stops motion;
an already acquired command lease remains until Disconnect. Reconnect in
Observe for a connection that has no command authority.

## Cameras and modules

Camera Settings shows real discovered sources, independent acquisition,
display and request rate (above 0 through 30 Hz), plus exactly one selected
visual/depth processing pair. Additional display feeds do not imply
multi-camera Spot mapping. Alignment verification is a physical operator
prerequisite, not a calibration function. Unverified/missing depth permits
image/query inspection but blocks map-based destinations.

Acquire and display the desired feeds for split view. The retained manual
session's panorama needs fresh front-left and front-right frames; stitching
and measured model telemetry currently use that command-connected session.
Read-only Observe connections still provide individual and split sources.
Human pose/model initialization is selected when connecting. Expand
Developer modules for enable/rate controls and detailed stage timing.

## Maps, Runs and Evaluate

Maps provides current occupancy/free/unknown, entities, robot pose, selected
target and route, layer toggles, world units and an entity inspector. Its
isometric projection is a view of the same occupancy data, not a reconstructed
point cloud. **Advanced 3D / Rerun** explains the existing viewer workflow;
**Runs → choose .rrd → Open in Rerun** launches recorded 3D inspection.
Point clouds, meshes, skeletons, pointing rays, camera frustums, memory layers,
entity history and playback timelines retain their existing CLI/Rerun paths.

Runs discovers the latest 500 supported local artifacts in configured roots.
JSON inspection is limited to 2 MB. Memory SQLite databases are read only,
with up to 100 recent episode, belief and event payloads each. Saved maps
show metadata/counts and their viewer command; meshes/RGB-D episodes retain
CLI replay. Symbolic links and paths outside configured roots cannot open
private files. No cloud storage or automatic capture upload is introduced.
Use a narrower `--runs-dir` to browse older sessions beyond the index limit.

Evaluate offers existing perception, mapping, pointing/human, memory,
entity and test commands, results through Runs, and current module health.
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

## Safety and verification limits

Stop advances the command epoch before waiting for perception. It clears
held inputs, gesture mode, navigation ownership and confirmation. Focus
loss, hidden tabs, failed connections and shutdown request zero velocity.
The browser heartbeat expires after 0.30 s, manual commands after 0.35 s,
and existing supervised trajectory commands after 0.75 s. GO performs the
existing fresh target, robot, geometry, route and single-use checks. A
queued or in-progress manual request blocks final navigation dispatch.
The class E-stop remains a separate process; GUI Stop is a zero-velocity
request, not an E-stop.

Offline verification: 118 unittest cases, 115 passed and 3 existing optional
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
