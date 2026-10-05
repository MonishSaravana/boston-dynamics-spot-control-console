# Milestone 5: live Spot interaction

Status: **READY FOR PHYSICAL VALIDATION**. The implementation and tests in this branch prepare the target-selection and dry-run path. No Spot was connected for this milestone in this workspace. Camera compatibility, depth registration, live latency, destination quality, and supervised movement have not been measured on the physical robot. Do not call M5 complete until those checks pass.

The new interaction console is separate from the existing manual-control and browser consoles. The simulator uses SCOPE's synthetic room geometry and a virtual robot. In Spot mode, the default connection reads images and robot state without a lease. Its GO button records a `WOULD_EXECUTE_NO_MOTION` proposal in memory and the process log. A separately opt-in supervised mode is prepared for the final Monday gate; it was exercised only with a fake command client.

## What is implemented

- Typed query, pointing ranking, and direct entity selection produce the same `TargetCandidate` and use the same **CONFIRM TARGET → destination preview → GO** sequence. Rejection cycles to another candidate or reports no match. An unresolved pointing ranking abstains.
- A candidate keeps its source, entity ID, world bounds, camera/box evidence, source timestamp, and host receipt time. Candidate scores are evidence, not calibrated probabilities. 2D-only detections can be highlighted but cannot produce a destination.
- The planner proposes a pose outside the target bounds, facing the target. It samples the full robot footprint plus 0.15 m clearance along a straight preview route. Any unknown or occupied map cell blocks the route. The route is a conservative proposal, not a promise that Spot's mobility system will accept it.
- Confirm and GO are separate actions. GO rechecks target freshness, stable geometry, fresh standing robot state, map clearance, and destination consistency. The authorization is single-use. A target that moves, disappears, becomes stale, or loses depth cannot dispatch.
- The Spot read-only source lists actual image sources and offers independent acquire/process/display toggles and acquisition rate limits. It records received frames, decode and RPC times, counts, observed request-limited rate, calibration, source identity, source/host times, and odom transforms. It converts grayscale/RGB and registered U16 depth into the same `RgbdFrame` used by replay; supported Kannala–Brandt fisheye pairs are rectified to a pinhole view. It refuses RGB-D geometry until an operator explicitly verifies alignment on the real robot. One failed source does not stop other streams.
- The console overlays candidate and tracked boxes, visible skeleton segments, and pointing direction when the corresponding modules return evidence. The world view shows occupied/free/unknown map cells, entities, robot pose, selected target, destination, and straight route. Detailed stage timing stays in the expandable developer panel.
- A supervised Spot executor uses the SDK's odom-frame SE2 trajectory command with a 0.20 m/s linear and 0.30 rad/s angular limit, short 0.75 s command expiry renewed while robot state and route remain valid, SDK feedback, and a zero-velocity request on Stop, focus loss, hide, failure, or exit. It is **not** active in default Spot mode. The opt-in mode acquires a lease and requires an already standing robot and the separate class E-stop procedure. No physical execution has been attempted.

The current limitations matter: the console processes one selected visual/depth pair while it can acquire other listed streams; processing still calls the mapping core on the UI thread. Live image/depth registration and odom consistency are unverified. Spot's own obstacle handling does not make an unknown SCOPE map cell safe. The supervised mode needs a physical stop/latency check before any ordinary-room demonstration. The existing manual-control Stop, zero-on-release/focus/failure/shutdown, short keyboard command expiry, and separate E-stop workflow were not changed.

## Run the synthetic interaction

From the repository with the existing environment:

```sh
.venv/bin/python -m scope.m5_console --demo
```

Type `chair`, select `chair_a` or `chair_b`, press **CONFIRM TARGET**, inspect the destination, then press **GO · virtual**. **Simulate pointing** proposes a simulated `chair_a` ranking through the same confirmation path. Clicking an entity directly uses the same path. The camera image and map are generated from a deterministic room fixture; they are not Spot captures or a measured performance result.

## Monday physical validation checklist

Keep the class E-stop available under the class procedure. Use the robot address and source names actually reported on site. Keep diagnostic output under ignored `runs/` if saved. Do not save private camera images, credentials, environment files, or logs in Git.

1. **Sensor diagnostic only; no lease or motion.** Run `.venv/bin/python -m scope spot-sensors --hostname ROBOT_IP --samples 10`. Record available sources, formats, resolution, request-limited FPS, acquisition timestamps, image/depth offset, transforms, RPC time, and decode time. RPC round trip is not one-way network latency.
2. **Live cameras only; no motion.** Choose a reported visual source: `.venv/bin/python -m scope.m5_console --spot ROBOT_IP --visual-source VISUAL_SOURCE`. Verify the displayed view and each source's acquire/display toggle and rate. Default Spot mode has no lease.
3. **Perception overlays; no motion.** In the same dry-run console, use a simple typed query and optionally restart with `--human-pose` for skeleton/pointing overlays. Check overlay registration and failure states.
4. **Live mapping only if geometry supports it; no motion.** Inspect the actual visual/depth pair, pixel alignment, intrinsics, depth scale, timestamps, and odom transforms. Only after those checks restart with `--depth-source DEPTH_SOURCE --alignment-verified`. Reject misaligned or stale pairs.
5. **Typed target query; no motion.** Query a physical object and check its camera highlight, entity bounds, score, age, and world position against the actual room. Reject wrong or ambiguous proposals.
6. **Pointing; no motion.** With `--human-pose`, verify the skeleton, ray, ranking, and selected physical object. An unresolved ranking must abstain.
7. **Dry-run target → destination; no motion.** Confirm a target, inspect the proposed pose and the entire route against the room, and press **GO · DRY RUN**. Verify the visible/logged `WOULD_EXECUTE_NO_MOTION` record and that Spot does not move.
8. **Only after all previous gates:** establish the separate class E-stop, place Spot standing in clear open space using the approved class procedure, release any other command lease, and restart with the verified pair plus `--supervised-go`. A human ready at the class E-stop checks a conservative destination, then explicitly presses **GO · Spot (supervised)**. Verify measured robot state and SDK feedback, Stop, network-loss behavior, and actual arrival. Stop and record any mismatch. Do not skip a gate because offline tests passed.

Physical validation records should include actual camera/source configuration, failures, target correctness, destination clearance, command/ack timing, measured robot response, and Stop behavior. Keep private captures outside the repository. M5 remains **READY FOR PHYSICAL VALIDATION** until camera acquisition, perception, spatial consistency, target and destination correctness, acceptable latency, and supervised execution pass on Spot.
