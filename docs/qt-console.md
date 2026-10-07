# Desktop (Qt) console

The original PySide6 console. The browser console covers the same manual controls and the target workflow; this one is still the place for supervised gesture mode and the front panorama stitch. Start the browser console from the [README](../README.md) unless you need those.

Contents: [Install](#install-the-desktop-console), [Qt target interaction console](#qt-target-interaction-console), [Run with the class E-stop](#run-with-the-class-e-stop), [Controls and views](#controls-and-views), [Offline demo](#offline-demo).

## Install the desktop console

Use the existing `.venv` and `spot-sdk` checkout in this workspace. The installed environment was checked with Python 3.14.2, Spot SDK 5.2.0, PySide6 6.11.2, Pillow 12.3.0, OpenCV 5.0.0.93, and MediaPipe 0.10.35. The app does not import SciPy from the SDK image-viewer example.

### macOS Apple Silicon

In Terminal, from this workspace:

```sh
cd /path/to/boston-dynamics-spot
source .venv/bin/activate
python -c 'import bosdyn.client, PySide6, PIL, cv2, mediapipe; print("Dependencies ready")'
```

If that import fails in a fresh environment, install only the missing package into `.venv`. The app uses `bosdyn-client==5.2.0`, `PySide6==6.11.2`, Pillow, `opencv-contrib-python==5.0.0.93`, NumPy, and optional `mediapipe==0.10.35` for gesture mode. The local SDK's `python/examples/wasd/requirements.txt` and `python/examples/get_image/requirements.txt` were inspected; SciPy is only needed by the latter example, not this app.

### Optional gesture models

Gesture mode needs two local MediaPipe bundles. Download them from Google's model pages and save them with these names under `models/`:

| Download | Local filename |
| --- | --- |
| [Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer#models) | `gesture_recognizer.task` |
| [Pose Landmarker Lite](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker#models) | `pose_landmarker_lite.task` |

The rest of the GUI runs without them; gesture mode cannot start if they are missing. The bundles are excluded from this repository while their redistribution terms are reviewed.

## Qt target interaction console

The browser console has the same workflow. This is the original Qt version. It runs the demo and the dry-run path:

```sh
.venv/bin/python -m scope.m5_console --demo
```

Type `chair`, select one of the two candidates, press **CONFIRM TARGET**, inspect the highlighted object and proposed route, then press **GO · virtual**. **Simulate pointing** and direct entity selection use the same confirmation path. Synthetic images and map geometry are test fixtures, not Spot captures.

To inspect physical image sources without a lease or movement, run:

```sh
.venv/bin/python -m scope spot-sensors --hostname ROBOT_IP --samples 10
```

Then start the default dry-run console using a source name from that report:

```sh
.venv/bin/python -m scope.m5_console --spot ROBOT_IP --visual-source VISUAL_SOURCE
```

Camera acquisition and query overlays can run without depth. Mapping and destination generation require a measured aligned depth pair, calibration, source timestamps, and odom transforms. The console refuses RGB-D geometry until the operator has checked alignment on the robot and explicitly passes `--depth-source DEPTH_SOURCE --alignment-verified`. Spot GO records `WOULD_EXECUTE_NO_MOTION` and sends no command. `--supervised-go` exists only on `codex/m5-live-spot-interaction` and exits with an error here. See [the M5 checklist](milestone-5.md) for the ordered validation procedure and remaining limits.

## Run with the class E-stop

1. Connect the computer to Spot's network and **confirm the robot IP with your instructor**. The commands use the class handout example `192.168.80.3`; replace it in both commands if needed.
2. Start the SDK's class E-stop in one Terminal and keep it open.
3. Start the GUI in a second Terminal. Enter credentials only when the SDK prompts in Terminal.

### macOS

**Terminal 1: class E-stop**

```sh
cd /path/to/boston-dynamics-spot
source .venv/bin/activate
python spot-sdk/python/examples/estop/estop_nogui.py 192.168.80.3
```

**Terminal 2: GUI**

```sh
cd /path/to/boston-dynamics-spot
source .venv/bin/activate
python spot_control_gui.py --hostname 192.168.80.3
```

In the E-stop Terminal, **Space** triggers the E-stop, **r** releases it, and **q** quits. Follow your instructor's operating procedure. The GUI's **STOP MOVEMENT** button requests zero velocity; it does not operate the class E-stop.

## Controls and views

These instructions describe the Qt manual console. The browser has the same manual requests in the **Robot** tab of the Live inspector.

### Cameras and panorama

The default dashboard shows Camera, Robot Model, and Controls together. Drag a panel title to move it, drag the divider to resize it, or use its title-bar buttons to float or hide it. The **Panels and layout** menu in the header restores hidden panels and offers **Balanced**, **Camera Focus**, **Model Focus**, and **Restore Default Layout**. Your arrangement is saved when the window closes. Stop and connection status stay visible above the panels. The Camera panel's **View** selector changes feeds, split screen, and panorama without replacing the robot model.

| View | What it shows |
| --- | --- |
| Camera View selector | One choice per available built-in fisheye feed. |
| Split screen | The available built-in feeds together. |
| Front panorama | An approximate stitch of the front-left and front-right feeds. Drag to pan and use the wheel to zoom. |

Camera views report when they are loading, receiving images, or stopped after a connection error. They do not require Spot CAM or another payload camera.

**Continuously update** requests a new stitch about every two seconds and is on by default. Turn it off to pause, or select **Stitch current front frames once**. Stitching runs off the UI thread and pauses during gesture mode.

OpenCV may fail to align the frames or distort the scene. The result is neither calibrated nor a 360° view; individual feeds continue after a stitch failure. See Boston Dynamics' [front-stitching example](https://dev.bostondynamics.com/python/examples/stitch_front_images/readme) for comparison.

### Drive and stop

| Control | Action |
| --- | --- |
| **Power On**, then **Stand** | Prepare Spot for movement if it is not already powered and standing. |
| Hold **W/A/S/D** or arrow keys | Move forward, left, backward, or right. Diagonal input stays within the same speed cap. |
| **Speed limit** | Requests 0.05–0.35 m/s; starts at 0.20 m/s. |
| **STOP MOVEMENT** | Clears held keys, disables gesture mode, and requests zero velocity when powered. It does not power Spot off. |

Releasing any movement key requests zero velocity before the remaining held direction resumes. Motion commands expire after **0.35 seconds** without updates. The app also requests zero velocity on Stop, focus loss, connection failure, and exit. Keyboard control unlocks only after fresh robot state confirms standing and near-zero odometry velocity.

### Read-only robot geometry and standing posture

The resizable dashboard keeps the selected camera view and robot mesh visible together. The mesh is loaded from the local SDK's `spot-sdk/files/spot_base_urdf.zip`; the [RobotStateService joint positions](https://dev.bostondynamics.com/docs/concepts/robot_services) update its leg geometry when fresh telemetry is available **and** the robot reports a matching base skeleton. If its skeleton differs or cannot be verified, geometry is hidden rather than shown as the wrong robot. Mouse orbit, pan, and zoom only change the view. The mesh has a body-frame reference grid; it does not represent measured floor contact, collision clearance, balance, or the robot's actual body orientation. If the SDK ZIP is missing, the model reports that in the UI while the other controls remain available. Demo mode shows the URDF zero configuration and labels it as simulated.

The standing posture sliders edit a request; they do not move the on-screen mesh or Spot until **Apply requested standing posture** sends the SDK's high-level stand command after a fresh stationary check. No individual joint control is exposed. Boston Dynamics describes [joint control](https://dev.bostondynamics.com/docs/concepts/joint_control/readme) as a controlled-access beta requiring a special-permissions license and high-rate streaming; that access cannot be verified offline.

| Request | App range | Step |
| --- | --- | --- |
| Height offset | 0 to +10 cm | 1 cm |
| Roll | −5° to +5° | 1° |
| Pitch | −5° to +5° | 1° |

**Apply requested standing posture** sends a stand request only when fresh robot state reports standing and near-zero body velocity, with no movement or gesture input active. The roll and pitch ±5° values are conservative **app request limits**, below the SDK Xbox example's example clamps; no robot-specific safe range has been verified. Height 0 to +10 cm is likewise an app limit based on the SDK tutorial's +0.1 m request. These are not physical guarantees: a 1 cm request may not move Spot by 1 cm. Driving or Stop can return roll and pitch to nominal.

The SDK defines [body height relative to nominal stand height](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot_command.html), and its [programming tutorial](https://dev.bostondynamics.com/docs/python/understanding_spot_programming.html) shows a +0.1 m request. The SDK Xbox example's ±0.30 m clamp is an example constant, not a verified safe range. No safe lowering bound was verified for this robot.

### Supervised gesture mode

Enable this mode after **Stand** is confirmed. It uses the front-left visual feed and aligned depth source, and disables keyboard driving. The other camera feeds keep updating; panorama stitching pauses while gesture recognition is active.

| Signal | Hold for | Request |
| --- | --- | --- |
| Open palm | At least 0.5 s | Arm or rearm gesture mode. |
| One raised index finger | At least 0.6 s | Approach at up to 0.10 m/s while valid person and depth readings remain farther than 2.5 m. |
| Two raised fingers | At least 0.6 s | Back up at up to 0.10 m/s for at most 0.5 s. |

Show an open palm again before another command. The detector requires one associated visible hand and person, a high-confidence gesture, fresh synchronized visual/depth frames, and a reliable segmented torso depth for approach. Missing or stale observations, unclear signals, focus loss, or Stop cancel motion. The 2.5 m software cutoff provides margin above 2 m, but camera depth, segmentation, latency, and unseen obstacles have **not** been validated on this robot; it is not a certified person-distance safeguard. Keep the class E-stop available and supervise closely.

## Offline demo

Run the full console without a robot:

```sh
python spot_control_gui.py --demo
```

Demo mode creates no robot client, authentication, lease, network request, or robot command.

- Five static grayscale scenes have **SIMULATED** labels. The demo panorama has known overlap for UI review; it does not validate live stitching.
- **Power On**, **Stand**, high-level **Apply**, and gesture control stay disabled.
- Use `--offline-preview` for a camera-free view of the posture controls and model empty state.
