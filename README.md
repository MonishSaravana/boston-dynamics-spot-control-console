# Boston Dynamics Spot Control Console (SCOPE)

**SCOPE** stands for **Spot Control, Observation, and Preview Environment**. The existing desktop app displays Spot's built-in fisheye cameras and requests power, stand, movement, and standing-posture commands through the Spot SDK. An independent offline mapping command now reconstructs a 3D scene from RGB-D frames with supplied camera poses. It does not connect to Spot.

The front panorama is an approximate stitch of two cameras. The robot mesh displays fresh measured joint positions when available; the posture controls show requested offsets. SCOPE is an independent project, not an official or endorsed Boston Dynamics product.

## Contents

- [Install](#install)
- [Run with the class E-stop](#run-with-the-class-e-stop)
- [Controls and views](#controls-and-views)
- [Offline demo and screenshots](#offline-demo-and-screenshots)
- [Offline 3D mapping](#offline-3d-mapping)
- [Future plans](#future-plans)
- [Contact](#contact)
- [Licensing and compatibility](#licensing-and-compatibility)
- [Development history](docs/development-history.md)

## Install

Use the existing `.venv` and `spot-sdk` checkout in this workspace. The installed environment was checked with Python 3.14.2, Spot SDK 5.2.0, PySide6 6.11.2, Pillow 12.3.0, OpenCV 5.0.0.93, and MediaPipe 0.10.35. The app does not import SciPy from the SDK image-viewer example.

### macOS Apple Silicon

In Terminal, from this workspace:

```sh
cd /Users/monishsaravana/work/boston-dynamics-spot
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

## Run with the class E-stop

1. Connect the computer to Spot's network and **confirm the robot IP with your instructor**. The commands use the class handout example `192.168.80.3`; replace it in both commands if needed.
2. Start the SDK's class E-stop in one Terminal and keep it open.
3. Start the GUI in a second Terminal. Enter credentials only when the SDK prompts in Terminal.

### macOS

**Terminal 1 — class E-stop**

```sh
cd /Users/monishsaravana/work/boston-dynamics-spot
source .venv/bin/activate
python spot-sdk/python/examples/estop/estop_nogui.py 192.168.80.3
```

**Terminal 2 — GUI**

```sh
cd /Users/monishsaravana/work/boston-dynamics-spot
source .venv/bin/activate
python spot_control_gui.py --hostname 192.168.80.3
```

In the E-stop Terminal, **Space** triggers the E-stop, **r** releases it, and **q** quits. Follow your instructor's operating procedure. The GUI's **STOP MOVEMENT** button requests zero velocity; it does not operate the class E-stop.

## Controls and views

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

## Offline demo and screenshots

Run the full console without a robot:

```sh
python spot_control_gui.py --demo
```

Demo mode creates no robot client, authentication, lease, network request, or robot command.

- Five static grayscale scenes have **SIMULATED** labels. The demo panorama has known overlap for UI review; it does not validate live stitching.
- **Power On**, **Stand**, high-level **Apply**, and gesture control stay disabled.
- Use `--offline-preview` for a camera-free view of the posture controls and model empty state.

These screenshots show the actual app window in offline demo mode. The original grayscale camera imagery was generated for this documentation and was **not captured from Spot**.

![Offline demo dashboard with simulated camera and URDF reference mesh](docs/illustrative-dashboard-ui.png)

![Offline demo split screen with illustrative lab scenes](docs/illustrative-split-screen-ui.png)

## Offline 3D mapping

The `scope` command maps a supplied-pose RGB-D sequence into a colored point cloud, observed free/occupied/unknown voxels, and a TSDF surface. It opens a Rerun recording with RGB, depth in meters, camera trajectory and frustum, a growing 3D map, and a top-down occupancy view. Use the **frame** timeline to scrub or press Play to watch observations accumulate; drag in the 3D view to orbit. Orange means occupied, teal means observed free, and dark gray means unknown. A depth hole leaves space unknown rather than marking it free. Map voxel size defaults to **0.10 m**; depth integration accepts **0.15–5.0 m**.

This is a known-pose mapper, not a localization system. A wrong but geometrically valid camera pose can distort the map; the synthetic accuracy report exposes that error, but the overlap warning does not detect every bad pose. TSDF fusion uses a room-scale dense grid capped at two million voxels. Input RGB and depth are synchronized by the source adapter; frame validation rejects delayed depth, changing intrinsics, invalid transforms, missing valid depth, and nonincreasing timestamps. The existing Spot GUI is separate and continues to run as described above.

From this repository, install the mapping command in a Python 3.11+ environment. In this workspace, the existing `.venv` was used:

```sh
cd /Users/monishsaravana/work/boston-dynamics-spot
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
QT_QPA_PLATFORM=offscreen python -m unittest -v test_scope_mapping test_spot_gesture test_spot_gui_offline test_spot_model_view
```


## Future plans

These are ideas for future work, not features in the current app.

| Area | Direction |
| --- | --- |
| Voice input | Connect a microphone for spoken commands. |
| Wearable interfaces | Explore control or viewing through VR headsets or Meta Ray-Ban smart glasses. |
| Emotion-aware interaction | Investigate visual cues for human emotion recognition. |
| Recognition and personalities | Explore opt-in facial recognition and distinct interaction personalities for known people. |

## Contact

For ideas or questions, email [monishsaravana@college.harvard.edu](mailto:monishsaravana@college.harvard.edu).

## Licensing and compatibility

- The Spot SDK checkout and virtual environment are excluded. Boston Dynamics' [SDK license](https://github.com/boston-dynamics/spot-sdk/blob/master/LICENSE) requires its full license and retained notices if SDK files are redistributed, and restricts trademark use that implies endorsement.
- The local MediaPipe model bundles are excluded until their redistribution terms are confirmed.
- SCOPE's original files use the [MIT License](LICENSE); it does not cover the SDK or third-party model bundles.

**Compatibility:** macOS Apple Silicon offline UI checked with Python 3.14.2; Windows and Linux not tested. Live robot operation has not been validated for this publication.
