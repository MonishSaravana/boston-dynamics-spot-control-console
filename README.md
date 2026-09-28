# Boston Dynamics Spot Control Console (SCOPE)

**SCOPE** expands to **Spot Control, Observation, and Preview Environment**. This independent Python desktop app uses the Spot SDK to display Spot's built-in fisheye cameras and request power, stand, movement, and standing-posture commands. It has an optional depth-assisted gesture mode. The panorama is an uncalibrated front-camera stitch, and the posture diagram is a visual guide to requested offsets, not a prediction of Spot's physical pose. This is not an official or endorsed Boston Dynamics product.

- [Install](#install)
- [Run with the class E-stop](#run-with-the-class-e-stop)
- [Controls and views](#controls-and-views)
- [SDK and model files](#sdk-and-model-files)

## Development history

The early feature commits are explicitly reconstructed approximations, not recovered source snapshots. See [development history](docs/development-history.md) for the dates and scope.

## Install

Use **Python 3.14**, Git, and a local virtual environment. The commands below place the project and the Spot SDK examples in separate folders under your home directory. The project clone URL will work **after this repository is published**. `bosdyn-client` is installed from PyPI; the SDK checkout supplies `estop_nogui.py`.

### macOS Apple Silicon

Install [Homebrew](https://brew.sh/) if needed. In Terminal:

```sh
brew install git python@3.14
cd "$HOME"
git clone https://github.com/monishsaravana/boston-dynamics-spot-control-console.git
cd boston-dynamics-spot-control-console
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'bosdyn-client==5.2.0' 'PySide6==6.11.2' Pillow 'opencv-contrib-python==5.0.0.93' 'mediapipe==0.10.35'
git clone --branch v5.2.0 --depth 1 https://github.com/boston-dynamics/spot-sdk.git "$HOME/spot-sdk"
```

### Windows 64-bit (PowerShell)

Install [Python 3.14 for Windows](https://www.python.org/downloads/windows/) with the Python launcher, and [Git for Windows](https://git-scm.com/download/win). In PowerShell:

```powershell
Set-Location $HOME
git clone https://github.com/monishsaravana/boston-dynamics-spot-control-console.git
Set-Location .\boston-dynamics-spot-control-console
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install bosdyn-client==5.2.0 PySide6==6.11.2 Pillow opencv-contrib-python==5.0.0.93 mediapipe==0.10.35 windows-curses
git clone --branch v5.2.0 --depth 1 https://github.com/boston-dynamics/spot-sdk.git "$HOME\spot-sdk"
```

`windows-curses` supports the terminal interface in `estop_nogui.py`. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that window, then activate the environment again.

### Optional gesture models

Gesture mode needs two `.task` files in `models/`. They are **not included in the proposed repository** while their redistribution terms are reviewed. For local use, download the [Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer#models) and [Pose Landmarker Lite](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker#models) bundles directly from Google's model pages, retaining these filenames:

```text
models/gesture_recognizer.task
models/pose_landmarker_lite.task
```

The rest of the GUI can launch without these files; enabling gesture mode will fail if they are missing.

## Run with the class E-stop

Connect the computer to Spot's network and **confirm the robot IP with your instructor**. `192.168.80.3` is the class handout example; replace it in both launch commands if your instructor gives a different IP. Run the SDK's class E-stop in one Terminal and keep it open while using the GUI in another. The SDK authentication helper prompts for credentials when needed. Enter them in the Terminal prompt; do not put them in source files or documentation.

**macOS — first Terminal, E-stop:**

```sh
cd "$HOME/boston-dynamics-spot-control-console"
source .venv/bin/activate
python "$HOME/spot-sdk/python/examples/estop/estop_nogui.py" 192.168.80.3
```

**macOS — second Terminal, GUI:**

```sh
cd "$HOME/boston-dynamics-spot-control-console"
source .venv/bin/activate
python spot_control_gui.py --hostname 192.168.80.3
```

**Windows — first PowerShell window, E-stop:**

```powershell
Set-Location "$HOME\boston-dynamics-spot-control-console"
.\.venv\Scripts\Activate.ps1
python "$HOME\spot-sdk\python\examples\estop\estop_nogui.py" 192.168.80.3
```

**Windows — second PowerShell window, GUI:**

```powershell
Set-Location "$HOME\boston-dynamics-spot-control-console"
.\.venv\Scripts\Activate.ps1
python spot_control_gui.py --hostname 192.168.80.3
```

In the E-stop Terminal, **Space** triggers the E-stop, **r** releases it, and **q** quits. Follow the instructor's operating procedure. The GUI's **STOP MOVEMENT** button requests zero velocity; it does not operate the class E-stop.

## Controls and views

- **Camera tabs:** Each available built-in fisheye source gets a tab. **Split screen** shows the available feeds together. Each view indicates loading, receiving images, or stopped after a connection error. The code does not require Spot CAM or another payload camera.
- **Panorama:** **Continuously update** is on by default and requests a new front-left/front-right stitch about every two seconds; uncheck it to pause or use **Stitch current front frames once**. Stitching runs off the UI thread with bounded image sizes. OpenCV alignment can fail or distort a scene; this is not calibrated or a 360° view. Individual feeds continue after a stitch failure. Drag to pan and use the wheel to zoom. Stitching pauses during gesture mode.
- **Movement:** Use **Power On** if needed, then **Stand**. Hold **W/A/S/D** or the arrow keys to move forward, left, backward, or right. The **Speed limit** slider requests **0.05–0.35 m/s**, with **0.20 m/s** as the initial value. Diagonal input is normalized to the same speed cap. Releasing a movement key requests zero velocity; motion commands expire after **0.35 seconds** without updates.
- **Stop:** **STOP MOVEMENT** clears held keys, disables gesture mode, and requests zero velocity when powered. The app also requests zero velocity on focus loss, connection failure, and exit. Stop does not power Spot off.
- **Posture preview:** The box shows requested body offsets relative to nominal stand; it has no leg or foot model and is not to scale. Moving a slider changes only the preview. **Apply requested standing posture** sends a stand request only when fresh robot state reports standing and near-zero body velocity, with no movement or gesture input active. Requested height is **0 to +10 cm** in 1 cm steps; roll and pitch are each **−5° to +5°** in 1° steps. These are app request limits, not physical guarantees. A 1 cm request may not cause 1 cm of movement. Driving or Stop can return roll and pitch to nominal.
- **Supervised gesture mode:** After **Stand**, this optional mode uses the front-left visual feed and aligned depth source; it disables keyboard driving. Hold an open palm for at least **0.5 s** to arm, then hold one raised index finger or two raised fingers for at least **0.6 s**. One finger requests a forward approach at up to **0.10 m/s** while valid person and depth observations remain farther than **2.5 m**. Two fingers request a backup at up to **0.10 m/s** for at most **0.5 s**. Show an open palm again before another command. Missing or stale observations, unclear signals, focus loss, or Stop cancel motion. Image recognition and depth measurement can be wrong; keep the class E-stop available.

The installed SDK is **5.2.0**. Its [command builder documentation](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot_command.html) defines `body_height` as relative to nominal stand height, and its [programming tutorial](https://dev.bostondynamics.com/docs/python/understanding_spot_programming.html) demonstrates a +0.1 m request. The SDK Xbox controller example clamps height at ±0.30 m, but that is an example constant, not a verified safe physical range. No safe lowering bound was verified for this robot. Boston Dynamics' [front stitching example](https://dev.bostondynamics.com/python/examples/stitch_front_images/readme) uses the two front sources and calibration; this app's OpenCV stitch remains approximate.


## SDK and model files

The Spot SDK checkout and virtual environment are excluded from this repository. Boston Dynamics' [SDK license](https://github.com/boston-dynamics/spot-sdk/blob/master/LICENSE) requires its full license and retained notices when SDK files are redistributed, and restricts trademark use that implies endorsement. The local MediaPipe model bundles are also excluded until their redistribution terms are confirmed. SCOPE's original project files are available under the [MIT License](LICENSE); that license does not cover the Spot SDK or third-party model bundles.

**Compatibility:** macOS Apple Silicon offline UI checked with Python 3.14.2; live Spot operation tested by the project owner; Windows and Linux not tested.
