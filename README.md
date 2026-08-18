<p align="center">
  <img src="p/gestures.pn.png" width="500">
</p>


## This is a project that takes your hand location and face location and tracks them to see when they overlap.

When the fingertip enters the forgiving nose touch zone, Gestures immediately
presses one configurable keyboard shortcut by default. It fires once, enters a
cooldown/release state, and will not fire again until the fingertip has moved away.

### QuickStart
##### In a folder of your choosing

```
gh repo clone lllons/Gestures
```
##### In the same folder
```
python -m venv venv
```
##### Then..
```
venv\Scripts\activate
```
##### Then...
```
python -m pip install -r requirements.txt
```
##### Then....
```
python scripts\download_models.py
```
##### Then.....
```
python -m app.main
```
Once you run this last command the GUI will launch and all you need to do it press **"START"**

The application does not open the webcam until **Start Detection** is pressed.

### Freebuff preview configuration

This repository is configured with these Freebuff preview commands:

```text
install: python3 -m pip install -r requirements.txt
preview: python3 -m app.main (port metadata: 8000)
build:   node -p 1
```

The preview command is the desktop application's local launcher, not an HTTP
server. Freebuff's managed hosting build image is Node-only and cannot build or
run a Windows Python/Tkinter GUI, so the build entry is only a hosting-side
compatibility check; use the Windows PyInstaller command below for the real
executable. The webcam and GUI must be started on the Windows laptop itself.

To stop capture and release the camera, press **Stop**. When stopped or when
**Enable detection** is unchecked, no keyboard shortcut is emitted.

## Using the application

1. Choose a camera. Press the refresh button if a USB camera was connected
   after the window opened.
2. Enter a shortcut, for example `Alt + Tab`, `Ctrl + C`,
   `Ctrl + Shift + S`, `Space`, or `Escape`.
3. Set the activation zone size, activation delay, and cooldown. The default
   touch zone is 10% of face width, adjustable up to 50% with the slider; the
   red circle on the preview resizes live to show the zone. The default
   activation delay is 0 ms, so the shortcut fires on the first qualifying
   frame. The default cooldown is 500 ms.
4. Press **Start Detection**.
5. Optionally press **Calibrate**. Keep your face still during the first phase,
   then touch the nose with the index fingertip and hold. The measured relative
   distance plus a safety margin is saved in the per-user settings file.

The camera preview is mirrored to feel natural. Hand landmarks, the face mesh,
the selected index fingertip, the nose point, the connector line, and the
activation zone are drawn on the preview; the zone is shown as a red circle
around the nose whose size matches the zone slider. Debug diagnostics also show:

- FPS
- hand count and face/hand detection state
- normalized index fingertip and nose coordinates
- fingertip-to-nose distance relative to face width
- state and cooldown/release status

## How detection works

- MediaPipe Hand Landmarker returns up to two hands and 21 landmarks per hand.
- MediaPipe Face Landmarker returns a face mesh. Landmark 1 is used as the nose
  tip, and cheek landmarks 234 and 454 provide the face-width reference.
- For every hand, only landmark 8 — the actual index-finger tip — is considered.
  If two hands are visible, the hand whose index tip is closest to the detected
  nose is selected.
- The normalized Euclidean distance is calculated as:

  ```text
  relative_distance = distance(index_tip, nose) / face_width
  ```

  This avoids a raw-pixel threshold changing when the user moves closer to the
  webcam.
- A moving average over five frames reduces jitter. The detector reports
  `READY`, `APPROACHING`, `TOUCH DETECTED`, and `COOLDOWN`.
- The activation zone size is 10% of face width by default, and the slider in
  the settings panel can increase it up to 50% for an even larger zone.
- The default activation delay is 0 ms, so the first qualifying frame triggers
  the shortcut. After it is sent, a cooldown and a separate release hysteresis
  require the fingertip to move outside the zone before a new trigger is armed.

This is an inference-only application. It does not train a custom model and
contains no cloud AI or language-model feature.

----

## Settings storage and privacy

Settings are stored locally as JSON:

```text
%LOCALAPPDATA%\Gestures\settings.json
```

On non-Windows development machines the fallback is `~/.gestures/settings.json`.
The file contains UI values such as the camera index, threshold, shortcut, and
calibration result. It does not contain camera frames. The optional **Start with
Windows** setting creates/removes a per-user `HKCU` Run entry and does not need
administrator rights.

Windows may still show its normal camera privacy indicator while capture is
active. To use the app offline after setup, disconnect the network; no runtime
network request is needed.

## Packaging a standalone `.exe`

The requirements include PyInstaller. From an activated Windows virtual
environment, run:

```powershell
pyinstaller --noconfirm --clean --windowed --name Gestures `
  --collect-all mediapipe `
  --collect-all cv2 `
  --collect-all pynput `
  --add-data "models;models" `
  app\main.py
```

The executable is created at `dist\Gestures\Gestures.exe` (or the equivalent
PyInstaller output for your installed version). Test it on the target machine
with the two `.task` files included in the `models` directory. If you prefer a
single-file build, add `--onefile`; a one-file build extracts the models to a
temporary local directory at launch and still does not send them over the
network.

The first packaged launch may take longer while MediaPipe's native libraries
load. Windows Defender or camera privacy settings can also require an explicit
permission the first time.

## Troubleshooting

- **No webcam found**: close Teams/Zoom/OBS and other camera clients, verify
  Windows Settings → Privacy & security → Camera, then refresh the camera list.
- **MediaPipe model unavailable**: run `python scripts\download_models.py` from
  the repository root and confirm both `.task` files are under `models\`.
- **Face or hand not detected**: improve lighting, keep the face in view, and
  make sure the index fingertip is visible. The state remains safe and no key is
  sent without both landmarks.
- **Shortcut failure**: use the documented `pynput` names and test a simple
  shortcut such as `Space`. Some elevated applications may reject synthetic
  input unless Gestures is run with an appropriate Windows integrity level.
- **Slow FPS**: leave the preview off, use a 640×480 webcam mode, close other
  camera consumers, and allow MediaPipe to run on the CPU. The application does
  not require a dedicated GPU.

----

## Project layout

```text
app/
    main.py                 # python -m app.main entry point
    gui.py                  # Tkinter window and controls
    worker.py               # background camera/inference loop
    camera.py               # OpenCV camera enumeration/reconnect
    hand_tracker.py         # MediaPipe Hand Landmarker adapter
    face_tracker.py         # MediaPipe Face Landmarker adapter
    gesture_detector.py     # smoothing and trigger state machine
    keyboard_controller.py  # pynput shortcut parser/emitter
    calibration.py          # local two-stage calibration
    settings.py             # validated local JSON settings
    model_paths.py          # source/PyInstaller model lookup
    startup.py              # optional HKCU Windows startup entry
models/
    README.md
scripts/
    download_models.py
requirements.txt
```

Licence MIT
