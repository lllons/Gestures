<p align="center">
  <img src="p/gestures.pn.png" width="500">
</p>

# Gestures — shortcut controller

Gestures is a Windows desktop application that watches the laptop webcam for one
specific gesture:

> **INDEX FINGER TIP → NOSE**

When the fingertip stays inside the nose touch zone for the configured duration,
Gestures presses one configurable keyboard shortcut. It fires once, enters a
cooldown/release state, and will not fire again until the fingertip has moved away.

All camera capture, MediaPipe inference, landmark smoothing, calibration, and
keyboard input run on the local computer. Camera frames are not uploaded,
stored, logged, or sent to a cloud service.

## Requirements

- Windows 10 or Windows 11
- Python 3.10–3.12 (64-bit recommended; these versions have the broadest MediaPipe wheel support)
- A working webcam and permission for desktop applications to use it
- Internet access only during the one-time Python package/model installation
  (after that, the app is local/offline)

The UI uses Python's built-in Tkinter. Computer vision uses OpenCV and the
MediaPipe Tasks Hand Landmarker and Face Landmarker. Keyboard events use
`pynput`.

## Setup on Windows

Open PowerShell in the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\download_models.py
```

If PowerShell blocks activation, either run this once as the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

or use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models.py
```

The model helper downloads the official local task bundles to:

```text
models\hand_landmarker.task
models\face_landmarker.task
```

Those files are needed by MediaPipe at runtime. They are deliberately not
bundled in the repository because of their size. Do not download camera or
model data from inside the application; the app fails with a clear message if a
model file is missing.

## Run

```powershell
python -m app.main
```

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
3. Set the relative touch threshold, hold duration, and cooldown. The default
   hold duration is 150 ms and the default cooldown is 500 ms.
4. Press **Start Detection**.
5. Optionally press **Calibrate**. Keep your face still during the first phase,
   then touch the nose with the index fingertip and hold. The measured relative
   distance plus a safety margin is saved in the per-user settings file.

The camera preview is mirrored to feel natural. Hand landmarks, the face mesh,
the selected index fingertip, the nose point, the connector line, and the
relative touch zone are drawn on the preview. Debug diagnostics also show:

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
- A touch must remain inside the configured threshold for the required duration.
  After the shortcut is sent, a cooldown and a separate release hysteresis
  require the fingertip to move outside the zone before a new trigger is armed.

This is an inference-only application. It does not train a custom model and
contains no cloud AI or language-model feature.

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
