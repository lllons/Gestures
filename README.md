<p align="center">
  <img src="p/gestures.pn.png" width="500">
</p>


## This is a project that takes your hand location and face location and tracks them to see when they overlap.

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
