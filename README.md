# Gestures

Gestures is a local/offline Windows webcam controller built on the existing
MediaPipe Hand Landmarker and Face Landmarker pipeline. It keeps the original
nose-touch shortcuts, pinch shortcuts, Air Mouse, scrolling, camera selection,
calibration, and local JSON settings, and adds an opt-in two-hand analog
controller for Blender.

The Blender feature does **not** replace hand tracking or simulate key presses.
The application calculates continuous navigation values and sends them over a
localhost-only UDP connection to a Blender add-on. Blender applies those values
to its actual `RegionView3D` or camera transform.

## Features

- MediaPipe detects up to two hands and 21 landmarks per hand.
- Face tracking and the original nose-touch gesture remain available.
- Thumb/index pinch remains a one-shot shortcut or Air Mouse click.
- Air Mouse and two-finger scrolling remain available.
- Two-hand navigation has explicit activation, confidence gating, smoothing,
  dead zones, velocity limits, inversion, calibration, and loss-of-hand stop
  behavior.
- Blender integration is local-only; no camera frames, landmarks, or commands
  are sent over the internet.
- The GUI shows both-hand geometry, distance, angle, confidence, vectors,
  gesture state, and Blender connection state.

## Quick start

Use Python 3.10, 3.11, or 3.12 on Windows.

```powershell
gh repo clone lllons/Gestures
cd Gestures
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts\download_models.py
python -m app.main
```

The GUI opens without accessing the webcam. Press **Start Detection** to open
the selected camera and load the local MediaPipe models.

If the repository is already present, only the last four commands are needed.
The model helper downloads the official model bundles once; runtime inference
reads the `.task` files locally and does not require a network connection.

## Blender add-on installation

The add-on supports Blender 3.0 and newer Blender releases that expose the
`RegionView3D` properties used below.

### Install through Blender Preferences

1. In Blender, choose **Edit → Preferences → Add-ons → Install…**.
2. Create a zip whose top level contains the `gestures_navigation` folder. For
   example, the zip should contain:

   ```text
   gestures_navigation/__init__.py
   ```

   The source folder to zip is `blender_addon/gestures_navigation`.
3. Select the zip, install it, and enable **Gestures 3D Navigation**.
4. In a 3D Viewport, press **N**, open the **Gestures** tab, and confirm the
   receiver says `WAITING`.
5. Leave the UDP port at `8765` unless the application settings use another
   localhost port. If the port was changed, press **Restart Local Receiver**.
6. Press **Enable Navigation** in the Blender panel. This is a second safety
   switch; Blender will not move while it is disabled.

Alternatively, copy `blender_addon/gestures_navigation` into Blender's user
`scripts/addons` directory as `gestures_navigation`, then enable it from
Preferences. Blender's bundled Python is used; no separate Blender package is
required.

The add-on listens on `127.0.0.1` only and runs its socket reader in a daemon
thread. Blender API mutations happen on Blender's main thread using a timer.

## Starting the complete system

1. Install and enable the Blender add-on as described above.
2. Open a `.blend` file with a visible 3D Viewport. For Camera mode, select a
   camera or make one the active scene camera.
3. Start Gestures with `python -m app.main`.
4. Press **Start Detection**.
5. In the **3D NAVIGATION** section, press **Enable 3D Navigation**.
6. Confirm the application shows `CONNECTED` after the Blender add-on receives
   a packet. Blender's panel should also show a connected receiver.
7. Put both hands in view. Hold the default open-hand activation pose for about
   0.7 seconds. The application then shows `NAVIGATION: ACTIVE`.
8. Optionally press **Calibrate** first, place both hands in a comfortable
   neutral position, and hold them while the calibration counter completes.
9. Move both hands slowly. Press the navigation button again or remove both
   hands to stop movement.

The application sends zero-valued movement packets whenever navigation is
inactive, confidence is too low, a hand is lost, the camera fails, or Gestures
stops. The add-on also stops applying movement after a packet timeout.

## Complete gesture and control map

### Activation and safety

- **Default activation:** both hands open and visible for approximately 700 ms.
- **Default deactivation:** remove both hands from the frame. Movement stops on
  the first missing-hand packet.
- Optional activation pose: **Two closed hands**.
- Optional held deactivation pose: **Two closed hands** or **Two open hands**.
- The **Enable 3D Navigation** button and the Blender add-on's **Enable
  Navigation** control must both be enabled.
- One visible hand never produces two-hand navigation.
- The confidence threshold, activation hold time, smoothing, dead zone, and
  maximum speed are configurable in the GUI.

### Full 3D mode (default)

| Hand movement | Continuous value | Blender result |
| --- | --- | --- |
| Both hands translate left/right | `midpoint_delta.x` | Horizontal orbit and pan |
| Both hands translate up/down | `midpoint_delta.y` | Vertical orbit and pan |
| Hands move farther apart | positive `distance_delta` | Zoom in |
| Hands move closer together | negative `distance_delta` | Zoom out |
| Line between hands rotates | `angle_delta` | Optional roll |

In `FULL 3D`, midpoint translation intentionally exposes both orbit and pan
channels so orbit, pan, and zoom can be combined in one continuous packet. The
Blender add-on applies each non-zero channel during the same main-thread timer
tick. Use `ORBIT`, `PAN`, or `ZOOM` when a single-axis mode is preferred.

`distance` is the normalized 2D distance between the palm centers. Zoom does
not require pinching: opening the hands farther apart produces a positive zoom
value, while closing the distance produces a negative value. `angle` is
calculated with:

```text
angle = atan2(right_y - left_y, right_x - left_x)
angle_delta = shortest_wrapped(current_angle - previous_angle)
```

The application packet also contains left/right centers, midpoint, individual
hand velocities, confidence, `orbit_x`, `orbit_y`, `pan_x`, `pan_y`, `zoom`,
and `roll`.

### Target modes

- **Viewport:** directly updates every visible Blender `RegionView3D`.
- **Camera:** applies rotation, lateral/vertical translation, and forward/back
  movement to the selected camera, or the active scene camera when no camera is
  selected. This changes the camera's actual transform.

The application and Blender panel each provide sensitivity multipliers.
Start with the defaults and increase them gradually. The app's **Invert X**,
**Invert Y**, and **Invert zoom direction** options can correct camera or
webcam orientation preferences.

## Calibration and smoothing

Press the navigation **Calibrate** button after enabling the feature. The next
stable two-hand samples are recorded for about 1.2 seconds. The median midpoint,
distance, and line angle become the neutral reference. Calibration resets the
movement baseline, so it does not cause a jump when navigation is activated.
The system then uses frame-to-frame deltas, not an ever-growing absolute offset,
so drift does not accumulate and the user does not need to hold the exact
neutral position forever.

The navigation layer performs the following on every valid frame:

1. Stable left/right pairing using MediaPipe handedness and previous positions.
2. Palm-center smoothing with a configurable moving-average window.
3. Distance, midpoint, angle, movement delta, and velocity calculation.
4. Dead-zone filtering and normalized per-frame velocity limits.
5. Sensitivity and inversion mapping.
6. A final output clamp before the packet is sent.

## Settings and privacy

Settings are stored locally as JSON:

```text
%LOCALAPPDATA%\Gestures\settings.json
```

On non-Windows development machines the fallback is:

```text
~/.gestures/settings.json
```

The file stores the camera index, shortcuts, existing gesture settings,
navigation target/control mode, activation/deactivation pose, sensitivities,
smoothing, confidence/dead-zone values, inversion choices, roll option, and
localhost ports. It does not store camera frames. Blender communication is
restricted to `127.0.0.1` / `localhost`.

## Testing

Run the deterministic unit tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The tests cover the existing gesture behavior plus two-hand detection, hand
pairing inputs, distance, midpoint, angle, distance delta, movement delta,
angle wrapping, dead zones, activation, calibration, roll, zoom direction,
confidence/loss handling, and UDP delivery/acknowledgement. To check Python
syntax, including the Blender add-on without importing `bpy`, run:

```powershell
python -m py_compile app\navigation.py app\blender_transport.py app\settings.py app\worker.py app\gui.py blender_addon\gestures_navigation\__init__.py
```

## Troubleshooting

- **No webcam found:** close Teams, Zoom, OBS, or another camera client;
  verify Windows camera privacy permissions; then press the camera refresh
  button.
- **MediaPipe model unavailable:** run `python scripts\download_models.py` and
  verify both `hand_landmarker.task` and `face_landmarker.task` are under
  `models`.
- **Blender says WAITING or the app says DISCONNECTED:** enable the add-on,
  confirm both sides use UDP port `8765`, confirm the reply port `8766` is free,
  and press **Restart Local Receiver** after changing Blender's port.
- **Blender is connected but does not move:** enable navigation in the Blender
  Gestures sidebar as well as in the app, show two hands, hold the activation
  pose, and check that the app reports `ACTIVE`. Viewport mode needs a visible
  3D Viewport; Camera mode needs a selected or active camera.
- **Activation does not complete:** use good lighting, show all four fingers
  clearly for the open-hand pose, keep both hands in frame, and lower the
  confidence threshold slightly only if the landmarks are stable.
- **Jitter or accidental movement:** increase smoothing and dead zone, lower
  maximum speed, use the calibration step, and keep the hands farther from the
  edge of the camera frame. A missing or low-confidence hand always stops
  movement.
- **Controls feel inverted:** use Invert X, Invert Y, or Invert zoom direction
  in the application settings. Blender sensitivity values are independent
  multipliers.
- **Slow FPS:** use a 640×480 camera mode, hide the preview while controlling
  Blender, close other camera consumers, and let MediaPipe run on the CPU.
  Camera inference, GUI updates, and UDP communication are kept off the
  Blender main thread where possible.
- **Original shortcuts or Air Mouse changed behavior:** leave 3D Navigation
  disabled. It is an independent mode and does not replace nose-touch, pinch,
  face tracking, or Air Mouse code paths.

## Performance and safety notes

The camera worker owns OpenCV and MediaPipe. The Tkinter thread consumes a
bounded newest-frame queue, and the Blender add-on consumes a bounded newest-
packet queue. UDP is local and non-blocking. No command is sent to Blender
unless two hands meet the activation and confidence requirements. No keyboard
shortcuts are used for viewport movement.

## Project layout

```text
app/
    main.py                 # python -m app.main entry point
    gui.py                  # Tkinter window and controls
    worker.py               # background camera/inference loop
    camera.py               # OpenCV camera enumeration/reconnect
    hand_tracker.py         # MediaPipe Hand Landmarker adapter
    face_tracker.py         # MediaPipe Face Landmarker adapter
    gesture_detector.py     # nose-touch, pinch, and Air Mouse gesture state
    navigation.py           # two-hand geometry/state/analog controls
    blender_transport.py    # localhost UDP sender/status listener
    keyboard_controller.py  # pynput shortcut parser/emitter
    mouse_controller.py     # Air Mouse pointer/click/scroll output
    calibration.py          # existing local face calibration
    settings.py             # validated local JSON settings
    model_paths.py           # source/PyInstaller model lookup
    startup.py              # optional HKCU Windows startup entry
blender_addon/
    gestures_navigation/
        __init__.py         # Blender receiver and viewport/camera controller
models/
    README.md
scripts/
    download_models.py
tests/
    test_gesture_detector.py
    test_mouse_controller.py
    test_navigation.py
    test_blender_transport.py

requirements.txt
```

## Existing standalone packaging

The existing PyInstaller workflow remains available:

```powershell
pyinstaller --noconfirm --clean --windowed --name Gestures `
  --collect-all mediapipe `
  --collect-all cv2 `
  --collect-all pynput `
  --add-data "models;models" `
  app\main.py
```

The executable is created under `dist\Gestures\Gestures.exe`. The Blender
add-on is installed separately in Blender and is not bundled into the Windows
executable.

## Limitations

- The add-on requires a desktop Blender build with a visible 3D Viewport for
  Viewport mode. Headless Blender has no `RegionView3D` to control.
- Camera mode moves the selected/active camera transform; it does not infer a
  scene-specific orbit pivot or alter camera lens settings.
- MediaPipe confidence and open-hand classification depend on lighting,
  occlusion, camera placement, and the chosen hand model.
- UDP packets are intentionally local and best-effort. The newest packet wins;
  the add-on stops on a short timeout rather than replaying stale movement.
- Existing `pynput` keyboard and pointer controls may require appropriate
  Windows permissions when the focused application is elevated.

License: MIT
