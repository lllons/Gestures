"""Background camera/inference worker.

Tkinter is single-threaded, so all camera access and MediaPipe inference stay in
this worker.  The UI receives only the newest result through a bounded queue.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import cv2

from .calibration import CalibrationSession, CalibrationUpdate
from .camera import CameraCapture
from .face_tracker import FaceDetection, FaceTracker
from .gesture_detector import (
    DetectionState,
    GestureDetector,
    GestureSnapshot,
    normalized_finger_separation,
    normalized_pinch_distance,
)
from .hand_tracker import HandDetection, HandTracker
from .input_controller import InputStatus, NavigationInputController
from .keyboard_controller import KeyboardController
from .model_paths import get_model_paths, model_install_message
from .mouse_controller import MouseController
from .navigation import NavigationSnapshot, TwoHandNavigation
from .settings import AppSettings, SettingsStore


@dataclass(frozen=True)
class WorkerEvent:
    kind: str
    message: str


@dataclass(frozen=True)
class FrameResult:
    preview_frame: Any | None
    snapshot: GestureSnapshot
    fps: float
    camera_index: int
    calibration: CalibrationUpdate | None = None
    navigation: NavigationSnapshot | None = None
    input_status: InputStatus | None = None


class CameraWorker:
    """Own the webcam and all inference objects on one daemon thread."""

    def __init__(
        self,
        settings_store: SettingsStore,
        settings: AppSettings,
        result_queue: queue.Queue[FrameResult],
        event_queue: queue.Queue[WorkerEvent],
    ) -> None:
        self._settings_store = settings_store
        self._settings = settings
        self._settings_lock = threading.Lock()
        self._commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._results = result_queue
        self._events = event_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if self.is_running():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="gestures-camera",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update_settings(self, settings: AppSettings) -> None:
        settings.validate()
        # Publish immediately so settings changed before a restart are used by
        # the next worker loop as well as by a currently running loop.
        self._replace_settings(settings)
        self._commands.put(("settings", settings))

    def begin_calibration(self) -> None:
        self._commands.put(("calibrate", None))

    def begin_navigation_calibration(self) -> None:
        self._commands.put(("navigation_calibrate", None))

    def _run(self) -> None:
        camera: CameraCapture | None = None
        hand_tracker: HandTracker | None = None
        face_tracker: FaceTracker | None = None
        keyboard: KeyboardController | None = None
        mouse_controller: MouseController | None = None
        input_controller: NavigationInputController | None = None
        navigation: TwoHandNavigation | None = None
        calibration: CalibrationSession | None = None
        fps_times: deque[float] = deque()
        last_camera_notice = 0.0
        active_camera_index: int | None = None
        frame_number = 0
        timestamp_base = time.monotonic()
        detector: GestureDetector | None = None

        try:
            model_paths = get_model_paths()
            if model_paths.missing():
                self._notify("error", model_install_message(model_paths))
                return

            try:
                hand_tracker = HandTracker(model_paths.hand)
                face_tracker = FaceTracker(model_paths.face)
            except Exception as exc:
                self._notify("error", f"MediaPipe could not start: {exc}")
                return

            try:
                keyboard = KeyboardController()
            except Exception as exc:
                self._notify(
                    "warning",
                    f"Keyboard control is unavailable; detection can preview only: {exc}",
                )

            try:
                mouse_controller = MouseController()
            except Exception as exc:
                self._notify(
                    "warning",
                    f"Air Mouse control is unavailable; gesture shortcuts still work: {exc}",
                )

            settings = self._read_settings()
            detector = GestureDetector(settings)
            navigation = TwoHandNavigation(settings)
            try:
                input_controller = NavigationInputController(settings)
                if not input_controller.start_hotkeys():
                    self._notify("warning", input_controller.last_hotkey_error)
            except Exception as exc:
                self._notify("warning", f"OS navigation input is unavailable: {exc}")

            while not self._stop_event.is_set():
                settings, calibration = self._apply_commands(
                    settings,
                    detector,
                    navigation,
                    input_controller,
                    calibration,
                    camera,
                )

                if active_camera_index != settings.camera_index:
                    if camera is not None:
                        camera.release()
                    camera = CameraCapture(settings.camera_index)
                    active_camera_index = settings.camera_index

                if camera is None or not camera.is_open:
                    if camera is None:
                        camera = CameraCapture(settings.camera_index)
                        active_camera_index = settings.camera_index
                    if not camera.open():
                        if navigation is not None:
                            navigation.reset()
                        if input_controller is not None:
                            input_controller.stop()
                        now = time.monotonic()
                        if now - last_camera_notice > 3.0:
                            self._notify(
                                "error",
                                f"Camera {settings.camera_index} is unavailable. "
                                "Check Windows camera permissions or reconnect it.",
                            )
                            last_camera_notice = now
                        self._stop_event.wait(0.5)
                        continue
                    self._notify("info", f"Camera {settings.camera_index} connected.")
                    last_camera_notice = 0.0

                ok, frame = camera.read()
                if not ok or frame is None:
                    camera.release()
                    detector.reset()
                    if navigation is not None:
                        navigation.reset()
                    if input_controller is not None:
                        input_controller.stop()
                    self._notify(
                        "error",
                        f"Camera {settings.camera_index} stopped responding; retrying...",
                    )
                    self._stop_event.wait(0.5)
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_number += 1
                elapsed_ms = int((time.monotonic() - timestamp_base) * 1000)
                timestamp_ms = max(frame_number, elapsed_ms)

                try:
                    hands = hand_tracker.detect(rgb_frame, timestamp_ms)
                    face = face_tracker.detect(rgb_frame, timestamp_ms)
                except Exception as exc:
                    self._notify("error", f"MediaPipe inference stopped: {exc}")
                    return

                is_calibrating = calibration is not None and calibration.active
                snapshot = detector.process(
                    hands,
                    face,
                    force_disabled=is_calibrating,
                )
                navigation_snapshot = (
                    navigation.process([] if is_calibrating else hands)
                    if navigation
                    else None
                )
                input_status = (
                    input_controller.apply(None if is_calibrating else navigation_snapshot)
                    if input_controller is not None
                    else None
                )
                air_mouse_active = settings.air_mouse_enabled and not settings.navigation_enabled

                if (
                    air_mouse_active
                    and settings.detection_enabled
                    and not is_calibrating
                    and mouse_controller is not None
                    and snapshot.index_tip is not None
                ):
                    try:
                        mouse_controller.move_to(snapshot.index_tip)
                    except Exception as exc:
                        self._notify("warning", f"Air Mouse movement stopped: {exc}")
                        mouse_controller = None

                if (
                    air_mouse_active
                    and settings.detection_enabled
                    and not is_calibrating
                    and mouse_controller is not None
                    and snapshot.scroll_active
                    and (snapshot.scroll_delta_x or snapshot.scroll_delta_y)
                ):
                    try:
                        mouse_controller.scroll(
                            snapshot.scroll_delta_x * GestureDetector.SCROLL_SENSITIVITY,
                            -snapshot.scroll_delta_y * GestureDetector.SCROLL_SENSITIVITY,
                        )
                    except Exception as exc:
                        self._notify("warning", f"Air Mouse scrolling stopped: {exc}")
                        mouse_controller = None

                if snapshot.triggered and keyboard is not None:
                    try:
                        keyboard.press_shortcut(settings.shortcut)
                        self._notify("info", f"Sent nose shortcut: {settings.shortcut}")
                    except Exception as exc:
                        self._notify("error", f"Could not send {settings.shortcut!r}: {exc}")

                if snapshot.pinch_triggered:
                    if air_mouse_active and mouse_controller is not None:
                        try:
                            mouse_controller.click()
                            self._notify("info", "Air Mouse click")
                        except Exception as exc:
                            self._notify("error", f"Could not click with Air Mouse: {exc}")
                    elif not air_mouse_active and keyboard is not None:
                        try:
                            keyboard.press_shortcut(settings.pinch_shortcut)
                            self._notify(
                                "info",
                                f"Sent pinch shortcut: {settings.pinch_shortcut}",
                            )
                        except Exception as exc:
                            self._notify(
                                "error",
                                f"Could not send {settings.pinch_shortcut!r}: {exc}",
                            )

                calibration_update = None
                if calibration is not None:
                    calibration_update = calibration.update(
                        face_width=face.face_width if face else None,
                        relative_distance=snapshot.relative_distance,
                        hand_detected=snapshot.hand_detected,
                        face_detected=snapshot.face_detected,
                    )
                    if calibration_update and calibration_update.completed:
                        if calibration_update.succeeded and calibration_update.threshold:
                            settings = replace(
                                settings,
                                touch_threshold=calibration_update.threshold,
                                calibrated_face_width=calibration_update.face_width,
                            )
                            settings.validate()
                            self._replace_settings(settings)
                            detector.update_settings(settings)
                            try:
                                self._settings_store.save(settings)
                            except OSError as exc:
                                self._notify(
                                    "warning",
                                    f"Calibration worked but settings could not be saved: {exc}",
                                )
                        calibration = None
                        detector.reset()

                now = time.monotonic()
                fps_times.append(now)
                while fps_times and now - fps_times[0] > 1.0:
                    fps_times.popleft()
                fps = float(len(fps_times))

                preview = None
                if settings.preview_visible:
                    preview = _draw_overlay(
                        frame,
                        hands,
                        face,
                        snapshot,
                        settings,
                        fps,
                        calibration_update,
                        navigation_snapshot,
                    )

                self._publish(
                    FrameResult(
                        preview_frame=preview,
                        snapshot=snapshot,
                        fps=fps,
                        camera_index=settings.camera_index,
                        calibration=calibration_update,
                        navigation=navigation_snapshot,
                        input_status=input_status,
                    )
                )
                self._stop_event.wait(0.001)
        finally:
            if input_controller is not None:
                try:
                    input_controller.close()
                except Exception:
                    pass
            if navigation is not None:
                navigation.reset()
            if camera is not None:
                camera.release()
            if hand_tracker is not None:
                try:
                    hand_tracker.close()
                except Exception:
                    pass
            if face_tracker is not None:
                try:
                    face_tracker.close()
                except Exception:
                    pass

    def _apply_commands(
        self,
        settings: AppSettings,
        detector: GestureDetector,
        navigation: TwoHandNavigation,
        input_controller: NavigationInputController | None,
        calibration: CalibrationSession | None,
        camera: CameraCapture | None,
    ) -> tuple[AppSettings, CalibrationSession | None]:
        while True:
            try:
                command, value = self._commands.get_nowait()
            except queue.Empty:
                break
            if command == "settings":
                settings = value
                detector.update_settings(settings)
                navigation.update_settings(settings)
                if input_controller is not None:
                    input_controller.update_settings(settings)
                self._replace_settings(settings)
                if camera is not None and camera.index != settings.camera_index:
                    camera.release()
            elif command == "navigation_calibrate":
                navigation.begin_calibration()
                self._notify(
                    "info",
                    "3D calibration started: hold both hands in a comfortable neutral position.",
                )
            elif command == "calibrate":
                calibration = CalibrationSession()
                calibration.begin()
                detector.reset()
                self._notify(
                    "info",
                    "Calibration started: keep your face centered and still.",
                )
        return settings, calibration

    def _read_settings(self) -> AppSettings:
        with self._settings_lock:
            return self._settings

    def _replace_settings(self, settings: AppSettings) -> None:
        with self._settings_lock:
            self._settings = settings

    def _notify(self, kind: str, message: str) -> None:
        try:
            self._events.put_nowait(WorkerEvent(kind=kind, message=message))
        except queue.Full:
            try:
                self._events.get_nowait()
            except queue.Empty:
                pass
            try:
                self._events.put_nowait(WorkerEvent(kind=kind, message=message))
            except queue.Full:
                pass

    def _publish(self, result: FrameResult) -> None:
        try:
            self._results.put_nowait(result)
        except queue.Full:
            try:
                self._results.get_nowait()
            except queue.Empty:
                pass
            try:
                self._results.put_nowait(result)
            except queue.Full:
                pass


def _draw_overlay(
    frame: Any,
    hands: list[HandDetection],
    face: FaceDetection | None,
    snapshot: GestureSnapshot,
    settings: AppSettings,
    fps: float,
    calibration: CalibrationUpdate | None,
    navigation: NavigationSnapshot | None = None,
) -> Any:
    """Draw hand/face landmarks and debug information onto the local preview."""

    height, width = frame.shape[:2]

    for face_point in face.landmarks if face else ():
        cv2.circle(
            frame,
            (_px(face_point.x, width), _px(face_point.y, height)),
            1,
            (90, 180, 90),
            -1,
        )

    hand_connections = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    )
    for hand in hands:
        for first, second in hand_connections:
            if len(hand.landmarks) > max(first, second):
                start = hand.landmarks[first]
                end = hand.landmarks[second]
                cv2.line(
                    frame,
                    (_px(start.x, width), _px(start.y, height)),
                    (_px(end.x, width), _px(end.y, height)),
                    (255, 170, 40),
                    2,
                )
        if len(hand.landmarks) > 8:
            thumb = hand.landmarks[4]
            index = hand.landmarks[8]
            pinch_distance = normalized_pinch_distance(hand)
            pinch_color = (
                (0, 255, 0)
                if pinch_distance is not None
                and pinch_distance <= GestureDetector.PINCH_THRESHOLD
                else (255, 170, 40)
            )
            cv2.line(
                frame,
                (_px(thumb.x, width), _px(thumb.y, height)),
                (_px(index.x, width), _px(index.y, height)),
                pinch_color,
                3,
            )
        if len(hand.landmarks) > 12:
            index = hand.landmarks[8]
            middle = hand.landmarks[12]
            finger_separation = normalized_finger_separation(hand)
            scroll_color = (
                (0, 255, 0)
                if finger_separation is not None
                and finger_separation
                >= GestureDetector.SCROLL_FINGER_SEPARATION_THRESHOLD
                else (255, 170, 40)
            )
            cv2.line(
                frame,
                (_px(index.x, width), _px(index.y, height)),
                (_px(middle.x, width), _px(middle.y, height)),
                scroll_color,
                3,
            )
        for point in hand.landmarks:
            cv2.circle(
                frame,
                (_px(point.x, width), _px(point.y, height)),
                3,
                (255, 210, 80),
                -1,
            )

    if navigation is not None:
        _draw_navigation_overlay(frame, navigation, width, height)

    if snapshot.nose is not None:
        nose_point = (_px(snapshot.nose.x, width), _px(snapshot.nose.y, height))
        if snapshot.face_scale:
            radius = max(4, int(snapshot.face_scale * width * settings.touch_threshold))
            cv2.circle(frame, nose_point, radius, (25, 35, 200), -1)
            cv2.circle(frame, nose_point, radius, (60, 80, 255), 2)
            cv2.putText(
                frame,
                "ZONE",
                (nose_point[0] - radius + 6, nose_point[1] - radius + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.circle(frame, nose_point, 7, (40, 80, 255), 2)
    if snapshot.index_tip is not None:
        fingertip = (_px(snapshot.index_tip.x, width), _px(snapshot.index_tip.y, height))
        cv2.circle(frame, fingertip, 8, (0, 220, 255), 2)
        if snapshot.nose is not None:
            cv2.line(frame, fingertip, nose_point, (0, 220, 255), 2)

    if snapshot.pinch_detected:
        cv2.putText(
            frame,
            "PINCH ACTIVE",
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    if snapshot.scroll_active:
        cv2.putText(
            frame,
            "SCROLL ACTIVE",
            (18, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    state_color = {
        DetectionState.READY: (100, 220, 120),
        DetectionState.APPROACHING: (0, 210, 255),
        DetectionState.TOUCH_DETECTED: (0, 255, 0),
        DetectionState.COOLDOWN: (80, 120, 255),
    }[snapshot.state]
    cv2.putText(
        frame,
        snapshot.state.value,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        state_color,
        2,
        cv2.LINE_AA,
    )

    if calibration and calibration.active:
        cv2.putText(
            frame,
            calibration.message,
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 220, 80),
            1,
            cv2.LINE_AA,
        )

    if settings.debug_mode:
        distance = (
            f"{snapshot.relative_distance:.3f}"
            if snapshot.relative_distance is not None
            else "--"
        )
        zone_text = "--"
        if snapshot.nose is not None and snapshot.face_scale:
            zone_radius = max(4, int(snapshot.face_scale * width * settings.touch_threshold))
            zone_text = f"{settings.touch_threshold * 100:.0f}% ({zone_radius}px)"
        scroll_text = (
            f"{snapshot.scroll_delta_x:.4f}, {snapshot.scroll_delta_y:.4f}"
            if snapshot.scroll_active
            else "--"
        )
        debug_lines = (
            f"FPS: {fps:.1f}",
            f"Hand: {'yes' if snapshot.hand_detected else 'no'} ({snapshot.hand_count})",
            f"Face: {'yes' if snapshot.face_detected else 'no'}",
            f"Distance: {distance}",
            f"Zone: {zone_text}",
            f"Pinch: {'yes' if snapshot.pinch_detected else 'no'}",
            f"Scroll: {scroll_text}",
        )
        for line_number, text in enumerate(debug_lines, start=1):
            cv2.putText(
                frame,
                text,
                (18, 90 + line_number * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
    return frame


def _draw_navigation_overlay(
    frame: Any,
    navigation: NavigationSnapshot,
    width: int,
    height: int,
) -> None:
    """Draw the two-hand navigation geometry and current analog vectors."""

    active_color = (80, 255, 150) if navigation.active else (180, 180, 80)
    if navigation.left_hand is not None and navigation.right_hand is not None:
        left = (_px(navigation.left_hand.x, width), _px(navigation.left_hand.y, height))
        right = (_px(navigation.right_hand.x, width), _px(navigation.right_hand.y, height))
        cv2.line(frame, left, right, active_color, 3)
        cv2.circle(frame, left, 9, (255, 120, 60), 2)
        cv2.circle(frame, right, 9, (60, 180, 255), 2)
    if navigation.midpoint is not None:
        midpoint = (
            _px(navigation.midpoint.x, width),
            _px(navigation.midpoint.y, height),
        )
        cv2.circle(frame, midpoint, 6, active_color, -1)
        vector_end = (
            _px(navigation.midpoint.x + navigation.midpoint_delta_x * 3.0, width),
            _px(navigation.midpoint.y + navigation.midpoint_delta_y * 3.0, height),
        )
        cv2.arrowedLine(frame, midpoint, vector_end, (255, 255, 255), 2, tipLength=0.25)
        cv2.putText(
            frame,
            f"NAV {navigation.state.value}: {navigation.gesture}",
            (18, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            active_color,
            2,
            cv2.LINE_AA,
        )
        distance = f"{navigation.distance:.3f}" if navigation.distance is not None else "--"
        angle = (
            f"{navigation.angle * 180.0 / 3.14159265:.1f} deg"
            if navigation.angle is not None
            else "--"
        )
        cv2.putText(
            frame,
            f"2H distance {distance} | angle {angle} | confidence {navigation.confidence:.2f}",
            (18, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"vel {navigation.midpoint_velocity_x:+.3f},"
                f" {navigation.midpoint_velocity_y:+.3f} | "
                f"zoom v {navigation.distance_velocity:+.3f}"
            ),
            (18, 158),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"smooth {navigation.smoothing_amount * 100:.0f}% | "
                f"dead-zone {'ACTIVE' if navigation.dead_zone_active else 'clear'} | "
                f"outlier {'REJECTED' if navigation.outlier_rejected else 'clear'}"
            ),
            (18, 178),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def _px(value: float, dimension: int) -> int:
    return max(0, min(dimension - 1, round(value * dimension)))
