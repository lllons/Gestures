"""MediaPipe Tasks Face Landmarker adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hand_tracker import Landmark


NOSE_TIP_INDEX = 1
FACE_LEFT_SCALE_INDEX = 234
FACE_RIGHT_SCALE_INDEX = 454


@dataclass(frozen=True)
class FaceDetection:
    landmarks: tuple[Landmark, ...]
    nose: Landmark
    face_width: float


class FaceTracker:
    """Run the official MediaPipe Face Landmarker in video mode."""

    def __init__(self, model_path: Path) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "MediaPipe is not installed. Run python -m pip install -r requirements.txt."
            ) from exc

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb_frame: Any, timestamp_ms: int) -> FaceDetection | None:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        raw_faces = getattr(result, "face_landmarks", []) or []
        if not raw_faces:
            return None

        landmarks = tuple(
            Landmark(
                x=float(getattr(point, "x", 0.0)),
                y=float(getattr(point, "y", 0.0)),
                z=float(getattr(point, "z", 0.0)),
            )
            for point in raw_faces[0]
        )
        if len(landmarks) <= NOSE_TIP_INDEX:
            return None

        nose = landmarks[NOSE_TIP_INDEX]
        face_width = _face_width(landmarks)
        if face_width <= 0:
            return None
        return FaceDetection(landmarks=landmarks, nose=nose, face_width=face_width)

    def close(self) -> None:
        self._landmarker.close()


def _face_width(landmarks: tuple[Landmark, ...]) -> float:
    """Use cheek landmarks for scale and fall back to the mesh bounding box."""

    if len(landmarks) > FACE_RIGHT_SCALE_INDEX:
        left = landmarks[FACE_LEFT_SCALE_INDEX]
        right = landmarks[FACE_RIGHT_SCALE_INDEX]
        width = ((left.x - right.x) ** 2 + (left.y - right.y) ** 2) ** 0.5
        if width > 0:
            return width

    xs = [point.x for point in landmarks]
    if not xs:
        return 0.0
    return max(xs) - min(xs)
