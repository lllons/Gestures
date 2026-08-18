"""MediaPipe Tasks Hand Landmarker adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


INDEX_FINGER_TIP_INDEX = 8


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class HandDetection:
    landmarks: tuple[Landmark, ...]
    handedness: str | None = None
    confidence: float | None = None


class HandTracker:
    """Run the official MediaPipe Hand Landmarker in video mode."""

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
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame: Any, timestamp_ms: int) -> list[HandDetection]:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        hands: list[HandDetection] = []
        raw_handedness = getattr(result, "handedness", []) or []
        for hand_index, raw_landmarks in enumerate(getattr(result, "hand_landmarks", []) or []):
            landmarks = tuple(
                Landmark(
                    x=float(getattr(point, "x", 0.0)),
                    y=float(getattr(point, "y", 0.0)),
                    z=float(getattr(point, "z", 0.0)),
                )
                for point in raw_landmarks
            )
            handedness: str | None = None
            confidence: float | None = None
            if hand_index < len(raw_handedness) and raw_handedness[hand_index]:
                category = raw_handedness[hand_index][0]
                handedness = getattr(category, "category_name", None) or getattr(
                    category, "display_name", None
                )
                raw_score = getattr(category, "score", None)
                confidence = float(raw_score) if raw_score is not None else None
            hands.append(
                HandDetection(
                    landmarks=landmarks,
                    handedness=handedness,
                    confidence=confidence,
                )
            )
        return hands

    def close(self) -> None:
        self._landmarker.close()
