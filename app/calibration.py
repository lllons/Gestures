"""Interactive calibration for a user's normal webcam position."""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class CalibrationUpdate:
    active: bool
    phase: str
    message: str
    completed: bool = False
    succeeded: bool = False
    threshold: float | None = None
    face_width: float | None = None
    sample_count: int = 0


class CalibrationSession:
    """Capture a stable face width, then a stable fingertip-to-nose distance."""

    FACE_REFERENCE_SECONDS = 1.8
    TOUCH_HOLD_SECONDS = 0.8
    TIMEOUT_SECONDS = 25.0
    TOUCH_GATE = 0.50

    def __init__(self) -> None:
        self.active = False
        self.phase = "IDLE"
        self._started_at = 0.0
        self._phase_started_at = 0.0
        self._face_widths: list[float] = []
        self._touch_distances: list[float] = []
        self._touch_started_at: float | None = None
        self._face_reference: float | None = None

    def begin(self, now: float | None = None) -> CalibrationUpdate:
        current_time = time.monotonic() if now is None else now
        self.active = True
        self.phase = "FACE REFERENCE"
        self._started_at = current_time
        self._phase_started_at = current_time
        self._face_widths.clear()
        self._touch_distances.clear()
        self._touch_started_at = None
        self._face_reference = None
        return CalibrationUpdate(
            active=True,
            phase=self.phase,
            message="Keep your face centered and still...",
        )

    def cancel(self) -> None:
        self.active = False
        self.phase = "IDLE"
        self._touch_started_at = None

    def update(
        self,
        face_width: float | None,
        relative_distance: float | None,
        hand_detected: bool,
        face_detected: bool,
        now: float | None = None,
    ) -> CalibrationUpdate | None:
        if not self.active:
            return None
        current_time = time.monotonic() if now is None else now
        if current_time - self._started_at > self.TIMEOUT_SECONDS:
            self.cancel()
            return CalibrationUpdate(
                active=False,
                phase="FAILED",
                message="Calibration timed out. Try again in better lighting.",
                completed=True,
                succeeded=False,
            )

        if self.phase == "FACE REFERENCE":
            if face_detected and face_width and face_width > 0:
                self._face_widths.append(face_width)
            if (
                current_time - self._phase_started_at >= self.FACE_REFERENCE_SECONDS
                and len(self._face_widths) >= 8
            ):
                self._face_reference = median(self._face_widths)
                self.phase = "TOUCH NOSE"
                self._phase_started_at = current_time
                self._touch_distances.clear()
                return CalibrationUpdate(
                    active=True,
                    phase=self.phase,
                    message="Now touch your nose with the index fingertip and hold...",
                    face_width=self._face_reference,
                )
            return CalibrationUpdate(
                active=True,
                phase=self.phase,
                message=(
                    "Keep your face centered and still..."
                    if face_detected
                    else "Face not detected; move into view..."
                ),
                sample_count=len(self._face_widths),
            )

        if self.phase == "TOUCH NOSE":
            close_enough = (
                hand_detected
                and face_detected
                and relative_distance is not None
                and relative_distance <= self.TOUCH_GATE
            )
            if close_enough:
                if self._touch_started_at is None:
                    self._touch_started_at = current_time
                    self._touch_distances.clear()
                self._touch_distances.append(relative_distance)
                if current_time - self._touch_started_at >= self.TOUCH_HOLD_SECONDS:
                    measured = median(self._touch_distances)
                    threshold = max(0.025, min(0.50, measured * 1.35))
                    face_width = self._face_reference
                    samples = len(self._touch_distances)
                    self.cancel()
                    return CalibrationUpdate(
                        active=False,
                        phase="COMPLETE",
                        message=f"Calibration complete. Threshold set to {threshold:.3f}.",
                        completed=True,
                        succeeded=True,
                        threshold=threshold,
                        face_width=face_width,
                        sample_count=samples,
                    )
                return CalibrationUpdate(
                    active=True,
                    phase=self.phase,
                    message="Hold your fingertip at the nose...",
                    face_width=self._face_reference,
                    sample_count=len(self._touch_distances),
                )

            self._touch_started_at = None
            self._touch_distances.clear()
            return CalibrationUpdate(
                active=True,
                phase=self.phase,
                message="Bring the index fingertip to your nose and hold...",
                face_width=self._face_reference,
            )

        return None
