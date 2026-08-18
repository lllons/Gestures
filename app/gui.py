"""Tkinter user interface for the local Gestures controller."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, ttk
from typing import Any

import cv2
from PIL import Image, ImageTk

from .camera import CameraInfo, enumerate_cameras
from .keyboard_controller import ShortcutError, parse_shortcut
from .settings import AppSettings, SettingsError, SettingsStore
from .startup import StartupError, set_start_with_windows
from .worker import CameraWorker, FrameResult, WorkerEvent


class GesturesApp:
    """Main window; camera work is delegated to :class:`CameraWorker`."""

    BG = "#101820"
    PANEL = "#182632"
    TEXT = "#edf4f7"
    MUTED = "#9cb1bc"
    ACCENT = "#43d6bd"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Gestures — Nose Touch Controller")
        self.root.geometry("1120x760")
        self.root.minsize(960, 680)
        self.root.configure(bg=self.BG)

        self.store = SettingsStore()
        self.settings = self.store.load()
        self.result_queue: queue.Queue[FrameResult] = queue.Queue(maxsize=3)
        self.event_queue: queue.Queue[WorkerEvent] = queue.Queue(maxsize=100)
        self.worker = CameraWorker(
            settings_store=self.store,
            settings=self.settings,
            result_queue=self.result_queue,
            event_queue=self.event_queue,
        )
        self._photo: ImageTk.PhotoImage | None = None
        self._camera_options: dict[str, int] = {}
        self._last_result: FrameResult | None = None
        self._calibration_active = False
        self._calibration_completion_shown = False
        self._poll_id: str | None = None
        self._build_variables()
        self._build_styles()
        self._build_ui()
        self._refresh_cameras()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._poll_id = self.root.after(40, self._poll_worker)

    def _build_variables(self) -> None:
        self.camera_var = tk.StringVar(value="Searching for cameras...")
        self.detection_var = tk.BooleanVar(value=self.settings.detection_enabled)
        self.preview_var = tk.BooleanVar(value=self.settings.preview_visible)
        self.debug_var = tk.BooleanVar(value=self.settings.debug_mode)
        self.startup_var = tk.BooleanVar(value=self.settings.start_with_windows)
        self.shortcut_var = tk.StringVar(value=self.settings.shortcut)
        self.threshold_var = tk.DoubleVar(value=self.settings.touch_threshold)
        self.duration_var = tk.DoubleVar(value=self.settings.touch_duration_ms)
        self.cooldown_var = tk.DoubleVar(value=self.settings.cooldown_ms)

        self.status_var = tk.StringVar(value="READY")
        self.detail_var = tk.StringVar(value="Press Start Detection to begin.")
        self.hand_var = tk.StringVar(value="Not detected")
        self.face_var = tk.StringVar(value="Not detected")
        self.distance_var = tk.StringVar(value="—")
        self.fps_var = tk.StringVar(value="0")
        self.cooldown_status_var = tk.StringVar(value="Released")
        self.debug_text_var = tk.StringVar(value="No frames received yet.")

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background=self.BG, foreground=self.MUTED, font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 11, "bold"))
        style.configure("Panel.TLabel", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=self.PANEL, foreground=self.MUTED, font=("Segoe UI", 9))
        style.configure("Metric.TLabel", background=self.PANEL, foreground=self.ACCENT, font=("Segoe UI", 11, "bold"))
        style.configure("TCheckbutton", background=self.PANEL, foreground=self.TEXT)
        style.map("TCheckbutton", background=[("active", self.PANEL)])
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("TButton", padding=(8, 6))
        style.configure("TCombobox", fieldbackground="#223746", background="#223746", foreground=self.TEXT)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=20)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.rowconfigure(1, weight=1)

        ttk.Label(outer, text="GESTURES", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 2)
        )
        ttk.Label(
            outer,
            text="A private, local nose-touch shortcut controller",
            style="Subtitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(38, 0))

        preview_section = ttk.Frame(outer, style="App.TFrame")
        preview_section.grid(row=1, column=0, sticky="nsew", padx=(0, 18), pady=(20, 0))
        preview_section.rowconfigure(1, weight=1)
        preview_section.columnconfigure(0, weight=1)
        ttk.Label(preview_section, text="CAMERA PREVIEW", style="Subtitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.preview_label = tk.Label(
            preview_section,
            text="Camera stopped\n\nPress Start Detection to begin",
            bg="#0b1014",
            fg=self.MUTED,
            font=("Segoe UI", 14),
            compound=tk.CENTER,
            anchor=tk.CENTER,
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew")

        self._build_sidebar(outer)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        sidebar = ttk.Frame(parent, style="Panel.TFrame", padding=18)
        sidebar.grid(row=0, column=1, rowspan=2, sticky="nsew")
        sidebar.configure(width=350)

        ttk.Label(sidebar, text="LIVE STATUS", style="PanelTitle.TLabel").pack(anchor="w")
        status_row = ttk.Frame(sidebar, style="Panel.TFrame")
        status_row.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(status_row, textvariable=self.status_var, style="Metric.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_row, text="index fingertip → nose", style="Muted.TLabel").pack(
            side=tk.RIGHT, pady=3
        )
        ttk.Label(
            sidebar,
            textvariable=self.detail_var,
            style="Muted.TLabel",
            wraplength=310,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X, pady=(0, 12))

        metrics = ttk.Frame(sidebar, style="Panel.TFrame")
        metrics.pack(fill=tk.X, pady=(0, 16))
        self._metric(metrics, "Hand", self.hand_var, 0)
        self._metric(metrics, "Face", self.face_var, 1)
        self._metric(metrics, "Distance", self.distance_var, 2)
        self._metric(metrics, "FPS", self.fps_var, 3)
        self._metric(metrics, "Cooldown", self.cooldown_status_var, 4)

        ttk.Separator(sidebar).pack(fill=tk.X, pady=(0, 15))
        ttk.Label(sidebar, text="SETTINGS", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 8))

        camera_row = ttk.Frame(sidebar, style="Panel.TFrame")
        camera_row.pack(fill=tk.X, pady=3)
        ttk.Label(camera_row, text="Camera", style="Panel.TLabel").pack(side=tk.LEFT)
        self.camera_combo = ttk.Combobox(
            camera_row,
            textvariable=self.camera_var,
            state="readonly",
            width=18,
        )
        self.camera_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(12, 4))
        ttk.Button(camera_row, text="↻", width=3, command=self._refresh_cameras).pack(side=tk.RIGHT)

        toggles = ttk.Frame(sidebar, style="Panel.TFrame")
        toggles.pack(fill=tk.X, pady=(8, 4))
        ttk.Checkbutton(
            toggles,
            text="Enable detection",
            variable=self.detection_var,
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            toggles,
            text="Show camera preview",
            variable=self.preview_var,
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            toggles,
            text="Debug overlay and diagnostics",
            variable=self.debug_var,
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            toggles,
            text="Start with Windows",
            variable=self.startup_var,
        ).pack(anchor="w", pady=2)

        self._scale_control(
            sidebar,
            "Touch threshold (relative)",
            self.threshold_var,
            0.01,
            0.18,
            0.005,
        )
        self._scale_control(
            sidebar,
            "Required touch duration (ms)",
            self.duration_var,
            20,
            1000,
            10,
        )
        self._scale_control(
            sidebar,
            "Cooldown (ms)",
            self.cooldown_var,
            50,
            3000,
            10,
        )

        shortcut_row = ttk.Frame(sidebar, style="Panel.TFrame")
        shortcut_row.pack(fill=tk.X, pady=(5, 3))
        ttk.Label(shortcut_row, text="Shortcut", style="Panel.TLabel").pack(side=tk.LEFT)
        self.shortcut_entry = ttk.Entry(shortcut_row, textvariable=self.shortcut_var, width=20)
        self.shortcut_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(12, 0))
        ttk.Label(
            sidebar,
            text="Examples: Alt + Tab · Ctrl + Shift + S · Space · Escape",
            style="Muted.TLabel",
            wraplength=310,
        ).pack(anchor="w", pady=(0, 10))

        actions = ttk.Frame(sidebar, style="Panel.TFrame")
        actions.pack(fill=tk.X, pady=(5, 8))
        ttk.Button(
            actions,
            text="Start Detection",
            style="Accent.TButton",
            command=self.start,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(actions, text="Stop", command=self.stop).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        secondary = ttk.Frame(sidebar, style="Panel.TFrame")
        secondary.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(secondary, text="Apply Settings", command=self._apply_settings).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4)
        )
        ttk.Button(secondary, text="Calibrate", command=self.calibrate).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0)
        )

        ttk.Label(sidebar, text="DEBUG", style="PanelTitle.TLabel").pack(anchor="w", pady=(4, 5))
        self.debug_box = tk.Text(
            sidebar,
            height=8,
            width=38,
            bg="#0e171e",
            fg="#b9e7da",
            insertbackground=self.TEXT,
            relief=tk.FLAT,
            borderwidth=0,
            font=("Consolas", 9),
            padx=8,
            pady=7,
            state=tk.DISABLED,
        )
        self.debug_box.pack(fill=tk.BOTH, expand=True)

    def _metric(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=f"{label}:", style="Muted.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        ttk.Label(parent, textvariable=variable, style="Panel.TLabel").grid(
            row=row, column=1, sticky="e", pady=2
        )
        parent.columnconfigure(1, weight=1)

    def _scale_control(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        minimum: float,
        maximum: float,
        resolution: float,
    ) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=(7, 0))
        value_label = ttk.Label(row, textvariable=variable, style="Muted.TLabel", width=7, anchor="e")
        value_label.pack(side=tk.RIGHT)
        ttk.Label(row, text=label, style="Panel.TLabel").pack(side=tk.LEFT)
        scale = tk.Scale(
            parent,
            variable=variable,
            from_=minimum,
            to=maximum,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            showvalue=False,
            highlightthickness=0,
            bd=0,
            bg=self.PANEL,
            fg=self.TEXT,
            troughcolor="#294453",
            activebackground=self.ACCENT,
            sliderrelief=tk.FLAT,
        )
        scale.pack(fill=tk.X, pady=(0, 2))

    def _refresh_cameras(self) -> None:
        self.camera_var.set("Searching for cameras...")
        self.camera_combo.configure(state="disabled")

        def probe() -> None:
            cameras = enumerate_cameras()
            self.root.after(0, lambda: self._set_cameras(cameras))

        threading.Thread(target=probe, name="camera-probe", daemon=True).start()

    def _set_cameras(self, cameras: list[CameraInfo]) -> None:
        self._camera_options = {camera.label: camera.index for camera in cameras}
        values = list(self._camera_options)
        if not values:
            values = ["No camera detected"]
        self.camera_combo.configure(values=values, state="readonly")
        preferred = next(
            (label for label, index in self._camera_options.items() if index == self.settings.camera_index),
            values[0],
        )
        self.camera_var.set(preferred)
        if not self.worker.is_running() and not self._camera_options:
            self.detail_var.set("No webcam found. Connect one, then press ↻.")

    def _selected_camera_index(self) -> int:
        return self._camera_options.get(self.camera_var.get(), self.settings.camera_index)

    def _collect_settings(self) -> AppSettings:
        settings = replace(
            self.settings,
            camera_index=self._selected_camera_index(),
            detection_enabled=bool(self.detection_var.get()),
            preview_visible=bool(self.preview_var.get()),
            debug_mode=bool(self.debug_var.get()),
            start_with_windows=bool(self.startup_var.get()),
            shortcut=self.shortcut_var.get().strip(),
            touch_threshold=round(float(self.threshold_var.get()), 3),
            touch_duration_ms=int(round(float(self.duration_var.get()))),
            cooldown_ms=int(round(float(self.cooldown_var.get()))),
        )
        settings.validate()
        parse_shortcut(settings.shortcut)
        return settings

    def _apply_settings(self, show_errors: bool = True) -> bool:
        try:
            settings = self._collect_settings()
            if settings.start_with_windows != self.settings.start_with_windows:
                set_start_with_windows(settings.start_with_windows)
            self.store.save(settings)
        except (SettingsError, ShortcutError, StartupError, OSError, ValueError) as exc:
            if show_errors:
                messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            return False

        self.settings = settings
        if self.worker.is_running():
            self.worker.update_settings(settings)
        self.detail_var.set("Settings saved locally.")
        if not settings.preview_visible:
            self._show_preview_placeholder("Camera preview hidden\n\nDetection can continue in the background")
        return True

    def start(self) -> None:
        if not self._apply_settings():
            return
        if self.worker.start():
            self.status_var.set("STARTING")
            self.detail_var.set("Opening camera and loading local MediaPipe models...")

    def stop(self) -> None:
        self.worker.stop()
        self._calibration_active = False
        self.status_var.set("STOPPED")
        self.detail_var.set("Camera released. No keyboard input is sent while stopped.")
        self._show_preview_placeholder("Camera stopped\n\nPress Start Detection to begin")
        self.hand_var.set("Not detected")
        self.face_var.set("Not detected")
        self.distance_var.set("—")
        self.fps_var.set("0")
        self.cooldown_status_var.set("Released")

    def calibrate(self) -> None:
        if not self.worker.is_running():
            messagebox.showinfo(
                "Start detection first",
                "Start Detection, then press Calibrate while the camera preview is running.",
                parent=self.root,
            )
            return
        messagebox.showinfo(
            "Calibration",
            "First keep your face centered and still.\n\n"
            "When prompted, touch your nose with the index fingertip and hold for a moment.\n\n"
            "Calibration uses only local landmark distances and saves the threshold locally.",
            parent=self.root,
        )
        self._calibration_completion_shown = False
        self._calibration_active = True
        self.worker.begin_calibration()
        self.status_var.set("CALIBRATING")
        self.detail_var.set("Keep your face centered and still...")

    def _poll_worker(self) -> None:
        newest: FrameResult | None = None
        while True:
            try:
                newest = self.result_queue.get_nowait()
            except queue.Empty:
                break
        if newest is not None:
            self._last_result = newest
            self._update_from_result(newest)

        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)

        if self.worker.is_running() and self.status_var.get() == "STOPPED":
            self.status_var.set("STARTING")
        self._poll_id = self.root.after(40, self._poll_worker)

    def _update_from_result(self, result: FrameResult) -> None:
        snapshot = result.snapshot
        settings = self.settings
        if self._calibration_active and result.calibration:
            self.detail_var.set(result.calibration.message)
            self.status_var.set(result.calibration.phase)
            if result.calibration.completed:
                self._calibration_active = False
                if result.calibration.succeeded and result.calibration.threshold is not None:
                    self.threshold_var.set(result.calibration.threshold)
                    self.settings = replace(
                        self.settings,
                        touch_threshold=result.calibration.threshold,
                        calibrated_face_width=result.calibration.face_width,
                    )
                    if not self._calibration_completion_shown:
                        self._calibration_completion_shown = True
                        messagebox.showinfo(
                            "Calibration complete",
                            result.calibration.message,
                            parent=self.root,
                        )
                elif not self._calibration_completion_shown:
                    self._calibration_completion_shown = True
                    messagebox.showwarning(
                        "Calibration not completed",
                        result.calibration.message,
                        parent=self.root,
                    )
        elif not settings.detection_enabled:
            self.status_var.set("DISABLED")
            self.detail_var.set("Detection is disabled; the camera is still available for preview.")
        else:
            self.status_var.set(snapshot.state.value)
            self.detail_var.set(snapshot.message)

        self.hand_var.set(
            f"Detected ({snapshot.hand_count})" if snapshot.hand_detected else "Not detected"
        )
        self.face_var.set("Detected" if snapshot.face_detected else "Not detected")
        self.distance_var.set(
            f"{snapshot.relative_distance:.3f}" if snapshot.relative_distance is not None else "—"
        )
        self.fps_var.set(f"{result.fps:.0f}")
        self.cooldown_status_var.set(
            f"{snapshot.cooldown_remaining_ms} ms"
            if snapshot.cooldown_remaining_ms > 0
            else ("Move finger away" if snapshot.awaiting_release else "Released")
        )
        self._update_debug(result)

        if result.preview_frame is not None and self.preview_var.get():
            self._show_frame(result.preview_frame)
        elif not self.preview_var.get():
            self._show_preview_placeholder("Camera preview hidden\n\nDetection can continue in the background")

    def _update_debug(self, result: FrameResult) -> None:
        snapshot = result.snapshot
        fingertip = _point_text(snapshot.index_tip)
        nose = _point_text(snapshot.nose)
        distance = (
            f"{snapshot.relative_distance:.4f}" if snapshot.relative_distance is not None else "--"
        )
        cooldown = f"{snapshot.cooldown_remaining_ms} ms"
        text = "\n".join(
            (
                f"FPS                 {result.fps:.1f}",
                f"Hand detected       {'yes' if snapshot.hand_detected else 'no'} ({snapshot.hand_count})",
                f"Face detected       {'yes' if snapshot.face_detected else 'no'}",
                f"Index fingertip     {fingertip}",
                f"Nose                {nose}",
                f"Relative distance   {distance}",
                f"State               {snapshot.state.value}",
                f"Cooldown            {cooldown}",
                f"Awaiting release    {'yes' if snapshot.awaiting_release else 'no'}",
            )
        )
        self.debug_box.configure(state=tk.NORMAL)
        self.debug_box.delete("1.0", tk.END)
        self.debug_box.insert("1.0", text)
        self.debug_box.configure(state=tk.DISABLED)

    def _handle_event(self, event: WorkerEvent) -> None:
        if event.kind == "error":
            first_line = event.message.splitlines()[0]
            self.detail_var.set(first_line)
            self.status_var.set("ERROR")
            if "model" in event.message.lower() or "mediapipe" in event.message.lower():
                messagebox.showerror("Gestures could not start", event.message, parent=self.root)
        elif event.kind == "warning":
            self.detail_var.set(event.message)
        elif event.kind == "info" and not self._calibration_active:
            self.detail_var.set(event.message)

    def _show_frame(self, frame: Any) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_size = (760, 610)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=self._photo, text="")

    def _show_preview_placeholder(self, text: str) -> None:
        self._photo = None
        self.preview_label.configure(image="", text=text)

    def close(self) -> None:
        if self._poll_id is not None:
            try:
                self.root.after_cancel(self._poll_id)
            except tk.TclError:
                pass
        try:
            self._apply_settings(show_errors=False)
        finally:
            self.worker.stop()
            self.root.destroy()


def _point_text(point: Any) -> str:
    if point is None:
        return "--"
    return f"({point.x:.3f}, {point.y:.3f})"


def launch() -> None:
    root = tk.Tk()
    GesturesApp(root)
    root.mainloop()
