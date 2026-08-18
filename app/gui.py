"""Tkinter user interface for the local Gestures controller."""

from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, ttk
from typing import Any, Callable

import cv2
from PIL import Image, ImageTk

from .camera import CameraInfo, enumerate_cameras
from .input_profile import (
    MODIFIER_TEXT_OPTIONS,
    MOUSE_BUTTON_OPTIONS,
    ZOOM_DIRECTION_OPTIONS,
    InputProfile,
    get_profile,
    profile_names,
)
from .keyboard_controller import SHORTCUT_OPTIONS, ShortcutError, parse_shortcut
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
        self.root.title("Gestures — Universal 3D Hand Controller")
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
        self.pinch_shortcut_var = tk.StringVar(value=self.settings.pinch_shortcut)
        self.air_mouse_var = tk.BooleanVar(value=self.settings.air_mouse_enabled)
        self.air_mouse_button_var = tk.StringVar()
        self._set_air_mouse_button_text()
        self.threshold_var = tk.DoubleVar(value=self.settings.touch_threshold)
        self.duration_var = tk.DoubleVar(value=self.settings.touch_duration_ms)
        self.cooldown_var = tk.DoubleVar(value=self.settings.cooldown_ms)

        self.navigation_enabled_var = tk.BooleanVar(value=self.settings.navigation_enabled)
        self.navigation_profile_var = tk.StringVar(value=self.settings.navigation_profile)
        self.navigation_profile_name_var = tk.StringVar(value=self.settings.navigation_profile)
        profile = get_profile(
            self.settings.navigation_profile,
            self.settings.navigation_profiles,
        )
        self.navigation_orbit_button_var = tk.StringVar(value=profile.orbit_button)
        self.navigation_orbit_modifier_var = tk.StringVar(
            value=" + ".join(profile.orbit_modifiers) if profile.orbit_modifiers else "none"
        )
        self.navigation_pan_button_var = tk.StringVar(value=profile.pan_button)
        self.navigation_pan_modifier_var = tk.StringVar(
            value=" + ".join(profile.pan_modifiers) if profile.pan_modifiers else "none"
        )
        self.navigation_zoom_direction_var = tk.StringVar(
            value=profile.zoom_direction_label
        )
        self.navigation_control_mode_var = tk.StringVar(
            value=self.settings.navigation_control_mode
        )
        self.navigation_activation_var = tk.StringVar(
            value=self.settings.navigation_activation_gesture
        )
        self.navigation_deactivation_var = tk.StringVar(
            value=self.settings.navigation_deactivation_gesture
        )
        self.navigation_pan_gesture_var = tk.StringVar(
            value=self.settings.navigation_pan_gesture
        )
        self.navigation_orbit_var = tk.DoubleVar(
            value=self.settings.navigation_orbit_sensitivity
        )
        self.navigation_pan_var = tk.DoubleVar(value=self.settings.navigation_pan_sensitivity)
        self.navigation_zoom_var = tk.DoubleVar(value=self.settings.navigation_zoom_sensitivity)
        self.navigation_roll_var = tk.DoubleVar(value=self.settings.navigation_roll_sensitivity)
        self.navigation_smoothing_var = tk.DoubleVar(
            value=self.settings.navigation_smoothing_frames
        )
        self.navigation_dead_zone_var = tk.DoubleVar(value=self.settings.navigation_dead_zone)
        self.navigation_max_speed_var = tk.DoubleVar(value=self.settings.navigation_max_speed)
        self.navigation_acceleration_var = tk.DoubleVar(value=self.settings.navigation_acceleration)
        self.navigation_mouse_scale_var = tk.DoubleVar(value=self.settings.navigation_mouse_scale)
        self.navigation_zoom_wheel_scale_var = tk.DoubleVar(
            value=self.settings.navigation_zoom_wheel_scale
        )
        self.navigation_confidence_threshold_var = tk.DoubleVar(
            value=self.settings.navigation_min_confidence
        )
        self.navigation_activation_hold_var = tk.DoubleVar(
            value=self.settings.navigation_activation_hold_ms
        )
        self.navigation_invert_x_var = tk.BooleanVar(value=self.settings.navigation_invert_x)
        self.navigation_invert_y_var = tk.BooleanVar(value=self.settings.navigation_invert_y)
        self.navigation_invert_zoom_var = tk.BooleanVar(
            value=self.settings.navigation_invert_zoom
        )
        self.navigation_roll_enabled_var = tk.BooleanVar(
            value=self.settings.navigation_roll_enabled
        )
        self.navigation_global_hotkey_var = tk.StringVar(
            value=self.settings.navigation_global_hotkey
        )
        self.navigation_emergency_hotkey_var = tk.StringVar(
            value=self.settings.navigation_emergency_hotkey
        )

        self.navigation_button_var = tk.StringVar()
        self._set_navigation_button_text()
        self.navigation_status_var = tk.StringVar(value="DISABLED")
        self.navigation_hands_var = tk.StringVar(value="0")
        self.navigation_mode_status_var = tk.StringVar(
            value=self.settings.navigation_control_mode
        )
        self.navigation_gesture_var = tk.StringVar(value="Idle")
        self.navigation_distance_var = tk.StringVar(value="—")
        self.navigation_angle_var = tk.StringVar(value="—")
        self.navigation_vector_var = tk.StringVar(value="0.000, 0.000")
        self.navigation_confidence_var = tk.StringVar(value="0%")
        self.navigation_mouse_var = tk.StringVar(value="Released")
        self.navigation_modifiers_var = tk.StringVar(value="None")

        self.status_var = tk.StringVar(value="READY")
        self.detail_var = tk.StringVar(value="")
        self.hand_var = tk.StringVar(value="Not detected")
        self.face_var = tk.StringVar(value="Not detected")
        self.distance_var = tk.StringVar(value="—")
        self.fps_var = tk.StringVar(value="0")
        self.cooldown_status_var = tk.StringVar(value="Released")
        self.pinch_var = tk.StringVar(value="Open")
        self.scroll_var = tk.StringVar(value="Off")
        self.debug_text_var = tk.StringVar(value="No frames received yet.")

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("Panel.TLabel", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=self.PANEL, foreground=self.MUTED, font=("Segoe UI", 9))
        style.configure("Metric.TLabel", background=self.PANEL, foreground=self.ACCENT, font=("Segoe UI", 11, "bold"))
        style.configure("Section.TLabel", background=self.PANEL, foreground=self.ACCENT, font=("Segoe UI", 10, "bold"))
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
        outer.rowconfigure(0, weight=1)

        preview_section = ttk.Frame(outer, style="App.TFrame")
        preview_section.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        preview_section.rowconfigure(0, weight=1)
        preview_section.columnconfigure(0, weight=1)
        self.preview_label = tk.Label(
            preview_section,
            text="Camera stopped",
            bg="#0b1014",
            fg=self.MUTED,
            font=("Segoe UI", 14),
            compound=tk.CENTER,
            anchor=tk.CENTER,
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        actions = ttk.Frame(preview_section, style="App.TFrame")
        actions.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        ttk.Button(
            actions,
            text="Start Detection",
            style="Accent.TButton",
            command=self.start,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(actions, text="Stop", command=self.stop).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        self.air_mouse_button = ttk.Button(
            preview_section,
            textvariable=self.air_mouse_button_var,
            command=self.toggle_air_mouse,
        )
        self.air_mouse_button.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self._build_sidebar(outer)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        sidebar = ttk.Frame(parent, style="Panel.TFrame")
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.configure(width=350)

        self.sidebar_canvas = tk.Canvas(
            sidebar,
            bg=self.PANEL,
            highlightthickness=0,
            bd=0,
        )
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        content = ttk.Frame(self.sidebar_canvas, style="Panel.TFrame", padding=(18, 16, 10, 18))
        self._sidebar_window = self.sidebar_canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _event: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")),
        )
        self.sidebar_canvas.bind(
            "<Configure>",
            lambda event: self.sidebar_canvas.itemconfigure(
                self._sidebar_window, width=event.width
            ),
        )
        self._pointer_x = 0
        self._pointer_y = 0
        self.root.bind_all(
            "<Motion>",
            lambda event: self._track_pointer(event.x_root, event.y_root),
        )
        self.root.bind_all(
            "<MouseWheel>",
            lambda event: self._scroll_sidebar_if_hovered(event),
        )
        self._build_sidebar_content(content)

    def _track_pointer(self, x_root: int, y_root: int) -> None:
        self._pointer_x = x_root
        self._pointer_y = y_root

    def _scroll_sidebar_if_hovered(self, event: Any) -> None:
        left = self.sidebar_canvas.winfo_rootx()
        top = self.sidebar_canvas.winfo_rooty()
        right = left + self.sidebar_canvas.winfo_width()
        bottom = top + self.sidebar_canvas.winfo_height()
        if left <= self._pointer_x <= right and top <= self._pointer_y <= bottom:
            self.sidebar_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _build_sidebar_content(self, sidebar: ttk.Frame) -> None:
        status_row = ttk.Frame(sidebar, style="Panel.TFrame")
        status_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(status_row, textvariable=self.status_var, style="Metric.TLabel").pack(side=tk.LEFT)
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
        self._metric(metrics, "Pinch", self.pinch_var, 5)
        self._metric(metrics, "Scroll", self.scroll_var, 6)

        ttk.Separator(sidebar).pack(fill=tk.X, pady=(0, 15))

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
            "Activation zone size (% of face width)",
            self.threshold_var,
            0.01,
            0.50,
            0.005,
            format_value=lambda value: f"{value * 100:.0f}%",
            command=self._apply_settings_live,
        )
        self._scale_control(
            sidebar,
            "Activation delay (ms)",
            self.duration_var,
            0,
            1000,
            10,
        )
        self._scale_control(
            sidebar,
            "Cooldown (ms)",
            self.cooldown_var,
            0,
            3000,
            10,
        )

        self._shortcut_control(sidebar, "Nose touch key", self.shortcut_var)
        self._shortcut_control(sidebar, "Thumb + index key", self.pinch_shortcut_var)
        secondary = ttk.Frame(sidebar, style="Panel.TFrame")
        secondary.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(secondary, text="Apply Settings", command=self._apply_settings).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4)
        )
        ttk.Button(secondary, text="Calibrate", command=self.calibrate).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0)
        )

        self._build_navigation_controls(sidebar)

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
        self.debug_box.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    def _build_navigation_controls(self, parent: ttk.Frame) -> None:
        """Build the opt-in universal 3D navigation panel inside the existing sidebar."""

        ttk.Separator(parent).pack(fill=tk.X, pady=(8, 15))
        ttk.Label(parent, text="3D NAVIGATION", style="Section.TLabel").pack(
            anchor="w", pady=(0, 8)
        )

        nav_button_row = ttk.Frame(parent, style="Panel.TFrame")
        nav_button_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            nav_button_row,
            textvariable=self.navigation_button_var,
            style="Accent.TButton",
            command=self.toggle_navigation,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(
            nav_button_row,
            text="Calibrate",
            command=self.calibrate_navigation,
        ).pack(side=tk.RIGHT, padx=(4, 0))

        nav_metrics = ttk.Frame(parent, style="Panel.TFrame")
        nav_metrics.pack(fill=tk.X, pady=(0, 9))
        self._metric(nav_metrics, "Status", self.navigation_status_var, 0)
        self._metric(nav_metrics, "Hands", self.navigation_hands_var, 1)
        self._metric(nav_metrics, "Mode", self.navigation_mode_status_var, 2)
        self._metric(nav_metrics, "Gesture", self.navigation_gesture_var, 3)
        self._metric(nav_metrics, "Distance", self.navigation_distance_var, 4)
        self._metric(nav_metrics, "Angle", self.navigation_angle_var, 5)
        self._metric(nav_metrics, "Vector", self.navigation_vector_var, 6)
        self._metric(nav_metrics, "Confidence", self.navigation_confidence_var, 7)
        self._metric(nav_metrics, "Mouse", self.navigation_mouse_var, 8)
        self._metric(nav_metrics, "Modifiers", self.navigation_modifiers_var, 9)

        profile_row = ttk.Frame(parent, style="Panel.TFrame")
        profile_row.pack(fill=tk.X, pady=3)
        ttk.Label(profile_row, text="Input profile", style="Panel.TLabel").pack(side=tk.LEFT)
        self.navigation_profile_combo = ttk.Combobox(
            profile_row,
            textvariable=self.navigation_profile_var,
            values=list(profile_names(self.settings.navigation_profiles)),
            state="readonly",
            width=18,
        )
        self.navigation_profile_combo.pack(
            side=tk.RIGHT, fill=tk.X, expand=True, padx=(12, 0)
        )
        self.navigation_profile_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._select_profile(),
        )
        self._navigation_combo(
            parent,
            "Control mode",
            self.navigation_control_mode_var,
            ("FULL 3D", "ORBIT", "PAN", "ZOOM"),
        )
        self._navigation_combo(
            parent,
            "Activation",
            self.navigation_activation_var,
            ("Two open hands", "Two closed hands"),
        )
        self._navigation_combo(
            parent,
            "Deactivation",
            self.navigation_deactivation_var,
            ("Hands removed", "Two closed hands", "Two open hands"),
        )
        self._navigation_combo(
            parent,
            "Pan pose in Full 3D",
            self.navigation_pan_gesture_var,
            ("Two closed hands", "Two open hands", "Disabled"),
        )

        ttk.Label(parent, text="Profile mapping", style="Muted.TLabel").pack(
            anchor="w", pady=(8, 3)
        )
        self._entry_control(parent, "Profile name", self.navigation_profile_name_var)
        self._navigation_combo(
            parent,
            "Orbit button",
            self.navigation_orbit_button_var,
            MOUSE_BUTTON_OPTIONS,
        )
        self._navigation_combo(
            parent,
            "Orbit modifier",
            self.navigation_orbit_modifier_var,
            MODIFIER_TEXT_OPTIONS,
        )
        self._navigation_combo(
            parent,
            "Pan button",
            self.navigation_pan_button_var,
            MOUSE_BUTTON_OPTIONS,
        )
        self._navigation_combo(
            parent,
            "Pan modifier",
            self.navigation_pan_modifier_var,
            MODIFIER_TEXT_OPTIONS,
        )
        self._navigation_combo(
            parent,
            "Zoom direction",
            self.navigation_zoom_direction_var,
            ZOOM_DIRECTION_OPTIONS,
        )
        ttk.Button(parent, text="Save Input Profile", command=self._save_input_profile).pack(
            fill=tk.X, pady=(3, 6)
        )

        self._scale_control(
            parent,
            "Orbit sensitivity",
            self.navigation_orbit_var,
            0.1,
            12.0,
            0.1,
            format_value=lambda value: f"{value:.1f}",
            command=self._apply_settings_live,
        )
        self._scale_control(
            parent,
            "Pan sensitivity",
            self.navigation_pan_var,
            0.1,
            6.0,
            0.1,
            format_value=lambda value: f"{value:.1f}",
            command=self._apply_settings_live,
        )
        self._scale_control(
            parent,
            "Zoom sensitivity",
            self.navigation_zoom_var,
            0.1,
            12.0,
            0.1,
            format_value=lambda value: f"{value:.1f}",
            command=self._apply_settings_live,
        )
        self._scale_control(
            parent,
            "Roll sensitivity",
            self.navigation_roll_var,
            0.1,
            12.0,
            0.1,
            format_value=lambda value: f"{value:.1f}",
            command=self._apply_settings_live,
        )
        self._scale_control(
            parent,
            "Smoothing (frames)",
            self.navigation_smoothing_var,
            1,
            20,
            1,
            format_value=lambda value: f"{value:.0f}",
            command=self._apply_settings_live,
        )
        self._scale_control(
            parent,
            "Dead zone",
            self.navigation_dead_zone_var,
            0,
            0.10,
            0.001,
            format_value=lambda value: f"{value:.3f}",
            command=self._apply_settings_live,
        )
        self._scale_control(
            parent,
            "Maximum speed",
            self.navigation_max_speed_var,
            0.1,
            4.0,
            0.05,
            format_value=lambda value: f"{value:.2f}",
            command=self._apply_settings_live,
        )
        self._scale_control(
            parent,
            "Acceleration",
            self.navigation_acceleration_var,
            0,
            3.0,
            0.05,
            format_value=lambda value: f"{value:.2f}",
            command=self._apply_settings_live,
        )
        self._scale_control(
            parent,
            "Relative mouse scale",
            self.navigation_mouse_scale_var,
            10,
            1000,
            10,
            format_value=lambda value: f"{value:.0f}",
            command=self._apply_settings_live,
        )
        self._scale_control(
            parent,
            "Zoom wheel scale",
            self.navigation_zoom_wheel_scale_var,
            0.1,
            10.0,
            0.1,
            format_value=lambda value: f"{value:.1f}",
            command=self._apply_settings_live,
        )
        self._scale_control(
            parent,
            "Confidence threshold",
            self.navigation_confidence_threshold_var,
            0.1,
            1.0,
            0.01,
            format_value=lambda value: f"{value:.2f}",
            command=self._apply_settings_live,
        )
        self._scale_control(
            parent,
            "Activation hold (ms)",
            self.navigation_activation_hold_var,
            250,
            3000,
            50,
            format_value=lambda value: f"{value:.0f}",
            command=self._apply_settings_live,
        )

        nav_toggles = ttk.Frame(parent, style="Panel.TFrame")
        nav_toggles.pack(fill=tk.X, pady=(5, 3))
        ttk.Checkbutton(
            nav_toggles,
            text="Invert X",
            variable=self.navigation_invert_x_var,
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            nav_toggles,
            text="Invert Y",
            variable=self.navigation_invert_y_var,
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            nav_toggles,
            text="Invert zoom direction",
            variable=self.navigation_invert_zoom_var,
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            nav_toggles,
            text="Enable roll from hand angle",
            variable=self.navigation_roll_enabled_var,
        ).pack(anchor="w", pady=2)

        ttk.Label(parent, text="Safety hotkeys", style="Muted.TLabel").pack(
            anchor="w", pady=(8, 3)
        )
        self._entry_control(parent, "Global toggle", self.navigation_global_hotkey_var)
        self._entry_control(parent, "Emergency stop", self.navigation_emergency_hotkey_var)
        ttk.Label(
            parent,
            text="F8 toggles OS input. F9 releases every simulated button/modifier.",
            style="Muted.TLabel",
            wraplength=300,
        ).pack(anchor="w", pady=(2, 4))
        ttk.Button(parent, text="Apply 3D Settings", command=self._apply_settings).pack(
            fill=tk.X, pady=(7, 4)
        )

    def _select_profile(self) -> None:
        self._load_profile_controls()
        self._apply_settings(show_errors=False)

    def _load_profile_controls(self) -> None:
        """Load the selected profile into the editable mapping controls."""

        profile = get_profile(
            self.navigation_profile_var.get(),
            self.settings.navigation_profiles,
        )
        self.navigation_profile_var.set(profile.name)
        self.navigation_profile_name_var.set(profile.name)
        self.navigation_orbit_button_var.set(profile.orbit_button)
        self.navigation_orbit_modifier_var.set(
            " + ".join(profile.orbit_modifiers) if profile.orbit_modifiers else "none"
        )
        self.navigation_pan_button_var.set(profile.pan_button)
        self.navigation_pan_modifier_var.set(
            " + ".join(profile.pan_modifiers) if profile.pan_modifiers else "none"
        )
        self.navigation_zoom_direction_var.set(profile.zoom_direction_label)

    def _save_input_profile(self) -> None:
        """Validate and persist a named mapping without application detection."""

        name = self.navigation_profile_name_var.get().strip()
        try:
            profile = get_profile(
                self.navigation_profile_var.get(),
                self.settings.navigation_profiles,
            )
            edited = InputProfile.from_dict(
                name,
                {
                    **profile.to_dict(),
                    "orbit_button": self.navigation_orbit_button_var.get(),
                    "orbit_modifiers": self.navigation_orbit_modifier_var.get(),
                    "pan_button": self.navigation_pan_button_var.get(),
                    "pan_modifiers": self.navigation_pan_modifier_var.get(),
                    "zoom_in_direction": self.navigation_zoom_direction_var.get(),
                },
            )
            profiles = dict(self.settings.navigation_profiles)
            profiles[name] = edited.to_dict()
            settings = replace(
                self.settings,
                navigation_profile=name,
                navigation_profiles=profiles,
            )
            settings.validate()
            self.settings = settings
            self.store.save(settings)
            self.navigation_profile_var.set(name)
            self.navigation_profile_combo.configure(values=list(profile_names(profiles)))
            self.detail_var.set(f"Input profile {name!r} saved locally.")
            if self.worker.is_running():
                self.worker.update_settings(settings)
        except (SettingsError, ValueError, OSError) as exc:
            messagebox.showerror("Invalid input profile", str(exc), parent=self.root)

    def _navigation_combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
    ) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, style="Panel.TLabel").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=variable,
            values=list(values),
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(12, 0))

    def _entry_control(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, style="Panel.TLabel").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=18).pack(
            side=tk.RIGHT, fill=tk.X, expand=True, padx=(12, 0)
        )

    def _set_navigation_button_text(self) -> None:
        self.navigation_button_var.set(
            "Disable 3D Navigation"
            if self.navigation_enabled_var.get()
            else "Enable 3D Navigation"
        )

    def toggle_navigation(self) -> None:
        previous = bool(self.navigation_enabled_var.get())
        self.navigation_enabled_var.set(not previous)
        if self.navigation_enabled_var.get():
            self.air_mouse_var.set(False)
            self._set_air_mouse_button_text()
        if not self._apply_settings():
            self.navigation_enabled_var.set(previous)
            self._set_navigation_button_text()

    def calibrate_navigation(self) -> None:
        if not self.worker.is_running():
            messagebox.showinfo(
                "Start detection first",
                "Start Detection, enable 3D Navigation, then press Calibrate.",
                parent=self.root,
            )
            return
        if not self.navigation_enabled_var.get():
            messagebox.showinfo(
                "Enable 3D navigation",
                "Enable 3D Navigation before calibrating the neutral hand position.",
                parent=self.root,
            )
            return
        messagebox.showinfo(
            "3D navigation calibration",
            "Place both hands in a comfortable neutral position and hold them still.\n\n"
            "The next few frames are recorded locally as the neutral reference. "
            "You do not need to remain perfectly still after calibration.",
            parent=self.root,
        )
        self.worker.begin_navigation_calibration()
        self.navigation_status_var.set("CALIBRATING")
        self.navigation_gesture_var.set("Calibration")

    def _set_air_mouse_button_text(self) -> None:
        self.air_mouse_button_var.set(
            "Air Mouse: ON" if self.air_mouse_var.get() else "Air Mouse: OFF"
        )

    def toggle_air_mouse(self) -> None:
        """Toggle pointer control and persist the choice immediately."""

        previous = bool(self.air_mouse_var.get())
        self.air_mouse_var.set(not previous)
        if not self._apply_settings():
            self.air_mouse_var.set(previous)
            self._set_air_mouse_button_text()

    def _shortcut_control(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, style="Panel.TLabel").pack(side=tk.LEFT)
        values = list(SHORTCUT_OPTIONS)
        current = variable.get().strip()
        if current and current not in values:
            values.insert(0, current)
        ttk.Combobox(
            row,
            textvariable=variable,
            values=values,
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(12, 0))

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
        format_value: Callable[[float], str] | None = None,
        command: Callable[[], None] | None = None,
    ) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=(7, 0))
        if format_value is None:
            value_label: ttk.Label = ttk.Label(
                row, textvariable=variable, style="Muted.TLabel", width=7, anchor="e"
            )
        else:
            value_label = ttk.Label(row, style="Muted.TLabel", width=7, anchor="e")

            def update_value(*_: Any) -> None:
                value_label.configure(text=format_value(float(variable.get())))

            variable.trace_add("write", update_value)
            update_value()
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
            command=lambda _value: command() if command else None,
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
        profile_name = self.navigation_profile_var.get().strip()
        selected_profile = get_profile(profile_name, self.settings.navigation_profiles)
        edited_profile = InputProfile.from_dict(
            profile_name,
            {
                **selected_profile.to_dict(),
                "orbit_button": self.navigation_orbit_button_var.get(),
                "orbit_modifiers": self.navigation_orbit_modifier_var.get(),
                "pan_button": self.navigation_pan_button_var.get(),
                "pan_modifiers": self.navigation_pan_modifier_var.get(),
                "zoom_in_direction": self.navigation_zoom_direction_var.get(),
            },
        )
        profiles = dict(self.settings.navigation_profiles)
        profiles[profile_name] = edited_profile.to_dict()
        navigation_enabled = bool(self.navigation_enabled_var.get())
        settings = replace(
            self.settings,
            camera_index=self._selected_camera_index(),
            detection_enabled=bool(self.detection_var.get()),
            preview_visible=bool(self.preview_var.get()),
            debug_mode=bool(self.debug_var.get()),
            start_with_windows=bool(self.startup_var.get()),
            shortcut=self.shortcut_var.get().strip(),
            pinch_shortcut=self.pinch_shortcut_var.get().strip(),
            # Air Mouse and 3D Navigation are mutually exclusive control modes.
            air_mouse_enabled=bool(self.air_mouse_var.get()) and not navigation_enabled,
            touch_threshold=round(float(self.threshold_var.get()), 3),
            touch_duration_ms=int(round(float(self.duration_var.get()))),
            cooldown_ms=int(round(float(self.cooldown_var.get()))),
            navigation_enabled=navigation_enabled,
            navigation_profile=profile_name,
            navigation_profiles=profiles,
            navigation_control_mode=self.navigation_control_mode_var.get().strip(),
            navigation_activation_gesture=self.navigation_activation_var.get().strip(),
            navigation_deactivation_gesture=self.navigation_deactivation_var.get().strip(),
            navigation_pan_gesture=self.navigation_pan_gesture_var.get().strip(),
            navigation_orbit_sensitivity=round(float(self.navigation_orbit_var.get()), 2),
            navigation_pan_sensitivity=round(float(self.navigation_pan_var.get()), 2),
            navigation_zoom_sensitivity=round(float(self.navigation_zoom_var.get()), 2),
            navigation_roll_sensitivity=round(float(self.navigation_roll_var.get()), 2),
            navigation_smoothing_frames=int(round(float(self.navigation_smoothing_var.get()))),
            navigation_dead_zone=round(float(self.navigation_dead_zone_var.get()), 3),
            navigation_max_speed=round(float(self.navigation_max_speed_var.get()), 2),
            navigation_acceleration=round(float(self.navigation_acceleration_var.get()), 2),
            navigation_mouse_scale=round(float(self.navigation_mouse_scale_var.get()), 1),
            navigation_zoom_wheel_scale=round(
                float(self.navigation_zoom_wheel_scale_var.get()), 2
            ),
            navigation_min_confidence=round(
                float(self.navigation_confidence_threshold_var.get()), 2
            ),
            navigation_activation_hold_ms=int(
                round(float(self.navigation_activation_hold_var.get()))
            ),
            navigation_invert_x=bool(self.navigation_invert_x_var.get()),
            navigation_invert_y=bool(self.navigation_invert_y_var.get()),
            navigation_invert_zoom=bool(self.navigation_invert_zoom_var.get()),
            navigation_roll_enabled=bool(self.navigation_roll_enabled_var.get()),
            navigation_global_hotkey=self.navigation_global_hotkey_var.get().strip(),
            navigation_emergency_hotkey=self.navigation_emergency_hotkey_var.get().strip(),
        )
        settings.validate()
        parse_shortcut(settings.shortcut)
        parse_shortcut(settings.pinch_shortcut)
        return settings

    def _apply_settings_live(self) -> None:
        """Apply slider changes immediately so the zone preview updates live."""

        self._apply_settings(show_errors=False)

    def _apply_settings(self, show_errors: bool = True) -> bool:
        try:
            if self.navigation_enabled_var.get():
                self.air_mouse_var.set(False)
            settings = self._collect_settings()
            if settings.start_with_windows != self.settings.start_with_windows:
                set_start_with_windows(settings.start_with_windows)
            self.store.save(settings)
        except (SettingsError, ShortcutError, StartupError, OSError, ValueError) as exc:
            if show_errors:
                messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            return False

        self.settings = settings
        self._set_air_mouse_button_text()
        self._set_navigation_button_text()
        if hasattr(self, "navigation_profile_combo"):
            self.navigation_profile_combo.configure(
                values=list(profile_names(settings.navigation_profiles))
            )
        if self.worker.is_running():
            self.worker.update_settings(settings)
        self.detail_var.set("Settings saved locally.")
        if not settings.preview_visible:
            self._show_preview_placeholder("Camera preview hidden")
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
        self.detail_var.set("Camera released. No simulated 3D input remains held.")
        self._show_preview_placeholder("Camera stopped")
        self.hand_var.set("Not detected")
        self.face_var.set("Not detected")
        self.distance_var.set("—")
        self.fps_var.set("0")
        self.cooldown_status_var.set("Released")
        self.pinch_var.set("Open")
        self.scroll_var.set("Off")
        self.navigation_status_var.set("DISABLED")
        self.navigation_hands_var.set("0")
        self.navigation_mode_status_var.set(self.navigation_control_mode_var.get())
        self.navigation_gesture_var.set("Idle")
        self.navigation_distance_var.set("—")
        self.navigation_angle_var.set("—")
        self.navigation_vector_var.set("0.000, 0.000")
        self.navigation_confidence_var.set("0%")

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
        self.pinch_var.set(
            f"{snapshot.pinch_cooldown_remaining_ms} ms"
            if snapshot.pinch_cooldown_remaining_ms > 0
            else (
                "Closed"
                if snapshot.pinch_detected
                else ("Release" if snapshot.pinch_awaiting_release else "Open")
            )
        )
        self.scroll_var.set(
            "Active" if snapshot.scroll_active else "Off"
        )
        self._update_navigation(result)
        self._update_debug(result)

        if result.preview_frame is not None and self.preview_var.get():
            self._show_frame(result.preview_frame)
        elif not self.preview_var.get():
            self._show_preview_placeholder("Camera preview hidden")

    def _update_navigation(self, result: FrameResult) -> None:
        navigation = result.navigation
        input_status = result.input_status
        if navigation is None:
            self.navigation_status_var.set("DISABLED")
            return

        if not navigation.enabled:
            status = "DISABLED"
        elif navigation.calibration_active:
            status = "CALIBRATING"
        elif input_status is not None and not input_status.global_enabled:
            status = "DISABLED"
        elif navigation.active:
            status = "ACTIVE"
        else:
            status = "READY"
        self.navigation_status_var.set(status)
        if input_status is not None and input_status.message:
            self.detail_var.set(input_status.message)

        self.navigation_hands_var.set(str(navigation.hand_count))
        self.navigation_mode_status_var.set(navigation.control_mode)
        self.navigation_gesture_var.set(navigation.gesture)
        self.navigation_distance_var.set(
            f"{navigation.distance:.3f}" if navigation.distance is not None else "—"
        )
        self.navigation_angle_var.set(
            f"{math.degrees(navigation.angle):.1f}°"
            if navigation.angle is not None
            else "—"
        )
        self.navigation_vector_var.set(
            f"{navigation.midpoint_delta_x:+.3f}, {navigation.midpoint_delta_y:+.3f}"
        )
        self.navigation_confidence_var.set(f"{navigation.confidence * 100:.0f}%")
        if input_status is not None:
            self.navigation_mouse_var.set(
                ", ".join(input_status.held_buttons) if input_status.held_buttons else "Released"
            )
            self.navigation_modifiers_var.set(
                ", ".join(input_status.held_modifiers)
                if input_status.held_modifiers
                else "None"
            )
        if navigation.calibration_completed:
            self.detail_var.set(navigation.message)
        elif navigation.message and navigation.state.value in {"ACTIVATING", "LOST"}:
            self.detail_var.set(navigation.message)

    def _update_debug(self, result: FrameResult) -> None:
        snapshot = result.snapshot
        fingertip = _point_text(snapshot.index_tip)
        nose = _point_text(snapshot.nose)
        distance = (
            f"{snapshot.relative_distance:.4f}" if snapshot.relative_distance is not None else "--"
        )
        pinch_distance = (
            f"{snapshot.pinch_distance:.3f}"
            if snapshot.pinch_distance is not None
            else "--"
        )
        finger_separation = (
            f"{snapshot.finger_separation:.3f}"
            if snapshot.finger_separation is not None
            else "--"
        )
        scroll_delta = f"{snapshot.scroll_delta_x:.4f}, {snapshot.scroll_delta_y:.4f}"
        navigation = result.navigation
        input_status = result.input_status
        navigation_lines = (
            f"Navigation status   {self.navigation_status_var.get()}",
            f"Input state         {input_status.state.value if input_status else '--'}",
            f"Global input        {'ON' if input_status and input_status.global_enabled else 'OFF'}",
            f"Navigation hands    {navigation.hand_count}"
            if navigation
            else "Navigation hands    --",
            f"Navigation gesture  {navigation.gesture}"
            if navigation
            else "Navigation gesture  --",
            f"Navigation pose     {navigation.pose}"
            if navigation
            else "Navigation pose     --",
            f"Navigation distance {navigation.distance:.4f}"
            if navigation and navigation.distance is not None
            else "Navigation distance --",
            f"Distance delta      {navigation.distance_delta:+.4f}"
            if navigation
            else "Distance delta      --",
            f"Navigation angle    {math.degrees(navigation.angle):.2f}"
            if navigation and navigation.angle is not None
            else "Navigation angle    --",
            f"Angle delta         {navigation.angle_delta:+.4f}"
            if navigation
            else "Angle delta         --",
            f"Orbit vector        {navigation.orbit_x:+.4f}, {navigation.orbit_y:+.4f}"
            if navigation
            else "Orbit vector        --",
            f"Pan vector          {navigation.pan_x:+.4f}, {navigation.pan_y:+.4f}"
            if navigation
            else "Pan vector          --",
            f"Zoom / roll         {navigation.zoom:+.4f}, {navigation.roll:+.4f}"
            if navigation
            else "Zoom / roll         --",
            f"Mouse buttons       {', '.join(input_status.held_buttons) if input_status and input_status.held_buttons else 'NONE'}",
            f"Modifiers           {', '.join(input_status.held_modifiers) if input_status and input_status.held_modifiers else 'NONE'}",
        )
        text = "\n".join(
            (
                f"FPS                 {result.fps:.1f}",
                f"Hand detected       {'yes' if snapshot.hand_detected else 'no'} ({snapshot.hand_count})",
                f"Face detected       {'yes' if snapshot.face_detected else 'no'}",
                f"Index fingertip     {fingertip}",
                f"Nose                {nose}",
                f"Relative distance   {distance}",
                f"State               {snapshot.state.value}",
                f"Cooldown            {snapshot.cooldown_remaining_ms} ms",
                f"Awaiting release    {'yes' if snapshot.awaiting_release else 'no'}",
                f"Pinch distance      {pinch_distance}",
                f"Pinch cooldown      {snapshot.pinch_cooldown_remaining_ms} ms",
                f"Pinch detected      {'yes' if snapshot.pinch_detected else 'no'}",
                f"Pinch release       {'yes' if snapshot.pinch_awaiting_release else 'no'}",
                f"Finger separation   {finger_separation}",
                f"Scroll active       {'yes' if snapshot.scroll_active else 'no'}",
                f"Scroll delta        {scroll_delta}",
                *navigation_lines,
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
