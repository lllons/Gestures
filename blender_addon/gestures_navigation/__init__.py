"""Gestures two-hand navigation add-on for Blender 3.0+.

Install this package as a normal Blender add-on. It listens only on
127.0.0.1:8765, receives continuous per-frame navigation values from the
Gestures application, and applies them on Blender's main thread through a
short timer. No keyboard or mouse automation is used.
"""

from __future__ import annotations

bl_info = {
    "name": "Gestures 3D Navigation",
    "author": "Gestures contributors",
    "version": (1, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Gestures",
    "description": "Navigate a Blender viewport or camera with two webcam-tracked hands",
    "category": "3D View",
}

import json
import math
import queue
import socket
import threading
import time
from typing import Any

import bpy
from mathutils import Quaternion, Vector


_DEFAULT_PORT = 8765
_PACKET_TIMEOUT_SECONDS = 0.45
_LOCAL_HOSTS = {"127.0.0.1", "localhost"}
_server: "NavigationReceiver | None" = None
_timer_registered = False
_last_packet_at = 0.0
_last_status_message = "Waiting for Gestures app"


class NavigationReceiver:
    """Receive UDP packets away from Blender's main thread."""

    def __init__(self, port: int) -> None:
        self.port = int(port)
        self._socket: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._packets: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=4)
        self.error = ""

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.running:
            return True
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(("127.0.0.1", self.port))
            self._socket.settimeout(0.25)
        except OSError as exc:
            self.error = f"Could not bind 127.0.0.1:{self.port}: {exc}"
            self._close_socket()
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._receive_loop,
            name="gestures-blender-udp",
            daemon=True,
        )
        self._thread.start()
        self.error = ""
        return True

    def stop(self) -> None:
        self._stop.set()
        self._close_socket()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

    def pop_latest(self) -> dict[str, Any] | None:
        latest = None
        while True:
            try:
                latest = self._packets.get_nowait()
            except queue.Empty:
                return latest

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            if self._socket is None:
                return
            try:
                raw, address = self._socket.recvfrom(16384)
            except socket.timeout:
                continue
            except OSError:
                return
            if address[0] not in _LOCAL_HOSTS:
                continue
            try:
                packet = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(packet, dict) or packet.get("type") != "gestures_navigation":
                continue
            try:
                self._packets.put_nowait(packet)
            except queue.Full:
                try:
                    self._packets.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._packets.put_nowait(packet)
                except queue.Full:
                    pass
            reply_port = packet.get("reply_port")
            if isinstance(reply_port, int) and 1 <= reply_port <= 65535:
                self._send_status(address[0], reply_port, packet)

    def _send_status(self, host: str, port: int, packet: dict[str, Any]) -> None:
        if self._socket is None:
            return
        status = {
            "type": "gestures_navigation_status",
            "connected": True,
            "mode": str(packet.get("mode", "Viewport")),
            "message": "Blender add-on connected",
            "timestamp": time.time(),
        }
        try:
            self._socket.sendto(
                json.dumps(status, separators=(",", ":")).encode("utf-8"),
                (host, port),
            )
        except OSError:
            pass

    def _close_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None


def _timer_tick() -> float | None:
    """Apply the newest packet on Blender's main thread."""

    global _last_packet_at, _last_status_message
    if _server is None or not _server.running:
        _update_scene_status()
        return 0.25

    packet = _server.pop_latest()
    if packet is not None:
        _last_packet_at = time.monotonic()
        scene = bpy.context.scene
        if packet.get("active") and scene.gestures_nav_enabled:
            _last_status_message = _apply_packet(scene, packet)
        elif not scene.gestures_nav_enabled:
            _last_status_message = "Blender navigation is disabled"
        else:
            _last_status_message = str(packet.get("message", "Navigation idle"))
    elif _last_packet_at and time.monotonic() - _last_packet_at > _PACKET_TIMEOUT_SECONDS:
        _last_status_message = "Gestures packet timeout; movement stopped"

    _update_scene_status()
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    return 0.01


def _update_scene_status() -> None:
    scene = bpy.context.scene
    if _server is None:
        scene.gestures_nav_connection_status = "STOPPED"
    elif _server.error:
        scene.gestures_nav_connection_status = _server.error
    elif _last_packet_at and time.monotonic() - _last_packet_at <= _PACKET_TIMEOUT_SECONDS:
        scene.gestures_nav_connection_status = "CONNECTED"
    else:
        scene.gestures_nav_connection_status = "WAITING"
    scene.gestures_nav_last_message = _last_status_message


def _finite_value(packet: dict[str, Any], key: str) -> float:
    try:
        value = float(packet.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _apply_packet(scene: Any, packet: dict[str, Any]) -> str:
    """Apply one complete analog packet and return a user-facing status."""

    mode = scene.gestures_nav_mode
    if scene.gestures_nav_follow_app:
        requested_mode = packet.get("mode")
        if requested_mode in {"Viewport", "Camera"}:
            mode = requested_mode

    orbit_x = _finite_value(packet, "orbit_x") * scene.gestures_nav_orbit_sensitivity
    orbit_y = _finite_value(packet, "orbit_y") * scene.gestures_nav_orbit_sensitivity
    pan_x = _finite_value(packet, "pan_x") * scene.gestures_nav_pan_sensitivity
    pan_y = _finite_value(packet, "pan_y") * scene.gestures_nav_pan_sensitivity
    zoom = _finite_value(packet, "zoom") * scene.gestures_nav_zoom_sensitivity
    roll = _finite_value(packet, "roll") * scene.gestures_nav_roll_sensitivity

    if mode == "Camera":
        return _apply_camera(scene, orbit_x, orbit_y, pan_x, pan_y, zoom, roll)
    return _apply_viewport(orbit_x, orbit_y, pan_x, pan_y, zoom, roll)


def _view3d_regions() -> list[Any]:
    regions = []
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region_3d = getattr(area.spaces.active, "region_3d", None)
                if region_3d is not None:
                    regions.append(region_3d)
    return regions


def _apply_viewport(
    orbit_x: float,
    orbit_y: float,
    pan_x: float,
    pan_y: float,
    zoom: float,
    roll: float,
) -> str:
    """Mutate RegionView3D directly instead of faking mouse or key input."""

    regions = _view3d_regions()
    if not regions:
        return "No 3D viewport found"

    for region_3d in regions:
        rotation = region_3d.view_rotation.copy()
        if orbit_x:
            rotation = Quaternion((0.0, 0.0, 1.0), -orbit_x) @ rotation
        if orbit_y:
            view_right = rotation @ Vector((1.0, 0.0, 0.0))
            rotation = Quaternion(view_right, orbit_y) @ rotation
        if roll:
            view_forward = rotation @ Vector((0.0, 0.0, -1.0))
            rotation = Quaternion(view_forward, roll) @ rotation
        region_3d.view_rotation = rotation.normalized()

        if pan_x or pan_y:
            view_right = rotation @ Vector((1.0, 0.0, 0.0))
            view_up = rotation @ Vector((0.0, 1.0, 0.0))
            pan_scale = max(0.001, float(region_3d.view_distance)) * 0.45
            region_3d.view_location += (-view_right * pan_x + view_up * pan_y) * pan_scale

        if zoom:
            region_3d.view_distance = max(
                0.001,
                min(1_000_000.0, region_3d.view_distance * math.exp(-zoom)),
            )

    return "Viewport navigation active"


def _selected_camera(scene: Any) -> Any | None:
    active = getattr(bpy.context.view_layer.objects, "active", None)
    if active is not None and getattr(active, "type", None) == "CAMERA":
        return active
    camera = scene.camera
    return camera if camera is not None and camera.type == "CAMERA" else None


def _apply_camera(
    scene: Any,
    orbit_x: float,
    orbit_y: float,
    pan_x: float,
    pan_y: float,
    zoom: float,
    roll: float,
) -> str:
    """Apply analog values to the selected camera's actual transform."""

    camera = _selected_camera(scene)
    if camera is None:
        return "Camera mode needs a selected or active scene camera"

    rotation = camera.matrix_world.to_quaternion()
    if orbit_x:
        rotation = Quaternion((0.0, 0.0, 1.0), -orbit_x) @ rotation
    if orbit_y:
        camera_right = rotation @ Vector((1.0, 0.0, 0.0))
        rotation = Quaternion(camera_right, orbit_y) @ rotation
    if roll:
        camera_forward = rotation @ Vector((0.0, 0.0, -1.0))
        rotation = Quaternion(camera_forward, roll) @ rotation
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = rotation.normalized()

    if pan_x or pan_y:
        camera_right = rotation @ Vector((1.0, 0.0, 0.0))
        camera_up = rotation @ Vector((0.0, 1.0, 0.0))
        camera.location += (-camera_right * pan_x + camera_up * pan_y) * 0.25
    if zoom:
        camera.location += (rotation @ Vector((0.0, 0.0, -1.0))) * (zoom * 0.35)
    return f"Camera navigation active: {camera.name}"


class GESTURES_OT_toggle_navigation(bpy.types.Operator):
    bl_idname = "gestures.toggle_navigation"
    bl_label = "Toggle 3D Navigation"
    bl_description = "Allow active Gestures packets to move Blender"

    def execute(self, context: Any) -> set[str]:
        context.scene.gestures_nav_enabled = not context.scene.gestures_nav_enabled
        return {"FINISHED"}


class GESTURES_OT_restart_receiver(bpy.types.Operator):
    bl_idname = "gestures.restart_receiver"
    bl_label = "Restart Local Receiver"
    bl_description = "Restart the localhost UDP listener on the configured port"

    def execute(self, context: Any) -> set[str]:
        global _server, _last_packet_at, _last_status_message
        if _server is not None:
            _server.stop()
        _server = NavigationReceiver(context.scene.gestures_nav_port)
        _server.start()
        _last_packet_at = 0.0
        _last_status_message = "Waiting for Gestures app"
        return {"FINISHED"}


class GESTURES_PT_navigation(bpy.types.Panel):
    bl_label = "Gestures 3D Navigation"
    bl_idname = "GESTURES_PT_navigation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Gestures"

    def draw(self, context: Any) -> None:
        layout = self.layout
        scene = context.scene
        row = layout.row(align=True)
        row.operator(
            "gestures.toggle_navigation",
            text="Disable Navigation" if scene.gestures_nav_enabled else "Enable Navigation",
            icon="PAUSE" if scene.gestures_nav_enabled else "PLAY",
        )
        layout.label(text=f"Receiver: {scene.gestures_nav_connection_status}")
        layout.label(text=scene.gestures_nav_last_message)
        layout.prop(scene, "gestures_nav_follow_app")
        layout.prop(scene, "gestures_nav_mode")
        layout.prop(scene, "gestures_nav_orbit_sensitivity")
        layout.prop(scene, "gestures_nav_pan_sensitivity")
        layout.prop(scene, "gestures_nav_zoom_sensitivity")
        layout.prop(scene, "gestures_nav_roll_sensitivity")
        layout.prop(scene, "gestures_nav_port")
        layout.operator("gestures.restart_receiver", icon="FILE_REFRESH")
        layout.label(text="Gestures sends only localhost UDP packets.")


_CLASSES = (
    GESTURES_OT_toggle_navigation,
    GESTURES_OT_restart_receiver,
    GESTURES_PT_navigation,
)


def register() -> None:
    global _server, _timer_registered
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.gestures_nav_enabled = bpy.props.BoolProperty(
        name="Enable Navigation",
        description="Permit active Gestures packets to control Blender",
        default=False,
    )
    bpy.types.Scene.gestures_nav_follow_app = bpy.props.BoolProperty(
        name="Follow Gestures mode",
        description="Use the Viewport/Camera mode selected in the Gestures app",
        default=True,
    )
    bpy.types.Scene.gestures_nav_mode = bpy.props.EnumProperty(
        name="Target",
        items=(
            ("Viewport", "Viewport", "Navigate the current 3D viewport"),
            ("Camera", "Camera", "Move the selected or active scene camera"),
        ),
        default="Viewport",
    )
    bpy.types.Scene.gestures_nav_orbit_sensitivity = bpy.props.FloatProperty(
        name="Orbit sensitivity",
        min=0.1,
        max=5.0,
        default=1.0,
    )
    bpy.types.Scene.gestures_nav_pan_sensitivity = bpy.props.FloatProperty(
        name="Pan sensitivity",
        min=0.1,
        max=5.0,
        default=1.0,
    )
    bpy.types.Scene.gestures_nav_zoom_sensitivity = bpy.props.FloatProperty(
        name="Zoom sensitivity",
        min=0.1,
        max=5.0,
        default=1.0,
    )
    bpy.types.Scene.gestures_nav_roll_sensitivity = bpy.props.FloatProperty(
        name="Roll sensitivity",
        min=0.1,
        max=5.0,
        default=1.0,
    )
    bpy.types.Scene.gestures_nav_port = bpy.props.IntProperty(
        name="UDP port",
        min=1,
        max=65535,
        default=_DEFAULT_PORT,
    )
    bpy.types.Scene.gestures_nav_connection_status = bpy.props.StringProperty(
        name="Connection status",
        default="WAITING",
    )
    bpy.types.Scene.gestures_nav_last_message = bpy.props.StringProperty(
        name="Last message",
        default="Waiting for Gestures app",
    )

    _server = NavigationReceiver(_DEFAULT_PORT)
    _server.start()
    if not _timer_registered:
        bpy.app.timers.register(_timer_tick, first_interval=0.01, persistent=True)
        _timer_registered = True


def unregister() -> None:
    global _server, _timer_registered
    if _timer_registered and bpy.app.timers.is_registered(_timer_tick):
        bpy.app.timers.unregister(_timer_tick)
    _timer_registered = False
    if _server is not None:
        _server.stop()
        _server = None
    for property_name in (
        "gestures_nav_enabled",
        "gestures_nav_follow_app",
        "gestures_nav_mode",
        "gestures_nav_orbit_sensitivity",
        "gestures_nav_pan_sensitivity",
        "gestures_nav_zoom_sensitivity",
        "gestures_nav_roll_sensitivity",
        "gestures_nav_port",
        "gestures_nav_connection_status",
        "gestures_nav_last_message",
    ):
        if hasattr(bpy.types.Scene, property_name):
            delattr(bpy.types.Scene, property_name)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
