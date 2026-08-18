"""Local UDP transport between Gestures and the Blender add-on.

The transport is intentionally localhost-oriented and nonblocking. A missing
Blender listener never stalls MediaPipe or the Tkinter UI; it only changes the
connection indicator. Every frame is a complete state packet, so stopping or
losing hands can send an explicit neutral packet.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Any

from .navigation import NavigationSnapshot


@dataclass(frozen=True)
class BlenderStatus:
    connected: bool = False
    enabled: bool = False
    active: bool = False
    message: str = "Blender add-on not detected"
    last_seen: float = 0.0
    mode: str = "Viewport"


class BlenderTransport:
    """Send navigation packets over UDP and listen for add-on acknowledgements."""

    STATUS_TIMEOUT_SECONDS = 2.5

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        reply_port: int = 8766,
    ) -> None:
        self._host = host
        self._port = port
        self._reply_port = reply_port
        self._send_socket: socket.socket | None = None
        self._reply_socket: socket.socket | None = None
        self._status = BlenderStatus()
        self._last_error = ""
        self._sequence = 0
        self.configure(host, port, reply_port)

    @property
    def status(self) -> BlenderStatus:
        self._poll_status()
        if (
            self._status.connected
            and time.monotonic() - self._status.last_seen > self.STATUS_TIMEOUT_SECONDS
        ):
            self._status = BlenderStatus(
                connected=False,
                enabled=self._status.enabled,
                active=False,
                message="Blender add-on heartbeat timed out",
                mode=self._status.mode,
            )
        return self._status

    @property
    def last_error(self) -> str:
        return self._last_error

    def configure(self, host: str, port: int, reply_port: int) -> None:
        """Recreate sockets only when connection settings change."""

        changed = (host, port, reply_port) != (self._host, self._port, self._reply_port)
        self._host = host
        self._port = port
        self._reply_port = reply_port
        if self._send_socket is not None and not changed:
            return

        self._close_sockets()
        self._last_error = ""
        try:
            send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            send_socket.setblocking(False)
            self._send_socket = send_socket
        except OSError as exc:
            self._send_socket = None
            self._last_error = f"Blender sender unavailable: {exc}"

        try:
            reply_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            reply_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            reply_socket.bind(("127.0.0.1", reply_port))
            reply_socket.setblocking(False)
            self._reply_socket = reply_socket
        except OSError as exc:
            self._reply_socket = None
            self._last_error = f"Blender reply port {reply_port} unavailable: {exc}"

        message = self._last_error or "Waiting for Blender add-on"
        self._status = BlenderStatus(message=message)

    def send(self, snapshot: NavigationSnapshot) -> BlenderStatus:
        """Send the newest frame and return the current add-on status."""

        self._sequence += 1
        payload = snapshot.to_payload()
        payload.update(
            {
                "timestamp": time.time(),
                "sequence": self._sequence,
                "reply_port": self._reply_port,
                "source": "gestures",
            }
        )
        self._send_payload(payload)
        return self.status

    def send_stop(self) -> BlenderStatus:
        """Send one explicit neutral packet before the worker shuts down."""

        self._sequence += 1
        payload = {
            "type": "gestures_navigation",
            "version": 1,
            "state": "LOST",
            "enabled": False,
            "active": False,
            "hands": 0,
            "mode": "Viewport",
            "control_mode": "FULL 3D",
            "orbit_x": 0.0,
            "orbit_y": 0.0,
            "pan_x": 0.0,
            "pan_y": 0.0,
            "zoom": 0.0,
            "roll": 0.0,
            "confidence": 0.0,
            "gesture": "Idle",
            "message": "Gestures stopped",
            "timestamp": time.time(),
            "sequence": self._sequence,
            "reply_port": self._reply_port,
            "source": "gestures",
        }
        self._send_payload(payload)
        return self.status

    def close(self) -> None:
        self._close_sockets()
        self._status = BlenderStatus(message="Blender transport closed")

    def _send_payload(self, payload: dict[str, Any]) -> None:
        self._poll_status()
        if self._send_socket is None:
            return
        try:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._send_socket.sendto(encoded, (self._host, self._port))
        except (OSError, TypeError, ValueError) as exc:
            self._last_error = f"Blender connection unavailable: {exc}"
            self._status = BlenderStatus(
                connected=False,
                enabled=self._status.enabled,
                active=False,
                message=self._last_error,
                mode=self._status.mode,
            )

    def _poll_status(self) -> None:
        if self._reply_socket is None:
            return
        while True:
            try:
                raw, _address = self._reply_socket.recvfrom(8192)
            except BlockingIOError:
                return
            except OSError as exc:
                self._last_error = f"Blender status listener stopped: {exc}"
                return
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("type") != "gestures_navigation_status":
                continue
            self._status = BlenderStatus(
                connected=bool(payload.get("connected", True)),
                enabled=bool(payload.get("enabled", False)),
                active=bool(payload.get("active", False)),
                message=str(payload.get("message", "Blender add-on connected")),
                last_seen=time.monotonic(),
                mode=str(payload.get("mode", "Viewport")),
            )

    def _close_sockets(self) -> None:
        for name in ("_send_socket", "_reply_socket"):
            transport_socket = getattr(self, name)
            if transport_socket is not None:
                try:
                    transport_socket.close()
                except OSError:
                    pass
                setattr(self, name, None)
