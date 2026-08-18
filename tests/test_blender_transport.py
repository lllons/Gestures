from __future__ import annotations

import json
import socket
import unittest

from app.blender_transport import BlenderTransport
from app.navigation import TwoHandNavigation
from app.settings import AppSettings


class BlenderTransportTests(unittest.TestCase):
    def test_sends_neutral_packet_and_accepts_local_ack(self) -> None:
        command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_socket.bind(("127.0.0.1", 0))
        command_socket.settimeout(1.0)
        reply_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reply_socket.bind(("127.0.0.1", 0))
        command_port = command_socket.getsockname()[1]
        reply_port = reply_socket.getsockname()[1]
        reply_socket.close()
        transport = BlenderTransport("127.0.0.1", command_port, reply_port)

        try:
            snapshot = TwoHandNavigation(AppSettings()).process([], now=1.0)
            status = transport.send(snapshot)
            self.assertFalse(status.connected)

            raw_packet, source = command_socket.recvfrom(8192)
            packet = json.loads(raw_packet.decode("utf-8"))
            self.assertEqual(packet["type"], "gestures_navigation")
            self.assertFalse(packet["active"])
            self.assertEqual(packet["hands"], 0)
            self.assertEqual(packet["reply_port"], reply_port)

            acknowledgement = {
                "type": "gestures_navigation_status",
                "connected": True,
                "mode": "Viewport",
                "message": "Blender add-on connected",
            }
            command_socket.sendto(
                json.dumps(acknowledgement).encode("utf-8"),
                ("127.0.0.1", reply_port),
            )
            connected = transport.status
            self.assertTrue(connected.connected)
            self.assertEqual(connected.mode, "Viewport")
            self.assertEqual(source[0], "127.0.0.1")
        finally:
            transport.close()
            command_socket.close()
            if reply_socket.fileno() >= 0:
                reply_socket.close()


if __name__ == "__main__":
    unittest.main()
