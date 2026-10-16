"""Serve a MinesweeperScreen to a browser so people can play or watch."""

from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .screen import MinesweeperScreen

PAGE = b"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Minesweeper</title>
<style>
  body { background: #808080; display: flex; justify-content: center; padding-top: 40px; }
  img { border: 3px outset #fff; cursor: default; user-select: none; }
</style>
</head>
<body>
<img id="screen" src="/screen.png" draggable="false">
<script>
  const screen = document.getElementById("screen");
  const refresh = () => { screen.src = "/screen.png?t=" + Date.now(); };

  // Send pixel clicks to the game; the image is shown at its native size
  screen.addEventListener("contextmenu", (event) => event.preventDefault());
  screen.addEventListener("mousedown", async (event) => {
    const button = event.button === 2 ? "right" : "left";
    await fetch("/click", {
      method: "POST",
      body: JSON.stringify({x: event.offsetX, y: event.offsetY, button}),
    });
    refresh();
  });

  // Poll so agent moves appear
  setInterval(refresh, 500);
</script>
</body>
</html>
"""


def make_server(screen: MinesweeperScreen, port: int) -> ThreadingHTTPServer:
    """Build an HTTP server for the screen on localhost."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self.reply(PAGE, "text/html")
            elif self.path.startswith("/screen.png"):
                buffer = io.BytesIO()
                screen.screenshot().save(buffer, format="PNG")
                self.reply(buffer.getvalue(), "image/png")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/click":
                self.send_error(404)
                return
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            screen.click(int(data["x"]), int(data["y"]), data["button"])
            self.reply(b"{}", "application/json")

        def reply(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
