"""Dev server: serves webapp + artifacts for local development."""

import http.server
import mimetypes
import os
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/javascript", ".mjs")


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = urllib.parse.unquote(path.split("?")[0].split("#")[0])
        path = os.path.normpath(path)
        # Normalize to forward slashes on Windows
        path = path.replace("\\", "/")
        if path.startswith("/artifacts/"):
            rel = path[len("/artifacts/"):]
            return str(ROOT / "hf_repo" / rel)
        if path == "/" or path == "":
            return str(ROOT / "webapp" / "index.html")
        rel = path.lstrip("/")
        return str(ROOT / "webapp" / rel)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


if __name__ == "__main__":
    server = http.server.HTTPServer(("localhost", 8080), Handler)
    print("Dev server: http://localhost:8080")
    server.serve_forever()
