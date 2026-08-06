import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

os.chdir(Path(__file__).resolve().parent / "public")
port = int(os.environ.get("PORT", "8000"))

server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
print(f"Serving ISL Sign Detector on 0.0.0.0:{port}")
server.serve_forever()
