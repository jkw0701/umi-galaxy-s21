r"""
RobotDataLearning Local - PC Receive Server

Receives sensor data from the Android app via USB (adb reverse).
Saves files to: C:\Users\user\Downloads\robotdatalearning_local\{session_id}\{relative_path}

Usage:
    1. Connect phone via USB
    2. Run: adb reverse tcp:8080 tcp:8080
    3. Run: python receive_server.py
    4. Record on the app — files will be saved automatically
"""

import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Save directory
SAVE_DIR = Path(os.path.expanduser("~")) / "Downloads" / "robotdatalearning_local"
PORT = 8080


class UploadHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """Health check endpoint."""
        if self.path == "/ping":
            self._respond(200, "ok")
        else:
            self._respond(404, "not found")

    def do_POST(self):
        if self.path == "/upload":
            self._handle_upload()
        elif self.path == "/upload_chunk":
            self._handle_chunk_upload()
        else:
            self._respond(404, "not found")

    def _handle_upload(self):
        """Receive a complete file in a single request."""
        session_id = self.headers.get("X-Session-Id", "")
        relative_path = self.headers.get("X-Relative-Path", "")

        if not session_id or not relative_path:
            self._respond(400, "missing X-Session-Id or X-Relative-Path header")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        save_path = SAVE_DIR / session_id / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(body)

        print(f"  Saved: {session_id}/{relative_path} ({len(body):,} bytes)")
        self._respond(200, "ok")

    def _handle_chunk_upload(self):
        """Receive a chunk of a large file."""
        session_id = self.headers.get("X-Session-Id", "")
        relative_path = self.headers.get("X-Relative-Path", "")
        chunk_index = int(self.headers.get("X-Chunk-Index", "-1"))
        total_chunks = int(self.headers.get("X-Total-Chunks", "-1"))

        if not session_id or not relative_path or chunk_index < 0:
            self._respond(400, "missing required headers")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Write chunks to temp files, assemble on last chunk
        save_dir = SAVE_DIR / session_id
        save_dir.mkdir(parents=True, exist_ok=True)

        chunk_dir = save_dir / f".chunks_{relative_path.replace('/', '_')}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_file = chunk_dir / f"chunk_{chunk_index:04d}"
        chunk_file.write_bytes(body)

        print(f"  Chunk {chunk_index + 1}/{total_chunks}: {session_id}/{relative_path} ({len(body):,} bytes)")

        # If this is the last chunk, assemble the file
        if chunk_index == total_chunks - 1:
            final_path = save_dir / relative_path
            final_path.parent.mkdir(parents=True, exist_ok=True)

            with open(final_path, "wb") as f:
                for i in range(total_chunks):
                    cf = chunk_dir / f"chunk_{i:04d}"
                    f.write(cf.read_bytes())

            # Clean up chunk directory
            import shutil
            shutil.rmtree(chunk_dir)

            total_size = final_path.stat().st_size
            print(f"  Assembled: {session_id}/{relative_path} ({total_size:,} bytes)")

        self._respond(200, "ok")

    def _respond(self, code, message):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode())

    def log_message(self, format, *args):
        # Suppress default access logs (we print our own)
        pass


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RobotDataLearning Local - PC Receive Server")
    print("=" * 60)
    print(f"  Save directory: {SAVE_DIR}")
    print(f"  Listening on:   http://localhost:{PORT}")
    print()
    print("  Make sure to run:")
    print("    adb reverse tcp:8080 tcp:8080")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    server = HTTPServer(("0.0.0.0", PORT), UploadHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
