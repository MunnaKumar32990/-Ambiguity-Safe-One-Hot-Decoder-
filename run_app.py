"""
run_app.py
==========
Convenience launcher for the Ambiguity-Safe Inverse Decoding web app.

Starts the Flask backend (which also serves the frontend) and opens the
browser at the app URL.

Run:
    python run_app.py
"""

import os
import sys
import threading
import webbrowser

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app

HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"


def _open_browser():
    """Open the default browser once the server is up."""
    webbrowser.open(URL)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Ambiguity-Safe Decoder — Web App")
    print("=" * 60)
    print(f"  Frontend + API : {URL}")
    print(f"  API base       : {URL}/api")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    # Open the browser shortly after the server starts. Guard against the
    # Flask reloader launching this twice.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.2, _open_browser).start()

    app.run(host=HOST, port=PORT, debug=True)
