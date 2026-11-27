"""Start the dashboard:  cd app && uv run python -m skd_app   (then open http://localhost:8000)"""

from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="SKD demo dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()
    url = f"http://localhost:{args.port}"
    print(f"\n  SKD demo dashboard → {url}\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run("skd_app.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
