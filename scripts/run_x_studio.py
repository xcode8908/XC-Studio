#!/usr/bin/env python3
"""CLI entry point for X-Studio.

X-Studio
Dev: @SILENTXOP
YouTuber: @silentx_nomore
YouTube: https://youtube.com/@silentx_nomore
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from x_studio_server import start_server


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run X-Studio AI Video Generator")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind (default: 7860)")
    parser.add_argument("--share", action="store_true", default=True, help="Enable public URL for browser access")
    parser.add_argument("--no-share", dest="share", action="store_false", help="Localhost only")
    args = parser.parse_args()

    start_server(port=args.port, share=args.share)


if __name__ == "__main__":
    main()
