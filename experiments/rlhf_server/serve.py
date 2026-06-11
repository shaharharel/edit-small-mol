#!/usr/bin/env python3
"""Production server for the RLHF demo using waitress (a real WSGI server).

The Flask dev server (`app.py`) is fine for local use; this runs the same app
under waitress with a worker thread pool, suitable for a small team.

One-line run (from the repo root):
    RLHF_SECRET_KEY=$(openssl rand -hex 32) conda run -n quris \
        python experiments/rlhf_server/serve.py

Environment:
    RLHF_SECRET_KEY  signs session cookies — REQUIRED in production
    RLHF_HOST        bind host   (default 0.0.0.0)
    RLHF_PORT        bind port   (default 5055)
    RLHF_THREADS     worker threads (default 8)
    RLHF_DB_PATH     SQLite path (default data/rlhf_demo/rlhf.db)
"""
import os
import sys

# allow `python experiments/rlhf_server/serve.py` from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waitress import serve  # noqa: E402

from app import MOLECULES, N_PAIRS, app, init_db  # noqa: E402

if __name__ == "__main__":
    if not os.environ.get("RLHF_SECRET_KEY"):
        print(
            "WARNING: RLHF_SECRET_KEY not set — using an insecure dev key. "
            "In production: export RLHF_SECRET_KEY=$(openssl rand -hex 32)",
            file=sys.stderr,
        )
    init_db()
    host = os.environ.get("RLHF_HOST", "0.0.0.0")
    port = int(os.environ.get("RLHF_PORT", "5055"))
    threads = int(os.environ.get("RLHF_THREADS", "8"))
    print(f"RLHF server (waitress): {N_PAIRS} pairs, {len(MOLECULES)} molecules")
    print(f"listening on http://{host}:{port}  ({threads} threads)")
    serve(app, host=host, port=port, threads=threads)
