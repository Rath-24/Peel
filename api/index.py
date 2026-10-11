"""Vercel serverless entrypoint for the Peel Flask app.

Vercel's Python runtime looks for a top-level WSGI variable named `app`
in `api/index.py` and routes requests to it (see vercel.json rewrites).
The full app (routes + inline frontend) lives in `peel.py`; this file
just re-exports it so local runs (`python peel.py`) and Vercel share
one implementation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from peel import app  # noqa: F401  (Vercel resolves the `app` variable)
