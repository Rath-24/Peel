"""
Peel — AI Layer Extractor (single-file edition)
------------------------------------------------
Everything in one file: the segmentation/inpainting/OCR pipeline, the
Flask API, and the entire frontend (HTML/CSS/JS inlined as a string).

Why one file: easiest possible thing to hand in, copy, or run on a
machine with nothing set up — no project structure to navigate, no
static folder wiring, just `python peel.py`.

Pipeline:
  - Subject/background split : rembg (U^2-Net, pretrained ONNX model)
  - Background reconstruction : OpenCV Telea inpainting
  - Text detection            : Tesseract OCR (pytesseract)

Run:
    pip install flask opencv-python-headless numpy rembg onnxruntime pytesseract Pillow
    sudo apt-get install -y tesseract-ocr
    python peel.py
Then open http://localhost:5000
"""

import base64
import io
import logging
import os
import time

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("peel")

try:
    import pytesseract  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    pytesseract = None

from flask import Flask, jsonify, request, Response
from PIL import Image

try:
    from rembg import remove, new_session  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    remove = None
    new_session = None


def has_ocr_support():
    """Return True when pytesseract is importable and the Tesseract binary is available."""
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # pragma: no cover - depends on local OCR runtime
        return False

# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

SESSION = None  # lazy-loaded so the server starts instantly


def get_session():
    global SESSION
    if new_session is None or remove is None:
        raise RuntimeError("rembg is not installed. Install it with: pip install rembg onnxruntime")
    if SESSION is None:
        SESSION = new_session("u2net")
    return SESSION


def extract_subject_mask(image_bgr):
    """Runs U^2-Net salient object segmentation, returns a clean alpha mask (0-255, uint8)."""
    if remove is None or new_session is None:
        h, w = image_bgr.shape[:2]
        return np.full((h, w), 255, dtype=np.uint8)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    # No alpha_matting - get a clean binary-like mask from U2-Net directly
    result = remove(pil_img, session=get_session())
    alpha = np.array(result)[:, :, 3]

    # Threshold to get a clean sharp mask (no soft/translucent edges)
    _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    # Light cleanup: remove tiny noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

    return alpha


def refine_mask(alpha):
    """Very light smoothing - only at the boundary pixels."""
    # Find boundary pixels (where mask transitions from 0 to 255)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(alpha, kernel)
    dilated = cv2.dilate(alpha, kernel)
    boundary = cv2.bitwise_xor(dilated, eroded)

    # Smooth only the boundary, keep interior solid
    smoothed = cv2.GaussianBlur(alpha, (3, 3), 0)
    result = alpha.copy()
    result[boundary > 0] = smoothed[boundary > 0]
    return result


def build_subject_layer(image_bgr, alpha):
    """Foreground cut-out on a transparent canvas with clean edges."""
    refined = refine_mask(alpha)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return np.dstack([rgb, refined])


def build_background_layer(image_bgr, alpha, inpaint_radius=None):
    """Removes the subject and inpaints the hole so the background reads as complete."""
    h, w = image_bgr.shape[:2]
    refined = refine_mask(alpha)

    if inpaint_radius is None:
        inpaint_radius = max(3, min(12, int(min(h, w) / 200)))

    # Use the clean mask directly - just dilate slightly to cover edges
    mask = refined.copy()
    dilate_size = max(3, int(min(h, w) / 250)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    mask = cv2.dilate(mask, kernel, iterations=1)

    inpainted = cv2.inpaint(image_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    alpha_full = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    return np.dstack([rgb, alpha_full])


def build_text_layer(image_bgr):
    """Detects text via Tesseract and lifts just those pixel regions onto a transparent layer."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    alpha = np.zeros((h, w), dtype=np.uint8)

    if pytesseract is None or not has_ocr_support():
        return np.dstack([rgb, alpha]), []

    try:
        # Preprocess: convert to grayscale + increase contrast for better OCR
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        data = pytesseract.image_to_data(
            gray,
            output_type=pytesseract.Output.DICT,
            config="--psm 6 --oem 3",
        )
    except Exception:
        return np.dstack([rgb, alpha]), []

    boxes = []
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        conf_value = data["conf"][i]
        try:
            conf = int(conf_value)
        except (TypeError, ValueError):
            conf = -1
        if text and conf > 30:
            x, y, bw, bh = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            pad = max(3, int(min(bw, bh) * 0.15))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            alpha[y0:y1, x0:x1] = 255
            boxes.append({"text": text, "conf": conf, "bbox": [x0, y0, x1, y1]})

    # Smooth text mask edges
    if boxes:
        blur_k = max(3, int(min(h, w) / 400)) | 1
        alpha = cv2.GaussianBlur(alpha, (blur_k, blur_k), 0)

    return np.dstack([rgb, alpha]), boxes


def rgba_to_data_url(rgba_array):
    buf = io.BytesIO()
    Image.fromarray(rgba_array, mode="RGBA").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ----------------------------------------------------------------------
# Frontend (inlined)
# ----------------------------------------------------------------------

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Peel — AI Layer Extractor</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ink: #121316;
  --panel: #1B1D22;
  --panel-raised: #22242B;
  --border: #2A2D35;
  --text: #EDEDEF;
  --text-muted: #8B8E99;
  --accent: #C9A227;
  --accent-soft: rgba(201, 162, 39, 0.14);
  --select-blue: #6FA8FF;
  --danger: #E5674A;
  --radius: 8px;
  --font-display: 'Space Grotesk', sans-serif;
  --font-body: 'Inter', sans-serif;
  --font-mono: 'IBM Plex Mono', monospace;
}

* { box-sizing: border-box; }

[hidden] { display: none !important; }

html, body {
  margin: 0;
  height: 100%;
  background: var(--ink);
  color: var(--text);
  font-family: var(--font-body);
  -webkit-font-smoothing: antialiased;
}

button { font-family: inherit; }

.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* ---------- topbar ---------- */

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 60px;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
  flex-shrink: 0;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.brand-mark { flex-shrink: 0; }

.brand-name {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 20px;
  letter-spacing: -0.01em;
}

.brand-sub {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
  border-left: 1px solid var(--border);
  padding-left: 10px;
  margin-left: 2px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.topbar-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-muted);
}

.btn {
  border: 1px solid transparent;
  border-radius: var(--radius);
  font-size: 13px;
  font-weight: 500;
  padding: 9px 16px;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
}
.btn:active { transform: translateY(1px); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }

.btn-primary {
  background: var(--accent);
  color: #1A1608;
}
.btn-primary:hover:not(:disabled) { background: #d9b23a; }

.btn-ghost {
  background: transparent;
  border-color: var(--border);
  color: var(--text);
}
.btn-ghost:hover { border-color: var(--text-muted); }

/* ---------- workspace layout ---------- */

.workspace {
  flex: 1;
  display: flex;
  position: relative;
  min-height: 0;
}

/* ---------- dropzone ---------- */

.dropzone {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
  transition: background 0.15s ease;
}

.dropzone.drag-over {
  background: var(--accent-soft);
}

.dropzone-inner {
  text-align: center;
  max-width: 420px;
}

.dropzone-inner svg { margin-bottom: 20px; }

.dropzone-inner h1 {
  font-family: var(--font-display);
  font-size: 26px;
  font-weight: 600;
  margin: 0 0 10px;
  letter-spacing: -0.01em;
}

.dropzone-inner p {
  color: var(--text-muted);
  font-size: 14px;
  line-height: 1.5;
  margin: 0 0 22px;
}

.dropzone-hint {
  margin-top: 14px !important;
  font-family: var(--font-mono);
  font-size: 11px !important;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

/* ---------- canvas ---------- */

.canvas-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.canvas-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 18px;
  border-bottom: 1px solid var(--border);
}

.zoom-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-muted);
}

.icon-btn {
  width: 24px;
  height: 24px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--panel-raised);
  color: var(--text);
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
}
.icon-btn:hover { border-color: var(--accent); color: var(--accent); }

.processing-time {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
}

.canvas-scroll {
  flex: 1;
  overflow: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 30px;
}

.canvas-box {
  position: relative;
  background:
    repeating-conic-gradient(#26282F 0% 25%, #1E2026 0% 50%) 50% / 22px 22px;
  border: 1px solid var(--border);
  border-radius: 4px;
  box-shadow: 0 12px 40px rgba(0,0,0,0.45);
}

.layer-img {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: block;
  transition: opacity 0.15s ease;
}

/* ---------- processing / iris loader ---------- */

.processing-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 22px;
}

#processing-label {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.iris-loader {
  position: relative;
  width: 64px;
  height: 64px;
  animation: spin 2.2s linear infinite;
}
.iris-blade {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 30px;
  height: 12px;
  background: var(--accent);
  opacity: 0.85;
  border-radius: 2px;
  transform-origin: 0 50%;
}
.iris-blade.b1 { transform: rotate(0deg) translateX(2px); }
.iris-blade.b2 { transform: rotate(60deg) translateX(2px); }
.iris-blade.b3 { transform: rotate(120deg) translateX(2px); }
.iris-blade.b4 { transform: rotate(180deg) translateX(2px); }
.iris-blade.b5 { transform: rotate(240deg) translateX(2px); }
.iris-blade.b6 { transform: rotate(300deg) translateX(2px); }

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* ---------- layers panel ---------- */

.layers-panel {
  width: 300px;
  flex-shrink: 0;
  border-left: 1px solid var(--border);
  background: var(--panel);
  padding: 20px;
  overflow-y: auto;
}

.layers-panel h2 {
  font-family: var(--font-display);
  font-size: 15px;
  margin: 0 0 4px;
}

.layers-hint {
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.5;
  margin: 0 0 18px;
}

.layer-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.layer-item {
  background: var(--panel-raised);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 14px;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}

.layer-item:hover {
  transform: translateY(-2px) rotate(-0.3deg);
  box-shadow: 0 8px 18px rgba(0,0,0,0.35);
  border-color: #3A3E48;
}

.layer-item-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.layer-thumb {
  width: 34px;
  height: 34px;
  border-radius: 5px;
  background:
    repeating-conic-gradient(#2E313A 0% 25%, #26282F 0% 50%) 50% / 10px 10px;
  border: 1px solid var(--border);
  overflow: hidden;
  flex-shrink: 0;
}
.layer-thumb img { width: 100%; height: 100%; object-fit: cover; }

.layer-name {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
}

.eye-toggle {
  width: 26px;
  height: 26px;
  border-radius: 6px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}
.eye-toggle:hover { color: var(--text); background: rgba(255,255,255,0.06); }
.eye-toggle.off { color: #55575F; }

.layer-download {
  width: 26px;
  height: 26px;
  border-radius: 6px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}
.layer-download:hover { color: var(--accent); background: rgba(255,255,255,0.06); }

.opacity-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
}

.opacity-row input[type="range"] {
  flex: 1;
  accent-color: var(--accent);
}

.opacity-value {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
  width: 30px;
  text-align: right;
}

.text-detected {
  margin-top: 24px;
  padding-top: 18px;
  border-top: 1px solid var(--border);
}
.text-detected h3 {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin: 0 0 10px;
  font-weight: 500;
}
#text-detected-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
#text-detected-list li {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-muted);
  background: var(--panel-raised);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 9px;
}
#text-detected-list li span { color: var(--text); }

/* ---------- recent panel ---------- */

.recent-panel {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
}
.recent-panel h3 {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin: 0 0 10px;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.recent-panel h3 button {
  font-size: 11px;
  color: var(--danger);
  background: none;
  border: none;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.recent-panel h3 button:hover { text-decoration: underline; }

.recent-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 240px;
  overflow-y: auto;
}

.recent-item {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--panel-raised);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;
}
.recent-item:hover {
  border-color: var(--accent);
  background: rgba(201, 162, 39, 0.06);
}
.recent-thumb {
  width: 36px;
  height: 36px;
  border-radius: 4px;
  object-fit: cover;
  flex-shrink: 0;
  background: var(--ink);
}
.recent-info {
  flex: 1;
  min-width: 0;
}
.recent-name {
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.recent-meta {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 2px;
}
.recent-delete {
  width: 22px;
  height: 22px;
  border-radius: 4px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.recent-delete:hover { color: var(--danger); background: rgba(229, 103, 74, 0.1); }

/* ---------- toast ---------- */

.toast {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  background: var(--panel-raised);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 11px 18px;
  border-radius: var(--radius);
  font-size: 13px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  z-index: 50;
}
.toast.error { border-color: var(--danger); color: #FFD4C7; }

/* ---------- responsive ---------- */

@media (max-width: 820px) {
  .workspace { flex-direction: column; }
  .layers-panel {
    width: 100%;
    border-left: none;
    border-top: 1px solid var(--border);
  }
  .brand-sub { display: none; }
}

/* ---------- reduced motion ---------- */

@media (prefers-reduced-motion: reduce) {
  .iris-loader { animation: none; }
  .layer-item:hover { transform: none; }
}

/* ---------- focus visibility ---------- */

button:focus-visible, input:focus-visible {
  outline: 2px solid var(--select-blue);
  outline-offset: 2px;
}

</style>
</head>
<body>

<div class="app">

  <header class="topbar">
    <div class="brand">
      <svg class="brand-mark" width="26" height="26" viewBox="0 0 26 26" fill="none">
        <rect x="6" y="8" width="16" height="14" rx="2" fill="#2A2D35"/>
        <rect x="4" y="5" width="16" height="14" rx="2" fill="#3A3E48"/>
        <rect x="2" y="2" width="16" height="14" rx="2" fill="#C9A227"/>
      </svg>
      <span class="brand-name">Peel</span>
      <span class="brand-sub">AI Layer Extractor</span>
    </div>
    <div class="topbar-actions">
      <span class="status" id="status-badge"></span>
      <button class="btn btn-ghost" id="btn-recent" title="Recent images">Recent (<span id="recent-count">0</span>)</button>
      <button class="btn btn-ghost" id="btn-reset" hidden>New image</button>
      <button class="btn btn-primary" id="btn-export" disabled>Export layers (.zip)</button>
    </div>
  </header>

  <main class="workspace">

    <!-- Empty state / dropzone -->
    <section class="dropzone" id="dropzone">
      <input type="file" id="file-input" accept="image/png, image/jpeg" hidden>
      <div class="dropzone-inner">
        <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
          <rect x="10" y="16" width="30" height="26" rx="3" fill="#22242B" stroke="#3A3E48" stroke-width="1.5"/>
          <rect x="14" y="12" width="30" height="26" rx="3" fill="#1B1D22" stroke="#3A3E48" stroke-width="1.5"/>
          <rect x="18" y="8" width="30" height="26" rx="3" fill="#14151A" stroke="#C9A227" stroke-width="1.5"/>
          <path d="M27 18v10M22 23l5-5 5 5" stroke="#C9A227" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <h1>Drop an image to peel it apart</h1>
        <p>Splits one flat image into a background, subject, and text layer — each exportable as a transparent PNG.</p>
        <button class="btn btn-primary" id="btn-browse">Choose an image</button>
        <p class="dropzone-hint">JPG or PNG, up to 15MB</p>
      </div>
    </section>

    <!-- Working state -->
    <section class="canvas-area" id="canvas-area" hidden>
      <div class="canvas-toolbar">
        <div class="zoom-controls">
          <button class="icon-btn" id="btn-zoom-out" title="Zoom out">–</button>
          <span id="zoom-label">100%</span>
          <button class="icon-btn" id="btn-zoom-in" title="Zoom in">+</button>
        </div>
        <div class="processing-time" id="processing-time"></div>
      </div>
      <div class="canvas-scroll">
        <div class="canvas-box" id="canvas-box">
          <img id="layer-img-background" class="layer-img">
          <img id="layer-img-subject" class="layer-img">
          <img id="layer-img-text" class="layer-img">
        </div>
      </div>
    </section>

    <!-- Loading state -->
    <section class="processing-state" id="processing-state" hidden>
      <div class="iris-loader">
        <div class="iris-blade b1"></div>
        <div class="iris-blade b2"></div>
        <div class="iris-blade b3"></div>
        <div class="iris-blade b4"></div>
        <div class="iris-blade b5"></div>
        <div class="iris-blade b6"></div>
      </div>
      <p id="processing-label">Segmenting subject…</p>
    </section>

    <!-- Layers panel -->
    <aside class="layers-panel" id="layers-panel" hidden>
      <h2>Layers</h2>
      <p class="layers-hint">Toggle visibility, adjust opacity, or download each layer on its own.</p>
      <ul class="layer-list" id="layer-list"></ul>

      <div class="text-detected" id="text-detected" hidden>
        <h3>Text detected</h3>
        <ul id="text-detected-list"></ul>
      </div>

      <div class="recent-panel" id="recent-panel">
        <h3>Recent <button id="btn-clear-recent">Clear</button></h3>
        <ul class="recent-list" id="recent-list"></ul>
      </div>
    </aside>

  </main>

  <div class="toast" id="toast" hidden></div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const btnBrowse = document.getElementById('btn-browse');
const btnReset = document.getElementById('btn-reset');
const btnExport = document.getElementById('btn-export');
const canvasArea = document.getElementById('canvas-area');
const canvasBox = document.getElementById('canvas-box');
const processingState = document.getElementById('processing-state');
const processingLabel = document.getElementById('processing-label');
const processingTime = document.getElementById('processing-time');
const layersPanel = document.getElementById('layers-panel');
const layerList = document.getElementById('layer-list');
const statusBadge = document.getElementById('status-badge');
const zoomLabel = document.getElementById('zoom-label');
const textDetected = document.getElementById('text-detected');
const textDetectedList = document.getElementById('text-detected-list');
const toast = document.getElementById('toast');

let currentZoom = 1;
let lastResult = null;

if (window.__TEST_RESULT__) {
  lastResult = window.__TEST_RESULT__;
  setTimeout(() => { renderResult(lastResult); setState('result'); }, 0);
}

const PROCESSING_STEPS = [
  'Segmenting subject…',
  'Reconstructing background…',
  'Detecting text…',
  'Building layers…',
];

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle('error', isError);
  toast.hidden = false;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => { toast.hidden = true; }, 4000);
}

function setState(name) {
  dropzone.hidden = name !== 'empty';
  processingState.hidden = name !== 'processing';
  canvasArea.hidden = name !== 'result';
  layersPanel.hidden = name !== 'result';
  btnReset.hidden = name === 'empty';
  btnExport.disabled = name !== 'result';
}

// ---- drag & drop / browse ----

['dragenter', 'dragover'].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  })
);
['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
  })
);
dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
btnBrowse.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});
btnReset.addEventListener('click', () => {
  lastResult = null;
  fileInput.value = '';
  setState('empty');
});

document.getElementById('btn-recent').addEventListener('click', () => {
  // Toggle layers panel visibility to show recent
  if (layersPanel.hidden) {
    setState('result');
    renderRecent();
  } else {
    layersPanel.hidden = true;
  }
});

// ---- upload + process ----

async function handleFile(file) {
  if (!file.type.match(/image\/(png|jpeg)/)) {
    showToast('Please use a JPG or PNG image.', true);
    return;
  }

  setState('processing');
  let stepIndex = 0;
  processingLabel.textContent = PROCESSING_STEPS[0];
  const stepTimer = setInterval(() => {
    stepIndex = Math.min(stepIndex + 1, PROCESSING_STEPS.length - 1);
    processingLabel.textContent = PROCESSING_STEPS[stepIndex];
  }, 900);

  const formData = new FormData();
  formData.append('image', file);

  try {
    const res = await fetch('/api/extract', { method: 'POST', body: formData });
    const data = await res.json();
    clearInterval(stepTimer);

    if (!res.ok) {
      showToast(data.error || 'Something went wrong processing that image.', true);
      setState('empty');
      return;
    }

    lastResult = data;
    renderResult(data);
    setState('result');
    addToRecent(data);
  } catch (err) {
    clearInterval(stepTimer);
    showToast('Could not reach the extraction service. Is the backend running?', true);
    setState('empty');
  }
}

// ---- render result ----

function renderResult(data) {
  canvasBox.style.width = data.width + 'px';
  canvasBox.style.height = data.height + 'px';
  currentZoom = 1;
  applyZoom();

  processingTime.textContent = `Processed in ${data.processing_seconds}s · ${data.width}×${data.height}`;

  data.layers.forEach(layer => {
    const img = document.getElementById(`layer-img-${layer.id}`);
    img.src = layer.data_url;
  });

  layerList.innerHTML = '';
  // top of list = topmost layer visually (text over subject over background)
  [...data.layers].reverse().forEach(layer => {
    layerList.appendChild(buildLayerRow(layer));
  });

  if (data.detected_text && data.detected_text.length) {
    textDetected.hidden = false;
    textDetectedList.innerHTML = '';
    data.detected_text.forEach(t => {
      const li = document.createElement('li');
      li.innerHTML = `<span>${escapeHtml(t.text)}</span> · ${t.conf}% conf`;
      textDetectedList.appendChild(li);
    });
  } else {
    textDetected.hidden = true;
  }
}

function buildLayerRow(layer) {
  const li = document.createElement('li');
  li.className = 'layer-item';
  li.innerHTML = `
    <div class="layer-item-row">
      <div class="layer-thumb"><img src="${layer.data_url}" alt=""></div>
      <span class="layer-name">${layer.name}</span>
      <button class="eye-toggle" title="Toggle visibility" data-layer="${layer.id}">
        ${eyeIcon(true)}
      </button>
      <button class="layer-download" title="Download this layer" data-layer="${layer.id}">
        ${downloadIcon()}
      </button>
    </div>
    <div class="opacity-row">
      <input type="range" min="0" max="100" value="100" data-layer="${layer.id}">
      <span class="opacity-value">100%</span>
    </div>
  `;

  const eyeBtn = li.querySelector('.eye-toggle');
  eyeBtn.addEventListener('click', () => {
    const img = document.getElementById(`layer-img-${layer.id}`);
    const isOn = !eyeBtn.classList.contains('off');
    eyeBtn.classList.toggle('off', isOn);
    eyeBtn.innerHTML = eyeIcon(!isOn);
    img.style.display = isOn ? 'none' : 'block';
  });

  const downloadBtn = li.querySelector('.layer-download');
  downloadBtn.addEventListener('click', () => downloadDataUrl(layer.data_url, `${layer.id}.png`));

  const range = li.querySelector('input[type="range"]');
  const valueLabel = li.querySelector('.opacity-value');
  range.addEventListener('input', () => {
    const v = range.value;
    valueLabel.textContent = `${v}%`;
    document.getElementById(`layer-img-${layer.id}`).style.opacity = v / 100;
  });

  return li;
}

function eyeIcon(visible) {
  return visible
    ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.6"/></svg>`
    : `<svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M3 3l18 18M10.6 10.6a3 3 0 0 0 4.2 4.2M6.6 6.7C3.9 8.3 2 11 2 11s4 7 11 7c1.8 0 3.4-.4 4.8-1.1M12 5c7 0 11 7 11 7-.4.7-1.4 2-2.8 3.3" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;
}
function downloadIcon() {
  return `<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 21h16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  a.click();
}

// ---- zoom ----

document.getElementById('btn-zoom-in').addEventListener('click', () => { currentZoom = Math.min(3, currentZoom + 0.25); applyZoom(); });
document.getElementById('btn-zoom-out').addEventListener('click', () => { currentZoom = Math.max(0.25, currentZoom - 0.25); applyZoom(); });

function applyZoom() {
  canvasBox.style.transform = `scale(${currentZoom})`;
  zoomLabel.textContent = `${Math.round(currentZoom * 100)}%`;
}

// ---- export all layers as a zip (built client-side, no server round trip) ----

btnExport.addEventListener('click', async () => {
  if (!lastResult) return;
  showToast('Preparing download…');
  // Lightweight zip writer: store each PNG uncompressed (method 0) — no
  // external dependency needed for a handful of already-compressed PNGs.
  const files = lastResult.layers.map(l => ({
    name: `${l.id}.png`,
    data: dataUrlToBytes(l.data_url),
  }));
  const blob = buildZip(files);
  const url = URL.createObjectURL(blob);
  downloadDataUrl(url, 'peel-layers.zip');
  setTimeout(() => URL.revokeObjectURL(url), 2000);
});

function dataUrlToBytes(dataUrl) {
  const base64 = dataUrl.split(',')[1];
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// Minimal STORE-only ZIP writer (no compression, valid ZIP format)
function buildZip(files) {
  const encoder = new TextEncoder();
  const localParts = [];
  const centralParts = [];
  let offset = 0;

  function crc32(buf) {
    let c, crc = 0xFFFFFFFF;
    for (let i = 0; i < buf.length; i++) {
      c = (crc ^ buf[i]) & 0xFF;
      for (let j = 0; j < 8; j++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      crc = (crc >>> 8) ^ c;
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }

  files.forEach(f => {
    const nameBytes = encoder.encode(f.name);
    const crc = crc32(f.data);
    const size = f.data.length;

    const localHeader = new Uint8Array(30 + nameBytes.length);
    const dv = new DataView(localHeader.buffer);
    dv.setUint32(0, 0x04034b50, true);
    dv.setUint16(4, 20, true);
    dv.setUint16(6, 0, true);
    dv.setUint16(8, 0, true);
    dv.setUint16(10, 0, true);
    dv.setUint16(12, 0, true);
    dv.setUint32(14, crc, true);
    dv.setUint32(18, size, true);
    dv.setUint32(22, size, true);
    dv.setUint16(26, nameBytes.length, true);
    dv.setUint16(28, 0, true);
    localHeader.set(nameBytes, 30);

    localParts.push(localHeader, f.data);

    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const cdv = new DataView(centralHeader.buffer);
    cdv.setUint32(0, 0x02014b50, true);
    cdv.setUint16(4, 20, true);
    cdv.setUint16(6, 20, true);
    cdv.setUint16(8, 0, true);
    cdv.setUint16(10, 0, true);
    cdv.setUint16(12, 0, true);
    cdv.setUint32(16, crc, true);
    cdv.setUint32(20, size, true);
    cdv.setUint32(24, size, true);
    cdv.setUint16(28, nameBytes.length, true);
    cdv.setUint32(42, offset, true);
    centralHeader.set(nameBytes, 46);
    centralParts.push(centralHeader);

    offset += localHeader.length + f.data.length;
  });

  const centralStart = offset;
  const centralSize = centralParts.reduce((s, p) => s + p.length, 0);

  const eocd = new Uint8Array(22);
  const edv = new DataView(eocd.buffer);
  edv.setUint32(0, 0x06054b50, true);
  edv.setUint16(8, files.length, true);
  edv.setUint16(10, files.length, true);
  edv.setUint32(12, centralSize, true);
  edv.setUint32(16, centralStart, true);

  return new Blob([...localParts, ...centralParts, eocd], { type: 'application/zip' });
}

// ---- recent images (localStorage) ----

const RECENT_KEY = 'peel_recent';
const MAX_RECENT = 8;
const recentList = document.getElementById('recent-list');
const recentPanel = document.getElementById('recent-panel');
const recentCount = document.getElementById('recent-count');
let recentItems = [];

function addToRecent(data) {
  // Use the subject layer data URL directly as thumbnail (already a PNG)
  // Just shrink it via an img element
  const srcUrl = data.layers[1]?.data_url || data.layers[0]?.data_url || '';
  if (!srcUrl) return;

  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const thumbW = 100;
    const thumbH = Math.round((img.height / img.width) * thumbW);
    canvas.width = thumbW;
    canvas.height = thumbH;
    ctx.drawImage(img, 0, 0, thumbW, thumbH);
    try {
      const thumb = canvas.toDataURL('image/jpeg', 0.5);
      const entry = {
        id: Date.now(),
        name: new Date().toLocaleTimeString(),
        width: data.width,
        height: data.height,
        time: data.processing_seconds,
        thumb: thumb,
      };
      recentItems.unshift(entry);
      if (recentItems.length > MAX_RECENT) recentItems.length = MAX_RECENT;
      renderRecent();
      saveRecentToStorage();
    } catch (e) {
      // canvas tainted - skip thumbnail but still record metadata
      recentItems.unshift({
        id: Date.now(),
        name: new Date().toLocaleTimeString(),
        width: data.width,
        height: data.height,
        time: data.processing_seconds,
        thumb: '',
      });
      if (recentItems.length > MAX_RECENT) recentItems.length = MAX_RECENT;
      renderRecent();
      saveRecentToStorage();
    }
  };
  img.onerror = () => {
    // Image load failed - still record metadata
    recentItems.unshift({
      id: Date.now(),
      name: new Date().toLocaleTimeString(),
      width: data.width,
      height: data.height,
      time: data.processing_seconds,
      thumb: '',
    });
    if (recentItems.length > MAX_RECENT) recentItems.length = MAX_RECENT;
    renderRecent();
    saveRecentToStorage();
  };
  img.src = srcUrl;
}

function saveRecentToStorage() {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(recentItems));
  } catch {}
}

function loadRecentFromStorage() {
  try {
    recentItems = JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
  } catch { recentItems = []; }
}

function renderRecent() {
  recentList.innerHTML = '';
  recentCount.textContent = recentItems.length;
  if (!recentItems.length) {
    recentPanel.hidden = true;
    return;
  }
  recentPanel.hidden = false;
  recentItems.forEach(item => {
    const li = document.createElement('li');
    li.className = 'recent-item';
    const thumbHtml = item.thumb
      ? `<img class="recent-thumb" src="${item.thumb}" alt="">`
      : `<div class="recent-thumb" style="display:flex;align-items:center;justify-content:center;font-size:18px;color:var(--text-muted);">?</div>`;
    li.innerHTML = `
      ${thumbHtml}
      <div class="recent-info">
        <div class="recent-name">${escapeHtml(item.name)}</div>
        <div class="recent-meta">${item.width}x${item.height} &middot; ${item.time}s</div>
      </div>
      <button class="recent-delete" title="Remove">&times;</button>
    `;
    li.querySelector('.recent-delete').addEventListener('click', (e) => {
      e.stopPropagation();
      recentItems = recentItems.filter(i => i.id !== item.id);
      saveRecentToStorage();
      renderRecent();
    });
    li.addEventListener('click', () => {
      showToast('Re-upload this image to edit it again');
    });
    recentList.appendChild(li);
  });
}

document.getElementById('btn-clear-recent').addEventListener('click', () => {
  recentItems = [];
  saveRecentToStorage();
  renderRecent();
  showToast('Recent history cleared');
});

loadRecentFromStorage();
renderRecent();

</script>
</body>
</html>
"""

# ----------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------

MAX_UPLOAD_MB = int(os.environ.get("PEEL_MAX_UPLOAD_MB", 15))
MAX_DIMENSION = int(os.environ.get("PEEL_MAX_DIMENSION", 1600))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.route("/")
def index():
    return Response(
        FRONTEND_HTML,
        mimetype="text/html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.route("/api/extract", methods=["POST"])
def extract():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under field 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    raw = file.read()
    np_arr = np.frombuffer(raw, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Could not decode image. Use JPG or PNG."}), 400

    h, w = image_bgr.shape[:2]
    original_w, original_h = w, h
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        h, w = image_bgr.shape[:2]
        logger.info("Downscaled %dx%d -> %dx%d", original_w, original_h, w, h)

    try:
        t0 = time.time()
        alpha = extract_subject_mask(image_bgr)
        t_mask = time.time()
        logger.info("Subject mask: %.2fs", t_mask - t0)

        subject_rgba = build_subject_layer(image_bgr, alpha)
        background_rgba = build_background_layer(image_bgr, alpha)
        t_bg = time.time()
        logger.info("Background inpaint: %.2fs", t_bg - t_mask)

        text_rgba, boxes = build_text_layer(image_bgr)
        t_text = time.time()
        logger.info("Text detection: %.2fs", t_text - t_bg)

        subject_url = rgba_to_data_url(subject_rgba)
        background_url = rgba_to_data_url(background_rgba)
        text_url = rgba_to_data_url(text_rgba)
        elapsed = round(time.time() - t0, 2)
        logger.info("Total: %.2fs (%dx%d)", elapsed, w, h)

        return jsonify({
            "width": w,
            "height": h,
            "processing_seconds": elapsed,
            "layers": [
                {"id": "background", "name": "Background", "data_url": background_url},
                {"id": "subject", "name": "Subject", "data_url": subject_url},
                {"id": "text", "name": "Text", "data_url": text_url},
            ],
            "detected_text": boxes,
        })

    except MemoryError:
        logger.error("Out of memory processing %dx%d image", w, h)
        return jsonify({"error": "Image too large to process. Try a smaller image."}), 500
    except Exception as e:
        logger.exception("Pipeline failed")
        return jsonify({"error": f"Processing error: {e}"}), 500


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"File too large. Max {MAX_UPLOAD_MB}MB."}), 413


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "session_loaded": SESSION is not None,
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_dimension": MAX_DIMENSION,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PEEL_PORT", 5000))
    debug = os.environ.get("PEEL_DEBUG", "false").lower() in ("1", "true", "yes")
    logger.info("Peel running -> http://localhost:%d (max_dim=%d)", port, MAX_DIMENSION)
    app.run(host="0.0.0.0", port=port, debug=debug)
