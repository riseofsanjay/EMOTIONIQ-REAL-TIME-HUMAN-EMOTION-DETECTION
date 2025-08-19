# app.py
"""
EmotionIQ: Real‑Time Human Emotion Detection (no PyAV dependency)
- Streamlit UI with a tech‑inspired background
- Snapshot analysis via `st.camera_input` (browser) or file upload
- Optional local webcam mode using OpenCV's VideoCapture (desktop‑only)
- FER (https://github.com/justinshenk/fer) for pretrained emotion classification (if available)
- OpenCV for drawing/overlays

Why this rewrite?
  The prior version used `streamlit-webrtc` which depends on the `av` package. In this
  sandbox, `av` is unavailable, so this version removes that dependency and still lets you
  analyze frames for emotions.

Run locally:
  1) python -m venv .venv && source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
  2) pip install -r requirements.txt
  3) streamlit run app.py

Optional (to run tests):
  - pip install pytest
  - pytest -q
"""

from __future__ import annotations
import io
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Tuple, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# Try to import FER; degrade gracefully if not installed
try:
    from fer import FER  # type: ignore
except Exception:  # pragma: no cover
    FER = None  # sentinel; we will handle this in get_analyzer()

# ------------------------- Page & Theme -------------------------
st.set_page_config(page_title="EmotionIQ — Real‑Time Emotion Detection", page_icon="😃", layout="wide")

# Tech‑influenced background & glass UI
st.markdown(
    """
    <style>
    :root{ --accent:#6C63FF; --accent2:#00DBDE; --bg1:#0e1024; --bg2:#080a1a; --glass:rgba(255,255,255,0.08);}    
    [data-testid="stAppViewContainer"]{
        background:
            radial-gradient(900px 600px at 12% 10%, rgba(108,99,255,.16), transparent 60%),
            radial-gradient(900px 600px at 88% 12%, rgba(0,219,222,.12), transparent 55%),
            linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%);
    }
    [data-testid="stAppViewContainer"]::before{
        content:""; position:fixed; inset:0; pointer-events:none;
        background-image:
            linear-gradient(transparent 96%, rgba(255,255,255,.06) 96%),
            linear-gradient(90deg, transparent 96%, rgba(255,255,255,.06) 96%);
        background-size: 32px 32px, 32px 32px; opacity:.75; mix-blend-mode: screen;
    }
    .block-container{padding-top:2rem; padding-bottom:2rem;}
    .section-card{ background:var(--glass); border:1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(10px); border-radius:16px; padding:16px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.06);
    }
    [data-testid="stSidebar"]{ background: rgba(0,0,0,.35); backdrop-filter: blur(8px);
        border-right:1px solid rgba(255,255,255,.08); }
    .badge{display:inline-block; padding:4px 10px; border-radius:999px; font-size:.8rem;
        background: rgba(108,99,255,.18); border:1px solid rgba(108,99,255,.35);} 
    h1,h2,h3{letter-spacing:.5px}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Sidebar Controls -------------------------
st.sidebar.title("⚙️ EmotionIQ Controls")
threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.99, 0.6, 0.01)
smooth_k = st.sidebar.slider("Temporal smoothing (frames)", 1, 30, 7)
show_prob = st.sidebar.checkbox("Show probabilities", value=False)

# Only expose backend choice if FER is available
if FER is not None:
    face_detector_backend = st.sidebar.selectbox(
        "Face detector backend", ["opencv", "mtcnn", "retinaface"], index=0
    )
else:
    face_detector_backend = "opencv"

# ------------------------- Header -------------------------
st.markdown('<span class="badge">Snapshot • ML‑powered</span>', unsafe_allow_html=True)
st.title("EmotionIQ")
st.markdown("**Human Emotion Detection** — analyze webcam snapshots or uploaded images using a pretrained FER2013 model (if available).")
st.divider()

# ------------------------- Model Factory -------------------------
@st.cache_resource(show_spinner=False)
def get_analyzer(backend: str):
    """Return a FER analyzer using the selected face backend, or None if FER not installed."""
    if FER is None:
        return None
    if backend == "mtcnn":
        return FER(mtcnn=True)
    if backend == "retinaface":
        # Requires `retina-face` package installed
        return FER(mtcnn=False, retinaface=True)
    return FER()  # OpenCV

# ------------------------- Utilities & Smoothing -------------------------
@dataclass
class SmoothLabel:
    window: int
    buf: Deque[str]

    def update(self, label: str) -> None:
        self.buf.append(label)
        if len(self.buf) > self.window:
            self.buf.popleft()

    def mode(self) -> str:
        if not self.buf:
            return ""
        vals, counts = np.unique(np.array(self.buf), return_counts=True)
        return str(vals[int(np.argmax(counts))])

def pick_top_emotion(scores: Dict[str, float]) -> Tuple[str, float]:
    """Return (label, score) with the highest probability from a dict."""
    if not scores:
        return "", 0.0
    label = max(scores, key=scores.get)
    return label, float(scores[label])

# --------------- Robust image decoding (prevents UnidentifiedImageError) ---------------
class ImageDecodeError(Exception):
    pass


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes to an RGB numpy array using PIL first, then OpenCV fallback.
    Raises ImageDecodeError when both decoders fail or data is empty.
    """
    if not data:
        raise ImageDecodeError("Empty file uploaded.")

    # Try PIL first (handles EXIF orientation well)
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im.convert("RGB")) if hasattr(Image, "Image") else im.convert("RGB")
            return np.array(im)
    except Exception:
        pass

    # Fallback to OpenCV
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageDecodeError("Unsupported or corrupted image format.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


# ------------------------- Frame Analysis -------------------------
@st.cache_resource(show_spinner=False)
def get_smoother(window: int) -> SmoothLabel:
    return SmoothLabel(window=max(1, int(window)), buf=deque())


def analyze_frame(rgb_img: np.ndarray, analyzer, threshold: float, smoother: SmoothLabel, show_prob: bool) -> np.ndarray:
    """Run FER on an RGB image and return an annotated BGR image."""
    # Convert to BGR for drawing
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    if analyzer is None:
        # FER not available; draw a note
        cv2.putText(bgr, "FER model not installed — showing raw image.", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 255), 2)
        return bgr

    results = analyzer.detect_emotions(rgb_img)
    if results:
        for r in results:
            (x, y, w, h) = r["box"]
            emotions = r["emotions"]
            best_label, best_score = pick_top_emotion(emotions)

            if best_score >= float(threshold) and best_label:
                smoother.update(best_label)
                label_to_show = smoother.mode()
            else:
                label_to_show = "uncertain"

            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 200), 2)
            _text = f"{label_to_show} {best_score:.2f}"
            (tw, th), _ = cv2.getTextSize(_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(bgr, (x, y - th - 10), (x + tw + 10, y), (0, 0, 0), -1)
            cv2.putText(bgr, _text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if show_prob:
                # probs bar chart
                bar_h = max(12, h // 10)
                max_w = 120
                yy = y
                for k, v in sorted(emotions.items(), key=lambda kv: -kv[1]):
                    wbar = int(max_w * float(v))
                    cv2.rectangle(bgr, (x + w + 8, yy), (x + w + 8 + max_w, yy + bar_h), (40, 40, 40), 1)
                    cv2.rectangle(bgr, (x + w + 8, yy), (x + w + 8 + wbar, yy + bar_h), (60, 180, 75), -1)
                    cv2.putText(bgr, f"{k[:10]} {v:.2f}", (x + w + 12, yy + bar_h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    yy += bar_h + 4
    else:
        smoother.update("")

    cv2.putText(bgr, "EmotionIQ", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 255), 2)
    return bgr


# ------------------------- UI Modes -------------------------
mode = st.radio(
    "Choose input mode",
    ["Snapshot (Browser Camera)", "Upload Image", "Local Webcam (OpenCV; desktop only)"]
)

analyzer = get_analyzer(face_detector_backend)
smoother = get_smoother(smooth_k)

st.markdown('<div class="section-card">', unsafe_allow_html=True)

if mode == "Snapshot (Browser Camera)":
    snap = st.camera_input("Capture a frame")
    if snap is not None:
        try:
            rgb = decode_image_bytes(snap.getvalue())
        except ImageDecodeError as e:
            st.error(f"Could not read camera snapshot: {e}")
        else:
            out = analyze_frame(rgb, analyzer, threshold, smoother, show_prob)
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Analyzed snapshot", use_column_width=True)
    else:
        st.info("Use the camera widget above to capture a snapshot.")

elif mode == "Upload Image":
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up is not None:
        try:
            data = up.read()
            rgb = decode_image_bytes(data)
        except ImageDecodeError as e:
            st.error(f"Unsupported or corrupted image: {e}")
        else:
            out = analyze_frame(rgb, analyzer, threshold, smoother, show_prob)
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Analyzed image", use_column_width=True)
    else:
        st.info("Upload an image file (JPG/PNG/BMP/WEBP) to analyze emotions.")

else:  # Local Webcam (OpenCV)
    st.warning(
        "This mode tries to use OpenCV's VideoCapture(0). It only works when the Streamlit app runs on your own desktop with webcam access."
    )
    run = st.checkbox("Start local webcam preview")
    frame_limit = st.slider("Preview frames (to avoid endless loops)", 30, 600, 120, 10)
    if run:
        cap = cv2.VideoCapture(0)
        placeholder = st.empty()
        ok_frames = 0
        if not cap.isOpened():
            st.error("Could not open local webcam.")
        else:
            try:
                while ok_frames < frame_limit:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        st.error("Failed to read frame from webcam.")
                        break
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    out = analyze_frame(rgb, analyzer, threshold, smoother, show_prob)
                    placeholder.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)
                    ok_frames += 1
            finally:
                cap.release()

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------- Help -------------------------
with st.expander("🔧 Troubleshooting & Notes"):
    st.markdown(
        """
        - **No `av` needed**: This build avoids `streamlit-webrtc`/`PyAV` and uses snapshots for analysis.
        - **Real-time**: For browser-based real-time streaming you will need `streamlit-webrtc` + `av`.
        - **FER optional**: If `fer` isn't installed, the app still runs but won't classify emotions.
        - **Local webcam**: Works only when the app runs on your own desktop (OpenCV VideoCapture).
        - Backends: **OpenCV** (fast), **MTCNN/RetinaFace** (slower but often more accurate, require extra pkgs).
        - If uploads still fail for **HEIC/HEIF** photos (iPhone), install `pillow-heif` and add `from pillow_heif import register_heif; register_heif()` at the top.
        """
    )

# ------------------ requirements.txt (put in a separate file) ------------------
# streamlit==1.37.1
# opencv-python>=4.9.0.80
# numpy>=1.24
# pillow>=10.0.0
# pillow-heif>=0.15.0     # optional, to support HEIC/HEIF inputs
# fer==22.5.1             # optional, for emotion classification
# retina-face==0.0.16     # optional (for RetinaFace backend)
# mtcnn==0.1.1            # optional (for MTCNN backend)
# pytest>=8.0.0           # optional (to run tests)

# ------------------------- Tests (pytest auto-discovers) -------------------------
# These tests do not run during Streamlit; they run only when you call `pytest`.
# Keep them here so the module stays single‑file for convenience.

def test_pick_top_emotion_basic():
    scores = {"happy": 0.8, "sad": 0.1, "angry": 0.1}
    label, score = pick_top_emotion(scores)
    assert label == "happy"
    assert abs(score - 0.8) < 1e-9


def test_pick_top_emotion_empty():
    label, score = pick_top_emotion({})
    assert label == ""
    assert score == 0.0


def test_pick_top_emotion_tie_stability():
    # Python's max on dict keys is deterministic for a given dict insertion order
    scores = {"happy": 0.5, "sad": 0.5}
    label, score = pick_top_emotion(scores)
    assert label in {"happy", "sad"}
    assert abs(score - 0.5) < 1e-9


def test_smoothlabel_mode_majority():
    s = SmoothLabel(window=5, buf=deque())
    for lab in ["happy", "happy", "sad", "happy", "angry"]:
        s.update(lab)
    assert s.mode() == "happy"


def test_smoothlabel_window_slide():
    s = SmoothLabel(window=3, buf=deque())
    for lab in ["sad", "sad", "happy", "happy"]:
        s.update(lab)
    # Last 3 are: sad, happy, happy -> mode should be "happy"
    assert s.mode() == "happy"


def test_smoothlabel_empty_buffer():
    s = SmoothLabel(window=3, buf=deque())
    assert s.mode() == ""


def test_decode_image_bytes_with_pil_png():
    # Create a small RGB image via PIL and ensure decoder returns numpy array
    from PIL import Image as _Image
    buf = io.BytesIO()
    _Image.new("RGB", (4, 4), (123, 50, 200)).save(buf, format="PNG")
    arr = decode_image_bytes(buf.getvalue())
    assert isinstance(arr, np.ndarray) and arr.shape[2] == 3


def test_decode_image_bytes_invalid():
    import pytest
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"not-an-image")
