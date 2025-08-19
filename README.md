# EMOTIONIQ — Real-Time Human Emotion Detection (No PyAV Build)

## Overview
Snapshot-based emotion detection app for Streamlit that does not require `av` or `streamlit-webrtc`. Supports browser camera snapshots, image upload, and a desktop-only OpenCV webcam preview.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Notes
- To enable classification, install `fer` (and optionally `retina-face` or `mtcnn`).
- For true realtime streaming in-browser, use a separate build with `streamlit-webrtc` + `av`.
