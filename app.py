import argparse
import cv2
from fer import FER

def put_label(img, text, x, y):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    y_text = max(30, y - 10)
    cv2.rectangle(img, (x, y_text - th - 8), (x + tw + 6, y_text + 6), (0, 255, 0), -1)
    cv2.putText(img, text, (x + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def main():
    parser = argparse.ArgumentParser(description="Real-time Emotion Detection (FER)")
    parser.add_argument("--source", type=str, default="0", help="camera index or video path")
    parser.add_argument("--min_size", type=int, default=60, help="minimum face size (pixels)")
    parser.add_argument("--show_probs", action="store_true", help="show probability next to label")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    detector = FER(mtcnn=False)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_emotions(rgb)

        for r in results:
            (x, y, w, h) = r["box"]
            if w < args.min_size or h < args.min_size:
                continue
            emotions = r["emotions"]
            dom = max(emotions, key=emotions.get)
            prob = emotions[dom]
            label = f"{dom} {prob:.2f}" if args.show_probs else dom
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            put_label(frame, label, x, y)

        cv2.imshow("Emotion Detection - Real-time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
