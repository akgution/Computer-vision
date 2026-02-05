import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(OUT_DIR, exist_ok=True)


VIDEO_PATH = os.path.join(VIDEO_DIR, 'video.mp4')
OUTVIDEO_PATH = os.path.join(OUT_DIR, 'newvideo.mp4')

cap = cv2.VideoCapture(VIDEO_PATH)

fps_input = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTVIDEO_PATH, fourcc, fps_input, (width, height))

model = YOLO('yolov8n.pt')
CONF_THRESHOLD = 0.4

AUTO_CLASSES = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    auto_count = {name: 0 for name in AUTO_CLASSES.values()}

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in AUTO_CLASSES:
                continue

            label_name = AUTO_CLASSES[cls]
            auto_count[label_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'{label_name} {conf:.2f}',
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    y_offset = 30
    for name, count in auto_count.items():
        cv2.putText(
            frame,
            f'{name}: {count}',
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        y_offset += 30

    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 1.0 / dt

    cv2.putText(frame,f'FPS: {fps:.1f}',(20, y_offset + 10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 0),2)

    cv2.imshow('traffic', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()