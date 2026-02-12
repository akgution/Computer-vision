import cv2
import os
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')

INPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, 'video.mp4')
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'out_video.mp4')


USE_WEBCAM = True
if USE_WEBCAM:
    source = 0
else:
    source = INPUT_VIDEO_PATH

MODEL_PATH = 'yolo8n.pt'

CONF_THRESHOLD = 0.5

TRAKER = "bytetrack.yalm"
SAVE_VIDEO = True

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(source)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

writer = None

if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))


seen_id_total = set()

seen_id_class = {}#лічільник обєктів

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.track(frame, conf = CONF_THRESHOLD, tracker=TRAKER, persist=True, verbose=True)

    r = result[0]

    if r.boxes is None or len(r.boxes) == 0:
        cv2.imshow('frame', frame)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = r.boxes

    xyxy = boxes.xyxy.cpu().numpy()

    cls = boxes.classes.cpu().numpy()
    conf = boxes.scores.cpu().numpy()

    tracks_id = None
    if boxes.id is not None:
        tracks_id = boxes.id.cpu().numpy()

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        class_id = int(cls[i])

        class_name = model.names[class_id]

        score = conf[i]

        tid = -1
        if tracks_id is not None:
            tid = int(tracks_id[i])

        if tid != -1:
            seen_id_total.add(tid)

            if class_name not in seen_id_class:
                seen_id_class[class_name] = set()
            seen_id_class[class_name].add(tid)

        label = f'{class_name} {score:.2f}'
        if tid != -1:
            label += f'ID {tid}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw +10, y1), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        total = len(seen_id_class)
        cv2.putText(frame, f'unique objects {total}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('frame', frame)

        if writer is not None:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()