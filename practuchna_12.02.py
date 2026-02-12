import os
import cv2
import csv
import json
import time
import math
import torch
import subprocess
import numpy as np
from collections import Counter
from ultralytics import YOLO


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)


YOUTUBE_URL = "https://www.youtube.com/live/Lxqcg1qt0XU"
CSV_PATH = os.path.join(OUT_DIR, "traffic_stats.csv")
CONF_THRESHOLD = 0.4
# Calibration: adjusted for ~60km/h average on this specific stream
METER_PER_PIXEL = 0.025

CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


def get_stream_url(url):
    """Get direct video URL using yt-dlp."""
    print("Extracting stream URL...")
    cmd = ["yt-dlp", "-j", "-f", "best[ext=mp4]", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return info["url"]


stream_url = get_stream_url(YOUTUBE_URL)
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    raise RuntimeError("Failed to open video stream")

fps_input = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


device = 0 if torch.cuda.is_available() else "cpu"
print(f"Running on DEVICE: {device}")
model = YOLO("yolov8s.pt")  # Small model as requested
model.to(device)


prev_centers = {}  # {id: (x, y)}
speed_history = {}  # {id: [list_of_speeds]}
last_y_position = {}  # {id: last_y}
crossed_down = set()
crossed_up = set()
total_counter = Counter()

# Line positions for crossing logic (relative to screen height)
LINE_TOP_Y = int(height * 0.55)
LINE_BOTTOM_Y = int(height * 0.60)

print("Processing... Press 'q' to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking
        results = model.track(
            frame,
            conf=CONF_THRESHOLD,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        # Draw visual crossing boundaries (Optional)
        cv2.line(frame, (0, LINE_TOP_Y), (width, LINE_TOP_Y), (255, 255, 0), 1)
        cv2.line(frame, (0, LINE_BOTTOM_Y), (width, LINE_BOTTOM_Y), (255, 255, 0), 1)

        new_centers = {}
        r = results[0]

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            track_ids = boxes.id.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for i in range(len(xyxy)):
                class_id = int(class_ids[i])
                if class_id not in CLASSES:
                    continue

                tid = int(track_ids[i])
                x1, y1, x2, y2 = xyxy[i].astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = CLASSES[class_id]

                new_centers[tid] = (cx, cy)

                # 1. Update unique object counter
                if tid not in last_y_position:
                    total_counter[class_name] += 1

                # 2. Crossing logic (Direction detection)
                if tid in last_y_position:
                    prev_y = last_y_position[tid]
                    if prev_y < LINE_TOP_Y <= cy:
                        crossed_down.add(tid)
                    elif prev_y > LINE_BOTTOM_Y >= cy:
                        crossed_up.add(tid)
                last_y_position[tid] = cy

                # 3. Speed calculation
                avg_speed = 0.0
                if tid in prev_centers:
                    dx = cx - prev_centers[tid][0]
                    dy = cy - prev_centers[tid][1]
                    dist_px = math.sqrt(dx ** 2 + dy ** 2)

                    # Convert pixels to km/h
                    instant_speed = (dist_px * METER_PER_PIXEL) * fps_input * 3.6

                    # Smoothing
                    if tid not in speed_history: speed_history[tid] = []
                    speed_history[tid].append(instant_speed)
                    if len(speed_history[tid]) > 15: speed_history[tid].pop(0)
                    avg_speed = np.mean(speed_history[tid])

                # 4. Visualization
                color = (0, 255, 0) if class_id != 0 else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} ID:{tid} | {avg_speed:.1f} km/h"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        prev_centers = new_centers

        # 5. UI: Overlay Stats
        h_box = 70 + 26 * len(total_counter)
        cv2.rectangle(frame, (5, 5), (280, h_box), (0, 0, 0), -1)

        y_text = 30
        cv2.putText(frame, "LIVE TRAFFIC STATS", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_text += 30

        for obj_type, count in total_counter.items():
            cv2.putText(frame, f"{obj_type.capitalize()}: {count}", (15, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_text += 25

        cv2.putText(frame, f"Down: {len(crossed_down)} | Up: {len(crossed_up)}", (15, y_text + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("Advanced Traffic Analytics", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"Error occurred: {e}")


with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    for obj_type, count in total_counter.items():
        writer.writerow([f"Total {obj_type}", count])
    writer.writerow(["Crossed Down (Incoming)", len(crossed_down)])
    writer.writerow(["Crossed Up (Outgoing)", len(crossed_up)])

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Process completed.")