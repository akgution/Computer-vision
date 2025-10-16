from operator import truediv

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_red1 = np.array([0,100,100])
upper_red1 = np.array([10,255,255])

lower_red2 = np.array([160,100,100])
upper_red2 = np.array([180,255,255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    find = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            cv2.rectangle(frame, (2, 2), (637, 478), (0, 0, 255), 2)
            cv2.putText(frame, "object not detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        if area > 1000:
            cv2.rectangle(frame, (2, 2), (637, 478), (0, 255, 0), 2)
            cv2.putText(frame, "object detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)


    cv2.imshow('video', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()