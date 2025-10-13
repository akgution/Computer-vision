from colorsys import hsv_to_rgb

import cv2
import numpy as np
img = cv2.imread('images/rechi.jpg')
img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
img_copy = img.copy()
img = cv2.GaussianBlur(img,(5,5),2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([157, 42, 0])
upper_red = np.array([179, 255, 255])
mask_red = cv2.inRange(img, lower_red, upper_red)

lower_blue = np.array([65, 172, 54])
upper_blue = np.array([112, 255, 255])
mask_blue = cv2.inRange(img, lower_blue, upper_blue)

lower_green = np.array([33, 36, 0])
upper_green = np.array([94, 255, 255])
mask_green = cv2.inRange(img, lower_green, upper_green)

lower_yellow = np.array([5, 73, 85])
upper_yellow = np.array([40, 250, 255])
mask_yellow= cv2.inRange(img, lower_yellow, upper_yellow)

mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_yellow)
mask_total = cv2.bitwise_or(mask_total, mask_green)
img = cv2.bitwise_and(img, img, mask=mask_total)
contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)

        perimeter = round(cv2.arcLength(cnt, True), 2)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area)/(perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        # for h, s, v in hsv_to_rgb(h, s, v):
        #     if h > 10 or h >160 and h<179:
        #         a = "red"
        #     elif h >26 and h<35:
        #         a = "yellow"
        #     elif h > 36 and h < 85:
        #         a = "green"
        #     elif h > 101 and h < 130:
        #         a = "blue"
        if len(approx) == 4:
            shape = "square"
        elif len(approx) == 3:
            shape = "triangle"
        elif len(approx) > 8:
            shape = "oval"
        else:
            shape = "inshe"

        cv2.putText(img_copy, f'S:{area}, P:{perimeter}', (x,y -2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img_copy, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x,y -40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img_copy, f'shape:{shape}', (x, y -20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img_copy, f'color:{a}', (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


cv2.imshow('rechi', img)
cv2.imshow("mask", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()