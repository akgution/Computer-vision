import cv2
import numpy as np

img = np.zeros((400, 600, 3), np.uint8)
img[:] = 184, 154, 149

photo = cv2.imread("images/photo.jpg")
photo = cv2.resize(photo, (120, 150))

photo1 = cv2.imread("images/qrcode.png")
photo1 = cv2.resize(photo1, (150, 150))

img[240:390, 440:590] = photo1
img[10:160, 10:130] = photo

cv2.putText(img, "Ksenia Gorska", (150, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
cv2.putText(img, "Student", (150, 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
cv2.putText(img, "email: ksu.gorska@gmail.com", (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
cv2.putText(img, "phone num: +380987416464", (150, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
cv2.putText(img, "birth day: 21/06/2010", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
cv2.putText(img, "OpenCV Business Card ", (20, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))

cv2.imshow("business_card", img)
cv2.imwrite("business_card.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()