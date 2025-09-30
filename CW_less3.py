import cv2
import numpy as np
img = np.zeros((500,400,3), np.uint8)
# img[:] = 0, 0, 79
# rgb = bgr

# img[100:150, 200:250] = 0, 0, 79 #y потім х

cv2.rectangle(img,(100,100),(200,200),(0,0,79),1)

cv2.line(img,(100,100),(200,200),(0,0,79),1)

print(img.shape)
cv2.line(img, (0, img.shape[0]//2), (img.shape[1], img.shape[0]//2), (0, 0, 79), 1)
cv2.line(img, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (0, 0, 79), 1)

cv2.circle(img, (200, 200), 30, (0,0,79), -1)
cv2.putText(img, "Ksenia Gorska", (250,150), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,79))


cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()