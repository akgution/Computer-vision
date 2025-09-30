# 1
# import cv2
# image = cv2.imread('images/myfoto.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
# image = cv2.Canny(image, 70, 70)
# print(image.shape)
# cv2.imshow('my photo', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 2
# import cv2
# import numpy as np
# image = cv2.imread('images/email1.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
# image = cv2.resize(image,(image.shape[1]//4,image.shape[0]//4))
# image = cv2.Canny(image,110,110)
# kernel = np.ones((3, 3), np.uint8)
# image = cv2.dilate(image, kernel, iterations=1)
# cv2.imshow('gmail', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()