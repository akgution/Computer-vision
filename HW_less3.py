import cv2
image = cv2.imread('images/myfoto.jpg')
image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
print(image.shape)
cv2.rectangle(image,(120,120),(175,175),(0,0,255),4)
cv2.putText(image, "Ksenia Gorska", (100,200), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255))
cv2.imshow('myfoto', image)

cv2.waitKey(0)
cv2.destroyAllWindows()