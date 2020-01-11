#Simple program to read and show an image
import cv2
img=cv2.imread('Sketchpad.png')
gray=cv2.imread('Sketchpad.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Sketchpad chutti',img)
cv2.imshow('grey',gray)
cv2.waitKey(0) #0 means wait for infinite amount of time,if 25 it means wait for 25millisec before window destroy,programs stops if any key pressed
cv2.destroyAllWindows()