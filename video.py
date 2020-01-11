#Read a Video Stream from Camera(Frame by Frame) video is made up frame by frame
import cv2

cap= cv2.VideoCapture(0)    #capturing device from which video, (id) default web cam

while True:
	ret,frame= cap.read()   #this method returns boolean value (if False the frame has not been captured properly)
	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   #frame,const                     #and frame that has been captured

	if ret == False:
		continue #continues the loop in that case


	cv2.imshow("Video Frame",frame)
	cv2.imshow("gray_frame",gray_frame)


	#Wait for user input -q , then you will stop the loop
	key_pressed= cv2.waitKey(1) & 0xFF #program will wait for 1 millisec before next iter comes in,bitwise
	if key_pressed == ord('q'): #ord method in python tells the ascii val of the char
		break 


cap.release()
cv2.destroyAllWindows()

