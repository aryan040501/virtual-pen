import cv2
import numpy as np
import time

#doing nothing
def nothing(x):
	pass

#clear variable 
clear = False

#noise threshold
noiseth = 500

#wiper threshold
wiperth = 40000

#webcam

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5,5),np.uint8)

#initializinf canvas
canvas = None
x1,y1= 0,0

#creating window

cv2.namedWindow("Trackbars")

#trackbar control
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
	#reading  frame by frame
	ret, frame = cap.read()
	if not ret:
		break
	#flip (required)....................................................
	frame = cv2.flip(frame, 1)

	#initialize canvas as frame
	if canvas is None:
		canvas = np.zeros_like(frame)

	#convert bgr to hsv image
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#new values to trackbar
	l_h = cv2.getTrackbarPos("L - H", "Trackbars")
	l_s = cv2.getTrackbarPos("L - S", "Trackbars")
	l_v = cv2.getTrackbarPos("L - V", "Trackbars")
	u_h = cv2.getTrackbarPos("U - H", "Trackbars")
	u_s = cv2.getTrackbarPos("U - S", "Trackbars")
	u_v = cv2.getTrackbarPos("U - V", "Trackbars")

	#set lower and upper hsv range
	lower_range = np.array([99, 106, l_v])
	upper_range = np.array([u_h, u_s, u_v])

	#filter image where white represents target
	mask = cv2.inRange(hsv, lower_range, upper_range)

	#visualise target (not required).................................................
	res = cv2.bitwise_and(frame, frame, mask=mask)

	#convert binary mask to 3 channel image so we can stack with others
	mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

	#stack mask, frame and filtered result
	stacked = np.hstack((mask_3, frame, res))

	#find contours
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	#contour is big
	if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) >noiseth:
		#biggest contour
		c= max(contours, key=cv2.contourArea)

		#get bounding box coordinate
		x2,y2,w,h = cv2.boundingRect(c)

		#get area of contour
		area = cv2.contourArea(c)

		#Draw bounding box
		cv2.rectangle(frame,(x2,y2),(x2+w,y2+h),(0,25,255),2)

		#for writing
		if x1==0 and y1==0:
			x1,y1 = x2,y2

		else:
			#draw line
			canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 4)

		#afterdrawing change origin point
		x1,y1 = x2,y2

		#for clearing
		if area > wiperth:
			cv2.putText(canvas, 'Clearing Canvas',(100,200),
			cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255),5, cv2.LINE_AA)
			clear = True

	else:
		x1,y1 = 0,0

	#Merge the canvas and the frame
	frame = cv2.add(frame,canvas)

	stacked = np.hstack((canvas,frame))
	cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.6,fy=0.6))

	#esc sequence if 'esc is pressed'
	key = cv2.waitKey(1)
	if key == 27:
		break

	#clear canvas when 'c' is pressed
	if key == ord('c'):
		canvas = None

	#if 's' is pressed print
	if key == ord('s'):

		thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
		print(thearray)

		#save as penval.npy
		np.save('penval',thearray)
		break

	#clear canvas if clear is true
	if clear == True:
		time.sleep(1)
		canvas = None

		#reset clear
		clear = None
#realease camera and destroy windows
cap.release()
cv2.destroyAllWindows()

    