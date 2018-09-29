#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
import cv2
import numpy as np


if __name__ == "__main__":
	lowerBound=np.array([0,221,134])
	upperBound=np.array([65,255,255])
	#upperBound=np.array([26,255,255])
	cam= cv2.VideoCapture(6)
	kernelOpen=np.ones((5,5))
	kernelClose=np.ones((20,20))
	rospy.init_node('camera_data')
	pub = rospy.Publisher('camera_x', Int16, queue_size=10)
	pub1 = rospy.Publisher('camera_y', Int16, queue_size=10)
	pub2 = rospy.Publisher('depth', Int16, queue_size=10)
	try:
		while not rospy.is_shutdown():
			ret, img=cam.read()
			img=cv2.resize(img,(340,220))
			#img=cv2.resize(img,(480,360))
			
			#convert BGR to HSV
			imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
			# create the Mask
			mask=cv2.inRange(imgHSV,lowerBound,upperBound)
			#morphology
			maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
			maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
			
			maskFinal=maskClose
			(_,conts,_)=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
			
			cv2.drawContours(img,conts,-1,(255,0,0),3)
			for i in range(len(conts)):
				(x,y),radius = cv2.minEnclosingCircle(conts[i])
				center = (int(x),int(y))
				radius = int(radius)
				bola = cv2.circle(img,center,radius,(0,0,255),2)
				M = cv2.moments(conts[i])
				x = int(M['m10']/M['m00'])
				y = int(M['m01']/M['m00'])
        
				print (x,y,radius)
				pub.publish(x)
				pub1.publish(y)
				pub2.publish(radius)
			#cv2.imshow("maskClose",maskClose)
    
			cv2.imshow("cam",img)
			cv2.waitKey(10)
	except rospy.ROSInterruptException:
		pass
