import cv2
import numpy as np

lowerBound=np.array([0,221,134])
upperBound=np.array([26,255,255])

cam= cv2.VideoCapture(1)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

while True:
    ret, img=cam.read()
    #img=cv2.resize(img,(340,220))
    img=cv2.resize(img,(480,360))

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
        if(x>239):
            print("Kanan")
        else:
            print("Kiri")
        #print (x,y)
        #x,y,w,h=cv2.boundingRect(conts[i])
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        #cv2.PutText(cv2.cv.fromarray(img), str(i+1),(x,y+h))#,font,(0,255,255))
    cv2.imshow("maskClose",maskClose)
    #cv2.imshow("maskOpen",maskOpen)
    #cv2.imshow("mask",mask)
    cv2.imshow("cam",img)
    cv2.waitKey(10)
