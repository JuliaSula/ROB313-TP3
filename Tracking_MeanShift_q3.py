import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

roi_defined = False
 
def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True

#cap = cv2.VideoCapture('Test-Videos/VOT-Ball.mp4')
cap = cv2.VideoCapture('Test-Videos/VOT-Basket.mp4')
#cap = cv2.VideoCapture('Test-Videos/VOT-Car.mp4')
#cap = cv2.VideoCapture('Test-Videos/VOT-Sunshade.mp4')
#cap = cv2.VideoCapture('Test-Videos/VOT-Woman.mp4')
#cap = cv2.VideoCapture('Test-Videos/VOT-Antoine_Mug.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width, height)
# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r, c), (r+h, c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break
 
track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w,r:r+h]
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# computation mask of the histogram:
# Pixels with S<60 or V<32 are ignored 
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 2 )
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# Backproject the model histogram roi_hist onto the 
# current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

#dst[dst>123]=255

print(np.shape(dst))
cpt = 1
while(1):
    ret ,frame = cap.read()
    if ret == True: 
       
       
        # computation mask of the histogram:
        # Pixels with S<60 or V<32 are ignored 
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        # Marginal histogram of the Hue component
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        # Histogram values are normalised to [0,255]
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
       
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	    # Backproject the model histogram roi_hist onto the 
	    # current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        
        cv2.imshow('Backpropagation',dst)

        #Calculating gradient and argument 
        #Defining images
        img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img6 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
        img7 = cv2.copyMakeBorder(frame,0,0,0,0,cv2.BORDER_REPLICATE)
        img11=cv2.copyMakeBorder(frame,0,0,0,0,cv2.BORDER_REPLICATE)
        img10=cv2.copyMakeBorder(frame,0,0,0,0,cv2.BORDER_REPLICATE)
        
        # Calculating the partial derivatives
        cv2.imshow('Gradient', img6) 
        kernel1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        img11 = cv2.filter2D(img,-1,kernel1)
        kernel2 = np.array([[-1,-2,-1] ,[0,0,0], [1,2,1]])
        img10 = cv2.filter2D(img,-1,kernel2)
       
        # Defining gradient and argument images
        img6= np.hypot(img11,img10)/255
        img7=2*np.arctan2(np.float32(img11),np.float32(img10))
        
        
       
        
        # Argument du gradient: Histogram
        # Histogram calcul
        histOrientation= cv2.calcHist([img7], [0], None, [64], [0,64])
        hist1=cv2.normalize(histOrientation, histOrientation)
        # Histogram visualization
        cv2.imshow('Orientatio Hist', histOrientation)
        fig2, ax2 =plt.subplots()	
        ax2 = plt.gca()
        ax2.plot(histOrientation)
        plt.xlabel("Bins (entre 0 et 2 pi)")
        plt.ylabel("Nombre de pixels")
        plt.grid()
        plt.draw()
        fig2.canvas.draw()
        imgt = np.fromstring(fig2.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
        imgt  = imgt.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
        imgt = cv2.cvtColor(imgt,cv2.COLOR_RGB2BGR)
        cv2.imshow("Orientation",imgt)

        # Gradient visualisation
        imgf1=cv2.cvtColor(np.float32(img6), cv2.COLOR_GRAY2BGR)
        imgf=cv2.cvtColor(np.float32(img6), cv2.COLOR_GRAY2BGR)
        imgf[img6<0.1]=[0, 0, 255]
        cv2.imshow('Gradient', np.float32(imgf))
        cv2.imshow('Gradient', np.float32(imgf1))
        cv2.imshow('Orientation', img7)

        # apply meanshift to dst to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw a blue rectangle on the current image
        x,y,h,w = track_window
        frame_tracked = cv2.rectangle(frame, (x,y), (x+h,y+w), (255,0,0) ,2)
        cv2.imshow('Sequence',frame_tracked)
        
        # Roi visualization
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV ROI',hsv_roi)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
