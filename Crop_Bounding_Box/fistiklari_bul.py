import numpy as np
import cv2

# The input image.
image = cv2.imread("Basler_acA1440-73gc__40038474__20220325_161252565_0561.bmp", 1)
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
params = cv2.SimpleBlobDetector_Params()

#Define thresholds
#Can define thresholdStep. See documentation. 
params.minThreshold = 60
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 1500
params.maxArea = 100000

# Filter by Color (black=0)
params.filterByColor = False  #Set true for cast_iron as we'll be detecting black regions
params.blobColor = 0

# Filter by Circularity


# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
params.maxConvexity = 1

# Filter by InertiaRatio
params.filterByInertia = True
params.minInertiaRatio = 0
params.maxInertiaRatio = 1



detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(gray_image)
img_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

new_image = np.ones(shape=[600, 800, 3], dtype=np.uint8)

y_counter=55

cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)   
cv2.imshow("Keypoints", img_with_blobs)
cv2.waitKey(1800)
for i in range(len(keypoints)):
    y_counter+=55
    x = keypoints[i].pt[0] 
    y = keypoints[i].pt[1]
    blob_size = keypoints[i].size
    cv2.putText(new_image, "X,Y : {}, {}".format(int(x),int(y)), (5,y_counter), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 50, 200), 2)
    cv2.putText(img_with_blobs,str(i),(int(x)-5,int(y)+10),cv2.FONT_HERSHEY_PLAIN,2,(0,y_counter/5,i*40),2)
    #Crop images:
    x1=int(x)-int(blob_size/2)-20
    x2=int(x)+int(blob_size/2)+20
    
    y1=int(y)-int(blob_size/2)-20
    y2=int(y)+int(blob_size/2)+20
    
    w=x2-x1
    h=y2-y1
    
    y=int(y)
    x=int(x)

    crop_image=img_with_blobs[y1:y+h,x1:x+w]
    cv2.namedWindow("crop_images"+str(i), cv2.WINDOW_NORMAL)   
    cv2.imshow("crop_images"+str(i),crop_image)
    cv2.waitKey(1500)
    # if cv2.waitKey() & 0xFF == ord('q'):
    #     # cv2.destroyAllWindows()
    #     # cv2.destroyWindow("Keypoints")
    #     continue

cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)   
cv2.imshow("Keypoints", img_with_blobs)
    
cv2.putText(new_image, "Bulunan Blob Sayisi : {}".format(int(len(keypoints))), (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
cv2.namedWindow("new_image", cv2.WINDOW_NORMAL)
cv2.imshow("new_image", new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()