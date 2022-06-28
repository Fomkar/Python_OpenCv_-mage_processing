import cv2
import sys

# Get user supplied values

cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread("image2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=2,
    minSize=(50, 50),
    # flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi=image[y:y+h,x:x+w]
    cv2.imshow("rect"+str(len(faces)),roi)
    cv2.namedWindow("Faces found",cv2.WINDOW_NORMAL)
    cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()