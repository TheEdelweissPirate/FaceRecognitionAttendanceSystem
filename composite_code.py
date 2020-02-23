import cv2
import os
#import os.path
import argparse
import imutils
from imutils import paths
import face_recognition
import pickle
import numpy


#CODE TO CLICK PICTURES AND SAVE THEM IN SEPARATE FOLDERS ACCORDING TO THE ID NUMBER

knownEncodings = []
knownNames = []
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


#cv2.namedWindow("Capture")

ap = argparse.ArgumentParser()
#ap.add_argument("-w","--workdir",required=True,help="The path to your working directory")
ap.add_argument("-i","--ID",required=True,help="The identification number of student")
#ap.add_argument("-e","--encodings",required=True,help="path to serialised db of facial encodings")

args=vars(ap.parse_args())

print ("Press the space key to capture")

img_counter=0
#ID = raw_input("Enter the student ID")
#savePath=args["workdir"]-
savePath= "DATA"
if not os.path.exists(os.path.join(savePath,args["ID"])):
       os.mkdir(os.path.join(savePath,args["ID"]))
while True:
       ret, frame = cap.read()
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       faces=face_cascade.detectMultiScale(gray, 1.3, 5)
       for (x,y,w,h) in faces:
               cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
               font = cv2.FONT_HERSHEY_SIMPLEX
               cv2.putText(frame,"click now",(x,y), font, 1, (200,255,155), 2, cv2.LINE_AA)
       cv2.imshow('Click',frame)
       rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       rgb = imutils.resize(frame, width=750)
       r = frame.shape[1] / float(rgb.shape[1])
       boxes = face_recognition.face_locations(rgb,model="hog")
       for (top, right, bottom, left) in boxes:
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)
                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
       if img_counter<5:
              k= cv2.waitKey(1)
              
              #cv2.imshow("Capture",frame)
              clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
              grey=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
              cl1 = clahe.apply(grey)
              cv2.imshow("CLAHE",cl1)
              if k%256 == 32:
                     fileName = os.path.join(savePath,args["ID"],"IMAGE "+str(img_counter)+".png")
                     cv2.imwrite(fileName,cl1)
                     img_counter+=1
                     print("Image captured")
       else:

              print("5 images Captured")
              break


knownEncodings=[]
knownNames=[]
#CODE TO CREATE THE ENCODINGS


#taking the user input
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["ID"]))#creating a list of image paths



#Now we will loop over the pictures

for (i,imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i+1,len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2] #extract name from image path
    #load the input image and convert it from BGR to RGB (just dlib things) 
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb,model="hog")

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over the encodings
    for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)
# print(knownEncodings)
print(knownNames)
#creating a file with our encodings!
print("[INFO] serializing encodings...")
data={"encodings":knownEncodings,"names":knownNames}

f=open(os.path.join(savePath,args["ID"],"encodings.pickle"),"wb")
f.write(pickle.dumps(data))
f.close
