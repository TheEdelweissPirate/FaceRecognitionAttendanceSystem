import cv2
import os
#import os.path
import argparse
from imutils import paths
import face_recognition
import pickle

#CODE TO CLICK PICTURES AND SAVE THEM IN SEPARATE FOLDERS ACCORDING TO THE ID NUMBER

knownEncodings = []
knownNames = []
cam = cv2.VideoCapture(0)
cv2.namedWindow("Capture")

ap = argparse.ArgumentParser()
#ap.add_argument("-w","--workdir",required=True,help="The path to your working directory")
ap.add_argument("-i","--ID",required=True,help="The identification number of student")
ap.add_argument("-e","--encodings",required=True,help="path to serialised db of facial encodings")

args=vars(ap.parse_args())

print ("Press the space key to capture")


img_counter=0
#ID = raw_input("Enter the student ID")
#savePath=args["workdir"]
savePath= "C:\Users\Hari kumar\Desktop\PHASE01"
os.mkdir(os.path.join(savePath,args["ID"]))
while True:
       
       if img_counter<5:
              
              k= cv2.waitKey(1)
              ret,frame = cam.read()
              cv2.imshow("Capture",frame)
              if k%256 == 32:
                     
                     fileName = os.path.join(savePath,args["ID"],"IMAGE "+str(img_counter)+".png")
                     cv2.imwrite(fileName,frame)
                     img_counter+=1
                     print "Image captured"
       else:

              print "5 images Captured"
              break




knownEncodings=[]
knownNames=[]
#CODE TO CREATE THE ENCODINGS


#taking the user input
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["ID"]))#creating a list of image paths



#Now we will loop over handsome Leo's pictures

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

#creating a file with our encodings!
print("[INFO] serializing encodings...")
data={"encodings":knownEncodings,"names":knownNames}

f=open(os.path.join(savePath,args["ID"],"encodings.pickle"),"wb")
f.write(pickle.dumps(data))
f.close
