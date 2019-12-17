from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os



#now we will construct the argument parser

ap = argparse.ArgumentParser()
ap.add_argument("-i","--dataset",required=True,help="path to input directory of faces + images")
ap.add_argument("-e","--encodings",required=True,help="path to serialised db of facial encodings")
#ap.add_argument("-d","--detection-method",type=str,default="cnn",help="face detection model to use...either hog or cnn")
args = vars(ap.parse_args())

#taking the user input
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))#creating a list of image paths

knownEncodings=[]
knownNames=[]

#Now we will loop over handsome Leo's pictures

for (i,imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i+1,len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2] #extract name from image path
    #load the input image and convert it from BGR to RGB (just dlib things) 
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb,
            model="hog")

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
f=open(args["encodings"],"wb")
f.write(pickle.dumps(data))
f.close
