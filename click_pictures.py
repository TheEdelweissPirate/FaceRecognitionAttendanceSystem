import cv2
import os
import os.path
import argparse
cam = cv2.VideoCapture(0)
cv2.namedWindow("Capture")

ap = argparse.ArgumentParser()
#ap.add_argument("-w","--workdir",required=True,help="The path to your working directory")
ap.add_argument("-i","--ID",required=True,help="The identification number of student")
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


cam.release()
cv2.destroyAllWindows()
 
