import cv2

cam = cv2.VideoCapture(0)
cv2.namedWindow("Capture")

print ("Press the space key to capture")

img_counter=0
while True:
       if img_counter<5:
              k= cv2.waitKey(1)
              ret,frame = cam.read()
              cv2.imshow("Capture",frame)
              if k%256 == 32:
                     img_name="image {}.png".format(img_counter)
                     cv2.imwrite(img_name,frame)
                     img_counter+=1
                     print "Image captured"
       else:
              print "5 images Captured"
              break


cam.release()
cv2.destroyAllWindows()
