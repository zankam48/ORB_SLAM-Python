import cv2

def feature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2000)
    kp = orb.detect(gray, None)
    kp, des = orb.compute(gray, kp)
    img1 = cv2.drawKeypoints(gray, kp, None, (0,0,255), flags=0)





