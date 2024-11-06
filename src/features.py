import os
import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, nfeatures=1000):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)

    def detect_and_compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors


class FeatureMatcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_features(self, descriptors1, descriptors2):
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        return good_matches
    
    

def extract_orb_features(images):
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints_list = []
    descriptors_list = []
    for timestamp, image in images:

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        keypoints_list.append((timestamp, keypoints))
        descriptors_list.append((timestamp, descriptors))
    return keypoints_list, descriptors_list

def extract(img):
    orb = cv2.ORB_create()
 
    # Detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=-1).astype(np.uint8), 1000, qualityLevel=0.01, minDistance=10)
 
    # Extraction
    kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in pts]
    kps, des = orb.compute(img, kps)
 
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def visualize_keypoints(images, keypoints_list):
    for (timestamp, image), (_, keypoints) in zip(images, keypoints_list):
        image_with_keypoints = cv2.drawKeypoints(
            image, keypoints, None, color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imshow(f'ORB Keypoints - Timestamp: {timestamp}', image_with_keypoints)
        key = cv2.waitKey(0)
        if key == 27: 
            break
    cv2.destroyAllWindows()