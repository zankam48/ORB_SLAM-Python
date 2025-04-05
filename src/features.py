import os
import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, nfeatures=1000):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)

    def detect_and_compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.method == 'gftt_orb':
            pts = cv2.goodFeaturesToTrack(gray, 1000, qualityLevel=0.01, minDistance=10)
            if pts is None:
                return [], None
            keypoints = [cv2.KeyPoint(p[0][0], p[0][1], 20) for p in pts]
            keypoints, descriptors = self.orb.compute(gray, keypoints)
        else:
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors


class FeatureMatcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # previosly True

    def match_features(self, descriptors1, descriptors2):
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        return good_matches
    
    

# def extract(img):
#     orb = cv2.ORB_create()
 
#     # convert img to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # detection
#     pts = cv2.goodFeaturesToTrack(gray, 1000, qualityLevel=0.01, minDistance=10)
 
#     if pts is None:
#         return [], None
    
#     # extraction
#     kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in pts]
#     kps, des = orb.compute(gray, kps)
 
#     if des is None:
#         return [], None

#     return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

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


def visualize_matches(img1, keypoints1, img2, keypoints2, matches):
    match_img = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Feature Matches", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
