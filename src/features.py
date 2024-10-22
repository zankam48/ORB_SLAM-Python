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
        matches = self.matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches