import numpy as np
import cv2

class PoseEstimator:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix

    def estimate_motion(self, keypoints1, keypoints2, matches):
        # Extract matched keypoints
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # Compute Essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        return E, mask, pts1, pts2

    def recover_pose(self, E, pts1, pts2):
        retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        return R, t, mask


# Extract matched keypoints
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
 
# Camera intrinsic parameters (example values, replace with your camera's calibration data)
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
 
# Compute the Fundamental matrix using RANSAC
F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
 
# Compute the Essential matrix using the camera's intrinsic parameters 
E = K.T @ F @ K
 
# Decompose the Essential matrix to get R and t
_, R, t, mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)