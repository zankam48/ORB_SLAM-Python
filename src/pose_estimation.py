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
