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


def extractPose(F):
    # Define the W matrix used for computing the rotation matrix
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
     
    # Perform Singular Value Decomposition (SVD) on the Fundamental matrix F
    U, d, Vt = np.linalg.svd(F)
    assert np.linalg.det(U) > 0
 
    # Correct Vt if its determinant is negative to ensure it's a proper rotation matrix
    if np.linalg.det(Vt) < 0:
        Vt *= -1
 
    # Compute the initial rotation matrix R using U, W, and Vt
    R = np.dot(np.dot(U, W), Vt)
 
    # Check the diagonal sum of R to ensure it's a proper rotation matrix
    # If not, recompute R using the transpose of W
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
 
    # Extract the translation vector t from the third column of U
    t = U[:, 2]
 
    # Initialize a 4x4 identity matrix to store the pose
    ret = np.eye(4)
 
    # Set the top-left 3x3 submatrix to the rotation matrix R
    ret[:3, :3] = R
 
    # Set the top-right 3x1 submatrix to the translation vector t
    ret[:3, 3] = t
 
    print(d)
 
    # Return the 4x4 homogeneous transformation matrix representing the pose
    return ret

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