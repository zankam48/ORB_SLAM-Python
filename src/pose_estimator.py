import numpy as np
import cv2

class PoseEstimator:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix

    def estimate_motion(self, keypoints1, keypoints2, matches):
        # extract matched keypoints
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # compute Essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        return E, mask, pts1, pts2

    def recover_pose(self, E, pts1, pts2):
        retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        return R, t, mask


def extractPose(F):
    # define the W matrix used for computing the rotation matrix
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
     
    # perform Singular Value Decomposition (SVD) on the Fundamental matrix F
    U, d, Vt = np.linalg.svd(F)
    assert np.linalg.det(U) > 0
 
    # correct Vt if its determinant is negative to ensure it's a proper rotation matrix
    if np.linalg.det(Vt) < 0:
        Vt *= -1
 
    # compute the initial rotation matrix R using U, W, and Vt
    R = np.dot(np.dot(U, W), Vt)
 
    # check the diagonal sum of R to ensure it's a proper rotation matrix
    # if not, recompute R using the transpose of W
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
 
    # extract the translation vector t from the third column of U
    t = U[:, 2]
 
    # initialize a 4x4 identity matrix to store the pose
    ret = np.eye(4)
 
    # set the top-left 3x3 submatrix to the rotation matrix R
    ret[:3, :3] = R
 
    # set the top-right 3x1 submatrix to the translation vector t
    ret[:3, 3] = t
 
    print(d)
 
    # return the 4x4 homogeneous transformation matrix representing the pose
    return ret

# extract matched keypoints
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
 
# camera intrinsic parameters (example values, replace with your camera's calibration data)
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
 
# compute the Fundamental matrix using RANSAC
F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
 
# compute the Essential matrix using the camera's intrinsic parameters 
E = K.T @ F @ K
 
# decompose the Essential matrix to get R and t
_, R, t, mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)