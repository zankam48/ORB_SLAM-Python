import cv2
import numpy as np
from utils import normalize

# class Frame(object):
 
#     def __init__(self, mapp, img, K):
#         self.K = K  # Intrinsic camera matrix
#         self.Kinv = np.linalg.inv(self.K)  # Inverse of the intrinsic camera matrix
#         self.pose = IRt  # Initial pose of the frame (assuming IRt is predefined)
 
#         self.id = len(mapp.frames)  # Unique ID for the frame based on the current number of frames in the map
#         mapp.frames.append(self)  # Add this frame to the map's list of frames
 
#         pts, self.des = extract(img)  # Extract feature points and descriptors from the image
#         self.pts = normalize(self.Kinv, pts)  # Normalize the feature points using the inverse intrinsic matrix


class Frame:
    frame_id = 0 

    def __init__(self, image, timestamp, camera, feature_extractor):
        self.id = Frame.frame_id
        Frame.frame_id += 1

        self.timestamp = timestamp
        self.original_image = image
        self.camera = camera
        self.keypoints = None
        self.descriptors = None
        self.pose = None  # 4x4 transformation matrix
        self.map_points = []  # associated map points

        # preprocess and extract features
        self.preprocess(feature_extractor)

    def preprocess(self, feature_extractor):
        # undistort image
        self.image = self.camera.undistort_image(self.original_image)
        # extract features
        self.keypoints, self.descriptors = feature_extractor.detect_and_compute(self.image)
        # convert keypoints to numpy array of points
        pts = np.array([kp.pt for kp in self.keypoints])
        # normalize points using inverse of K
        self.pts_normalized = normalize(self.Kinv, pts)

    def to_keyframe(self):
        # convert Frame to KeyFrame
        keyframe = KeyFrame(
            image=self.image,
            timestamp=self.timestamp,
            camera=self.camera,
            keypoints=self.keypoints,
            descriptors=self.descriptors,
            pose=self.pose,
            id=self.id
        )
        keyframe.map_points = self.map_points.copy()
        return keyframe

def match_frames(f1, f2):
    # BFMathcer with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(f1.des, f2.des, k=2)
    
    good_matches = []
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            p1 = f1.pts[m.queryIdx] 
            p2 = f2.pts[m.trainIdx]
             
            # Euclidean distance between p1 and p2 is less than 0.1
            if np.linalg.norm((p1-p2)) < 0.1:
                # Keep idxs
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                good_matches.append((p1, p2))
                pass
 
 
    assert len(good_matches) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
 
    # Fit matrix
    model, inliers = ransac((ret[:, 0], 
                            ret[:, 1]), FundamentalMatrixTransform, 
                            min_samples=8, residual_threshold=0.005, 
                            max_trials=200)
     
    
    # extract pose from fundamental matrices
    F = model.params
    E = f2.K.T @ F @ f1.K
    
    Rt = extract_pose(E, matched_pts1[inliers], matched_pts2[inliers], f1.K)
 
    return idx1[inliers], idx2[inliers], Rt

def extract_pose(E, pts1, pts2, K):
    """
    Extract rotation and translation from the essential matrix.
    """
    # convert points to homogeneous pixel coordinates
    pts1_px = pts1 * np.array([[K[0, 0], K[1, 1]]]) + np.array([[K[0, 2], K[1, 2]]])
    pts2_px = pts2 * np.array([[K[0, 0], K[1, 1]]]) + np.array([[K[0, 2], K[1, 2]]])

    # ensure points are of type float32
    pts1_px = pts1_px.astype(np.float32)
    pts2_px = pts2_px.astype(np.float32)

    # recover pose from the essential matrix
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_px, pts2_px, K)

    # 4x4 transformation matrix
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t.squeeze()

    return Rt
