

class Frame(object):
 
    def __init__(self, mapp, img, K):
        self.K = K  # Intrinsic camera matrix
        self.Kinv = np.linalg.inv(self.K)  # Inverse of the intrinsic camera matrix
        self.pose = IRt  # Initial pose of the frame (assuming IRt is predefined)
 
        self.id = len(mapp.frames)  # Unique ID for the frame based on the current number of frames in the map
        mapp.frames.append(self)  # Add this frame to the map's list of frames
 
        pts, self.des = extract(img)  # Extract feature points and descriptors from the image
        self.pts = normalize(self.Kinv, pts)  # Normalize the feature points using the inverse intrinsic matrix


# src/frame.py

class Frame:
    frame_id = 0  # Class variable to assign unique IDs

    def __init__(self, image, timestamp, camera, feature_extractor):
        self.id = Frame.frame_id
        Frame.frame_id += 1

        self.timestamp = timestamp
        self.original_image = image
        self.camera = camera
        self.keypoints = None
        self.descriptors = None
        self.pose = None  # 4x4 transformation matrix
        self.map_points = []  # Associated map points

        # Preprocess and extract features
        self.preprocess(feature_extractor)

    def preprocess(self, feature_extractor):
        # Undistort image
        self.image = self.camera.undistort_image(self.original_image)
        # Extract features
        self.keypoints, self.descriptors = feature_extractor.detect_and_compute(self.image)

    def to_keyframe(self):
        # Convert Frame to KeyFrame
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
    # The code performs k-nearest neighbors matching on feature descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)
 
    # applies Lowe's ratio test to filter out good 
    # matches based on a distance threshold.
    ret = []
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            p1 = f1.pts[m.queryIdx] 
            p2 = f2.pts[m.trainIdx]
             
            # Distance test
            # dditional distance test, ensuring that the 
            # Euclidean distance between p1 and p2 is less than 0.1
            if np.linalg.norm((p1-p2)) < 0.1:
                # Keep idxs
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((p1, p2))
                pass
 
 
    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
 
    # Fit matrix
    model, inliers = ransac((ret[:, 0], 
                            ret[:, 1]), FundamentalMatrixTransform, 
                            min_samples=8, residual_threshold=0.005, 
                            max_trials=200)
     
    # Ignore outliers
    ret = ret[inliers]
    Rt = extractPose(model.params)
 
    return idx1[inliers], idx2[inliers], Rt

