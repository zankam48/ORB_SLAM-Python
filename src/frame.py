

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

