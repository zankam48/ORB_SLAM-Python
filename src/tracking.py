# src/tracking.py

import cv2
import numpy as np
from src.frame import Frame
from src.map import Map
from src.pose_estimator import PoseEstimator

class Tracking:
    def __init__(self, camera, feature_extractor, map):
        self.camera = camera
        self.feature_extractor = feature_extractor
        self.map = map
        self.pose_estimator = PoseEstimator(camera.camera_matrix)
        self.last_frame = None
        self.current_frame = None

    def process_frame(self, image, timestamp):
        # Create a new Frame
        self.current_frame = Frame(image, timestamp, self.camera, self.feature_extractor)

        if self.last_frame is None:
            # Initialization
            self.last_frame = self.current_frame
            return

        # Feature matching with the last frame
        matches = self.match_frames(self.last_frame, self.current_frame)

        # Estimate motion using matches
        success = self.estimate_current_pose(matches)

        if not success:
            # Handle tracking failure
            print("Tracking failed.")
            return

        # Track the local map to refine pose
        self.track_local_map()

        # Decide if we need a new keyframe
        if self.need_new_keyframe():
            self.create_new_keyframe()

        self.last_frame = self.current_frame

    def match_frames(self, last_frame, current_frame):
        # Implement feature matching between frames
        # Similar to the code in feature_matcher.py
        pass  # Replace with actual implementation

    def estimate_current_pose(self, matches):
        # Use matches to estimate the current frame's pose
        # Similar to the code in pose_estimator.py
        pass  # Replace with actual implementation

    def track_local_map(self):
        # Get local map points
        local_map_points = self.map.get_local_map_points(self.current_frame)

        if len(local_map_points) < 20:
            print("Not enough local map points to track.")
            return False

        # Prepare 2D-3D correspondences
        keypoints = self.current_frame.keypoints
        descriptors = self.current_frame.descriptors

        # Match current frame descriptors with local map points
        map_descriptors = np.array([mp.descriptor for mp in local_map_points])
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf_matcher.match(descriptors, map_descriptors)

        # Prepare data for PnP
        object_points = []
        image_points = []
        for m in matches:
            idx = m.queryIdx
            mp_idx = m.trainIdx
            object_points.append(local_map_points[mp_idx].position)
            image_points.append(keypoints[idx].pt)

        object_points = np.array(object_points)
        image_points = np.array(image_points)

        # Solve PnP to refine pose
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.camera.camera_matrix,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if retval:
            R, _ = cv2.Rodrigues(rvec)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = tvec.ravel()
            self.current_frame.pose = pose
            print("Pose refined using local map.")
            return True
        else:
            print("PnP failed during local map tracking.")
            return False

    def need_new_keyframe(self):
        # Implement criteria for deciding if a new keyframe is needed
        # This will be explained in the next section
        pass  # Replace with actual implementation

    def create_new_keyframe(self):
        # Convert current frame to a keyframe and add to the map
        keyframe = self.current_frame.to_keyframe()
        self.map.add_keyframe(keyframe)
        print("New keyframe created.")

        # Optionally, add new map points
        # Implement triangulation and add new map points to the map

