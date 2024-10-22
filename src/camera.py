# src/camera.py

import numpy as np
import cv2

class Camera:
    def __init__(self):
        # Camera parameters
        self.fx = 517.3
        self.fy = 516.5
        self.cx = 318.6
        self.cy = 255.3
        self.d0 = 0.2624
        self.d1 = -0.9531
        self.d2 = -0.0054
        self.d3 = 0.0026
        self.d4 = 1.1633

        # Construct camera intrinsic matrix
        self.camera_matrix = np.array([
            [self.fx,  0,  self.cx],
            [0, self.fy,  self.cy],
            [0,  0,   1]
        ], dtype=np.float64)

        # Distortion coefficients
        self.dist_coeffs = np.array([self.d0, self.d1, self.d2, self.d3, self.d4], dtype=np.float64)

    def undistort_image(self, image):
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        return undistorted_image
