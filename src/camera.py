# src/camera.py

import numpy as np
import cv2

class Camera:
    def __init__(self, fx=None, fy=None, cx=None, cy=None, dist_coeffs=None):
        # camera params
        self.fx = fx if fx is not None else 517.3
        self.fy = fy if fy is not None else 516.5
        self.cx = cx if cx is not None else 318.6
        self.cy = cy if cy is not None else 255.3
        self.d0 = 0.2624
        self.d1 = -0.9531
        self.d2 = -0.0054
        self.d3 = 0.0026
        self.d4 = 1.1633

        # intrinsic matrix
        self.camera_matrix = np.array([
            [self.fx,  0,  self.cx],
            [0, self.fy,  self.cy],
            [0,  0,   1]
        ], dtype=np.float64)

        # distortion coefficients
        self.dist_coeffs = np.array([self.d0, self.d1, self.d2, self.d3, self.d4], dtype=np.float64)

    def undistort_image(self, image):
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        return undistorted_image

    def pixel_to_camera(self, pixel, depth):
        x = (pixel[0] - self.cx) * depth / self.fx
        y = (pixel[1] - self.cy) * depth / self.fy
        return np.array([x, y, depth])

    def camera_to_pixel(self, point3D):
        u = (point3D[0] * self.fx) / point3D[2] + self.cx
        v = (point3D[1] * self.fy) / point3D[2] + self.cy
        return np.array([u, v])

    def get_intrinsics(self):
        return self.camera_matrix

    def get_distortion(self):
        return self.dist_coeffs

    # scale intrinsices func if i need to resize the image

