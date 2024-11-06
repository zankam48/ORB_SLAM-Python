import os
import cv2
import numpy as np

def read_image_file_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    image_files = []
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue  
        parts = line.strip().split()
        timestamp = float(parts[0])
        filename = parts[1]
        image_files.append((timestamp, filename))
    return image_files

def load_images(image_dir, image_file_list):
    images = []
    for timestamp, filename in image_file_list:
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            images.append((timestamp, image))
        else:
            print(f"Warning: Could not load image {image_path}")
    return images

def add_ones(pts):
    return np.hstack([pts, np.ones((pts.shape[0], 1))])

def add_ones(x):
    # creates homogenious coordinates given the point x
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
 
def triangulate(pose1, pose2, pts1, pts2):
    # Initialize the result array to store the homogeneous coordinates of the 3D points
    ret = np.zeros((pts1.shape[0], 4))
 
    # Invert the camera poses to get the projection matrices
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
 
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        # Initialize the matrix A to hold the linear equations
        A = np.zeros((4, 4))
 
        # Populate the matrix A with the equations derived from the projection matrices and the points
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
 
        # Perform SVD on A
        _, _, vt = np.linalg.svd(A)
 
        # The solution is the last row of V transposed (V^T), corresponding to the smallest singular value
        ret[i] = vt[3]
 
    # Return the 3D points in homogeneous coordinates
    return ret

def normalize(Kinv, pts):
    # The inverse camera intrinsic matrix 𝐾^(−1) transforms 2D homogeneous points 
    # from pixel coordinates to normalized image coordinates. 
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
    # Converts a normalized point to pixel coordinates by applying the intrinsic camera matrix and normalizing the result.
    ret = np.dot(K, [pt[0], pt[1], 1.0])
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))