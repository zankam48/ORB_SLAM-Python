# src/optimizer.py

import numpy as np
from scipy.optimize import least_squares

class Optimizer:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix

    def bundle_adjustment(self, frames, map_points):
        """
        Performs bundle adjustment on the provided frames and map points.

        Args:
            frames (list): List of Frame objects (keyframes).
            map_points (list): List of MapPoint objects observed in the frames.

        Returns:
            Updated frames and map_points with optimized poses and positions.
        """
        # Prepare data for optimization
        camera_params, point_3d, observations, cam_indices, point_indices = self.prepare_bundle_adjustment_data(frames, map_points)

        # Flatten parameters
        x0 = np.hstack((camera_params.ravel(), point_3d.ravel()))

        # Run optimization
        res = least_squares(
            fun=self.reprojection_error,
            x0=x0,
            jac='2-point',
            verbose=2,
            x_scale='jac',
            ftol=1e-4,
            method='lm',
            args=(self.camera_matrix, observations, cam_indices, point_indices)
        )

        # Update frames and map points with optimized parameters
        optimized_camera_params = res.x[:camera_params.size].reshape(camera_params.shape)
        optimized_point_3d = res.x[camera_params.size:].reshape(point_3d.shape)

        self.update_parameters(frames, map_points, optimized_camera_params, optimized_point_3d)

    def prepare_bundle_adjustment_data(self, frames, map_points):
        """
        Prepares the data needed for bundle adjustment.

        Returns:
            camera_params: Array of camera parameters.
            point_3d: Array of 3D point positions.
            observations: Array of observed 2D keypoints.
            cam_indices: Array of camera indices for each observation.
            point_indices: Array of point indices for each observation.
        """
        camera_params = []
        point_3d = []
        observations = []
        cam_indices = []
        point_indices = []

        # Mapping from MapPoint id to index in point_3d array
        point_id_to_index = {}
        for idx, p in enumerate(map_points):
            point_3d.append(p.position)
            point_id_to_index[p.id] = idx

        # Mapping from Frame id to index in camera_params array
        frame_id_to_index = {}
        for idx, f in enumerate(frames):
            R_vec, _ = cv2.Rodrigues(f.pose[:3, :3])
            t_vec = f.pose[:3, 3]
            camera_params.append(np.hstack((R_vec.ravel(), t_vec)))
            frame_id_to_index[f.id] = idx

        # Collect observations
        for i, f in enumerate(frames):
            for mp_id, kp_idx in f.observations.items():
                if mp_id in point_id_to_index:
                    mp_idx = point_id_to_index[mp_id]
                    kp = f.keypoints[kp_idx].pt
                    observations.append(kp)
                    cam_indices.append(i)
                    point_indices.append(mp_idx)

        camera_params = np.array(camera_params)
        point_3d = np.array(point_3d)
        observations = np.array(observations)
        cam_indices = np.array(cam_indices, dtype=int)
        point_indices = np.array(point_indices, dtype=int)

        return camera_params, point_3d, observations, cam_indices, point_indices

    def reprojection_error(self, params, camera_matrix, observations, cam_indices, point_indices):
        """
        Computes the reprojection error.

        Args:
            params: Flattened array of camera parameters and 3D points.
            camera_matrix: Camera intrinsic matrix.
            observations: Observed 2D keypoints.
            cam_indices: Indices of cameras for each observation.
            point_indices: Indices of 3D points for each observation.

        Returns:
            Residuals vector.
        """
        n_cameras = np.max(cam_indices) + 1
        n_points = np.max(point_indices) + 1
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        point_3d = params[n_cameras * 6:].reshape((n_points, 3))

        residuals = []
        for i in range(observations.shape[0]):
            cam_idx = cam_indices[i]
            pt_idx = point_indices[i]
            obs = observations[i]

            # Project the 3D point into the camera
            projected_pt = self.project_point(camera_params[cam_idx], point_3d[pt_idx], camera_matrix)

            # Compute the residual (reprojection error)
            residual = obs - projected_pt
            residuals.append(residual)

        residuals = np.array(residuals).ravel()
        return residuals

    def project_point(self, cam_params, point_3d, camera_matrix):
        """
        Projects a 3D point into the camera using the camera parameters.

        Args:
            cam_params: Camera parameters (rotation vector and translation vector).
            point_3d: 3D point coordinates.
            camera_matrix: Camera intrinsic matrix.

        Returns:
            2D point projection.
        """
        R_vec = cam_params[:3]
        t_vec = cam_params[3:6]

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(R_vec)

        # Transform the 3D point to camera coordinate system
        point_cam = R @ point_3d + t_vec

        # Project the point using the camera intrinsic parameters
        point_proj = camera_matrix @ point_cam

        # Normalize to get pixel coordinates
        point_proj = point_proj[:2] / point_proj[2]
        return point_proj

    def update_parameters(self, frames, map_points, optimized_camera_params, optimized_point_3d):
        """
        Updates the frames and map points with the optimized parameters.
        """
        # Update camera poses
        for idx, f in enumerate(frames):
            cam_params = optimized_camera_params[idx]
            R_vec = cam_params[:3]
            t_vec = cam_params[3:6]
            R, _ = cv2.Rodrigues(R_vec)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t_vec
            f.pose = pose

        # Update map point positions
        for idx, p in enumerate(map_points):
            p.position = optimized_point_3d[idx]


# learn opencv
def vertex_to_points(optimizer, first_point_id, last_point_id): 
    """ Converts g2o point vertices to their current 3d point estimates. """
    vertices_dict = optimizer.vertices()
    estimated_points = list()
    for idx in range(first_point_id, last_point_id): 
        estimated_points.append(vertices_dict[idx].estimate())
    estimated_points = np.array(estimated_points)
    xs, ys, zs = estimated_points.T
    return xs, ys, zs 
 
ipv.show()
for i in range(100):
    optimizer.optimize(1)
    xs, ys, zs = vertex_to_points(optimizer, first_point_id, last_point_id)
    scatter_plot.x = xs 
    scatter_plot.y = ys 
    scatter_plot.z = zs 
    time.sleep(0.25)

def compute_reprojection_error(intrinsics, extrinsics, points_3d, observations):
    total_error = 0  # Initialize the total error to zero
    num_points = 0  # Initialize the number of points to zero
     
    # Iterate through each camera's extrinsics and corresponding 2D observations
    for (rotation, translation), obs in zip(extrinsics, observations):
        # Project the 3D points to 2D using the current camera's intrinsics and extrinsics
        projected_points = project_points(points_3d, intrinsics, rotation, translation)
         
        # Calculate the Euclidean distance (reprojection error) between the projected points and the observed points
        error = np.linalg.norm(projected_points - obs, axis=1)
         
        # Accumulate the total error
        total_error += np.sum(error)
         
        # Accumulate the total number of points
        num_points += len(points_3d)
     
    # Calculate the mean reprojection error
    mean_error = total_error / num_points
     
    return mean_error  # Return the mean reprojection error