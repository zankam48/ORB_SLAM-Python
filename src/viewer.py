import open3d as o3d
import numpy as np

class Viewer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Open3D Viewer', width=1280, height=720)
        
        self.pcd = o3d.geometry.PointCloud()
        self.camera_trajectory = o3d.geometry.LineSet()
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        self.vis.add_geometry(coordinate_frame)
        
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.camera_trajectory)
    
    def update(self, points, camera_poses):
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.paint_uniform_color([1.0, 0.0, 0.0])  
        
        self.camera_trajectory.points = o3d.utility.Vector3dVector(camera_poses)
        lines = [[i, i+1] for i in range(len(camera_poses)-1)]
        self.camera_trajectory.lines = o3d.utility.Vector2iVector(lines)
        self.camera_trajectory.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0] for _ in lines])  # Green color for lines
        
        self.vis.update_geometry(self.pcd)
        self.vis.update_geometry(self.camera_trajectory)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run(self):
        while True:
            self.vis.poll_events()
            self.vis.update_renderer()

def main():
    points = np.array([
        [0, 0, 0],   
        [1, 0, 0],   
        [1, 1, 0],   
        [0, 1, 0],   
        [0, 0, 1],   
        [1, 0, 1],   
        [1, 1, 1],   
        [0, 1, 1],   
    ])

    camera_poses = np.array([
        [0, 0, -1],  
        [0, 0, -2],  
        [0, 0, -3],  
        [0, 0, -4],  
        [0, 0, -5],  
    ])

    viewer = Viewer()

    viewer.update(points, camera_poses)

    viewer.run()

if __name__ == '__main__':
    main()