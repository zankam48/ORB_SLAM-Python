from multiprocessing import Process, Queue
import numpy as np
import open3d as o3d
import g2o

# Global map // 3D map visualization using Open3D
class Map(object):
    def __init__(self):
        self.frames = []  # Camera frames (poses)
        self.points = []  # 3D map points
        self.state = None  # Variable to hold current state of the map and camera poses
        self.q = None  # Queue for inter-process communication

    def create_viewer(self):
        # Initialize the Queue for communication
        self.q = Queue()

        # Initialize the parallel process for visualization
        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True  # Exit when the main program exits
        p.start()

    def viewer_thread(self, q):
        # Initialize the Open3D visualizer
        self.viewer_init()

        # Main loop to update the visualization
        while True:
            self.viewer_refresh(q)

    def viewer_init(self):
        # Create an Open3D visualizer window
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Open3D', width=1280, height=720)

        # Initialize geometries
        self.pcd = o3d.geometry.PointCloud()
        self.cam_frames = o3d.geometry.LineSet()

        # Add coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        self.vis.add_geometry(coordinate_frame)

        # Add geometries to the visualizer
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.cam_frames)

    def viewer_refresh(self, q):
        # Check if there's new data to update
        if self.state is None or not q.empty():
            self.state = q.get()
            self.update_geometries()

        # Update the visualizer
        self.vis.update_geometry(self.pcd)
        self.vis.update_geometry(self.cam_frames)
        self.vis.poll_events()
        self.vis.update_renderer()

    def update_geometries(self):
        poses, pts = self.state

        # Update point cloud with map points
        self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for points

        # Update camera poses (trajectory)
        # Create lines between consecutive camera positions
        points = []
        lines = []
        colors = []
        for idx, pose in enumerate(poses):
            # Extract camera position from the pose (assuming pose is a 4x4 matrix)
            cam_pos = pose[:3, 3]
            points.append(cam_pos)

            # For visualization, connect the camera positions with lines
            if idx > 0:
                lines.append([idx - 1, idx])
                colors.append([0.0, 1.0, 0.0])  # Green color for trajectory lines

        # Update LineSet for camera trajectory
        self.cam_frames.points = o3d.utility.Vector3dVector(points)
        self.cam_frames.lines = o3d.utility.Vector2iVector(lines)
        self.cam_frames.colors = o3d.utility.Vector3dVector(colors)

    def display(self):
        if self.q is None:
            return

        poses, pts = [], []
        for f in self.frames:
            # Assuming f.pose is a 4x4 numpy array
            poses.append(f.pose)
        for p in self.points:
            # Assuming p.pt is a numpy array of shape (3,)
            pts.append(p.pt)

        # Put the updated state into the queue
        self.q.put((np.array(poses), np.array(pts)))


class Point(object):
    # A Point is a 3-D point in the world
    # Each point is observed in multiple frames
 
    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []
 
        # assigns a unique ID to the point based on the current number of points in the map.
        self.id = len(mapp.points)
        # adds the point instance to the map’s list of points.
        mapp.points.append(self)
 
    def add_observation(self, frame, idx):
        # Frame is the frame class
        self.frames.append(frame)
        self.idxs.append(idx)

