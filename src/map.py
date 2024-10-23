# src/map.py

class Map:
    def __init__(self):
        self.keyframes = []
        self.map_points = []

    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)
        # Update connections, covisibility graphs, etc.

    def add_map_point(self, map_point):
        self.map_points.append(map_point)

    def get_local_keyframes(self, current_frame, max_neighbors=5):
        # Find keyframes that share map points with the current frame
        # For simplicity, we can select the last few keyframes
        if len(self.keyframes) == 0:
            return []
        return self.keyframes[-max_neighbors:]

    def get_local_map_points(self, current_frame):
        local_keyframes = self.get_local_keyframes(current_frame)
        local_map_points = []
        for kf in local_keyframes:
            local_map_points.extend(kf.map_points)
        # Remove duplicates
        local_map_points = list(set(local_map_points))
        return local_map_points

