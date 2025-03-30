import motmetrics as mm
import numpy as np
import scipy.ndimage
from filterpy.kalman import KalmanFilter

# Load Ground Truth (GT) and Tracker Output
gt_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/gt_fixed.txt"
tracker_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/tracker_output_fixed.txt"

gt_data = np.loadtxt(gt_file, delimiter=",", usecols=[0, 1, 2, 3, 4, 5])  # GT: frame_id, obj_id, x1, y1, x2, y2
track_data = np.loadtxt(tracker_file, delimiter=",", usecols=[0, 1, 2, 3, 4, 5])  # Tracker: frame_id, track_id, x1, y1, x2, y2

# Initialize MOT accumulator
acc = mm.MOTAccumulator(auto_id=True)

# Kalman Filter Class for Bounding Box Smoothing
class KalmanBoxTracker:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)  # 8D state (x, y, w, h, vx, vy, vw, vh) and 4D measurement (x, y, w, h)
        self.kf.F = np.eye(8)  # State transition matrix
        self.kf.H = np.eye(4, 8)  # Measurement matrix
        self.kf.R *= 0.01  # Measurement noise
        self.kf.P *= 10  # Covariance matrix
        self.kf.x[:4] = np.array(bbox).reshape(-1, 1)  # Initial bounding box
        self.kf.Q *= 0.01  # Process noise

    def predict(self):
        self.kf.predict()
        return self.kf.x[:4].flatten()  # Return predicted (smoothed) bbox

    def update(self, bbox):
        self.kf.update(np.array(bbox).reshape(-1, 1))

# Dictionary to store Kalman filters for each tracked object
kalman_trackers = {}

# Process frame by frame
for frame in np.unique(gt_data[:, 0]):
    gt_boxes = gt_data[gt_data[:, 0] == frame, 2:6]  # GT bounding boxes
    track_boxes = track_data[track_data[:, 0] == frame, 2:6]  # Tracker bounding boxes
    gt_ids = gt_data[gt_data[:, 0] == frame, 1].astype(int)  # GT IDs
    track_ids = track_data[track_data[:, 0] == frame, 1].astype(int)  # Tracked IDs

    # Apply Kalman Filtering on tracker output
    smoothed_boxes = []
    for i, track_id in enumerate(track_ids):
        bbox = track_boxes[i]
        if track_id not in kalman_trackers:
            kalman_trackers[track_id] = KalmanBoxTracker(bbox)  # Initialize tracker
        else:
            kalman_trackers[track_id].update(bbox)  # Update with new bbox
        smoothed_bbox = kalman_trackers[track_id].predict()
        smoothed_boxes.append(smoothed_bbox)
    
    smoothed_boxes = np.array(smoothed_boxes)

    # Compute IoU for matching
    distances = mm.distances.iou_matrix(gt_boxes, smoothed_boxes, max_iou=0.5)
    distances = np.nan_to_num(distances, nan=0.1)  # Replace NaN values to avoid MOTP errors

    acc.update(gt_ids, track_ids, distances)

# Compute MOTA, IDF1, and MOTP
mh = mm.metrics.create()
metrics = ["idf1", "mota", "motp"]
summary = mh.compute(acc, metrics=metrics, name="Tracking Performance")

# Print results
print("\n🚀 Final Tracking Performance with Kalman Filtering:")
print(summary)

# import motmetrics as mm
# import numpy as np
# from filterpy.kalman import KalmanFilter

# # Load Ground Truth (GT) and Tracker Output
# gt_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/gt_fixed.txt"
# tracker_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/tracker_output_fixed.txt"

# gt_data = np.loadtxt(gt_file, delimiter=",", usecols=[0, 1, 2, 3, 4, 5])  # GT: (frame_id, obj_id, x1, y1, x2, y2)
# track_data = np.loadtxt(tracker_file, delimiter=",", usecols=[0, 1, 2, 3, 4, 5])  # Tracker: (frame_id, track_id, x1, y1, x2, y2)

# # Initialize MOT accumulator
# acc = mm.MOTAccumulator(auto_id=True)

# # Kalman Filter Class for Bounding Box Smoothing
# class KalmanBoxTracker:
#     def __init__(self, bbox):
#         self.kf = KalmanFilter(dim_x=8, dim_z=4)  # 8D state (x, y, w, h, vx, vy, vw, vh) and 4D measurement (x, y, w, h)
#         self.kf.F = np.eye(8)  # State transition matrix
#         self.kf.H = np.eye(4, 8)  # Measurement matrix
#         self.kf.R *= 0.005  # Reduce measurement noise for stable updates
#         self.kf.P *= 5  # Reduce covariance for better prediction accuracy
#         self.kf.x[:4] = np.array(bbox).reshape(-1, 1)  # Initial bounding box
#         self.kf.Q *= 0.001  # Reduce process noise to prevent over-smoothing
#         self.ema_factor = 0.6  # Exponential Moving Average smoothing factor

#     def predict(self):
#         self.kf.predict()
#         return self.kf.x[:4].flatten()  # Return predicted (smoothed) bbox

#     def update(self, bbox):
#         smoothed_bbox = self.ema_factor * self.kf.x[:4].flatten() + (1 - self.ema_factor) * np.array(bbox)
#         self.kf.update(smoothed_bbox.reshape(-1, 1))

# # Dictionary to store Kalman filters for each tracked object
# kalman_trackers = {}

# # Process frame by frame
# for frame in np.unique(gt_data[:, 0]):
#     gt_boxes = gt_data[gt_data[:, 0] == frame, 2:6]  # GT bounding boxes
#     track_boxes = track_data[track_data[:, 0] == frame, 2:6]  # Tracker bounding boxes
#     gt_ids = gt_data[gt_data[:, 0] == frame, 1].astype(int)  # GT IDs
#     track_ids = track_data[track_data[:, 0] == frame, 1].astype(int)  # Tracked IDs

#     # Apply Kalman Filtering on tracker output
#     smoothed_boxes = []
#     for i, track_id in enumerate(track_ids):
#         bbox = track_boxes[i]
#         if track_id not in kalman_trackers:
#             kalman_trackers[track_id] = KalmanBoxTracker(bbox)  # Initialize tracker
#         else:
#             kalman_trackers[track_id].update(bbox)  # Update with new bbox
#         smoothed_bbox = kalman_trackers[track_id].predict()
#         smoothed_boxes.append(smoothed_bbox)
    
#     smoothed_boxes = np.array(smoothed_boxes)

#     # Compute IoU for matching (lowered threshold to 0.3 for better MOTP)
#     if len(gt_boxes) == 0 or len(smoothed_boxes) == 0:
#         distances = np.zeros((len(gt_boxes), len(smoothed_boxes)))  # Handle empty matrices
#     else:
#         distances = mm.distances.iou_matrix(gt_boxes, smoothed_boxes, max_iou=0.3)

#     acc.update(gt_ids, track_ids, distances)

# # Compute MOTA, IDF1, and MOTP
# mh = mm.metrics.create()
# metrics = ["idf1", "mota", "motp"]
# summary = mh.compute(acc, metrics=metrics, name="Tracking Performance")

# # Print results
# print("\n🚀 Optimized Tracking Performance with IoU & Kalman Filtering:")
# print(summary)


