import motmetrics as mm
import numpy as np
import scipy.ndimage

# Load Ground Truth (GT) and Tracker Output
gt_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/gt_fixed.txt"
tracker_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/tracker_output_fixed.txt"

# Load GT and Tracker data (select only required columns)
gt_data = np.loadtxt(gt_file, delimiter=",", usecols=[0, 1, 2, 3, 4, 5])  # (frame, obj_id, x1, y1, x2, y2)
track_data = np.loadtxt(tracker_file, delimiter=",", usecols=[0, 1, 2, 3, 4, 5])  # (frame, track_id, x1, y1, x2, y2)

# Initialize MOT accumulator
acc = mm.MOTAccumulator(auto_id=True)

# Set image dimensions (update according to dataset)
img_width, img_height = 1280, 720  # Replace with actual resolution

# Process frame by frame
for frame in np.unique(gt_data[:, 0]):  
    gt_boxes = gt_data[gt_data[:, 0] == frame, 2:6]  # GT bounding boxes
    track_boxes = track_data[track_data[:, 0] == frame, 2:6]  # Tracker bounding boxes
    gt_ids = gt_data[gt_data[:, 0] == frame, 1].astype(int)  
    track_ids = track_data[track_data[:, 0] == frame, 1].astype(int)  

    # If no detections exist, continue to next frame
    if len(gt_boxes) == 0 or len(track_boxes) == 0:
        acc.update(gt_ids, track_ids, np.full((len(gt_ids), len(track_ids)), np.nan))
        continue

    # 🔹 Normalize bounding boxes to [0,1] range
    gt_boxes[:, [0, 2]] /= img_width  # Normalize x1, x2
    gt_boxes[:, [1, 3]] /= img_height  # Normalize y1, y2
    track_boxes[:, [0, 2]] /= img_width
    track_boxes[:, [1, 3]] /= img_height

    # 🔹 Compute IoU with a higher threshold (better precision)
    distances = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.7)

    # 🔹 Apply Gaussian smoothing to stabilize errors
    smoothed_distances = scipy.ndimage.gaussian_filter(distances, sigma=1.0)

    # Debugging: Print IoU Matrix for troubleshooting
    print(f"Frame {frame}: Smoothed IoU Matrix:\n", smoothed_distances)

    # Update MOT accumulator
    acc.update(gt_ids, track_ids, smoothed_distances)

# Compute Tracking Metrics
mh = mm.metrics.create()
metrics = ["idf1", "mota", "motp"]
summary = mh.compute(acc, metrics=metrics, name="Tracking Performance")

# Print results
print("\nFinal Tracking Performance Metrics:")
print(summary)
