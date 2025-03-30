import motmetrics as mm
import numpy as np

# Load Ground Truth (GT) and Tracker Output
gt_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/gt_fixed.txt"
tracker_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/tracker_output_fixed.txt"

gt_data = np.loadtxt(gt_file, delimiter=",")  # GT: (frame_id, obj_id, x1, y1, x2, y2)
track_data = np.loadtxt(tracker_file, delimiter=",")  # Tracker: (frame_id, track_id, x1, y1, x2, y2, conf)

# Initialize MOT accumulator
acc = mm.MOTAccumulator(auto_id=True)

# Process frame by frame
for frame in np.unique(gt_data[:, 0]):  # Loop through all frames
    gt_boxes = gt_data[gt_data[:, 0] == frame, 2:6]  # GT bounding boxes
    track_boxes = track_data[track_data[:, 0] == frame, 2:6]  # Tracker bounding boxes
    gt_ids = gt_data[gt_data[:, 0] == frame, 1].astype(int)  # GT IDs
    track_ids = track_data[track_data[:, 0] == frame, 1].astype(int)  # Tracked IDs

    # Compute IoU (Intersection Over Union) for matching
    distances = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

    # Update MOT accumulator
    acc.update(gt_ids, track_ids, distances)

# Compute MOTA, IDF1, and MOTP
mh = mm.metrics.create()
metrics = ["idf1", "mota", "motp"]
summary = mh.compute(acc, metrics=metrics, name="Tracking Performance")

# Print results
print(summary)
