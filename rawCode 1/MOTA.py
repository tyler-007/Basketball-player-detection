import motmetrics as mm
import numpy as np

# Initialize MOT accumulator
acc = mm.MOTAccumulator(auto_id=True)

# Load ground truth & tracker predictions (format: frame_id, obj_id, x1, y1, x2, y2)
gt_data = np.loadtxt("/Users/aayushjain/codes/projects/company assignements/Big Vision/gt_fixed.txt", delimiter=",")  # Replace with your GT file
track_data = np.loadtxt("/Users/aayushjain/codes/projects/company assignements/Big Vision/tracker_output_fixed.txt", delimiter=",")  # Replace with ByteTrack results

# Process frame by frame
for frame in np.unique(gt_data[:, 0]):  # Loop through all frames
    gt_boxes = gt_data[gt_data[:, 0] == frame, 2:6]  # Extract GT bbox
    track_boxes = track_data[track_data[:, 0] == frame, 2:6]  # Extract detected bbox
    gt_ids = gt_data[gt_data[:, 0] == frame, 1].astype(int)  # GT IDs
    track_ids = track_data[track_data[:, 0] == frame, 1].astype(int)  # Tracked IDs

    # Compute IoU (Intersection Over Union) for matching GT with predicted tracks
    distances = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

    # Update MOT accumulator with matches
    acc.update(gt_ids, track_ids, distances)

# Compute metrics
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=["num_misses", "num_false_positives", "num_switches", "mota"], name="MOTA Results")
print(summary)