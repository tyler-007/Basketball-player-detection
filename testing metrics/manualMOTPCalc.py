import numpy as np

# Load Ground Truth and Tracker Output
gt_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/testing metrics/gt_fixed.txt"
tracker_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/testing metrics/tracker_output_fixed.txt"

gt_data = np.loadtxt(gt_file, delimiter=",", usecols=[0, 1, 2, 3, 4, 5])  # (frame_id, obj_id, x1, y1, x2, y2)
track_data = np.loadtxt(tracker_file, delimiter=",", usecols=[0, 1, 2, 3, 4, 5])  # (frame_id, track_id, x1, y1, x2, y2)

# Function to compute IoU
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        return 0  # Avoid division by zero
    return interArea / unionArea  # IoU Calculation

# Initialize variables for MOTP calculation
total_iou = 0
total_matches = 0

# Process frame by frame
for frame in np.unique(gt_data[:, 0]):  # Loop through all frames
    gt_boxes = gt_data[gt_data[:, 0] == frame, 2:6]  # GT bounding boxes
    track_boxes = track_data[track_data[:, 0] == frame, 2:6]  # Tracker bounding boxes
    gt_ids = gt_data[gt_data[:, 0] == frame, 1].astype(int)  # GT IDs
    track_ids = track_data[track_data[:, 0] == frame, 1].astype(int)  # Tracker IDs

    # Find matches between GT and Tracker using object IDs
    for gt_id, gt_box in zip(gt_ids, gt_boxes):
        matching_tracker_boxes = track_boxes[track_ids == gt_id]  # Get the corresponding tracker box

        if len(matching_tracker_boxes) > 0:
            iou = compute_iou(gt_box, matching_tracker_boxes[0])  # Compute IoU for matched object
            total_iou += iou
            total_matches += 1

# Compute final MOTP
if total_matches == 0:
    motp = 0  # No matches found
else:
    motp = total_iou / total_matches  # Average IoU over all matches

# Print results
print(f"\n🚀 Final MOTP (Multiple Object Tracking Precision): {motp:.6f}")
