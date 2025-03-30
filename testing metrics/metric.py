import motmetrics as mm
import numpy as np

def evaluate_tracker():
    # Load ground truth and tracker output
    gt_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/testing metrics/gt_fixed.txt"
    tracker_file = "/Users/aayushjain/codes/projects/company assignements/Big Vision/testing metrics/tracker_output_fixed.txt"

    gt_data = np.loadtxt(gt_file, delimiter=",", usecols=[0, 1, 2, 3, 4, 5])
    track_data = np.loadtxt(tracker_file, delimiter=",", usecols=[0, 1, 2, 3, 4, 5])

    # Initialize MOT accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Process frame by frame
    for frame in np.unique(gt_data[:, 0]):
        gt_boxes = gt_data[gt_data[:, 0] == frame, 2:6]
        track_boxes = track_data[track_data[:, 0] == frame, 2:6]
        gt_ids = gt_data[gt_data[:, 0] == frame, 1].astype(int)
        track_ids = track_data[track_data[:, 0] == frame, 1].astype(int)

        # Compute IoU for matching
        distances = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)
        distances = np.nan_to_num(distances, nan=0.1)

        acc.update(gt_ids, track_ids, distances)

    # Compute MOTA, IDF1, and MOTP
    mh = mm.metrics.create()
    metrics = ["idf1", "mota", "motp"]
    summary = mh.compute(acc, metrics=metrics, name="Tracking Performance")

    return {
        "MOTA": summary["mota"].values[0],
        "IDF1": summary["idf1"].values[0],
        "MOTP": summary["motp"].values[0],
    }

# If run directly, compute metrics
if __name__ == "__main__":
    results = evaluate_tracker()
    print("\n🚀 Final Tracking Performance:")
    print(results)