import itertools
import numpy as np
import subprocess
from metric import evaluate_tracker  # Function to compute MOTA, IDF1, MOTP

# Define hyperparameter ranges
iou_values = [0.3, 0.5, 0.7]
conf_values = [0.25, 0.3, 0.35, 0.4]

# Store best results
best_params = None
best_metrics = {"MOTA": -1, "IDF1": -1, "MOTP": -1}

# Loop over different combinations of IoU and confidence threshold
for iou, conf in itertools.product(iou_values, conf_values):
    print(f"\n🚀 Running tracking with IoU={iou}, Conf={conf}")

    # Run tracking and generate tracker output
    subprocess.run(["python3", "output.py", str(conf), str(iou)], check=True)

    # Fix tracker output before evaluation
    subprocess.run(["python3", "trackerFix.py"], check=True)

    # Evaluate the tracker output
    metrics = evaluate_tracker()

    print(f"Results for IoU={iou}, Conf={conf} -> MOTA: {metrics['MOTA']}, IDF1: {metrics['IDF1']}, MOTP: {metrics['MOTP']}")

    # Update best parameters based on MOTA, then IDF1
    if metrics["MOTA"] > best_metrics["MOTA"] or (metrics["MOTA"] == best_metrics["MOTA"] and metrics["IDF1"] > best_metrics["IDF1"]):
        best_metrics = metrics
        best_params = {"iou": iou, "conf": conf}

# Print the best results
print(f"\n🏆 Best Parameters: {best_params}, Best Metrics: {best_metrics}")