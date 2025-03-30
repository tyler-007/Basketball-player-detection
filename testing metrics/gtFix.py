import numpy as np

# Load ground truth file
gt_data = np.loadtxt("/Users/aayushjain/codes/projects/company assignements/Big Vision/testing metrics/gt.txt", delimiter=",")

# Convert bbox format
gt_data[:, 4] += gt_data[:, 2]  # x2 = x1 + width
gt_data[:, 5] += gt_data[:, 3]  # y2 = y1 + height

# Save fixed GT
np.savetxt("/Users/aayushjain/codes/projects/company assignements/Big Vision/testing metrics/gt_fixed.txt", gt_data, delimiter=",", fmt="%.2f")

print("✅ Ground truth converted to (x1, y1, x2, y2) format and saved!")