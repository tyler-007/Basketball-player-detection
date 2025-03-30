import numpy as np

# Load data
gt_data = np.loadtxt("/Users/aayushjain/codes/projects/company assignements/Big Vision/gt_fixed.txt", delimiter=",")
track_data = np.loadtxt("/Users/aayushjain/codes/projects/company assignements/Big Vision/tracker_output.txt", delimiter=",")

# Check unique frames
print("Unique frames in GT:", np.unique(gt_data[:, 0]))
print("Unique frames in Tracker Output:", np.unique(track_data[:, 0]))
