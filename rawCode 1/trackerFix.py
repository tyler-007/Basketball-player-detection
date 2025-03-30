import numpy as np

# Load tracker output
track_data = np.loadtxt("/Users/aayushjain/codes/projects/company assignements/Big Vision/tracker_output.txt", delimiter=",")

# Increment frame IDs by 1
track_data[:, 0] += 1

# Save back
np.savetxt("/Users/aayushjain/codes/projects/company assignements/Big Vision/tracker_output_fixed.txt", track_data, delimiter=",", fmt="%.2f")

print("✅ Fixed tracker_output.txt saved as tracker_output_fixed.txt")
