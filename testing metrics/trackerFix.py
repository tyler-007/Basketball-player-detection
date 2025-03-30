import numpy as np
import os

# Corrected absolute path
file_path = "/Users/aayushjain/codes/projects/company assignements/Big Vision/testing metrics/tracker_output.txt"
output_path = "/Users/aayushjain/codes/projects/company assignements/Big Vision/testing metrics/tracker_output_fixed.txt"

# Ensure the file exists before loading
if not os.path.exists(file_path):
    print(f"❌ ERROR: File not found at {file_path}")
    exit(1)

# Load tracker output
track_data = np.loadtxt(file_path, delimiter=",")

# Ensure track_data is not empty
if track_data.size == 0:
    print("❌ ERROR: tracker_output.txt is empty or incorrectly formatted.")
    exit(1)

# Fix frame indexing
track_data[:, 0] += 1  # Increment frame numbers

# Save fixed output
np.savetxt(output_path, track_data, delimiter=",", fmt="%.2f")

print(f"✅ Fixed tracker_output.txt saved as {output_path}")
