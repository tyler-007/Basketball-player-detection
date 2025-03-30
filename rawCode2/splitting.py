import os
import shutil

# Define paths
ORIGINAL_DATASET_PATH = "/Users/aayushjain/Downloads/sportsmot_publish/dataset"  # Update this with your dataset path
NEW_DATASET_PATH = "/Users/aayushjain/Downloads/sportsmot_publish/basketball_dataset"  # Where the basketball-only dataset will be stored

# Read the basketball sequences from basketball.txt
basketball_sequences = set()
with open("/Users/aayushjain/Downloads/sportsmot_publish/splits_txt/basketball.txt", "r") as f:  # Update the correct path
    for line in f:
        basketball_sequences.add(line.strip())  # Store sequence names

# Process each split (train, val, test)
for split in ["train", "val", "test"]:
    original_split_path = os.path.join(ORIGINAL_DATASET_PATH, split)
    new_split_path = os.path.join(NEW_DATASET_PATH, split)
    os.makedirs(new_split_path, exist_ok=True)  # Create new split folder
    
    # Copy only basketball sequences
    for seq in os.listdir(original_split_path):
        if seq in basketball_sequences:
            shutil.copytree(
                os.path.join(original_split_path, seq),
                os.path.join(new_split_path, seq)
            )

print("Basketball dataset created successfully!")
