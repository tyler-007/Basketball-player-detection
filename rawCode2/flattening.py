import os
import shutil

# Define dataset path
dataset_path = "/Users/aayushjain/Downloads/sportsmot_publish/basketball_dataset_final"
splits = ["train", "val", "test"]  # Add "test" if needed

for split in splits:
    split_path = os.path.join(dataset_path, split)
    new_img_dir = os.path.join(dataset_path, f"{split}_images")
    new_lbl_dir = os.path.join(dataset_path, f"{split}_labels")

    os.makedirs(new_img_dir, exist_ok=True)
    os.makedirs(new_lbl_dir, exist_ok=True)

    for seq_folder in os.listdir(split_path):
        if seq_folder.startswith('.'):  # Ignore hidden files like .DS_Store
            continue

        seq_path = os.path.join(split_path, seq_folder)
        img1_path = os.path.join(seq_path, "img1")
        label_folder = os.path.join(dataset_path, "labels", split, seq_folder)

        # Move & rename images
        if os.path.exists(img1_path):
            for img_file in os.listdir(img1_path):
                if img_file.startswith('.') or not img_file.endswith('.jpg'):  # Ignore hidden/system files
                    continue
                frame_id = img_file.split('.')[0]  # Extract frame number (e.g., 0001)
                new_name = f"{seq_folder}_{frame_id}.jpg"  # Rename with sequence
                shutil.move(os.path.join(img1_path, img_file), os.path.join(new_img_dir, new_name))

        # Move & rename labels
        if os.path.exists(label_folder):
            for lbl_file in os.listdir(label_folder):
                if lbl_file.startswith('.') or not lbl_file.endswith('.txt'):  # Ignore hidden/system files
                    continue
                frame_id = lbl_file.split('.')[0]  # Extract frame number
                new_name = f"{seq_folder}_{frame_id}.txt"  # Rename with sequence
                shutil.move(os.path.join(label_folder, lbl_file), os.path.join(new_lbl_dir, new_name))

print("✅ Renamed and moved all images & labels successfully!")