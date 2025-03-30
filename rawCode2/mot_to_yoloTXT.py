import os
import cv2

# Define dataset paths
DATASET_PATH = "/Users/aayushjain/Downloads/sportsmot_publish/basketball_dataset"
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    data_path = os.path.join(DATASET_PATH, split)
    label_path = os.path.join(DATASET_PATH, "labels", split)
    os.makedirs(label_path, exist_ok=True)

    for seq in sorted(os.listdir(data_path)):  # Loop through all folders in train/val/test
        seq_path = os.path.join(data_path, seq)
        img_path = os.path.join(seq_path, "img1")
        ann_path = os.path.join(seq_path, "gt/gt.txt")

        if not os.path.exists(ann_path):
            print(f"Skipping {seq} (No annotation file found)")
            continue  # Skip if no annotations
        
        print(f"Processing: {seq}")

        with open(ann_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            data = line.strip().split(',')
            frame_id, obj_id, x, y, w, h, _, class_id, _ = map(float, data)

            # Normalize bounding box values
            img_file = f"{int(frame_id):06d}.jpg"
            img_path_full = os.path.join(img_path, img_file)

            if not os.path.exists(img_path_full):
                continue  # Skip if image does not exist
            
            # Load image dimensions
            img = cv2.imread(img_path_full)
            img_height, img_width, _ = img.shape
            
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w /= img_width
            h /= img_height

            # YOLO format: class_id x_center y_center width height
            yolo_format = f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"

            # Save the annotation in the corresponding text file
            seq_label_path = os.path.join(label_path, seq)
            os.makedirs(seq_label_path, exist_ok=True)

            txt_filename = f"{int(frame_id):06d}.txt"
            label_file = os.path.join(seq_label_path, txt_filename)

            with open(label_file, "a") as out_file:
                out_file.write(yolo_format)

print("✅ All MOT → YOLO conversions complete! Check 'labels/train' and 'labels/val'.")