import os

dataset_path = "/Users/aayushjain/Downloads/sportsmot_publish/basketball_dataset"
splits = ["train", "val"]

for split in splits:
    image_dir = os.path.join(dataset_path, split)
    label_dir = os.path.join(dataset_path, "labels", split)

    for seq in sorted(os.listdir(image_dir)):
        if seq.startswith('.'):  # Ignore hidden files
            continue

        img_folder = os.path.join(image_dir, seq, "img1")
        label_folder = os.path.join(label_dir, seq)

        if not os.path.exists(label_folder):
            print(f"❌ Missing labels for {seq}")
            continue

        img_files = set(f.split('.')[0] for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png')))
        txt_files = set(f.split('.')[0] for f in os.listdir(label_folder) if f.endswith('.txt'))

        missing_labels = img_files - txt_files
        missing_images = txt_files - img_files

        if missing_labels:
            print(f"⚠️ Missing labels for {len(missing_labels)} images in {seq}")
        if missing_images:
            print(f"⚠️ Extra labels found for {len(missing_images)} images in {seq}")

print("✅ Dataset verification complete!")