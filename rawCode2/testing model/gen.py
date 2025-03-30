from ultralytics import YOLO

# Load trained model
model = YOLO("/Users/aayushjain/Downloads/sportsmot_publish/Final models/runs/detect/train4/weights/best.pt")

# Run detection on the video
results = model.predict(source="/Users/aayushjain/Downloads/sportsmot_publish/output_video_1.mp4", save=True)

# Output will be saved in runs/detect/predict/
print("✅ Detection completed! Check runs/detect/predict/")