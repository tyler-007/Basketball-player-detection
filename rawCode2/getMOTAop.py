import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("/Users/aayushjain/Downloads/sportsmot_publish/Final models/runs/detect/train4/weights/best.pt")

# Open the video file
video_path = "/Users/aayushjain/Downloads/sportsmot_publish/output_video_MOTA.mp4"
cap = cv2.VideoCapture(video_path)

# Define output file for tracking results
output_txt = "/Users/aayushjain/Downloads/sportsmot_publish/tracker_output.txt"
f = open(output_txt, "w")

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Run YOLOv8 + ByteTrack
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, iou=0.7)

    # Ensure detections exist
    if results and len(results) > 0:
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                track_id = int(box.id[0]) if box.id is not None else -1  # Tracking ID
                conf = box.conf[0].item()  # Confidence score

                # Save to file (Format: frame_id, track_id, x1, y1, x2, y2, confidence)
                f.write(f"{frame_id},{track_id},{x1},{y1},{x2},{y2},{conf:.2f}\n")

    frame_id += 1

cap.release()
f.close()
print(f"✅ Tracking data saved at: {output_txt}")