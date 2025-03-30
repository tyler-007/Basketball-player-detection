import sys
import cv2
from ultralytics import YOLO

# Read command-line arguments (hyperparameters)
conf = float(sys.argv[1])
iou = float(sys.argv[2])

# Load YOLO model
model = YOLO("/Users/aayushjain/codes/projects/company assignements/Big Vision/testing metrics/best.pt")

# Open the video file
video_path = "/Users/aayushjain/codes/projects/company assignements/Big Vision/testing metrics/output_video_tracker_metrics.mp4"
cap = cv2.VideoCapture(video_path)

# Define output tracker results file
output_txt = "/Users/aayushjain/codes/projects/company assignements/Big Vision/testing metrics/tracker_output.txt"
f = open(output_txt, "w")

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 + DeepSORT/ByteTrack
    results = model.track(frame, persist=True, tracker="bytesort.yaml", conf=conf, iou=iou)

    # Save results
    if results and len(results) > 0:
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                conf_score = box.conf[0].item()
                f.write(f"{frame_id},{track_id},{x1},{y1},{x2},{y2},{conf_score:.2f}\n")

    frame_id += 1

cap.release()
f.close()
print(f"✅ Tracker output saved at: {output_txt}")