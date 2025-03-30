import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("/Users/aayushjain/Downloads/sportsmot_publish/Final models/runs/detect/train4/weights/best.pt")

# Open the video file
video_path = "/Users/aayushjain/Downloads/sportsmot_publish/output_video_1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break if the video ends

    # Use YOLOv8's built-in tracking function with DeepSORT
    results = model.track(frame, persist=True, tracker="deepsort.yaml", conf=0.4, iou=0.7)

    # Draw bounding boxes with tracking IDs
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            track_id = int(box.id[0]) if box.id is not None else -1  # Tracking ID
            conf = box.conf[0].item()  # Confidence score
            label = int(box.cls[0])  # Class label

            # Draw rectangle and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
