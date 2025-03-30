import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("/Users/aayushjain/Downloads/sportsmot_publish/Final models/runs/detect/train4/weights/best.pt")

# Open the video file
video_path = "/Users/aayushjain/Downloads/sportsmot_publish/output_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties (optional)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break if the video ends

    # Run YOLOv8 detection on the current frame
    results = model(frame)  # YOLO processes the frame

    # Draw bounding boxes on the frame
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            label = int(box.cls[0])  # Class label

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow("YOLOv8 Real-Time Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()