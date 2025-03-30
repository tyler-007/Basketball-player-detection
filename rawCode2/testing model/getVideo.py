import cv2
import os

image_folder = "/Users/aayushjain/Downloads/sportsmot_publish/dataset/train/v_-6Os86HzwCs_c001/img1"
output_video = "output_video_MOTA.mp4"
fps = 30  # Frames per second

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
h, w, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    video.write(frame)

video.release()
print("✅ Video saved as", output_video)