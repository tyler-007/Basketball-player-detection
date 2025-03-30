import argparse
import torch
from ultralytics import YOLO

def train_yolo(data_path, model_type, epochs, img_size, device):
    """
    Train YOLOv8 on a given dataset.

    Parameters:
    - data_path (str): Path to data.yaml file
    - model_type (str): Model to use (yolov8n, yolov8s, yolov8m, etc.)
    - epochs (int): Number of training epochs
    - img_size (int): Image size for training (e.g., 640, 1280)
    - device (str): "cuda" for GPU, "cpu" for CPU
    """
    # Load model
    model = YOLO(f"{model_type}.pt")

    # Train model
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        device=device
    )

    print("✅ Training Complete! Best model saved in 'runs/detect/train/weights/best.pt'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n", help="YOLO model type (e.g., yolov8n, yolov8s, yolov8m)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--img_size", type=int, default=640, help="Image size for training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    train_yolo(args.data, args.model, args.epochs, args.img_size, args.device)