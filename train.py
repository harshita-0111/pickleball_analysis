import argparse
from ultralytics import YOLO
from config import Config

def train(data_yaml, epochs=80, imgsz=1280, batch=8):
    model = YOLO('yolov8n.pt')
    
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,
        device=Config.DEVICE,
        workers=2,
        project='runs',
        name='pickleball_v2',
        exist_ok=True,
        fliplr=0.5,
        mosaic=1.0,
        degrees=10,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.3
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Pickleball Dataset")
    parser.add_argument("--data", default="pickleball_merged_v2/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    
    train(args.data, epochs=args.epochs)
