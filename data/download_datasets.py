import os
from roboflow import Roboflow
from config import Config

def download():
    api_key = Config.ROBOFLOW_API_KEY
    if not api_key:
        print("Error: ROBOFLOW_API_KEY not found in Config or environment.")
        return

    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace("pickleball-pczbl")

    print("Downloading ball-tracker-pczbl...")
    workspace.project("ball-tracker-pczbl").version(1).download("yolov8")

    print("Downloading pickleball-vision...")
    workspace.project("pickleball-vision").version(1).download("yolov8")

    print("Downloading pickleball-detection-1oqlw...")
    workspace.project("pickleball-detection-1oqlw").version(1).download("yolov8")

    print("✓ All datasets downloaded.")

if __name__ == "__main__":
    download()
