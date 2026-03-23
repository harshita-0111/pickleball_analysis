import os
import torch

class Config:
    # ─── Paths ────────────────────────────────────────────────────────
    VIDEO_IN = os.getenv("VIDEO_IN", "picckle_ball2.mp4")
    VIDEO_OUT = os.getenv("VIDEO_OUT", "tracked_final_v2.mp4")
    
    # Default directory for weights and outputs
    BASE_SAVE_DIR = os.getenv("SAVE_DIR", "pickleball_cv")
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    
    TN_BEST_WEIGHTS = os.path.join(BASE_SAVE_DIR, "tracknetv3_best.pt")
    YOLO_BALL_WEIGHTS = os.path.join(BASE_SAVE_DIR, "best.pt")
    YOLO_POSE_WEIGHTS = "yolov8m-pose.pt"
    
    # ─── Detection & Tracking ─────────────────────────────────────────
    ROI = (300, 150, 1600, 980)
    INPUT_W, INPUT_H = 320, 192  # TrackNetV3 input size
    TN_THRESH = 0.35
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Static detections to ignore
    MANUAL_BLACKLIST = [
        (784, 531, 50), (1884, 119, 80),
        (1280, 480, 40), (1088, 416, 40)
    ]
    
    # ─── Visuals ──────────────────────────────────────────────────────
    PLAYER_COLORS = {
        1: (0, 200, 255), 2: (0, 140, 200),
        3: (255, 220, 0), 4: (200, 160, 0)
    }
    
    SHOT_COLORS = {
        "DINK": (100, 220, 100), "DRIVE": (100, 100, 255),
        "LOB": (255, 200, 0), "SMASH": (0, 80, 255),
        "DROP": (255, 140, 0)
    }
    
    SHOT_ICONS = {
        "DINK": "●", "DRIVE": "→", "LOB": "↑",
        "SMASH": "⚡", "DROP": "↓"
    }
    
    # ─── API Keys ───────────────────────────────────────────────────
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
