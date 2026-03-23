import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO

from config import Config
from models.tracknet import TrackNetV3
from tracking.ball_detector import BallDetector
from tracking.player_detector import PlayerDetector
from tracking.kalman_tracker import BallKalman
from tracking.rally_state import RallyState
from analysis.shot_classifier import ShotClassifier
from visualization.trail import ShortTrail
from visualization.hud import (
    draw_player_circles, draw_shot_label, 
    draw_shot_panel, draw_speed_bar, draw_panel
)
from visualization.charts import create_shot_chart, create_heatmap

def run_pipeline(video_in, video_out, yolo_weights, tracknet_weights, roi=Config.ROI, device=Config.DEVICE):
    # ─── Initialization ──────────────────────────────────────────────
    print(f"Loading models on {device}...")
    
    # TrackNetV3
    model_tn = None
    try:
        model_tn = TrackNetV3().to(device)
        ckpt = torch.load(tracknet_weights, map_location=device)
        model_tn.load_state_dict(ckpt["model_state"])
        model_tn.eval()
        print(f"✓ TrackNetV3 loaded (F1={ckpt.get('f1', 0):.3f})")
    except Exception as e:
        print(f"⚠ Could not load TrackNetV3: {e}. Falling back to YOLO/HSV.")
        model_tn = None
    
    # YOLO Models
    try:
        yolo_ball = YOLO(yolo_weights)
    except:
        print(f"⚠ Could not load custom YOLO weights from {yolo_weights}. Using yolov8n.pt as fallback.")
        yolo_ball = YOLO("yolov8n.pt")
        
    yolo_pose = YOLO(Config.YOLO_POSE_WEIGHTS)
    print("✓ YOLO models loaded")
    
    # Video Setup
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Component Initialization
    detector = BallDetector(model_tn, yolo_ball, device=device)
    player_detector = PlayerDetector(yolo_pose, W, H)
    kalman = BallKalman(roi=roi)
    rally = RallyState()
    clf = ShotClassifier(H=H)
    trail = ShortTrail(n=6)
    
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    
    # ─── Processing Loop ─────────────────────────────────────────────
    frames = []
    print("Reading frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    total = len(frames)
    print(f"✓ {total} frames | {W}x{H} @ {fps:.0f}fps")
    
    positions = []
    stats = {"tracknet": 0, "yolo": 0, "hsv": 0, "none": 0}
    
    overlay_shot = None
    overlay_conf = 0.
    overlay_timer = 0
    overlay_frame = -999
    hitter_num = None
    OVERLAY_HOLD = 60
    MAX_AGE = 30
    
    print("\nProcessing pipeline...")
    for n in range(total):
        frame = frames[n].copy()
        
        # 1. Player Detection
        players = player_detector.detect(frames[n])
        
        # 2. Ball Detection (needs prev, curr, next frames)
        fp = frames[max(0, n-1)]
        fc = frames[n]
        fn = frames[min(total-1, n+1)]
        
        det, src, conf_val, hm = detector.detect(fp, fc, fn, W, H)
        
        # 3. State Updates
        is_detected = (det is not None and src != "none")
        should_draw_trail = rally.update(is_detected)
        
        if rally.is_dead:
            trail.clear()
            kalman.reset()
            clf.reset()
            overlay_shot = None
            overlay_timer = 0
            hitter_num = None
            
        pos = kalman.update(det if is_detected else None)
        
        # Statistics
        stats[src] += 1
        
        if pos:
            positions.append((n, pos[0], pos[1]))
            
        trail.push(det if is_detected and should_draw_trail else None)
        
        # 4. Shot Classification
        shot, s_conf = clf.update(pos, n)
        
        # Hitter Logic (Cell 58 logic)
        if shot and s_conf >= 0.70 and n - clf.last_frame < 3 and is_detected:
            hn = None
            if pos and players:
                bx, by = pos
                best_dist = 400
                for p in players:
                    # Wrist priority (KPS 9, 10)
                    if p["kps"] is not None:
                        for ki in [9, 10]:
                            if ki < len(p["kps"]):
                                kx, ky = p["kps"][ki]
                                if kx > 0 and ky > 0:
                                    d = np.hypot(bx - kx, by - ky)
                                    if d < best_dist:
                                        best_dist = d
                                        hn = p["num"]
                    # Fallback to center distance
                    d = np.hypot(bx - p["cx"], by - p["cy"])
                    if d < best_dist and hn is None:
                        best_dist = d
                        hn = p["num"]
            
            if hn is not None:
                overlay_shot = shot
                overlay_conf = s_conf
                overlay_timer = OVERLAY_HOLD
                overlay_frame = n
                hitter_num = hn
                
        if overlay_timer > 0: overlay_timer -= 1
        if n - overlay_frame > MAX_AGE:
            overlay_shot = None
            overlay_timer = 0
            
        # ─── Rendering ───────────────────────────────────────────────
        vis = frame.copy()
        if should_draw_trail:
            vis = trail.draw(vis)
            
        draw_player_circles(vis, players, hitter_num, overlay_timer)
        draw_shot_label(vis, overlay_shot, overlay_conf, overlay_timer, hitter_num, W=W)
        
        # Source Label
        src_colors = {"tracknet": (0, 255, 128), "yolo": (0, 255, 0), "hsv": (255, 200, 0)}
        src_labels = {"tracknet": "TrackNet", "yolo": "YOLO", "hsv": "Vision"}
        label = src_labels.get(src, "")
        if label:
            draw_panel(vis, 8, 8, 180, 40, 0.75)
            cv2.putText(vis, label, (16, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, 
                        src_colors.get(src, (200, 200, 200)), 2, cv2.LINE_AA)
                        
        draw_shot_panel(vis, clf.shot_counts, W=W)
        draw_speed_bar(vis, trail.current_speed, H=H)
        
        cv2.putText(vis, f"Frame {n}/{total}", (10, H-12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        
        writer.write(vis)
        
        if n % 100 == 0:
            print(f"  {n:4d}/{total} | TN:{stats['tracknet']} X:{stats['none']} | shot:{overlay_shot or '--'}")

    writer.release()
    print(f"\n✓ Processing complete. Output saved to {video_out}")
    
    # Generate Analytics
    print("Generating analytics...")
    create_shot_chart(clf.shot_counts, "shot_distribution.png")
    create_heatmap(positions, W, H, "court_heatmap.png")
    print("✓ Analytics generated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickleball Analysis Pipeline")
    parser.add_argument("--input", default=Config.VIDEO_IN, help="Input video path")
    parser.add_argument("--output", default=Config.VIDEO_OUT, help="Output video path")
    parser.add_argument("--yolo-weights", default=Config.YOLO_BALL_WEIGHTS, help="YOLO ball weights path")
    parser.add_argument("--tracknet-weights", default=Config.TN_BEST_WEIGHTS, help="TrackNetV3 weights path")
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output, args.yolo_weights, args.tracknet_weights)
