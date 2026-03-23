import numpy as np
from config import Config

class PlayerIDTracker:
    """
    Tracks player IDs based on position and velocity with fallback to quadrant-based assignment.
    IDs: 1 (Near Left), 2 (Near Right), 3 (Far Left), 4 (Far Right)
    """
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.positions = {1: None, 2: None, 3: None, 4: None}
        self.velocity = {1: (0, 0), 2: (0, 0), 3: (0, 0), 4: (0, 0)}
        self.missing = {1: 0, 2: 0, 3: 0, 4: 0}
        self.MAX_MISS = 30

    def _get_base_id(self, cx, cy):
        # Near/Far split at 48% height
        near = cy > self.H * 0.48
        left = cx < self.W * 0.50
        if near and left: return 1
        elif near and not left: return 2
        elif not near and left: return 3
        else: return 4

    def update(self, detections):
        """
        Assigns consistent IDs to detected players.
        detections: list of dicts with {"cx", "cy", "conf", "kps"}
        """
        if not detections:
            for k in self.missing:
                self.missing[k] += 1
            return []

        predicted = {}
        for pid in range(1, 5):
            if self.positions[pid] is not None:
                px, py = self.positions[pid]
                vx, vy = self.velocity[pid]
                predicted[pid] = (px + vx, py + vy)

        assigned = {}
        used_pids = set()
        
        # Assign best matches first based on confidence and closeness
        dets = sorted(enumerate(detections), key=lambda x: -x[1]["conf"])
        for di, det in dets:
            cx, cy = det["cx"], det["cy"]
            best_pid = None
            best_d = 200 # Max pixel distance for tracking
            
            for pid, pred in predicted.items():
                if pid in used_pids:
                    continue
                d = np.hypot(cx - pred[0], cy - pred[1])
                if d < best_d:
                    best_d = d
                    best_pid = pid
            
            if best_pid is None:
                # Fallback to base quadrant ID
                qp = self._get_base_id(cx, cy)
                best_pid = qp if qp not in used_pids else next((p for p in range(1, 5) if p not in used_pids), 1)
                
            assigned[di] = best_pid
            used_pids.add(best_pid)

        result = []
        seen_pids = set()
        for di, det in enumerate(detections):
            pid = assigned.get(di, self._get_base_id(det["cx"], det["cy"]))
            seen_pids.add(pid)
            
            if self.positions[pid]:
                ox, oy = self.positions[pid]
                # Smooth velocity update
                self.velocity[pid] = (0.7 * (det["cx"] - ox), 0.7 * (det["cy"] - oy))
            
            self.positions[pid] = (det["cx"], det["cy"])
            self.missing[pid] = 0
            
            player_info = dict(det)
            player_info["num"] = pid
            player_info["color"] = Config.PLAYER_COLORS.get(pid, (180, 180, 180))
            result.append(player_info)

        # Update missing counts
        for pid in range(1, 5):
            if pid not in seen_pids:
                self.missing[pid] += 1
                if self.missing[pid] > self.MAX_MISS:
                    self.positions[pid] = None
                    self.velocity[pid] = (0, 0)
        
        return result

class PlayerDetector:
    """
    Detects players using YOLOv8-pose and tracks them with PlayerIDTracker.
    """
    def __init__(self, pose_model, W, H):
        self.model = pose_model
        self.tracker = PlayerIDTracker(W, H)
        self.W, self.H = W, H
        self.roi = Config.ROI

    def detect(self, frame):
        """
        Detects players in frame and returns tracked list.
        """
        res = self.model(frame, conf=0.30, verbose=False)[0]
        raw = []
        if res.boxes is not None and len(res.boxes):
            for i, box in enumerate(res.boxes):
                if int(box.cls[0].item()) != 0: # Class 0 is person
                    continue
                    
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                bh, bw = y2 - y1, x2 - x1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Filtering logic
                if bh < 60 or bw < 30: continue
                if bh > self.H * 0.75 or bw > self.W * 0.30: continue
                
                rx1, ry1, rx2, ry2 = self.roi
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2): continue
                if cy < self.H * 0.12: continue
                
                kps = None
                if res.keypoints and i < len(res.keypoints):
                    kps = res.keypoints[i].xy[0].cpu().numpy()
                
                raw.append({
                    "cx": cx, "cy": cy,
                    "conf": float(box.conf[0].item()),
                    "kps": kps,
                    "box": xyxy
                })
        
        return self.tracker.update(raw)
