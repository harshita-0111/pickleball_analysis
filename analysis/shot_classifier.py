import numpy as np
from collections import deque, Counter

class ShotClassifier:
    """
    Classifies pickleball shots based on trajectory history.
    Shots: DINK, DRIVE, LOB, SMASH, DROP
    """
    def __init__(self, window=20, cooldown=18, H=1080):
        self.H = H
        self.hist = deque(maxlen=window)
        self.last_shot = None
        self.last_frame = -999
        self.cooldown = cooldown
        self.shot_counts = Counter()
        self.conf = 0.0

    def reset(self):
        self.hist.clear()
        self.last_shot = None
        self.conf = 0.0

    def update(self, pos, frame_idx):
        """
        Updates shot classifier with new position and returns (shot_type, confidence).
        """
        if pos:
            self.hist.append((frame_idx, float(pos[0]), float(pos[1])))
        
        if len(self.hist) < 15:
            return None, 0.0
            
        shot, conf = self._classify()
        
        # Apply cooldown to prevent double classification of the same shot
        if shot and frame_idx - self.last_frame >= self.cooldown:
            self.last_shot = shot
            self.last_frame = frame_idx
            self.conf = conf
            self.shot_counts[shot] += 1
            
        return self.last_shot, self.conf

    def _classify(self):
        pts = list(self.hist)
        xs = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        
        # Calculate velocities
        dx = np.diff(xs)
        dy = np.diff(ys)
        dist = np.sqrt(dx**2 + dy**2)
        
        avg_speed = float(np.mean(dist))
        max_speed = float(np.max(dist))
        
        # Vertical direction in the last few frames
        recent_dy = float(np.mean(dy[-6:])) if len(dy) >= 6 else float(np.mean(dy))
        
        mid = len(ys) // 2
        # Arc height (max y-coordinate difference in the first half of the window)
        arc = float(ys[0] - np.min(ys[:mid+1])) if mid > 1 else 0.
        
        # Horizontal straightness
        horiz_ratio = abs(float(np.sum(dx))) / (float(np.sum(dist)) + 1e-6)
        
        min_y = float(np.min(ys))
        
        # ── Shot Logic ────────────────────────────────────────────────
        if max_speed >= 65 and recent_dy >= 6.0 and min_y < self.H * 0.50:
            return "SMASH", 0.90
        elif arc >= 40 and avg_speed <= 25.:
            return "LOB", 0.86
        elif avg_speed >= 22 and horiz_ratio >= 0.40:
            return "DRIVE", 0.84
        elif recent_dy >= 4. and 10 < avg_speed <= 25:
            return "DROP", 0.80
        elif avg_speed <= 10:
            return "DINK", 0.82
        elif avg_speed > 22:
            return "DRIVE", 0.70
        else:
            return "DINK", 0.60
