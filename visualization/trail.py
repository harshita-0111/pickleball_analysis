import cv2
import numpy as np
from collections import deque

class ShortTrail:
    """
    Trajectory trail with speed-based coloring and recent history.
    """
    MAX_JUMP = 150

    def __init__(self, n=6):
        self.trail = deque(maxlen=n)
        self.current_speed = 0.

    def push(self, pos):
        if pos is None:
            return
        if self.trail:
            last = list(self.trail)[-1]
            if last and np.hypot(pos[0] - last[0], pos[1] - last[1]) > self.MAX_JUMP:
                self.trail.clear()
        self.trail.append(pos)

    def clear(self):
        self.trail.clear()
        self.current_speed = 0.

    def _get_color(self, spd, alpha=1.0):
        t = np.clip(spd / 50., 0, 1)
        if t < 0.5:
            s = t * 2
            r, g, b = int(50 * s), 255, int(255 * (1 - s))
        else:
            s = (t - 0.5) * 2
            r, g, b = int(50 + 205 * s), int(255 * (1 - s)), 0
        return (int(b * alpha), int(g * alpha), int(r * alpha))

    def draw(self, frame):
        pts = list(self.trail)
        if not pts:
            return frame
            
        cx, cy = int(pts[-1][0]), int(pts[-1][1])
        if len(pts) >= 2:
            self.current_speed = float(np.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]))
        else:
            self.current_speed = 0.
            
        # Draw the trail segment by segment
        for i in range(1, len(pts)):
            alpha = 0.3 + 0.7 * (i / len(pts))
            spd = np.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
            color = self._get_color(spd, alpha)
            thickness = max(1, int(1 + 2 * (i / len(pts))))
            
            p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
            p2 = (int(pts[i][0]), int(pts[i][1]))
            
            # Shadow
            cv2.line(frame, (p1[0] + 2, p1[1] + 2), (p2[0] + 2, p2[1] + 2), (15, 15, 15), thickness + 1, cv2.LINE_AA)
            # Main line
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
            
        # Draw current position marker
        spd = self.current_speed
        color = self._get_color(spd)
        
        if spd > 20 and len(pts) >= 2:
            # Motion blur effect
            dx = pts[-1][0] - pts[-2][0]
            dy = pts[-1][1] - pts[-2][1]
            norm = np.hypot(dx, dy) + 1e-6
            dx, dy = dx / norm, dy / norm
            blur_len = int(np.clip(spd * 0.6, 6, 30))
            x0, y0 = int(cx - dx * blur_len), int(cy - dy * blur_len)
            cv2.line(frame, (x0, y0), (cx, cy), color, 5)
            cv2.line(frame, (x0, y0), (cx, cy), (255, 255, 255), 1, cv2.LINE_AA)
            
        cv2.circle(frame, (cx, cy), 14, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 5, color, -1)
        
        cv2.putText(frame, f"{spd:.0f}px/f", (cx + 16, cy - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return frame
