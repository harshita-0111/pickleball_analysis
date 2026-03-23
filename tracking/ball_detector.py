import torch
import cv2
import numpy as np
from config import Config

class BallDetector:
    """
    Fused ball detector combining TrackNetV3, YOLOv8, and HSV color/motion fallback.
    """
    def __init__(self, tracknet_model, yolo_model, device=Config.DEVICE):
        self.tn_model = tracknet_model
        self.yolo_model = yolo_model
        self.device = device
        self.tn_thresh = Config.TN_THRESH
        self.roi = Config.ROI
        self.iW, self.iH = Config.INPUT_W, Config.INPUT_H

    def _is_blacklisted(self, cx, cy):
        for bx, by, r in Config.MANUAL_BLACKLIST:
            if abs(cx - bx) < r and abs(cy - by) < r:
                return True
        return False

    def _preprocess_tn(self, frame):
        r = cv2.resize(frame, (self.iW, self.iH))
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        return torch.tensor((r.astype(np.float32) / 255.).transpose(2, 0, 1))

    def _heatmap_peak(self, hm, thr, ow, oh):
        if hm.max() < thr:
            return None, float(hm.max())
        mask = (hm >= hm.max() * 0.8).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, float(hm.max())
        M = cv2.moments(max(cnts, key=cv2.contourArea))
        if M["m00"] == 0:
            return None, float(hm.max())
        return (int(M["m10"] / M["m00"] * ow / self.iW),
                int(M["m01"] / M["m00"] * oh / self.iH)), float(hm.max())

    def detect(self, fp, fc, fn, W, H):
        """
        Runs multi-tier detection logic.
        fp: previous frame, fc: current frame, fn: next frame
        """
        x1, y1, x2, y2 = self.roi
        
        # ── Tier 1: TrackNetV3 ────────────────────────────────────────
        hm = None
        if self.tn_model is not None:
            with torch.no_grad():
                inp = torch.stack([self._preprocess_tn(fp),
                                 self._preprocess_tn(fc),
                                 self._preprocess_tn(fn)], dim=0).reshape(1, 9, self.iH, self.iW).to(self.device)
                hm = self.tn_model(inp)[0, 1].cpu().numpy()
                
            pos, pk = self._heatmap_peak(hm, self.tn_thresh, W, H)
            if pos and x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2 and not self._is_blacklisted(*pos):
                # Adaptive thresholding
                self.tn_thresh = min(0.55, self.tn_thresh + 0.003)
                return pos, "tracknet", pk, hm
        else:
            # Create a dummy heatmap if needed for return consistency, or leave as None
            pass

        # ── Tier 2: YOLOv8 ────────────────────────────────────────────
        self.tn_thresh = max(0.25, self.tn_thresh - 0.005)
        res = self.yolo_model(fc, conf=0.08, verbose=False)[0]
        cands = []
        for box in res.boxes:
            if self.yolo_model.names[int(box.cls[0].item())] != "ball":
                continue
            xy = box.xyxy[0].cpu().numpy()
            cx = float((xy[0] + xy[2]) / 2)
            cy = float((xy[1] + xy[3]) / 2)
            bw = float(xy[2] - xy[0])
            bh = float(xy[3] - xy[1])
            
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            if bw > 80 or bh > 80:
                continue
            if self._is_blacklisted(int(cx), int(cy)):
                continue
            cands.append((float(box.conf[0]), int(cx), int(cy)))
            
        if cands:
            cands.sort(reverse=True)
            return (cands[0][1], cands[0][2]), "yolo", cands[0][0], hm

        # ── Tier 3: HSV + Motion ──────────────────────────────────────
        g0 = cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(fn, cv2.COLOR_BGR2GRAY)
        d1 = cv2.absdiff(g1, g0)
        d2 = cv2.absdiff(g1, g2)
        _, t1 = cv2.threshold(d1, 10, 255, cv2.THRESH_BINARY)
        _, t2 = cv2.threshold(d2, 10, 255, cv2.THRESH_BINARY)
        
        hsv = cv2.cvtColor(fc, cv2.COLOR_BGR2HSV)
        col = cv2.inRange(hsv, np.array([28, 60, 130]), np.array([60, 255, 255]))
        motion = cv2.bitwise_and(t1, t2)
        
        # ROI mask
        rm = np.zeros((H, W), dtype=np.uint8)
        rm[y1:y2, x1:x2] = 255
        comb = cv2.bitwise_and(cv2.bitwise_and(motion, rm), col)
        
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, kernel3)
        comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernel5)
        
        cnts, _ = cv2.findContours(comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 30 or area > 700:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            perim = cv2.arcLength(cnt, True)
            circ = (4 * np.pi * area) / (perim**2 + 1e-6)
            if circ < 0.30:
                continue
            cx, cy = int(cx), int(cy)
            if self._is_blacklisted(cx, cy):
                continue
            score = circ * min(area, 400)
            if best is None or score > best[0]:
                best = (score, cx, cy)
                
        if best:
            return (best[1], best[2]), "hsv", 0.0, hm
            
        return None, "none", 0.0, hm
