import numpy as np
from filterpy.kalman import KalmanFilter
from config import Config

class BallKalman:
    """
    Kalman filter for smooth ball tracking with ROI clamping.
    """
    def __init__(self, roi=Config.ROI):
        self.roi = roi
        self.init = False
        self.MAX_JUMP = 120
        self._setup_kf()

    def _setup_kf(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        # State transition matrix: x, y, vx, vy
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        # Measurement transition matrix
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        # Tuning parameters
        kf.R *= 4    # Measurement noise
        kf.Q *= 0.1  # Process noise
        kf.P *= 100  # Error covariance
        
        # Set explicitly as float64
        kf.R = kf.R.astype(np.float64)
        kf.Q = kf.Q.astype(np.float64)
        kf.P = kf.P.astype(np.float64)
        kf.x = np.zeros((4, 1), dtype=np.float64)
        self.kf = kf

    def _get_scalar(self, val):
        return float(np.asarray(val).flat[0])

    def _is_in_roi(self, x, y):
        x1, y1, x2, y2 = self.roi
        return x1 <= x <= x2 and y1 <= y <= y2

    def _clamp_to_roi(self, x, y):
        x1, y1, x2, y2 = self.roi
        return (float(np.clip(x, x1, x2)), float(np.clip(y, y1, y2)))

    def _is_jump_valid(self, cx, cy):
        if not self.init:
            return True
        lx = self._get_scalar(self.kf.x[0])
        ly = self._get_scalar(self.kf.x[1])
        vx = self._get_scalar(self.kf.x[2])
        vy = self._get_scalar(self.kf.x[3])
        # Allow larger jumps if velocity is high
        return np.hypot(cx - lx, cy - ly) < max(self.MAX_JUMP, np.hypot(vx, vy) * 3.5)

    def reset(self):
        self._setup_kf()
        self.init = False

    def update(self, det):
        """
        Updates the Kalman filter with a new detection.
        det: (x, y) or None
        """
        if det is None:
            return None
            
        cx, cy = float(det[0]), float(det[1])
        
        # Check ROI and jump validity
        if not self._is_in_roi(cx, cy) or not self._is_jump_valid(cx, cy):
            self.reset()
            return None
            
        if not self.init:
            self.kf.x = np.array([[cx], [cy], [0.], [0.]], dtype=np.float64)
            self.init = True
        else:
            self.kf.predict()
            self.kf.update(np.array([[cx], [cy]], dtype=np.float64))
            
        # Clamp estimated position to ROI
        px, py = self._clamp_to_roi(self._get_scalar(self.kf.x[0]), self._get_scalar(self.kf.x[1]))
        self.kf.x[0, 0] = px
        self.kf.x[1, 0] = py
        
        return (px, py)
