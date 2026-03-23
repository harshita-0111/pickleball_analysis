import numpy as np
from scipy.signal import savgol_filter, find_peaks

def detect_bounces(positions, window_size=7):
    """
    Detects ball bounces in a trajectory by looking for abrupt changes in vertical velocity.
    positions: list of (x, y) coordinates
    """
    if len(positions) < window_size * 2:
        return []
        
    ys = np.array([p[1] for p in positions])
    
    # Smooth the vertical trajectory
    try:
        smoothed_y = savgol_filter(ys, window_size, 2)
    except:
        return []
        
    # Vertical velocity (dy/dt)
    vy = np.diff(smoothed_y)
    
    # Vertical acceleration (dvy/dt)
    # A bounce often corresponds to a large positive peak in vertical acceleration (inflection point)
    ay = np.diff(vy)
    
    # Adjust thresholds based on scale if needed
    # In pickleball, a bounce is a sudden reversal from downward to upward velocity
    # So we look for peaks in the second derivative (acceleration)
    peaks, properties = find_peaks(ay, height=1.5, distance=10)
    
    bounce_indices = []
    for p in peaks:
        # p is index in ay, which corresponds to index p+1 in positions
        bounce_indices.append(p + 1)
        
    return bounce_indices
