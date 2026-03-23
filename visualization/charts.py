import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import Config

def create_shot_chart(shot_counts, output_path):
    """Creates a bar chart of shot distributions."""
    labels = list(Config.SHOT_COLORS.keys())
    counts = [shot_counts.get(l, 0) for l in labels]
    colors = [tuple(np.array(Config.SHOT_COLORS[l][::-1]) / 255.) for l in labels] # BGR to RGB normalized

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=colors)
    plt.title("Shot Distribution")
    plt.xlabel("Shot Type")
    plt.ylabel("Count")
    plt.savefig(output_path)
    plt.close()

def create_heatmap(positions, W, H, output_path):
    """Creates a heatmap of ball positions on the court."""
    heatmap = np.zeros((H, W), dtype=np.float32)
    for _, x, y in positions:
        ix, iy = int(x), int(y)
        if 0 <= ix < W and 0 <= iy < H:
            heatmap[iy, ix] += 1
            
    # Apply Gaussian blur for better visualization
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = heatmap / (heatmap.max() + 1e-6)
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    cv2.imwrite(output_path, heatmap_color)
