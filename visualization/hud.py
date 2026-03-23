import cv2
import numpy as np
from config import Config

def draw_panel(frame, x, y, w, h, alpha=0.78):
    """Draws a semi-transparent dark panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_player_circles(frame, players, hitter_num=None, timer=0):
    """Draws circles around players with highlighting for the current hitter."""
    for p in players:
        pcx, pcy = int(p["cx"]), int(p["cy"])
        num = p["num"]
        color = p["color"]
        is_hitter = (num == hitter_num and timer > 0)
        
        radius = 22 if is_hitter else 18
        if is_hitter:
            cv2.circle(frame, (pcx, pcy), radius + 6, color, 2, cv2.LINE_AA)
            
        cv2.circle(frame, (pcx, pcy), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (pcx, pcy), radius, (255, 255, 255), 2, cv2.LINE_AA)
        
        txt = str(num)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.putText(frame, txt, (pcx - tw // 2, pcy + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

def draw_shot_label(frame, shot, conf, timer, player_num=None, W=1920):
    """Draws the big shot label at the top center."""
    if not shot or timer <= 0 or player_num is None:
        return
        
    sc = Config.SHOT_COLORS.get(shot, (200, 200, 200))
    pcol = Config.PLAYER_COLORS.get(player_num, (200, 200, 200))
    
    alpha = min(1.0, timer / 12.0)
    color_alpha = tuple(int(c * alpha) for c in sc)
    player_color_alpha = tuple(int(c * alpha) for c in pcol)
    
    icon = Config.SHOT_ICONS.get(shot, "")
    lbl = f"{icon}  {shot}"
    sub = f"Player {player_num}  ·  {conf:.0%}"
    
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 1.6, 2)
    (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    
    max_w = max(tw, sw)
    lx = W // 2 - max_w // 2
    tot_h = th + sh + 36
    
    draw_panel(frame, lx - 24, 10, max_w + 48, tot_h, 0.85 * alpha)
    
    # Left decorative bar
    cv2.rectangle(frame, (lx - 24, 14), (lx - 16, 10 + tot_h - 4), player_color_alpha, -1)
    # Border
    cv2.rectangle(frame, (lx - 24, 10), (lx + max_w + 24, 10 + tot_h), color_alpha, 2)
    
    cv2.putText(frame, lbl, (lx, 10 + th + 6), cv2.FONT_HERSHEY_DUPLEX, 1.6, color_alpha, 2, cv2.LINE_AA)
    cv2.putText(frame, sub, (lx, 10 + th + sh + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1, cv2.LINE_AA)

def draw_shot_panel(frame, shot_counts, W=1920):
    """Draws a summary panel with shot counts in the top right."""
    px0 = W - 220
    py0 = 8
    ph = 28 + len(Config.SHOT_COLORS) * 28 + 8
    
    draw_panel(frame, px0, py0, 212, ph, 0.78)
    cv2.putText(frame, "SHOTS", (px0 + 10, py0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    
    for i, (shot_name, color) in enumerate(Config.SHOT_COLORS.items()):
        count = shot_counts.get(shot_name, 0)
        cv2.putText(frame, f"{shot_name:<6} {count}", (px0 + 10, py0 + 38 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

def draw_speed_bar(frame, spd, H=1080):
    """Draws a speed gauge at the bottom left."""
    lx, ly = 12, H - 65
    draw_panel(frame, lx - 4, ly - 18, 160, 58, 0.75)
    cv2.putText(frame, "SPEED", (lx, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
    
    for i in range(140):
        t = i / 140
        if t < 0.5:
            s = t * 2
            c = (int(255 * (1 - s)), 255, int(50 * s))
        else:
            s = (t - 0.5) * 2
            c = (0, int(255 * (1 - s)), int(50 + 205 * s))
        cv2.line(frame, (lx + i, ly + 2), (lx + i, ly + 14), c, 1)
        
    line_pos = int(np.clip(spd / 60, 0, 1) * 140)
    cv2.line(frame, (lx + line_pos, ly), (lx + line_pos, ly + 16), (255, 255, 255), 2)
    cv2.putText(frame, f"{spd:.0f} px/f", (lx, ly + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
