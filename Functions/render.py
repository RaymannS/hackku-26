import cv2
import numpy as np
import random

def draw_tree(canvas, x, y, size=12, color=(20, 80, 20)):
    pts = np.array([[x, y-size], [x-size//2, y], [x+size//2, y]], np.int32)
    cv2.fillPoly(canvas, [pts], color)
    cv2.rectangle(canvas, (x-2, y), (x+2, y+size//3), (60, 40, 20), -1)
    cv2.polylines(canvas, [pts], True, (10, 40, 10), 1)

def draw_cactus(canvas, x, y, size=14, color=(40, 120, 40)):
    cv2.rectangle(canvas, (x-2, y-size), (x+2, y+size//2), color, -1)
    cv2.rectangle(canvas, (x-size//2, y-size//3), (x-2, y), color, -1)
    cv2.rectangle(canvas, (x-size//2, y-size//2), (x-size//2+3, y-size//3), color, -1)
    cv2.rectangle(canvas, (x+2, y-size//3), (x+size//2, y), color, -1)
    cv2.rectangle(canvas, (x+size//2-3, y-size//2), (x+size//2, y-size//3), color, -1)

def draw_house(canvas, x, y, size=14):
    cv2.rectangle(canvas, (x-size//2, y-size//2), (x+size//2, y+size//2), (180, 160, 120), -1)
    cv2.rectangle(canvas, (x-size//2, y-size//2), (x+size//2, y+size//2), (80, 60, 40), 1)
    pts = np.array([[x, y-size], [x-size//2-2, y-size//2], [x+size//2+2, y-size//2]], np.int32)
    cv2.fillPoly(canvas, [pts], (30, 50, 120))
    cv2.polylines(canvas, [pts], True, (20, 35, 90), 1)
    cv2.rectangle(canvas, (x-3, y), (x+3, y+size//2), (80, 50, 20), -1)

def apply_desert_terrain(canvas, mask, Z):
    z_vals = Z[mask].astype(float)
    r = np.clip(194 + (z_vals - 90) * 0.3, 170, 220).astype(np.uint8)
    g = np.clip(154 + (z_vals - 90) * 0.2, 130, 175).astype(np.uint8)
    b = np.clip(80  + (z_vals - 90) * 0.1,  60, 110).astype(np.uint8)
    canvas[mask] = np.stack([b, g, r], axis=1)
    noise = np.random.randint(-15, 15, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas

def draw_bridge(canvas, p1, p2, color=(120, 80, 40), width=4):
    cv2.line(canvas, p1, p2, color, width)
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
    length = max(1, int(np.sqrt(dx**2 + dy**2)))
    for i in range(0, length, 10):
        t = i/length
        mx = int(p1[0]+t*dx); my = int(p1[1]+t*dy)
        px = int(-dy/length*5); py = int(dx/length*5)
        cv2.line(canvas, (mx-px, my-py), (mx+px, my+py), (80, 50, 20), 2)

def draw_path(canvas, points, color=(155, 195, 210), width=2):
    for i in range(len(points)-1):
        cv2.line(canvas, points[i], points[i+1], color, width, cv2.LINE_AA)

def add_winding(path, strength=10):
    winding = []
    for i, (x, y) in enumerate(path):
        if i == 0 or i == len(path) - 1:
            winding.append((x, y))
        else:
            nx = x + random.randint(-strength, strength)
            ny = y + random.randint(-strength, strength)
            winding.append((nx, ny))
    return winding

def draw_orc(canvas, x, y, size=12):
    # Body
    cv2.circle(canvas, (x, y), size // 2, (30, 90, 30), -1)
    # Head
    cv2.circle(canvas, (x, y - size), size // 3, (40, 110, 40), -1)
    # Eyes
    cv2.circle(canvas, (x - 3, y - size), 2, (0, 0, 180), -1)
    cv2.circle(canvas, (x + 3, y - size), 2, (0, 0, 180), -1)
    # Tusks
    cv2.line(canvas, (x - 3, y - size + 3), (x - 5, y - size + 7), (200, 200, 200), 1)
    cv2.line(canvas, (x + 3, y - size + 3), (x + 5, y - size + 7), (200, 200, 200), 1)

def spawn_orcs(canvas, player_x, player_y, n=5, radius=60):
    spawned = 0
    attempts = 0
    placed = []
    while spawned < n and attempts < 200:
        angle = random.uniform(0, 2 * np.pi)
        dist = random.uniform(radius * 0.3, radius)
        x = int(player_x + dist * np.cos(angle))
        y = int(player_y + dist * np.sin(angle))
        h, w = canvas.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            if all(abs(x - orc["x"]) > 15 or abs(y - orc["y"]) > 15 for orc in placed):
                draw_orc(canvas, x, y)
                placed.append({"x": x, "y": y, "defeated": False})
                spawned += 1
        attempts += 1
        
    print(f"Spawned {spawned} orcs near ({player_x}, {player_y})")
    return placed
    
def draw_label(canvas, x, y, name, size=0.5):
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(name, font, size, thickness)
    tx = x - tw // 2
    ty = y - 22
    pad = 6
    h_pad = 20
    x1, y1 = tx - h_pad, ty - th - pad
    x2, y2 = tx + tw + h_pad, ty + pad
    roll_w = 8
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (155, 195, 210), -1)
    cv2.rectangle(canvas, (x1, y1), (x1 + roll_w, y2), (120, 160, 180), -1)
    cv2.rectangle(canvas, (x2 - roll_w, y1), (x2, y2), (120, 160, 180), -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 75, 100), 1)
    mid_y = (y1 + y2) // 2
    #cv2.line(canvas, (x1 + roll_w, mid_y), (x2 - roll_w, mid_y), (138, 178, 195), 1)
    cv2.putText(canvas, name, (tx+1, ty+1), font, size, (80, 60, 30), thickness, cv2.LINE_AA)
    cv2.putText(canvas, name, (tx, ty), font, size, (20, 10, 5), thickness, cv2.LINE_AA)
    
def scatter_placer(mask, n=30, min_dist=40):
    coords = np.argwhere(mask).tolist()
    if not coords:
        return []
    random.shuffle(coords)
    placed = []
    for pt in coords:
        y, x = pt
        if all(abs(x-px) >= min_dist or abs(y-py) >= min_dist for px, py in placed):
            placed.append((x, y))
        if len(placed) >= n:
            break
    return placed