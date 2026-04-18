import numpy as np
import cv2
import heapq
import random
import re
import speech_recognition as sr

VOICE_MODE = False  # Change to True for voice, False for typing

# -----------------------------
# HEIGHT LEVELS
# -----------------------------
sea_level = 90
mountain_level = 180
snow_level = 240

# -----------------------------
# LOCATION REGISTRY
# -----------------------------
named_locations = {}

# -----------------------------
# BAND CONFIGS
# -----------------------------
water_cfg = {
    "in_min": 0, "in_max": sea_level,
    "out_min": 0, "out_max": 170,
    "colormap": cv2.COLORMAP_PARULA
}
land_cfg = {
    "in_min": sea_level, "in_max": mountain_level,
    "out_min": 60, "out_max": 120,
    "colormap": cv2.COLORMAP_DEEPGREEN
}
mount_cfg = {
    "in_min": mountain_level, "in_max": 255,
    "out_min": 25, "out_max": 85,
    "colormap": cv2.COLORMAP_BONE
}

# -----------------------------
# TERRAIN
# -----------------------------
depth = cv2.imread("MiDaS-master/output/sand-dpt_large_384.pfm", cv2.IMREAD_UNCHANGED)
Z = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
Z = Z.astype(np.uint8)
# -----------------------------
# SLOPE (cliffs)
# -----------------------------
gy, gx = np.gradient(Z.astype(float))
slope = np.sqrt(gx**2 + gy**2)
slope = cv2.normalize(slope, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# -----------------------------
# BAND FUNCTION
# -----------------------------
def apply_band(depth, cfg):
    x = np.clip(depth - cfg["in_min"], 0, cfg["in_max"] - cfg["in_min"])
    x = x / (cfg["in_max"] - cfg["in_min"] + 1e-6)
    x = x * (cfg["out_max"] - cfg["out_min"]) + cfg["out_min"]
    x = x.astype(np.uint8)
    return cv2.applyColorMap(x, cfg["colormap"])

# -----------------------------
# APPLY BANDS
# -----------------------------
water = apply_band(Z, water_cfg)
land = apply_band(Z, land_cfg)
mountain = apply_band(Z, mount_cfg)

# -----------------------------
# MASKS
# -----------------------------
water_mask = Z < sea_level
mountain_mask = Z >= mountain_level
snow_mask = Z >= snow_level
cliffs = slope > 120

# -----------------------------
# BASE COMPOSITION
# -----------------------------
final = land.copy()
final[water_mask] = water[water_mask]
final[mountain_mask] = mountain[mountain_mask]
final[snow_mask] = [245, 245, 245]
final[cliffs & ~water_mask] = (final[cliffs & ~water_mask] * 0.6).astype(np.uint8)

# PARCHMENT
paper = np.full_like(final, [235, 220, 190])
final = cv2.addWeighted(final, 0.85, paper, 0.15, 0)

# OUTLINES
edges = cv2.Canny(Z, 40, 120)
final[edges > 0] = [20, 20, 20]

# -----------------------------
# LABEL RENDERER
# -----------------------------
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

# -----------------------------
# NAME PARSERS
# -----------------------------
def parse_name(prompt):
    match = re.search(r'(?:called|named)\s+(.+?)$', prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip().title()
    return None

def parse_named_path(prompt):
    match = re.search(r'between\s+(.+?)\s+and\s+(.+?)$', prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower(), match.group(2).strip().lower()
    match = re.search(r'from\s+(.+?)\s+to\s+(.+?)$', prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower(), match.group(2).strip().lower()
    return None, None

# -----------------------------
# RENDERERS
# -----------------------------
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

# -----------------------------
# PLACERS
# -----------------------------
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

# -----------------------------
# PATHFINDING
# -----------------------------
def find_path(Z, start, end, avoid_mask=None, snow_lvl=240, scale=16):
    h_full, w_full = Z.shape
    Z_small = cv2.resize(Z, (w_full//scale, h_full//scale), interpolation=cv2.INTER_AREA)
    if avoid_mask is not None:
        avoid_small = cv2.resize(avoid_mask.astype(np.uint8), (w_full//scale, h_full//scale)) > 0
    else:
        avoid_small = None

    start_s = (start[0]//scale, start[1]//scale)
    end_s   = (end[0]//scale,   end[1]//scale)
    h, w = Z_small.shape

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def cost(pos):
        y, x = pos
        base = 1.0
        if avoid_small is not None and avoid_small[y, x]:
            base += 100
        if Z_small[y, x] >= snow_lvl:
            base += 40
        elif Z_small[y, x] >= 200:
            base += 15
        base += abs(float(Z_small[y, x]) - 120) * 0.05
        return base

    open_set = [(0, start_s)]
    came_from = {}
    g_score = {start_s: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end_s:
            path = []
            while current in came_from:
                path.append((current[1]*scale, current[0]*scale))
                current = came_from[current]
            path.append((start[1], start[0]))
            return path[::-1]
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = current[0]+dy, current[1]+dx
            if 0 <= ny < h and 0 <= nx < w:
                neighbor = (ny, nx)
                tentative_g = g_score[current] + cost(neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, end_s)
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current
    return []

# -----------------------------
# REGION + POINT PARSING
# -----------------------------
def get_region_mask(prompt, h, w):
    mask = np.ones((h, w), dtype=bool)
    if "top" in prompt:
        mask[h//3:, :] = False
    elif "bottom" in prompt:
        mask[:2*h//3, :] = False
    elif "middle" in prompt or "center" in prompt:
        mask[:h//3, :] = False
        mask[2*h//3:, :] = False
    if "left" in prompt:
        mask[:, w//3:] = False
    elif "right" in prompt:
        mask[:, :2*w//3] = False
    elif "middle" in prompt or "center" in prompt:
        mask[:, :w//3] = False
        mask[:, 2*w//3:] = False
    return mask

def parse_points(prompt, h, w):
    coords = re.findall(r'(\d+)\s*,\s*(\d+)', prompt)
    if len(coords) >= 2:
        return (int(coords[0][0]), int(coords[0][1])), (int(coords[1][0]), int(coords[1][1]))
    return (50, 50), (w-50, h-50)

# -----------------------------
# COMPOSITE
# -----------------------------
def composite(base, path_layer, feature_layer):
    """Merge layers: path under features"""
    result = path_layer.copy()
    diff = np.any(feature_layer != base, axis=2)
    result[diff] = feature_layer[diff]
    return result

# -----------------------------
# PROMPT PARSER
# -----------------------------
def parse_and_apply(prompt, feature_canvas, path_canvas, Z, sea_level, mountain_level, snow_level):
    p = prompt.lower()
    h, w = Z.shape
    water_mask = Z < sea_level

    if ("forest" in p) and not ("path" in p or "road" in p):
        region_mask = get_region_mask(p, h, w)
        mask = (Z >= sea_level) & (Z < mountain_level) & region_mask
        pts = scatter_placer(mask, n=50, min_dist=20)
        for x, y in pts:
            draw_tree(feature_canvas, x, y)
        name = parse_name(prompt)
        if name and pts:
            cx = int(np.mean([pt[0] for pt in pts]))
            cy = int(np.mean([pt[1] for pt in pts]))
            draw_label(feature_canvas, cx, cy, name)
            named_locations[name.lower()] = (cx, cy)
            print(f"Labelled forest as '{name}'")
        print(f"Placed {len(pts)} trees")

    if ("desert" in p) and not ("path" in p or "road" in p):
        region_mask = get_region_mask(p, h, w)
        mask = (Z >= sea_level) & (Z < sea_level + 50) & region_mask
        feature_canvas = apply_desert_terrain(feature_canvas, mask, Z)
        pts = scatter_placer(mask, n=35, min_dist=25)
        for x, y in pts:
            draw_cactus(feature_canvas, x, y)
        name = parse_name(prompt)
        if name and pts:
            cx = int(np.mean([pt[0] for pt in pts]))
            cy = int(np.mean([pt[1] for pt in pts]))
            draw_label(feature_canvas, cx, cy, name)
            named_locations[name.lower()] = (cx, cy)
            print(f"Labelled desert as '{name}'")
        print(f"Desert applied with {len(pts)} cacti")

    if ("town" in p or "village" in p or "city" in p) and not ("path" in p or "road" in p):
        region_mask = get_region_mask(p, h, w)
        n = 15 if "city" in p else 6
        mask = (Z >= sea_level + 5) & (Z < mountain_level - 20) & region_mask
        pts = scatter_placer(mask, n=n, min_dist=15)
        for x, y in pts:
            draw_house(feature_canvas, x, y)
        name = parse_name(prompt)
        if name and pts:
            cx = int(np.mean([pt[0] for pt in pts]))
            cy = int(np.mean([pt[1] for pt in pts]))
            draw_label(feature_canvas, cx, cy, name)
            named_locations[name.lower()] = (cx, cy)
            print(f"Labelled as '{name}'")
        print(f"Placed {len(pts)} buildings")

    if ("path" in p or "road" in p) and ("between" in p or "from" in p):
        name_a, name_b = parse_named_path(prompt)
        if name_a and name_b and name_a in named_locations and name_b in named_locations:
            start = named_locations[name_a]
            end = named_locations[name_b]
            print(f"Drawing path from '{name_a}' to '{name_b}'")
        else:
            if name_a and name_a not in named_locations:
                print(f"Location '{name_a}' not found, falling back to coordinates")
            if name_b and name_b not in named_locations:
                print(f"Location '{name_b}' not found, falling back to coordinates")
            start, end = parse_points(p, h, w)
        path_pts = find_path(Z, (start[1], start[0]), (end[1], end[0]), water_mask, snow_level)
        path_pts = path_pts[::3]
        path_pts = add_winding(path_pts, strength=8)
        draw_path(path_canvas, path_pts)  # draw to path layer
        print(f"Drew path with {len(path_pts)} points")

    if "bridge" in p:
        start, end = parse_points(p, h, w)
        draw_bridge(path_canvas, start, end)  # bridges also on path layer
        print(f"Drew bridge from {start} to {end}")

    return feature_canvas, path_canvas

# -----------------------------
# SPEECH SETUP
# -----------------------------
recognizer = sr.Recognizer()
mic = sr.Microphone()

if VOICE_MODE:
    print("Calibrating microphone...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Ready! Listening for commands...")

def listen_for_command():
    with mic as source:
        print("\nListening...")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)
            command = recognizer.recognize_google(audio)
            print(f"Heard: '{command}'")
            return command
        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError:
            print("Speech service unavailable")
            return None

# -----------------------------
# INTERACTIVE LOOP
# -----------------------------
feature_layer = final.copy()
path_layer = final.copy()

print(f"Mode: {'VOICE' if VOICE_MODE else 'TYPING'}")
print("Commands: add a forest/desert/town/village/city in [region] called [name]")
print("          draw a path/road between [name] and [name]")
print("          draw a bridge from x,y to x,y")
print("          list | reset | save | quit")

while True:
    current_map = composite(final, path_layer, feature_layer)
    cv2.imshow("D&D World Map", current_map)
    cv2.waitKey(1)

    if VOICE_MODE:
        prompt = listen_for_command()
    else:
        prompt = input("\nEnter command: ").strip()

    if prompt is None or prompt == "":
        continue

    if "quit" in prompt.lower():
        break
    elif "reset" in prompt.lower():
        feature_layer = final.copy()
        path_layer = final.copy()
        named_locations.clear()
        print("Map reset")
    elif "save" in prompt.lower():
        cv2.imwrite("map.png", composite(final, path_layer, feature_layer))
        print("Saved map.png")
    elif "list" in prompt.lower():
        print(f"Known locations: {list(named_locations.keys())}")
    else:
        feature_layer, path_layer = parse_and_apply(
            prompt, feature_layer, path_layer, Z, sea_level, mountain_level, snow_level
        )

cv2.destroyAllWindows()