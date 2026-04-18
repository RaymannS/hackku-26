import cv2
import numpy as np
import random

def overlay_image(canvas, img_path, x, y, size):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Could not load image: {img_path}")
        return
    img = cv2.resize(img, (size, size))
    
    x1, y1 = x - size//2, y - size//2
    x2, y2 = x1 + size, y1 + size

    # clamp to canvas bounds
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)
    
    # crop image to match
    ix1, iy1 = cx1 - x1, cy1 - y1
    ix2, iy2 = ix1 + (cx2 - cx1), iy1 + (cy2 - cy1)

    if cx2 <= cx1 or cy2 <= cy1:
        return

    crop = img[iy1:iy2, ix1:ix2]
    if img.shape[2] == 4:
        alpha = crop[:, :, 3:4] / 255.0
        canvas[cy1:cy2, cx1:cx2] = (crop[:, :, :3] * alpha + canvas[cy1:cy2, cx1:cx2] * (1 - alpha)).astype(np.uint8)
    else:
        canvas[cy1:cy2, cx1:cx2] = crop

def draw_tree(canvas, x, y, size=80):
    overlay_image(canvas, "Images/tree.png", x, y, size)

def draw_cactus(canvas, x, y, size=80):
    overlay_image(canvas, "Images/cactus.png", x, y, size)

def draw_house(canvas, x, y, size=90):
    overlay_image(canvas, "Images/house.png", x, y, size)
    
def draw_castle(canvas, x, y, size=150):
    overlay_image(canvas, "Images/castle.png", x, y, size)

def apply_desert_terrain(canvas, mask, Z):
    z_vals = Z[mask].astype(float)
    r = np.clip(194 + (z_vals - 90) * 0.3, 170, 220).astype(np.uint8)
    g = np.clip(154 + (z_vals - 90) * 0.2, 130, 175).astype(np.uint8)
    b = np.clip(80  + (z_vals - 90) * 0.1,  60, 110).astype(np.uint8)
    canvas[mask] = np.stack([b, g, r], axis=1)
    noise = np.random.randint(-15, 15, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def draw_contour_lines(canvas, Z, interval=20, color=(70, 45, 30), thickness=1, alpha=0.18):
    contour_overlay = np.zeros_like(canvas)
    for level in range(interval, 256, interval * 2):
        mask = (Z >= level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cv2.drawContours(contour_overlay, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    for level in range(interval // 2, 256, interval * 2):
        mask = (Z >= level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cv2.drawContours(contour_overlay, contours, -1, tuple(min(255, c + 30) for c in color), max(1, thickness - 1), lineType=cv2.LINE_AA)
    cv2.addWeighted(canvas, 1.0 - alpha, contour_overlay, alpha, 0, dst=canvas)
    return canvas


def apply_hillshade(canvas, Z, azimuth=0, altitude=90, strength=0.05):
    """Apply hillshade using an overhead light source for a top-down map view."""
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    gy, gx = np.gradient(Z.astype(np.float32))
    slope = np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(gy, -gx)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    normalized = ((shaded - shaded.min()) / (np.ptp(shaded) + 1e-6)).astype(np.float32)
    shade = (normalized * 255).astype(np.uint8)
    shade = cv2.cvtColor(shade, cv2.COLOR_GRAY2BGR)
    overlay = (1.0 - normalized[..., None] * strength) * canvas.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def apply_paper_texture(canvas, intensity=0.06):
    h, w = canvas.shape[:2]
    noise = np.random.randint(-12, 12, (h, w, 1), dtype=np.int16)
    texture = np.repeat(noise, 3, axis=2)
    blended = canvas.astype(np.int16) + (texture * intensity).astype(np.int16)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_vignette(canvas, strength=0.18):
    h, w = canvas.shape[:2]
    ys = np.linspace(-1, 1, h)[:, None]
    xs = np.linspace(-1, 1, w)[None, :]
    vignette = 1.0 - (xs**2 + ys**2)
    vignette = np.clip(vignette, 0, 1) ** 1.2
    alpha = 1 - (1 - vignette) * strength
    return np.clip(canvas.astype(np.float32) * alpha[..., None], 0, 255).astype(np.uint8)


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

from PIL import ImageFont, ImageDraw, Image

def draw_label(canvas, x, y, name, size=0.5):
    print("LABEL")
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
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