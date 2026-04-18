import cv2
import numpy as np
import random
import re
import heapq

named_locations = {}

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


def find_path(Z, start, end, avoid_mask=None, snow_lvl=240, scale=6):
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