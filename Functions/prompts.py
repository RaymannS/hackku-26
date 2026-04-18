import cv2
import numpy as np
import random
import re
from .location_determ import *
from .render import *

orc_list = []  # persistent state, lives outside your if-block
orc_canvas = None  # ← add this

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

def get_player_location():
    """Dummy — replace with your OpenCV implementation later."""
    return (0, 0)

def parse_and_apply(prompt, feature_canvas, path_canvas, Z, sea_level, mountain_level, snow_level):
    global orc_list, orc_canvas  # persistent state, lives outside your if-block

    # initialize orc_canvas once
    if orc_canvas is None:
        orc_canvas = np.zeros_like(feature_canvas)
        
    p = prompt.lower()
    h, w = Z.shape
    water_mask = Z < sea_level

    if ("forest" in p) and not ("path" in p or "road" in p):
        region_mask = get_region_mask(p, h, w)
        mask = (Z >= sea_level) & (Z < mountain_level) & region_mask
        pts = scatter_placer(mask, n=32, min_dist=20)
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
        n = 10 if "city" in p else 5
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
        

    if "orc" in p:
        player_x, player_y = get_player_location()
        orc_list = spawn_orcs(feature_canvas, player_x, player_y, n=5, radius=60)
        
    if any(word in p for word in ["defeated", "kill", "slain"]):
        orc_canvas[:] = 0  # wipe it — orcs gone, world intact
        orc_list = []
        
    
        
    return feature_canvas, path_canvas