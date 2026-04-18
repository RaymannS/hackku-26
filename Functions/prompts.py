import cv2
import numpy as np
import random
import re
from .capture_utils import *
from .audio_manager import *
from .location_determ import *
from .render import *
from .scene_generator import *
from .character_generator import *

# Global instances
scene_gen = SceneGenerator()
char_gen = CharacterGenerator()

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



def get_player_location(Z):
    h, w = Z.shape
    try:
        sandbox = capture_sandbox_frame()
        centers = find_red_targets(sandbox)
        if centers:
            avg_x = sum(x for x, _ in centers) / len(centers)
            avg_y = sum(y for _, y in centers) / len(centers)
            sh, sw = sandbox.shape[:2]
            player_x = int(avg_x / sw * w)
            player_y = int(avg_y / sh * h)
            # Rotate the detected player location 180° CCW to match the saved image orientation.
            player_x = w - 1 - player_x
            player_y = h - 1 - player_y
            print(f"Detected {len(centers)} red targets; player location: ({player_x}, {player_y})")
            return player_x, player_y
        print("No red targets found; falling back to center")
    except Exception as exc:
        print(f"Player location detection failed: {exc}")

    return w // 2, h // 2


def parse_and_apply(prompt, feature_canvas, path_canvas, orc_canvas, Z, sea_level, mountain_level, snow_level, named_locations):
        
    p = prompt.lower()
    h, w = Z.shape
    water_mask = Z < sea_level

    # Item generation mapping
    item_keywords = {
        ItemType.FOREST: ["forest"],
        ItemType.DESERT: ["desert"],
        ItemType.TOWN: ["town"],
        ItemType.VILLAGE: ["village"], 
        ItemType.CITY: ["city"],
        ItemType.CASTLE: ["castle"]
    }

    # Check for item generation commands
    for item_type, keywords in item_keywords.items():
        if any(keyword in p for keyword in keywords) and not ("path" in p or "road" in p):
            region_mask = get_region_mask(p, h, w)
            positions = scene_gen.generate_items(item_type, feature_canvas, Z, region_mask)
            
            name = parse_name(prompt)
            if name and positions:
                cx = int(np.mean([x for x, y in positions]))
                cy = int(np.mean([y for x, y in positions]))
                draw_label(feature_canvas, cx, cy, name)
                named_locations[name.lower()] = (cx, cy)
                print(f"Labelled {item_type.value} as '{name}' at '{(cx, cy)}'")
            print(f"Generated {len(positions)} {item_type.value} features")
            #return feature_canvas, path_canvas

    # Special handling for deserts (they modify terrain)
    if "desert" in p and not ("path" in p or "road" in p):
        region_mask = get_region_mask(p, h, w)
        positions = scene_gen.generate_desert_area(feature_canvas, Z, region_mask)

        name = parse_name(prompt)
        if name and positions:
            cx = int(np.mean([x for x, y in positions]))
            cy = int(np.mean([y for x, y in positions]))
            draw_label(feature_canvas, cx, cy, name)
            named_locations[name.lower()] = (cx, cy)
            print(f"Labelled desert as '{name}'")
        print(f"Desert applied with {len(positions)} cacti")
        return feature_canvas, path_canvas

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
        

    kill_words = ["defeated", "kill", "slain", "clear"]

    if "orc" in p and not any(word in p for word in kill_words):
        player_x, player_y = get_player_location(Z)
        char_gen.spawn_characters(orc_canvas, player_x, player_y, CharacterType.ORC, n=3, radius=90)
        audio_manager.play_orc()

    if any(word in p for word in kill_words):
        print(f"Before clear: {len(char_gen.active_characters)} characters")
        char_gen.clear_characters(CharacterType.ORC)
        char_gen.clear_characters(CharacterType.BOSS)
        audio_manager.play_normal()
        print(f"After clear: {len(char_gen.active_characters)} characters")
        
    if "boss" in p and not any(word in p for word in kill_words):
        player_x, player_y = get_player_location(Z)
        char_gen.spawn_characters(orc_canvas, player_x, player_y, CharacterType.BOSS, n=1, radius=50)
        audio_manager.play_boss()
        
    
        
    return feature_canvas, path_canvas