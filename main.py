import numpy as np
import cv2
import heapq
import random
import re
import speech_recognition as sr
import os
import pickle
import subprocess

VOICE_MODE = False  # Change to True for voice, False for typing

from Functions.render import *
from Functions.prompts import *
from Functions.location_determ import *
from Functions.scene_generator import *
from Functions.character_generator import *

# -----------------------------
# SCENE GENERATOR SETUP
# -----------------------------
scene_gen = SceneGenerator()

# -----------------------------
# CHARACTER GENERATOR SETUP
# -----------------------------
char_gen = CharacterGenerator()

# Get configuration from scene generator
terrain_config = scene_gen.terrain_config
sea_level = terrain_config.height_levels["sea_level"]
mountain_level = terrain_config.height_levels["mountain_level"]
snow_level = terrain_config.height_levels["snow_level"]

# Create band configs for backward compatibility
water_cfg = {
    "in_min": terrain_config.band_configs[TerrainType.WATER].in_min,
    "in_max": terrain_config.band_configs[TerrainType.WATER].in_max,
    "out_min": terrain_config.band_configs[TerrainType.WATER].out_min,
    "out_max": terrain_config.band_configs[TerrainType.WATER].out_max,
    "colormap": terrain_config.band_configs[TerrainType.WATER].colormap
}
land_cfg = {
    "in_min": terrain_config.band_configs[TerrainType.LAND].in_min,
    "in_max": terrain_config.band_configs[TerrainType.LAND].in_max,
    "out_min": terrain_config.band_configs[TerrainType.LAND].out_min,
    "out_max": terrain_config.band_configs[TerrainType.LAND].out_max,
    "colormap": terrain_config.band_configs[TerrainType.LAND].colormap
}
mount_cfg = {
    "in_min": terrain_config.band_configs[TerrainType.MOUNTAIN].in_min,
    "in_max": terrain_config.band_configs[TerrainType.MOUNTAIN].in_max,
    "out_min": terrain_config.band_configs[TerrainType.MOUNTAIN].out_min,
    "out_max": terrain_config.band_configs[TerrainType.MOUNTAIN].out_max,
    "colormap": terrain_config.band_configs[TerrainType.MOUNTAIN].colormap
}

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
# TERRAIN GENERATION
# -----------------------------
depth_path = "MiDaS-master/output/sand-dpt_large_384.pfm"
Z, water, land, mountain, water_mask, mountain_mask, snow_mask, cliffs, final = scene_gen.generate_terrain(depth_path)

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
# MAP STATE PERSISTENCE
# -----------------------------
def save_state(feature_layer, path_layer, named_locations, char_gen, filename="map_state.pkl"):
    """Save the current map state to a file."""
    state = {
        "feature_layer": feature_layer,
        "path_layer": path_layer,
        "named_locations": named_locations,
        "active_characters": char_gen.active_characters
    }
    try:
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"Map state saved to {filename}")
    except Exception as e:
        print(f"Error saving state: {e}")

def load_state(char_gen, filename="map_state.pkl"):
    """Load map state from a file."""
    if not os.path.exists(filename):
        print(f"No saved state file found: {filename}")
        return None
    
    try:
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        # Update character generator's active characters
        char_gen.active_characters = state.get("active_characters", [])
        print(f"Map state loaded from {filename}")
        return state
    except Exception as e:
        print(f"Error loading state: {e}")
        return None

# -----------------------------
# REDRAW MAP FUNCTION
# -----------------------------
def redraw_map(depth_path="MiDaS-master/output/sand-dpt_large_384.pfm"):
    """Capture new image, process it, and regenerate terrain."""
    print("Capturing new image and processing depth...")
    
    try:
        # Run grab_image.py to capture new image and generate depth
        # Since we're running from MiDaS-master directory, grab_image.py is in current dir
        result = subprocess.run(["python", "MiDaS-master/grab_image.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"Error running grab_image.py: {result.stderr}")
            return None
        
        print("Image captured and depth processed successfully.")
        
        # Regenerate terrain using scene generator
        return scene_gen.generate_terrain(depth_path)
        
    except subprocess.TimeoutExpired:
        print("Timeout: Image capture took too long")
        return None
    except Exception as e:
        print(f"Error during redraw: {e}")
        return None

# -----------------------------
# INITIAL TERRAIN GENERATION
# -----------------------------
depth_path = "MiDaS-master/output/sand-dpt_large_384.pfm"
Z, water, land, mountain, water_mask, mountain_mask, snow_mask, cliffs, final = scene_gen.generate_terrain(depth_path)

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
            audio = recognizer.listen(source, timeout=12, phrase_time_limit=20)
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
print("          spawn orcs | clear orcs")
print("          save state [filename] | load state [filename]")
print("          redraw map")
print("          list | reset | save [filename] | quit")

while True:
    # Draw active characters on the feature layer
    char_gen.draw_all_characters(feature_layer)
    
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
        char_gen.clear_characters()
        print("Map reset")
    elif "redraw map" in prompt.lower():
        print("Redrawing map...")
        new_terrain = redraw_map(depth_path)
        if new_terrain:
            Z, water, land, mountain, water_mask, mountain_mask, snow_mask, cliffs, final = new_terrain
            feature_layer = final.copy()
            path_layer = final.copy()
            named_locations.clear()
            char_gen.clear_characters()
            print("Map redrawn successfully!")
        else:
            print("Failed to redraw map")
    elif prompt.lower().startswith("save state"):
        parts = prompt.split()
        filename = "map_state.pkl"
        if len(parts) > 2:
            filename = " ".join(parts[2:])
        save_state(feature_layer, path_layer, named_locations, char_gen, filename)
    elif prompt.lower().startswith("load state"):
        parts = prompt.split()
        filename = "map_state.pkl"
        if len(parts) > 2:
            filename = " ".join(parts[2:])
        state = load_state(char_gen, filename)
        if state:
            feature_layer = state["feature_layer"]
            path_layer = state["path_layer"]
            named_locations = state["named_locations"]
            char_gen.active_characters = state.get("active_characters", [])
            print("Map state restored")
    elif prompt.lower().startswith("save "):
        filename = prompt[5:].strip() or "map.png"
        cv2.imwrite(filename, composite(final, path_layer, feature_layer))
        print(f"Saved map image to {filename}")
    elif "list" in prompt.lower():
        print(f"Known locations: {list(named_locations.keys())}")
    else:
        feature_layer, path_layer = parse_and_apply(
            prompt, feature_layer, path_layer, Z, sea_level, mountain_level, snow_level
        )

cv2.destroyAllWindows()