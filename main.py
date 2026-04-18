import numpy as np
import cv2
import heapq
import random
import re
import speech_recognition as sr
import os
import pickle
import subprocess
import sys

VOICE_MODE = False  # Change to True for voice, False for typing

from Functions.render import *
from Functions.prompts import *
from Functions.location_determ import *
from Functions.scene_generator import *
from Functions.character_generator import *
from Functions.audio_manager import *

# -----------------------------
# SCENE GENERATOR SETUP
# -----------------------------
scene_gen = SceneGenerator()
STYLE_MODE = True
scene_gen.set_style_enabled(STYLE_MODE)

audio_manager.play_normal()

# -----------------------------
# CHARACTER GENERATOR SETUP
# -----------------------------
#char_gen = CharacterGenerator()

# -----------------------------
# WINDOW SETUP
# -----------------------------
cv2.namedWindow("D&D World Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("D&D World Map", 1920, 1080)  # Adjust width and height as needed

PROJECTOR_X_OFFSET = -1920  # change to your main monitor's width

cv2.namedWindow("D&D World Map", cv2.WINDOW_NORMAL)
cv2.moveWindow("D&D World Map", PROJECTOR_X_OFFSET, 0)
cv2.setWindowProperty("D&D World Map", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



# Get configuration from scene generator
terrain_config = scene_gen.terrain_config
sea_level = terrain_config.height_levels["coastal_level"]  # Use coastal as sea level
mountain_level = terrain_config.height_levels["mountains_level"]
snow_level = terrain_config.height_levels["peaks_level"]

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
        repo_root = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(repo_root, "MiDaS-master", "grab_image.py")
        result = subprocess.run([sys.executable, script_path], cwd=repo_root,
                                capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"Error running grab_image.py: {result.stderr}")
            return None
        
        print("Image captured and depth processed successfully.")
        print(f"Loading depth file: {os.path.abspath(depth_path)}")
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
recognizer.pause_threshold = 0.8
recognizer.non_speaking_duration = 0.5
mic = sr.Microphone()

if VOICE_MODE:
    print("Calibrating microphone...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Ready! Waiting for wake word 'Dungeon Master'...")

def listen_for_wake_word():
    with mic as source:
        print("\nWaiting for wake word: 'Dungeon Master'")

        recognizer.pause_threshold = 1.0
        recognizer.non_speaking_duration = 0.5

        while True:
            try:
                # listen long enough for full sentence
                audio = recognizer.listen(
                    source,
                    timeout=10,
                    phrase_time_limit= 15
                )

                text = recognizer.recognize_google(audio)
                print(f"Heard: '{text}'")

                lowered = text.lower()

                if "dungeon master" in lowered:

                    # capture everything AFTER the wake word
                    parts = re.split(
                        r"dungeon master",
                        text,
                        flags=re.IGNORECASE,
                        maxsplit=1
                    )

                    command = parts[1].strip() if len(parts) > 1 else ""

                    if command:
                        print(f"Command detected: '{command}'")
                        return command
                    else:
                        print("Wake word detected but no command followed.")
                        continue

                else:
                    print("Wake word not detected.")

            except sr.WaitTimeoutError:
                print("No speech detected")
                continue

            except sr.UnknownValueError:
                print("Could not understand audio")
                continue

            except sr.RequestError as e:
                print(f"Speech service unavailable: {e}")
                return None

# -----------------------------
# INTERACTIVE LOOP
# -----------------------------
feature_layer = final.copy()
path_layer = final.copy()
named_locations = {}

print(f"Mode: {'VOICE' if VOICE_MODE else 'TYPING'}")
if VOICE_MODE:
    print("Say 'Dungeon Master' followed by your command.")
else:
    print("Type your commands below.")
print("Commands: add a forest/desert/town/village/city in [region] called [name]")
print("          draw a path/road between [name] and [name]")
print("          spawn orcs | clear orcs")
# print("          save state [filename] | load state [filename]")
print("          style on | style off | toggle style")
print("          redraw map")
print("          list | reset | save [filename] | quit")


while True:
    # Draw active characters on the feature layer
    #char_gen.draw_all_characters(feature_layer)
    orc_layer = np.zeros_like(final)
    char_gen.draw_all_characters(orc_layer)
    
    current_map = composite(final, path_layer, feature_layer)
    
    # overlay orcs
    orc_mask = np.any(orc_layer != 0, axis=2)
    current_map[orc_mask] = orc_layer[orc_mask]
    
    # Shrink and center with black margins
    MARGIN = 70
    h, w = current_map.shape[:2]
    print(f"HW: {h}, {w}")
    new_w = w - MARGIN * 2
    shrunk = cv2.resize(current_map, (new_w, h))
    padded = np.zeros((h, w, 3), dtype=np.uint8)
    padded[:, MARGIN:MARGIN + new_w] = shrunk
    current_map = cv2.rotate(padded, cv2.ROTATE_180)
    


    cv2.imshow("D&D World Map", current_map)
    cv2.waitKey(1)

    if VOICE_MODE:
        prompt = listen_for_wake_word()
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
        audio_manager.play_normal()
        print("Map reset")
    elif "toggle style" in prompt.lower() or "style toggle" in prompt.lower():
        STYLE_MODE = scene_gen.toggle_style()
        print(f"Terrain style {'enabled' if STYLE_MODE else 'disabled'}")
        Z, water, land, mountain, water_mask, mountain_mask, snow_mask, cliffs, final = scene_gen.generate_terrain(depth_path)
        feature_layer = final.copy()
        path_layer = final.copy()
        named_locations.clear()
        char_gen.clear_characters()
        audio_manager.play_normal()
        current_map = composite(final, path_layer, feature_layer)
        cv2.imshow("D&D World Map", current_map)
        cv2.waitKey(1)
    elif prompt.lower().startswith("style "):
        if "on" in prompt.lower():
            STYLE_MODE = True
        elif "off" in prompt.lower():
            STYLE_MODE = False
        else:
            print("Use 'style on' or 'style off' to control the terrain look.")
            continue
        scene_gen.set_style_enabled(STYLE_MODE)
        print(f"Terrain style {'enabled' if STYLE_MODE else 'disabled'}")
        Z, water, land, mountain, water_mask, mountain_mask, snow_mask, cliffs, final = scene_gen.generate_terrain(depth_path)
        feature_layer = final.copy()
        path_layer = final.copy()
        named_locations.clear()
        char_gen.clear_characters()
        audio_manager.play_normal()
        current_map = composite(final, path_layer, feature_layer)
        cv2.imshow("D&D World Map", current_map)
        cv2.waitKey(1)
    elif "redraw map" in prompt.lower():
        print("Redrawing map...")
        # Display black screen while processing
        black_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imshow("D&D World Map", black_screen)
        cv2.waitKey(1)
        
        new_terrain = redraw_map(depth_path)
        if new_terrain:
            Z, water, land, mountain, water_mask, mountain_mask, snow_mask, cliffs, final = new_terrain
            feature_layer = final.copy()
            path_layer = final.copy()
            named_locations.clear()
            char_gen.clear_characters()
            audio_manager.play_normal()
            current_map = composite(final, path_layer, feature_layer)
            cv2.imshow("D&D World Map", current_map)
            cv2.waitKey(1)
            print("Map redrawn successfully!")
        else:
            print("Failed to redraw map")
    elif prompt.lower().startswith("save state"):
        if VOICE_MODE:
            print("Save state is disabled in voice mode.")
            continue
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
        if VOICE_MODE:
            print("Saving images is disabled in voice mode.")
            continue
        filename = prompt[5:].strip() or "map.png"
        cv2.imwrite(filename, composite(final, path_layer, feature_layer))
        print(f"Saved map image to {filename}")
    elif "list" in prompt.lower():
        print(f"Known locations: {list(named_locations.keys())}")
    else:
        feature_layer, path_layer = parse_and_apply(
            prompt, feature_layer, path_layer, orc_layer, Z, sea_level, mountain_level, snow_level,
            named_locations
        )

cv2.destroyAllWindows()