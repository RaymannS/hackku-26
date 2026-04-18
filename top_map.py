import numpy as np
import cv2
import heapq
import random
import re
import speech_recognition as sr

VOICE_MODE = False  # Change to True for voice, False for typing

from Functions.render import *
from Functions.prompts import *
from Functions.location_determ import *

# -----------------------------
# HEIGHT LEVELS
# -----------------------------
sea_level = 90
mountain_level = 180
snow_level = 240


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
# REGION + POINT PARSING
# -----------------------------


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