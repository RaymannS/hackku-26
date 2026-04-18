import cv2
import os
import sys
import subprocess
import numpy as np
from datetime import datetime


def detect_and_crop_box(frame):
    """Detect rectangular box in frame and crop to its region."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No box detected, using full frame")
        return frame
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Fit rectangle to contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add small margin to avoid edge artifacts
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(frame.shape[1] - x, w + 2 * margin)
    h = min(frame.shape[0] - y, h + 2 * margin)
    
    # Only crop if the box is a reasonable size (at least 20% of frame)
    frame_area = frame.shape[0] * frame.shape[1]
    if area > frame_area * 0.2:
        cropped = frame[y:y+h, x:x+w]
        print(f"Box detected and cropped: {w}x{h} from position ({x}, {y})")
        return cropped
    else:
        print(f"Box too small ({area} < {frame_area * 0.2}), using full frame")
        return frame


def capture_image_for_midas():
    # Use fixed folder to avoid timestamp overhead
    folder_name = "current_capture"
    midas_dir = os.path.dirname(os.path.abspath(__file__))
    rel_dir = os.path.join("input", folder_name)
    full_dir = os.path.join(midas_dir, "input", folder_name)
    
    # Create directory if it doesn't exist (much faster than recreating each time)
    os.makedirs(full_dir, exist_ok=True)

    # Initialize camera with optimized settings
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use DirectShow for Windows speed
    
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # Set camera properties for faster capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Quick capture
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture image")

    # Detect and crop to box region
    frame = detect_and_crop_box(frame)

    # Resize to standard dimensions for MiDaS input
    frame = cv2.resize(frame, (576, 384), interpolation=cv2.INTER_AREA)

    # Use fixed filename for speed
    image_path = os.path.join(full_dir, "sand.jpg")  # JPG is faster than PNG
    
    # Save with optimized settings
    cv2.imwrite(image_path, frame)

    return rel_dir


if __name__ == "__main__":
    input_dir = capture_image_for_midas()
    # run MiDaS using the generated folder
    midas_dir = os.path.dirname(os.path.abspath(__file__))
    run_cmd = [sys.executable, "run.py", "--model_type", "dpt_large_384",
               "--input_path", input_dir, "--output_path", "output"]
    subprocess.run(run_cmd, cwd=midas_dir, check=True)