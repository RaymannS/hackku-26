import cv2
import os
from datetime import datetime


def capture_image_for_midas():
    # Use fixed folder to avoid timestamp overhead
    folder_name = "current_capture"
    rel_dir = os.path.join("input", folder_name)
    full_dir = os.path.join("MiDaS-master\input", folder_name)
    
    # Create directory if it doesn't exist (much faster than recreating each time)
    os.makedirs(full_dir, exist_ok=True)

    # Initialize camera with optimized settings
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows speed
    
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

    # Use fixed filename for speed
    image_path = os.path.join(full_dir, "sand.jpg")  # JPG is faster than PNG
    
    # Save with optimized settings
    cv2.imwrite(image_path, frame)

    return rel_dir


if __name__ == "__main__":
    input_dir = capture_image_for_midas()
    # run MiDaS using the generated folder
    os.system(
        f"cd MiDaS-master && python run.py --model_type dpt_large_384 --input_path {input_dir} --output_path output"
    )