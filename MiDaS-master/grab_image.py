import cv2
import os
import sys
import subprocess
import numpy as np
from datetime import datetime


def extract_sandbox_from_frame(
    frame,
    pad_left=25,
    pad_right=25,
    pad_top=25,
    pad_bottom=25,
    show_mask=False,
    show_detection=False,
    show_crop=False
):
    """
    Detect pink corner markers and crop sandbox interior.

    Parameters
    ----------
    frame : np.ndarray
        image already loaded (ex: webcam frame)

    padding values move the crop inward

    Returns
    -------
    sandbox : np.ndarray
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Target color is approximately RGB(219, 134, 176) -> BGR(176, 134, 219)
    pink_bgr = np.uint8([[[176, 134, 219]]])
    pink_hsv = cv2.cvtColor(pink_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(pink_hsv[0]), int(pink_hsv[1]), int(pink_hsv[2])

    lower = np.array([max(0, h - 12), max(40, s - 60), max(40, v - 80)])
    upper = np.array([min(179, h + 12), 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise RuntimeError("No markers detected")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    selected = [c for c in contours if cv2.contourArea(c) > 800]
    if len(selected) == 0:
        selected = contours[:4]
    else:
        selected = selected[:4]

    selected_mask = np.zeros_like(mask)
    cv2.drawContours(selected_mask, selected, -1, 255, thickness=cv2.FILLED)

    ys, xs = np.where(selected_mask > 0)

    if len(xs) == 0:
        raise RuntimeError("No markers detected after filtering small regions")

    x_min = np.min(xs) + pad_left
    x_max = np.max(xs) - pad_right
    y_min = np.min(ys) + pad_top
    y_max = np.max(ys) - pad_bottom

    sandbox = frame[y_min:y_max, x_min:x_max]



    # optional debugging
    if show_detection:
        debug = frame.copy()
        cv2.rectangle(debug,(x_min,y_min),(x_max,y_max),(0,255,0),3)
        cv2.imshow("sandbox bounds", debug)

    if show_mask:
        cv2.imshow("mask", mask)

    if show_crop:
        if sandbox.size == 0:
            cv2.waitKey(0)
            raise RuntimeError("Crop produced empty image")
        else:
            cv2.imshow("sandbox", sandbox)

    if sandbox.size == 0:
        raise RuntimeError("Crop produced empty image")

    if show_mask or show_detection or show_crop:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return sandbox


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

    # Set camera properties for higher-resolution capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Quick capture
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture image")

    # Detect and crop to box region
    frame = extract_sandbox_from_frame(
        frame,
        pad_left    =170,    # Increase number to move inward
        pad_right   =175,
        pad_top     =130,
        pad_bottom  =135,
        show_mask=False,
        show_detection=False,
        show_crop=False
    )

    # Use fixed filename for speed
    image_path = os.path.join(full_dir, "sand.jpg")  # JPG is faster than PNG
    
    # Save with optimized settings
    cv2.imwrite(image_path, frame)

    return rel_dir


if __name__ == "__main__":
    input_dir = capture_image_for_midas()
    print("File captured")
    # run MiDaS using the generated folder
    midas_dir = os.path.dirname(os.path.abspath(__file__))
    run_cmd = [sys.executable, "run.py", "--model_type", "dpt_large_384",
               "--input_path", input_dir, "--output_path", "output"] #, "--height", "576", "--width", "384"]
    subprocess.run(run_cmd, cwd=midas_dir, check=True)