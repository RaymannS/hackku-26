import cv2
import os
import sys
import subprocess
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Functions.capture_utils import (
    capture_sandbox_frame as shared_capture_sandbox_frame,
    extract_sandbox_from_frame as shared_extract_sandbox_from_frame,
)


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

    sandbox = shared_extract_sandbox_from_frame(
        frame,
        pad_left=pad_left,
        pad_right=pad_right,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
    )

    if show_crop:
        if sandbox.size == 0:
            cv2.waitKey(0)
            raise RuntimeError("Crop produced empty image")
        cv2.imshow("sandbox", sandbox)

    if sandbox.size == 0:
        raise RuntimeError("Crop produced empty image")

    if show_crop:
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

    # Capture a sharp sandbox crop using the shared helper
    frame = shared_capture_sandbox_frame()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

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