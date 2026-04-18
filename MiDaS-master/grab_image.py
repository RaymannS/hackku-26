import cv2
import os
from datetime import datetime


def capture_image_for_midas():
    """
    Captures an image from webcam and saves to:

        input/image_TIMESTAMP/image.png

    Assumes script is run from inside MiDaS-master directory.

    Returns:
        str: relative directory path (e.g. "input/image_20260417_1530")
    """

    # timestamp folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"image_{timestamp}"

    # create path relative to MiDaS-master
    full_dir = os.path.join("input", folder_name)

    # create directory if needed
    os.makedirs(full_dir, exist_ok=True)

    # open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture image")

    # save image
    image_path = os.path.join(full_dir, "sand.png")
    cv2.imwrite(image_path, frame)

    return full_dir


if __name__ == "__main__":
    input_dir = capture_image_for_midas()

    print("Saved image to:", input_dir)

    # run MiDaS using the generated folder
    os.system(
        f"python run.py --model_type dpt_large_384 --input_path {input_dir} --output_path output"
    )