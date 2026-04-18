import cv2
import os
from datetime import datetime


def capture_image_for_midas():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"image_{timestamp}"

    full_dir = os.path.join("input", folder_name)
    os.makedirs(full_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture image")

    image_path = os.path.join(full_dir, "sand.png")

    # save using standard PNG encoding
    cv2.imwrite(image_path, frame)

    print("Saved:", image_path)

    return full_dir


if __name__ == "__main__":
    input_dir = capture_image_for_midas()

    print("Saved image to:", input_dir)

    # run MiDaS using the generated folder
    os.system(
        f"python run.py --model_type dpt_large_384 --input_path {input_dir} --output_path output"
    )