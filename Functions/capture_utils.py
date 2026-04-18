import cv2
import numpy as np

PAD_LEFT = 80 # INVERTED LMAO
PAD_RIGHT = 90
PAD_TOP = 80
PAD_BOTTOM = 100


def _pink_hsv_target():
    pink_bgr = np.uint8([[[176, 134, 219]]])
    pink_hsv = cv2.cvtColor(pink_bgr, cv2.COLOR_BGR2HSV)[0][0]
    return int(pink_hsv[0]), int(pink_hsv[1]), int(pink_hsv[2])


def _pink_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = _pink_hsv_target()

    # Pink range (exclude very light/white values)
    pink_lower = np.array([max(0, h - 20), 90, 90])
    pink_upper = np.array([min(179, h + 20), 255, 220])
    pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)

    # Cotton candy light blue range
    blue_lower = np.array([80, 50, 100])
    blue_upper = np.array([120, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    mask = cv2.bitwise_or(pink_mask, blue_mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def extract_sandbox_from_frame(
    frame,
    pad_left=PAD_LEFT,
    pad_right=PAD_RIGHT,
    pad_top=PAD_TOP,
    pad_bottom=PAD_BOTTOM,
):
    mask = _pink_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise RuntimeError("No markers detected")
    # Save contours visualization before filtering
    contour_viz = frame.copy()
    cv2.drawContours(contour_viz, contours, -1, (0, 255, 0), 2)  # Draw all contours in green
    cv2.imwrite("contours_debug.png", contour_viz)
    print("Saved contours visualization to contours_debug.png")
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    selected = [c for c in contours if cv2.contourArea(c) > 800]
    if len(selected) == 0:
        selected = contours[:4]
    else:
        selected = selected[:4]

    print(f"DEBUG: Found {len(contours)} contours, selected {len(selected)} with area > 800")

    selected_mask = np.zeros_like(mask)
    cv2.drawContours(selected_mask, selected, -1, 255, thickness=cv2.FILLED)

    ys, xs = np.where(selected_mask > 0)
    if len(xs) == 0:
        raise RuntimeError("No markers detected after filtering small regions")

    raw_x_min = np.min(xs)
    raw_x_max = np.max(xs)
    raw_y_min = np.min(ys)
    raw_y_max = np.max(ys)
    
    raw_width = raw_x_max - raw_x_min
    raw_height = raw_y_max - raw_y_min
    print(f"DEBUG: Detected markers - raw bounds: ({raw_x_min}, {raw_y_min}) to ({raw_x_max}, {raw_y_max}), size: {raw_width}x{raw_height}")
    
    # Apply padding, but ensure bounds remain valid
    x_min = max(0, raw_x_min + pad_left)
    x_max = min(frame.shape[1], raw_x_max - pad_right)
    y_min = max(0, raw_y_min + pad_top)
    y_max = min(frame.shape[0], raw_y_max - pad_bottom)

    # If padding creates invalid bounds, reduce padding proportionally
    if x_min >= x_max or y_min >= y_max:
        # Use 75% of original padding
        pad_left = int(pad_left * 0.75)
        pad_right = int(pad_right * 0.75)
        pad_top = int(pad_top * 0.75)
        pad_bottom = int(pad_bottom * 0.75)
        
        x_min = max(0, raw_x_min + pad_left)
        x_max = min(frame.shape[1], raw_x_max - pad_right)
        y_min = max(0, raw_y_min + pad_top)
        y_max = min(frame.shape[0], raw_y_max - pad_bottom)
        
        # If still invalid, try 50% padding
        if x_min >= x_max or y_min >= y_max:
            pad_left = int(pad_left * 0.67)
            pad_right = int(pad_right * 0.67)
            pad_top = int(pad_top * 0.67)
            pad_bottom = int(pad_bottom * 0.67)
            
            x_min = max(0, raw_x_min + pad_left)
            x_max = min(frame.shape[1], raw_x_max - pad_right)
            y_min = max(0, raw_y_min + pad_top)
            y_max = min(frame.shape[0], raw_y_max - pad_bottom)
            
            # If still invalid, use raw bounds with no padding
            if x_min >= x_max or y_min >= y_max:
                x_min = raw_x_min
                x_max = raw_x_max
                y_min = raw_y_min
                y_max = raw_y_max

    final_width = x_max - x_min
    final_height = y_max - y_min
    print(f"DEBUG: Final crop bounds: ({x_min}, {y_min}) to ({x_max}, {y_max}), size: {final_width}x{final_height}")
    
    return frame[y_min:y_max, x_min:x_max]


def capture_sandbox_frame(video_index=1):
    cap = cv2.VideoCapture(video_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    best_frame = None
    best_score = -1.0
    for _ in range(10):
        ret, candidate = cap.read()
        if not ret or candidate is None:
            continue
        gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if score > best_score:
            best_score = score
            best_frame = candidate

    cap.release()
    if best_frame is None:
        raise RuntimeError("Failed to capture image")

    return extract_sandbox_from_frame(best_frame)


def find_red_targets(frame, min_area=50):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 80])
    upper2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))
    return centers
