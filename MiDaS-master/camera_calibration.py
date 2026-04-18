import cv2
import numpy as np

img = cv2.imread("MiDaS-master/input/current_capture/sand.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lime green range
lower_green = np.array([35, 80, 80])
upper_green = np.array([85, 255, 255])

mask = cv2.inRange(hsv, lower_green, upper_green)

# clean noise slightly
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("total contours found:", len(contours))

# store coordinates of all detected green pixels
ys, xs = np.where(mask > 0)

for i, c in enumerate(contours):

    area = cv2.contourArea(c)

    x, y, w, h = cv2.boundingRect(c)

    cx = x + w//2
    cy = y + h//2

    print(f"blob {i}: area={area}")

    # draw detected green regions
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

    # draw center point
    cv2.circle(img, (cx,cy), 5, (255,0,0), -1)

# compute rectangle inside markers
if len(xs) > 0:

    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)

    # adjust padding so rectangle is INSIDE tape corners
    padding = 40

    x_min += padding
    y_min += padding
    x_max -= padding
    y_max -= padding

    # draw final sandbox rectangle
    cv2.rectangle(
        img,
        (x_min, y_min),
        (x_max, y_max),
        (0,255,0),
        3
    )

    print("sandbox bounds:")
    print("x:", x_min, "to", x_max)
    print("y:", y_min, "to", y_max)

    # crop sandbox region
    sandbox = img[y_min:y_max, x_min:x_max]

    # print size
    h, w = sandbox.shape[:2]
    print(f"sandbox size: {w} x {h}")

    cv2.imshow("cropped sandbox", sandbox)

else:
    print("no green markers detected")

# show debugging windows
cv2.imshow("detected markers", img)
cv2.imshow("green mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()