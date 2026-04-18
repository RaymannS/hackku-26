import cv2
import numpy as np

depth = cv2.imread("MiDaS-master/output/sand3-dpt_large_384.pfm", cv2.IMREAD_UNCHANGED)

print(depth.shape)
print(depth.min(), depth.max())

depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_uint8 = depth_normalized.astype("uint8")

cv2.imwrite("output/depth_visual.png", depth_uint8)