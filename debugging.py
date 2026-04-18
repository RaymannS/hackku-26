import cv2
import numpy as np

depth = cv2.imread("output/sand-dpt_large_384.pfm", cv2.IMREAD_UNCHANGED)

print(depth.shape)
print(depth.min(), depth.max())