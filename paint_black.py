import cv2
import numpy as np

def paint_black(frame):
    img = frame
    stencil = np.zeros(img.shape).astype(img.dtype)
    contours = [np.array([[25, 268], [593, 1150], [1250, 146], [837, 52]])]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result = cv2.bitwise_and(img, stencil)


    return result
