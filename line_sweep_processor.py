import numpy as np
import cv2

def extract_signature_linesweep(image):
    if image is None or image.size == 0:
        return None

    temp = image.copy()
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image.copy()

    _, thresh = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY_INV)
    rows = thresh.shape[0]
    cols = thresh.shape[1]

    flagx = 0
    indexStartX = 0
    indexEndX = 0

    for i in range(rows):
        line = thresh[i, :]
        if flagx == 0:
            if 255 in line:
                indexStartX = i
                flagx = 1
        elif flagx == 1:
            if 255 in line:
                indexEndX = i
            elif indexStartX + 5 > indexEndX:
                indexStartX = 0
                flagx = 0
            else:
                break

    flagy = 0
    indexStartY = 0
    indexEndY = 0

    for j in range(cols):
        line = thresh[indexStartX:indexEndX, j:j + 20]
        if flagy == 0:
            if 255 in line:
                indexStartY = j
                flagy = 1
        elif flagy == 1:
            if 255 in line:
                indexEndY = j
            elif indexStartY + 20 > indexEndY:
                indexStartY = 0
                flagy = 0
            else:
                break

    if indexEndX <= indexStartX or indexEndY <= indexStartY:
        return None

    cropped = temp[indexStartX:indexEndX + 1, indexStartY:indexEndY + 1]
    if cropped.size == 0:
        return None

    return cropped
