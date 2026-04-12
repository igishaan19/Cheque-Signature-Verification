try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract
import cv2
import os
import numpy as np
import platform

# Configure Tesseract path conditionally
if platform.system() == "Windows":
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

def extract_signature_region(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, ""

    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([103, 79, 60])
    upper = np.array([129, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10:
            cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)

    mask = 255 - mask
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    try:
        data = pytesseract.image_to_data(Image.open(image_path))
    except Exception as e:
        print(f"OCR Failed. Is Tesseract installed? {e}")
        return None, ""

    pleaseCd = [0, 0, 0, 0]
    aboveCd = [0, 0, 0, 0]
    ifsc_code = ""

    for d in data.splitlines():
        d = d.split("\t")
        if len(d) == 12:
            if len(d[11]) == 11:
                s = d[11][:4]
                known_banks = ["SYNB", "SBIN", "HDFC", "CNRB", "PUNB", "UTIB", "ICIC"]
                if s in known_banks:
                    ifsc_code = d[11]
                elif s == "1C1C":
                    temp = list(d[11])
                    temp[0] = 'I'
                    temp[2] = 'I'
                    ifsc_code = ''.join(temp)

            if d[11].lower() == "please":
                pleaseCd = [int(d[6]), int(d[7]), int(d[8]), int(d[9])]
            if d[11].lower() == "above":
                aboveCd = [int(d[6]), int(d[7]), int(d[8]), int(d[9])]

    lengthSign = aboveCd[0] + aboveCd[3] - pleaseCd[0]
    if lengthSign <= 0:
        return None, ifsc_code

    scaleY = 2
    scaleXL = 2.5
    scaleXR = 0.5
    
    lengthSignCd = [0, 0]
    lengthSignCd[0] = int(pleaseCd[0] - lengthSign * 2.5)
    lengthSignCd[1] = int(pleaseCd[1] - lengthSign * 2)

    y1 = max(0, lengthSignCd[1])
    y2 = min(h, lengthSignCd[1] + int(scaleY * lengthSign))
    x1 = max(0, lengthSignCd[0])
    x2 = min(w, lengthSignCd[0] + int((scaleXL + scaleXR + 1) * lengthSign))

    crop_img = img[y1:y2, x1:x2]

    if crop_img.size == 0:
        return None, ifsc_code

    return crop_img, ifsc_code
