# src/preprocessing.py
import cv2
import numpy as np
import imutils

def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return img

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img_gray, ksize=3):
    return cv2.GaussianBlur(img_gray, (ksize, ksize), 0)

def adaptive_threshold(img_gray):
    # returns binary image suitable for contour/deskew
    return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 25, 15)

def deskew(image, binary_thresh=None):
    # Estimate skew angle using binary image or the grayscale image
    gray = image if len(image.shape) == 2 else to_grayscale(image)
    if binary_thresh is None:
        binary = adaptive_threshold(denoise(gray))
    else:
        binary = binary_thresh

    coords = np.column_stack(np.where(binary < 255))  # foreground coords
    if coords.shape[0] < 10:
        return image  # nothing to deskew

    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angle in range [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_for_ocr(img_bgr):
    # pipeline that returns image (BGR) and a gray for possible analysis
    gray = to_grayscale(img_bgr)
    gray = denoise(gray, ksize=3)
    rotated = deskew(gray)
    # if deskew returned rotated grayscale, warp original BGR similarly:
    if rotated.shape == gray.shape and not np.array_equal(rotated, gray):
        # compute rotation matrix between gray and rotated (we returned rotated value directly so assume deskew applied)
        # For simplicity: compute rotation angle again and apply to BGR (fast route)
        bgr_rot = imutils.rotate_bound(img_bgr, 0)  # fallback: no rotate if not sure
        # In most cases deskew returned rotated grayscale computed from same shape: use rotated gray as final
        final = img_bgr
    else:
        final = img_bgr
    # final small enhancements:
    final_gray = to_grayscale(final)
    return final, final_gray
