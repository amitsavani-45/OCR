# src/utils.py
import cv2
import numpy as np

def draw_bbox_on_image(img, bbox, text=None, thickness=2):
    """
    bbox: list of four points [[x1,y1],...]
    """
    out = img.copy()
    pts = np.array(bbox, dtype=np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=(0,255,0), thickness=thickness)
    if text:
        # put text near top-left point
        tl = pts[0]
        cv2.putText(out, text, (int(tl[0]), int(max(tl[1]-10, 0))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return out
