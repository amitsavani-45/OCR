# src/ocr_engine.py
import easyocr
import numpy as np

class OCREngine:
    def __init__(self, languages=['en'], gpu=False):
        # initialize once
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def run_ocr(self, image):
        """
        image: numpy array (BGR or grayscale)
        returns: list of dicts: [{'bbox': [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], 'text': '...', 'conf': 0.95}, ...]
        """
        # EasyOCR expects RGB or grayscale; convert BGR->RGB if needed
        import cv2
        img = image
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        results = self.reader.readtext(img_rgb, detail=1, paragraph=False)
        output = []
        for bbox, text, conf in results:
            output.append({
                'bbox': bbox,   # list of four points
                'text': text,
                'conf': float(conf)
            })
        return output

    def text_only(self, image):
        res = self.run_ocr(image)
        return [r['text'] for r in res]
