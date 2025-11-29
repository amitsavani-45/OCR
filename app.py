# app.py
import streamlit as st
import numpy as np
import cv2
from src.preprocessing import preprocess_for_ocr, load_image
from src.ocr_engine import OCREngine
from src.text_extraction import extract_target_lines, pick_best_match
from src.utils import draw_bbox_on_image
from PIL import Image
import io

st.set_page_config(page_title="Shipping Label OCR (_1_ extractor)", layout="wide")

st.title("Shipping Label OCR — extract `_1_` token (EasyOCR)")
st.markdown("Upload a shipping label / waybill image. The app will run EasyOCR, search for tokens containing `_1_` (e.g. `..._1_...`) and highlight them.")

uploaded = st.file_uploader("Upload image (jpg/png) or drag-and-drop", type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'])
use_gpu = st.checkbox("Use GPU for EasyOCR (if available)", value=False)

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Couldn't read image. Try a different format.")
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)
        # Preprocess
        _, gray = preprocess_for_ocr(image)
        st.write("Running OCR...")
        ocr = OCREngine(languages=['en'], gpu=use_gpu)
        results = ocr.run_ocr(image)

        st.write(f"Detected {len(results)} text blocks.")
        # Show results table
        rows = []
        for r in results:
            rows.append({'text': r['text'], 'conf': round(r['conf'], 3)})
        st.table(rows[:50])

        matches = extract_target_lines(results)
        st.write("Matches found:", len(matches))
        if matches:
            best = pick_best_match(matches)
            st.success(f"Best match: `{best['text']}` (conf: {best['conf']:.3f})")
            # highlight on image
            draw_img = draw_bbox_on_image(image, best['bbox'], text=best['text'])
            st.image(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB), caption="Highlighted best match", use_column_width=True)
            st.write("All matches (sorted by confidence):")
            for m in matches:
                st.write(f"- `{m['text']}`  — conf {m['conf']:.3f}")
        else:
            st.warning("No `_1_`-pattern match found. Try adjusting image quality or use cropping.")
