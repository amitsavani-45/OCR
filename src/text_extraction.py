# src/text_extraction.py
import re
from statistics import mean

# Primary pattern to match: any token containing '_1_' inside it (e.g., 163233702292313922_1_lWV)
PRIMARY_REGEX = re.compile(r'\b\S*_1_\S*\b')  # \S* to capture non-space tokens having _1_

def extract_target_lines(ocr_results, prefer_high_conf=True):
    """
    Given OCR results list of dicts {'text', 'conf', 'bbox'}, return matches list.
    Returns list of dicts: [{'text':..., 'conf':..., 'bbox':...}, ...]
    """
    matches = []
    for r in ocr_results:
        text = r['text'].strip()
        # look for primary regex anywhere in the text
        m = PRIMARY_REGEX.search(text)
        if m:
            matches.append({
                'text': m.group(0),
                'full_text': text,
                'conf': r['conf'],
                'bbox': r.get('bbox')
            })
    # If no exact matches, try fuzzy search: token containing '_1' or '1_' or '_ 1 _' etc.
    if not matches:
        alt_regex = re.compile(r'\b\S*[_\s]1[_\s]\S*\b|\b\S*_1\S*\b|\b\S*1_\S*\b')
        for r in ocr_results:
            text = r['text'].strip().replace(' ', '')
            m = alt_regex.search(text)
            if m:
                matches.append({
                    'text': m.group(0),
                    'full_text': r['text'],
                    'conf': r['conf'],
                    'bbox': r.get('bbox')
                })
    # Sort by confidence descending
    matches.sort(key=lambda x: x.get('conf', 0), reverse=True)
    return matches

def pick_best_match(matches):
    if not matches:
        return None
    # Already sorted by conf, return top
    return matches[0]
