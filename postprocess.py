import re
from typing import List, Tuple, Optional

GENERIC_RE = re.compile(r'^[A-Z0-9]{4,12}$')
INDIA_RE   = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$')

def filter_by_conf(results: List[Tuple], thr: float = 0.45) -> List[Tuple]:
    """results: iterable of (bbox, text, conf) from EasyOCR"""
    kept = []
    for item in results:
        # Support either (bbox, text, conf) or dict-like with 'text','conf'
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            _, text, conf = item[:3]
        elif isinstance(item, dict):
            text, conf = item.get('text',''), item.get('conf', 0.0)
        else:
            continue
        if conf is None: conf = 0.0
        if conf >= thr and text and isinstance(text, str):
            kept.append((text, float(conf)))
    return kept

def clean_text(s: str) -> str:
    s = s.upper()
    return re.sub(r'[^A-Z0-9]', '', s)

# Conservative confusion maps
TO_LETTER = str.maketrans({
    '0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B','9':'G'
})
TO_DIGIT = { 'O':'0','I':'1','Z':'2','S':'5','G':'6','B':'8','Q':'9' }

def try_normalize_indian(s: str) -> str:
    """
    Try to coerce into AA 99 AA 9999 pattern family by
    lightly correcting early letters & numeric blocks.
    We avoid heavy over-correction; only flip if it helps.
    """
    if not s: return s
    t = s

    # If length < 4 or > 12, don't force it
    if not (4 <= len(t) <= 12):
        return t

    # Pass 1: assume first two are letters; coerce digits -> letters there
    if len(t) >= 1 and t[0].isdigit():
        t = chr(TO_LETTER.get(t[0], ord(t[0]))) + t[1:]
    if len(t) >= 2 and t[1].isdigit():
        t = t[0] + chr(TO_LETTER.get(t[1], ord(t[1])))

    # Pass 2: for long trailing numeric block, coerce obvious letters → digits
    # Heuristic: last 2–4 should be digits in most Indian plates
    tail_start = max(0, len(t) - 4)
    prefix, tail = t[:tail_start], t[tail_start:]
    tail_fixed = ''.join(TO_DIGIT.get(ch, ch) for ch in tail)
    t2 = prefix + tail_fixed

    # Choose the variant that matches India regex if possible
    for cand in (t2, t):
        if INDIA_RE.fullmatch(cand):
            return cand
    return t2 if GENERIC_RE.fullmatch(t2) else t

def score_candidate(raw_text: str, conf: float, use_indian: bool=True):
    cleaned = clean_text(raw_text)
    normalized = cleaned
    bonus = 0.0
    matched_generic = bool(GENERIC_RE.fullmatch(cleaned))
    if matched_generic: bonus += 0.05

    if use_indian:
        if INDIA_RE.fullmatch(cleaned):
            bonus += 0.15
        else:
            fixed = try_normalize_indian(cleaned)
            if fixed != cleaned:
                normalized = fixed
                if INDIA_RE.fullmatch(normalized):
                    bonus += 0.15
                bonus += 0.02  # small bonus for a successful normalization
            else:
                normalized = cleaned
    else:
        # For non-India, you can plug your country regex here.
        pass

    matched = INDIA_RE.fullmatch(normalized) if use_indian else GENERIC_RE.fullmatch(normalized)
    total = conf + bonus
    reason = []
    reason.append(f"conf={conf:.2f}")
    if matched_generic: reason.append("generic_ok")
    if use_indian and INDIA_RE.fullmatch(normalized): reason.append("india_ok")
    if normalized != cleaned: reason.append("normalized")
    return total, cleaned, normalized, " | ".join(reason), matched

def format_indian_plate(s: str) -> str:
    """
    Insert spaces: AA 9/99 AA 9/99/9999
    We split as: [2 letters][1-2 digits][1-2 letters][1-4 digits]
    """
    m = INDIA_RE.fullmatch(s)
    if not m:
        return s
    # Build by walking the regex manually
    # Find indices by scanning
    # First 2 letters
    p = 0
    part1 = s[p:p+2]; p += 2
    # 1-2 digits
    i = p
    while i < len(s) and s[i].isdigit():
        i += 1
        if i - p == 2: break
    part2 = s[p:i]; p = i
    # 1-2 letters
    j = p
    while j < len(s) and s[j].isalpha():
        j += 1
        if j - p == 2: break
    part3 = s[p:j]; p = j
    # rest digits (1-4)
    part4 = s[p:]
    return f"{part1} {part2} {part3} {part4}".strip()

def postprocess(results: List[Tuple], use_indian: bool=True, conf_thr: float=0.45):
    """
    Input: EasyOCR results [(bbox, text, conf), ...]
    Output: (best_plate, info_dict) or (None, reason)
    """
    kept = filter_by_conf(results, conf_thr)
    if not kept:
        return None, {"reason": "no parts above confidence"}

    scored = []
    for text, conf in kept:
        total, cleaned, normalized, why, matched = score_candidate(text, conf, use_indian)
        display = format_indian_plate(normalized) if use_indian else normalized
        scored.append({
            "raw": text,
            "conf": conf,
            "cleaned": cleaned,
            "normalized": normalized,
            "pretty": display,
            "score": total,
            "why": why,
            "valid": bool(matched)
        })

    # Prefer valid matches; otherwise take highest score anyway
    valid = [x for x in scored if x["valid"]]
    pool = valid if valid else scored
    best = max(pool, key=lambda x: x["score"])

    return best["normalized"], {
        "pretty": best["pretty"],
        "score": round(best["score"], 3),
        "why": best["why"],
        "valid": best["valid"],
        "candidates": scored
    }
if __name__ == "__main__":
    # Correct order: (bbox, text, conf)
    sample_results = [
        (None, "mh 12 ab 1234", 0.92),  # good confidence
        (None, "random", 0.60),         # medium confidence
        (None, "lowconf", 0.30),        # below threshold
    ]

    from pprint import pprint
    plate, info = postprocess(sample_results, use_indian=True, conf_thr=0.45)
    pprint((plate, info))
