"""Classical license-plate pipeline adapted for Ontario plates.

Stages (same as the original project, but each is a function now):
  1. localize()   - find plate candidates (edges + morphology + CCA)
  2. segment()    - split a plate crop into character images
  3. read_plate() - classify characters with the trained SVC + Ontario cleanup
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.io import imread
from skimage.filters import sobel_v, threshold_otsu, threshold_local
from skimage.morphology import closing, footprint_rectangle
from skimage.measure import label, regionprops
from skimage.transform import resize

HERE = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(HERE, "models/svc/svc.pkl")


# ---------------------------------------------------------------- 1. LOCALIZE
def localize(gray, show=False):
    """Return candidate plate boxes (min_row, min_col, max_row, max_col).

    Otsu on the whole photo doesn't work outdoors. Instead we use the fact
    that a plate is a dense cluster of vertical edges (character strokes):
    vertical Sobel -> threshold -> close horizontally -> connected components
    -> keep blobs with a plate-like size and aspect ratio (12in x 6in = 2.0).
    """
    h, w = gray.shape
    edges = np.abs(sobel_v(gray))
    edges = edges > threshold_otsu(edges)
    # smear edges sideways so the characters merge into one blob
    blobs = closing(edges, footprint_rectangle((3, 15)))

    candidates = []
    for r in regionprops(label(blobs)):
        y0, x0, y1, x1 = r.bbox
        bh, bw = y1 - y0, x1 - x0
        if bh == 0 or r.area < 150:
            continue
        aspect = bw / bh
        if (0.03 * h <= bh <= 0.2 * h and 0.05 * w <= bw <= 0.4 * w
                and 1.6 <= aspect <= 3.0):
            candidates.append((y0, x0, y1, x1))

    if show:
        fig, ax = plt.subplots(1)
        ax.imshow(gray, cmap="gray")
        for y0, x0, y1, x1 in candidates:
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           edgecolor="red", fill=False, lw=2))
        plt.show()
    return candidates


def crop(gray, box, pad=0):
    y0, x0, y1, x1 = box
    return gray[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]


# --------------------------------------------------------------- 2. SEGMENT
def binarize_plate(gray_plate):
    """Local threshold: handles Ontario's white-to-blue gradient background.
    Returns True for dark (character) pixels."""
    block = max(3, (gray_plate.shape[0] // 2) * 2 + 1)  # odd, ~plate height
    return gray_plate < threshold_local(gray_plate, block_size=block, offset=0.02)


def segment(gray_plate, show=False):
    """Return (characters, x_positions) sorted left to right.
    Each character is a 20x20 boolean image (True = ink)."""
    binary = binarize_plate(gray_plate)
    H, W = binary.shape
    chars = []
    for r in regionprops(label(binary)):
        y0, x0, y1, x1 = r.bbox
        h, w = y1 - y0, x1 - x0
        # characters are tall (35-80% of plate height), narrower than tall
        # ('1' is thin, so allow down to 0.15), and not tiny blobs
        if 0.35 * H <= h <= 0.80 * H and 0.15 <= w / h <= 1.0 and r.area > 30:
            roi = binary[y0:y1, x0:x1]
            chars.append((x0, (y0, x0, y1, x1), resize(roi, (20, 20))))
    # real characters share a baseline: drop blobs (border slivers, stickers)
    # whose top/bottom is far from the median top/bottom of the candidates
    if chars:
        med_top = np.median([c[1][0] for c in chars])
        med_bot = np.median([c[1][2] for c in chars])
        chars = [c for c in chars
                 if abs(c[1][0] - med_top) <= 0.08 * H
                 and abs(c[1][2] - med_bot) <= 0.08 * H]
    chars.sort(key=lambda c: c[0])

    if show:
        fig, ax = plt.subplots(1)
        ax.imshow(binary, cmap="gray_r")
        for _, (y0, x0, y1, x1), _ in chars:
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           edgecolor="red", fill=False, lw=2))
        plt.show()
    return [c[2] for c in chars], [c[0] for c in chars]


# ---------------------------------------------------------------- 3. RECOGNIZE
TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}
TO_DIGIT = {v: k for k, v in TO_LETTER.items()}


def fix_ontario(raw):
    """Standard Ontario plate = 4 letters + 3 digits. Use position to fix
    confusable characters. Returns None if it can't be a valid plate."""
    if len(raw) != 7:
        return None
    letters = ''.join(TO_LETTER.get(c, c) for c in raw[:4])
    digits = ''.join(TO_DIGIT.get(c, c) for c in raw[4:])
    out = letters + digits
    return out if re.fullmatch(r'[A-Z]{4}\d{3}', out) else None


def read_plate(characters):
    import joblib
    model = joblib.load(MODEL_PATH)
    raw = ''.join(model.predict(c.reshape(1, -1))[0] for c in characters)
    return raw, fix_ontario(raw)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "car.jpg"
    gray = imread(path, as_gray=True)
    boxes = localize(gray, show=True)
    print("plate candidates:", boxes)
    for box in boxes:
        chars, _ = segment(crop(gray, box, pad=2), show=True)
        print(len(chars), "characters found")
        if len(chars) == 7 and os.path.exists(MODEL_PATH):
            print(read_plate(chars))
