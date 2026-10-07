"""Build an Ontario character dataset and train the SVC.

Step A (harvest):  python train_ontario.py harvest photos/ plates_out/
    Runs localize+segment on every photo in photos/ and saves each character
    crop into plates_out/unlabeled/<photo>_<i>.png, then prints the plate's
    position order so you can label by hand: type the true plate text and the
    crops are filed into plates_out/<CHAR>/ for you.

Step B (train):    python train_ontario.py train plates_out/
    Loads every labeled crop (any number per class), augments, runs
    cross-validation, trains and saves models/svc/svc.pkl.
"""
import os
import sys
import glob
import numpy as np
import joblib
from skimage.io import imread, imsave
from skimage.transform import rotate, resize
from skimage.filters import threshold_otsu
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import ontario_pipeline as op


def harvest(photo_dir, out_dir):
    for path in sorted(glob.glob(os.path.join(photo_dir, "*.jp*g"))):
        gray = imread(path, as_gray=True)
        for box in op.localize(gray):
            chars, _ = op.segment(op.crop(gray, box, pad=2))
            if len(chars) != 7:          # standard Ontario plates only
                continue
            print(os.path.basename(path), box)
            label = input("  true plate text (blank to skip): ").strip().upper()
            if len(label) != 7:
                continue
            for i, (ch, c) in enumerate(zip(chars, label)):
                folder = os.path.join(out_dir, c)
                os.makedirs(folder, exist_ok=True)
                n = len(os.listdir(folder))
                imsave(os.path.join(folder, f"{c}_{n}.png"),
                       (ch * 255).astype(np.uint8), check_contrast=False)


def augment(img):
    """Yield small variations of a 20x20 boolean/float character image."""
    img = img.astype(float)
    yield img
    for angle in (-8, -4, 4, 8):
        yield rotate(img, angle, mode="edge") > 0.5
    # slight shifts
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield np.roll(np.roll(img, dy, axis=0), dx, axis=1) > 0.5


def load_dataset(data_dir):
    X, y = [], []
    for folder in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, folder)
        if not os.path.isdir(label_dir) or len(folder) != 1:
            continue
        for f in glob.glob(os.path.join(label_dir, "*.png")):
            img = imread(f, as_gray=True)
            img = resize(img, (20, 20))
            binary = img > threshold_otsu(img)   # True = ink (same as segment())
            for aug in augment(binary):
                X.append(np.asarray(aug).reshape(-1))
                y.append(folder)
    return np.array(X), np.array(y)


def train(data_dir):
    X, y = load_dataset(data_dir)
    classes, counts = np.unique(y, return_counts=True)
    print("classes:", dict(zip(classes, counts)))
    # cv folds can't exceed the smallest class size
    folds = int(min(4, counts.min()))
    model = SVC(kernel="rbf", probability=True)  # try 'linear' to compare
    if folds >= 2:
        print("CV accuracy:", cross_val_score(model, X, y, cv=folds) * 100)
    model.fit(X, y)
    os.makedirs(os.path.dirname(op.MODEL_PATH), exist_ok=True)
    joblib.dump(model, op.MODEL_PATH)
    print("saved", op.MODEL_PATH)


if __name__ == "__main__":
    if sys.argv[1] == "harvest":
        harvest(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "train":
        train(sys.argv[2])
