# Ontario License Plate Recognition

A simple computer-vision pipeline that finds and reads Ontario license plates in photos. No deep learning: just image processing and a Support Vector Classifier (SVC) trained on 20×20 character crops. Built as a learning project, adapted from a pipeline originally designed for Nigerian plates.

```
photo → localize plate → segment characters → classify each → clean up with Ontario rules → "AAAA001"
```

## How it works

| Stage | What it does | Where |
|---|---|---|
| **1. Localization** | Vertical Sobel edges → threshold → horizontal closing → connected components. Keeps blobs with plate-like size and aspect ratio (Ontario plates are 12″×6″, ≈2.0). | `localize()` |
| **2. Segmentation** | Local (adaptive) threshold to handle the white-to-blue gradient background, then connected components filtered by height, aspect ratio, and shared baseline. Characters are returned left to right as 20×20 images. | `segment()` |
| **3. Recognition** | An SVC classifies each 20×20 crop (400 pixel features). | `read_plate()` |
| **4. Post-processing** | Standard Ontario plates are 4 letters + 3 digits, so position is used to fix confusable characters (`0/O`, `1/I`, `5/S`, `8/B`, …). | `fix_ontario()` |

## Project layout

```
ontario_pipeline.py      # localize → segment → recognize (run this)
train_ontario.py         # harvest labeled crops from photos / train the SVC
make_dataset.py          # build a starter dataset (real crops + synthetic characters)
training_data_ontario/
  plates_out/            # one folder per character: 0-9, A-Z
models/svc/svc.pkl       # trained model (created by training, not committed)
```

## Setup

Requires Python 3.10+.

```bash
conda create -n plates python=3.12 -y
conda activate plates
conda install -c conda-forge numpy scikit-image scikit-learn matplotlib joblib pillow -y
```

Or with pip: `pip install numpy scikit-image scikit-learn matplotlib joblib pillow`

## Usage

**1. Train the model** (a few seconds to a minute):

```bash
python train_ontario.py train training_data_ontario/plates_out/
```

This runs cross-validation and saves `models/svc/svc.pkl`.

**2. Read a plate:**

```bash
python ontario_pipeline.py path/to/photo.jpg
```

Matplotlib windows show the detected plate and the segmented characters. Close them to continue. The script prints the raw prediction and the cleaned Ontario plate string.

## Building and improving the dataset

The dataset is organized as `<CHAR>/<CHAR>_<n>.png`, one folder per character.

**Starter set.** `make_dataset.py` generates one from two sources:
- `real_*` files: crops cut from photos with known plate text
- `syn_*` files: characters rendered in bold sans-serif fonts, then rotated, sheared, downscaled, blurred, and noised to resemble low-resolution plate crops

```bash
python make_dataset.py
```

**Adding real data (the biggest accuracy win).** Put Ontario plate photos in a `photos/` folder, then:

```bash
python train_ontario.py harvest photos/ training_data_ontario/plates_out/
```

For each photo where 7 characters are segmented, you type the true plate text and each crop is filed into the right folder automatically. Then retrain.

## Limitations

- **Font mismatch.** Synthetic characters use fonts that resemble, but are not, the Ontario plate font. Expect weaker results on `1`, `Q`, `G`, `6` until you add real crops.
- **Standard plates only.** The cleanup and the 7-character check assume the `AAAA 000` format. Vanity, commercial, and older plate formats are not handled.
- **Resolution matters.** Plates narrower than ~80 px in the photo produce blocky characters that are hard to classify.
- **Fragile localization.** Edge-based detection can pick up emblems, grilles, and other textured regions (the pipeline keeps the candidate that yields 7 characters). Heavy angles, glare, and dealer plate frames can break it.
- **Inflated validation scores.** Augmented copies of the same crop can land in both train and test folds, so cross-validation accuracy overstates real performance. Test on plates the model has never seen.
- **Model portability.** `svc.pkl` depends on the scikit-learn version it was trained with. Retrain on each machine rather than sharing the file.

## Ideas for next steps

- Collect 30–100 real Ontario plate photos and harvest crops
- Split train/test by original crop before augmenting for honest accuracy
- Swap the SVC for a small CNN and compare
- Replace the CCA localizer with a trained detector (e.g. YOLO)
- Support more Ontario plate formats

## Privacy note

License plates are personal information. Only use photos you have the right to use, and avoid publishing images of identifiable plates.
