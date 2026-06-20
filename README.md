# Real vs AI-Generated Face Classifier

A deep learning project that detects whether a human face image is **real (photographic)** or **AI-generated**, using transfer learning on **ResNet50**. The trained model can be used for real-time classification through a webcam feed.

## Overview

This project has two parts:

1. **Training** (`real-fake-image.ipynb`) — Trains a binary image classifier on a dataset of real and AI-generated human face images, and saves the trained model as `fake.h5`.
2. **Inference** (`first.py`, `second.py`) — Loads `fake.h5` and runs real-time predictions on a live webcam feed using OpenCV, labeling each frame as **Real** or **AI-Generated** along with a confidence score.

## Project Structure

```
.
├── real-fake-image.ipynb   # Model training notebook
├── first.py                 # Webcam inference script (basic version)
├── second.py                 # Webcam inference script (with confidence handling)
├── fake.h5                  # Trained model (generated after training)
└── README.md
```

## Dataset

The model is trained on the **[Human Faces Dataset](https://www.kaggle.com/datasets/cashbowman/human-faces-dataset)** from Kaggle, which contains two classes:

- `Real Images`
- `AI-Generated Images`

The notebook automatically splits the dataset into:

| Split | Ratio |
|-------|-------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

## Model Architecture

The classifier uses **transfer learning** with a pretrained **ResNet50** (ImageNet weights) as the feature extractor:

- **Base model:** ResNet50 (`include_top=False`), layers frozen
- **Head:** `GlobalAveragePooling2D` → `Dense(128, activation='relu')` → `Dense(1, activation='sigmoid')`
- **Input size:** 224 × 224 × 3
- **Loss:** Binary cross-entropy
- **Optimizer:** Adam
- **Output:** Single sigmoid value — probability of the image being "Real"

## Training Pipeline

1. Split the raw dataset into train/val/test folders.
2. Load images using `ImageDataGenerator` with pixel rescaling (`1./255`).
3. Build the ResNet50-based model with a custom classification head.
4. Train for 10 epochs, validating on the validation set.
5. Evaluate on the held-out test set.
6. Visualize predictions on a sample of validation images with true vs. predicted labels.
7. Save the trained model as `fake.h5`.

## Real-Time Webcam Inference

Two inference scripts are included, demonstrating different levels of post-processing on the model's output.

### `first.py` — Basic version
- Captures webcam frames, resizes to 224×224, normalizes, and feeds them to the model.
- Labels the frame **"Real"** if confidence > 50%, otherwise **"AI"**.
- Displays the label and confidence on-screen in green text.

### `second.py` — Improved version
- Same pipeline as `first.py`, but with refined classification logic:
  - Confidence > 70% → labeled **"Real"** (green text)
  - Confidence ≤ 70% → labeled **"AI-Generated"** (red text, with confidence flipped to reflect the AI-Generated probability)
- Color-coded output makes it easier to visually distinguish predictions.

Press **`q`** to quit the webcam window in either script.

## Requirements

```
tensorflow
opencv-python
numpy
pandas
matplotlib
```

Install with:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib
```

> **Note:** The training notebook is designed to run on **Kaggle** (it references `/kaggle/input/...` and `/kaggle/working/...` paths). To run it locally, update these paths to point to your local dataset and output directories.

## Usage

### 1. Train the model (optional — or use a pre-trained `fake.h5`)

Run all cells in `real-fake-image.ipynb` on Kaggle or in a Jupyter environment with GPU support. This produces `fake.h5`.

### 2. Run real-time classification

Place `fake.h5` in the same directory as the script, then run:

```bash
python second.py
```

(or `python first.py` for the simpler version)

A webcam window will open, classifying each frame as **Real** or **AI-Generated** in real time.

## Results

After training, the notebook reports test loss/accuracy and displays a grid of validation images with their true label, predicted label, and whether the prediction was correct — useful for a quick visual sanity check of model performance.

## Future Improvements

- Fine-tune the ResNet50 base layers instead of freezing them entirely, for potentially higher accuracy.
- Expand the dataset with more diverse and recent AI image generators (e.g., diffusion-based models) to improve generalization.
- Add data augmentation (rotation, flipping, brightness jitter) to reduce overfitting.
- Package the webcam scripts into a single configurable CLI tool instead of two near-duplicate files.
- Add unit tests and a `requirements.txt` for easier setup.

## License

Specify your preferred license here (e.g., MIT).
