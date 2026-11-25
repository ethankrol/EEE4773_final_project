# We can import our augmentations using this.
# import train_aug, eval_aug, IMG_SIZE, etc

import glob, os
import numpy as np
import cv2
import albumentations as A
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = "data"
SEED = 33541943

CLASS_NAMES = {
    0: "Classroom",
    1: "Library",
    2: "Cafeteria",
    3: "Parking Area",
    4: "Outdoor Campus Area"
}

X = np.load(f"{DATA_DIR}/data_train.npy")
y = np.load(f"{DATA_DIR}/labels_train.npy")

X = X.reshape(-1, 224, 224, 3)
print(X.shape, X.dtype, y.shape)

X_trval, X_test, y_trval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.20, stratify=y_trval, random_state=SEED)

for name, labels in [("train", y_train), ("val", y_val), ("test", y_test)]:
    counts = [(CLASS_NAMES[c],int((labels == c).sum())) for c in CLASS_NAMES.keys()]
    print(f"{name:5s} {len(labels):4d}  per class: {counts}")

# Right now, we randomly choose different transformations to apply together. We can change this to be one transformation of each type for each image and see if this improves performance

train_aug = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0), ratio=(0.85, 1.15)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    # Types of distortion
    A.OneOf([
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.08, 0.08), rotate=(-20, 20), shear=(-8, 8)),
        A.Perspective(scale=(0.03, 0.08)),
        A.ElasticTransform(alpha=30, sigma=6),
        A.GridDistortion(),
        A.OpticalDistortion(),
    ], p=0.5),
    # Color differences
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
        A.ToGray(),
    ], p=0.5),
    # Brightness differneces
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
        A.RandomGamma(gamma_limit=(70, 130)),
        A.CLAHE(),
    ], p=0.4),
    # Blurring
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.Defocus(),
    ], p=0.25),
    # Sharpening
    A.OneOf([A.Sharpen(), A.Emboss()], p=0.15),
    # Noise
    A.OneOf([A.GaussNoise(), A.ISONoise(), A.MultiplicativeNoise()], p=0.25),
    # Weather
    A.OneOf([
        A.RandomRain(), A.RandomFog(), A.RandomSnow(), A.RandomSunFlare(), A.RandomShadow(),
    ], p=0.15),
    A.ImageCompression(p=0.2),
    A.CoarseDropout(p=0.2),
])

eval_aug = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])