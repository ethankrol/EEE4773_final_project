"""Dataset loading, splitting, and image augmentation helpers."""

import albumentations as A
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_and_split_data(
    data_dir="data",
    image_size=224,
    seed=33541943,
    test_size=0.20,
    validation_size=0.20,
):
    """Load and split data, returning raw training images and normalized evaluation images."""
    images, labels = load_dataset(data_dir, image_size)
    splits = split_dataset(
        images,
        labels,
        seed=seed,
        test_size=test_size,
        validation_size=validation_size,
    )
    splits["validation"]["images"] = prepare_evaluation_images(
        splits["validation"]["images"], image_size
    )
    splits["test"]["images"] = prepare_evaluation_images(
        splits["test"]["images"], image_size
    )
    return splits


def augment_training_data(images, labels, image_size=224, copies_per_image=1):
    """Create fresh augmented and normalized training images for one epoch."""
    pipelines = {
        label: make_train_aug(label, image_size=image_size)
        for label in np.unique(labels)
    }
    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
        for _ in range(copies_per_image):
            augmented = pipelines[label](image=image)["image"]
            augmented_images.append(augmented.astype(np.float32) / 255.0)
            augmented_labels.append(label)

    return np.stack(augmented_images), np.asarray(augmented_labels)


def load_dataset(data_dir="data", image_size=224):
    """Load the saved image and label arrays from data_dir."""
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent / data_path
    images = np.load(data_path / "data_train.npy")
    labels = np.load(data_path / "labels_train.npy")
    images = images.reshape(-1, image_size, image_size, 3)
    return images, labels


def split_dataset(
    images,
    labels,
    seed=33541943,
    test_size=0.20,
    validation_size=0.20,
):
    """Make stratified train, validation, and test splits."""
    train_validation_images, test_images, train_validation_labels, test_labels = (
        train_test_split(
            images,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=seed,
        )
    )
    train_images, validation_images, train_labels, validation_labels = train_test_split(
        train_validation_images,
        train_validation_labels,
        test_size=validation_size,
        stratify=train_validation_labels,
        random_state=seed,
    )
    return {
        "train": {"images": train_images, "labels": train_labels},
        "validation": {"images": validation_images, "labels": validation_labels},
        "test": {"images": test_images, "labels": test_labels},
    }


def make_train_aug(class_label, image_size=224, weather_p=0.1):
    """Build the training transforms appropriate for one class label."""
    outdoor_classes = {3, 4}
    class_names = {
        0: "Classroom",
        1: "Library",
        2: "Cafeteria",
        3: "Parking Area",
        4: "Outdoor Campus Area",
    }
    if isinstance(class_label, str):
        name_to_label = {name.lower(): label for label, name in class_names.items()}
        class_label = name_to_label[class_label.lower()]
    class_weather_p = weather_p if class_label in outdoor_classes else 0.0

    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.85, 1.15)
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.OneOf([
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.08, 0.08),
                rotate=(-10, 10),
                shear=(-8, 8),
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.Perspective(
                scale=(0.03, 0.08),
                border_mode=cv2.BORDER_REFLECT_101,
            ),
        ], p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15
            ),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
            A.RandomGamma(gamma_limit=(70, 130)),
            A.CLAHE(),
        ], p=0.1),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3)),
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.Defocus(radius=(2, 2), alias_blur=(0.1, 0.1)),
        ], p=0.05),
        A.Sharpen(p=0.1),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.05)),
            A.ISONoise(intensity=(0.02, 0.10), color_shift=(0.002, 0.01)),
            A.MultiplicativeNoise(multiplier=(0.98, 1.02)),
        ], p=0.10),
        A.OneOf([
            A.RandomRain(
                rain_type="drizzle",
                drop_length=8,
                blur_value=3,
                brightness_coefficient=0.9,
            ),
            A.RandomFog(
                alpha_coef=0.04,
                fog_coef_range=(0.1, 0.25),
            ),
            A.RandomSnow(
                brightness_coeff=1.2,
                snow_point_range=(0.2, 0.3),
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.3),
                src_radius=15,
                num_flare_circles_range=(1, 2),
            ),
            A.RandomShadow(
                num_shadows_limit=(1, 1),
                shadow_intensity_range=(0.15, 0.25),
            ),
        ], p=class_weather_p),
        A.ImageCompression(p=0.05),
        A.CoarseDropout(p=0.05),
    ])


def make_eval_aug(image_size=224):
    """Build deterministic resizing for validation and test images."""
    return A.Compose([A.Resize(image_size, image_size)])


def prepare_evaluation_images(images, image_size=224):
    """Resize and normalize evaluation images without random augmentation."""
    resize = make_eval_aug(image_size)
    prepared = np.stack([resize(image=image)["image"] for image in images])
    return prepared.astype(np.float32) / 255.0


def make_mild_augmentation(image_shape, seed=42):
    """Build a restrained augmentation pipeline for training-fold images only."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-8, 8), border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.RandomResizedCrop(
            size=(image_shape[0], image_shape[1]), scale=(0.9, 1.0), ratio=(0.95, 1.05), p=0.2
        ),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=8, p=0.25),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.15),
        A.GaussianBlur(blur_limit=(3, 3), p=0.1),
    ], seed=seed)

def augment_training_images(images, labels, copies_per_image=1, seed=42):
    """Append the requested mild augmented copies to training images while preserving labels."""
    if copies_per_image == 0:
        return np.asarray(images), np.asarray(labels)
    augmentation = make_mild_augmentation(images[0].shape, seed=seed)
    expanded_images = [image for image in images]
    expanded_labels = [label for label in labels]
    for image, label in zip(images, labels):
        for _ in range(copies_per_image):
            expanded_images.append(augmentation(image=image)["image"])
            expanded_labels.append(label)
    return np.stack(expanded_images), np.asarray(expanded_labels)
