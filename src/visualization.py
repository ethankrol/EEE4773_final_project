"""Provide shared visualization helpers for model training and evaluation."""

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, exposure
from skimage.feature import hog, local_binary_pattern

from src.features.handcrafted_features import _to_rgb_float, extract_image_features
from src.preprocessing import make_mild_augmentation, make_train_aug


def preview_full_augmentations(image, label, n=16):
    """Display the main class-aware training augmentation beside its original image."""
    transform = make_train_aug(label)
    title = f"Full training augmentations: {label}"
    return _preview_augmentation_grid(image, transform, title, n)


def preview_mild_augmentations(image, n=16, seed=42):
    """Display mild classical-CV training augmentations beside their original image."""
    transform = make_mild_augmentation(image.shape, seed=seed)
    return _preview_augmentation_grid(image, transform, "Mild classical-CV augmentations", n)


def _preview_augmentation_grid(image, transform, title, n):
    """Draw an original image and n minus one transformed variants in a subplot grid."""
    columns = min(4, n)
    rows = (n + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    flat_axes = list(axes.flat)
    _draw_augmentation_panels(flat_axes, image, transform, n)
    for axis in flat_axes:
        axis.axis("off")
    fig.tight_layout()
    fig.suptitle(title)
    plt.show()
    return fig, axes


def _draw_augmentation_panels(axes, image, transform, n):
    """Draw the original image and requested random augmentation variants."""
    axes[0].imshow(image)
    axes[0].set_title("original")
    for axis in axes[1:n]:
        axis.imshow(transform(image=image)["image"])


def plot_training_history(history, save_path=None):
    """Plot training and validation mean squared error by epoch."""
    train_mse = history.history["loss"]
    validation_mse = history.history["val_loss"]
    best_epoch = int(np.argmin(validation_mse)) + 1
    fig, axis = plt.subplots(figsize=(8, 5), dpi=120)
    _draw_mse_history(axis, train_mse, validation_mse, best_epoch)
    _style_history_axis(axis)
    _finish_figure(fig, save_path)
    return fig, axis


def _draw_mse_history(axis, train_mse, validation_mse, best_epoch):
    """Draw training and validation MSE series and mark the best validation epoch."""
    epochs = range(1, len(train_mse) + 1)
    _draw_mse_series(axis, epochs, train_mse, validation_mse)
    _mark_best_epoch(axis, best_epoch, validation_mse)
    axis.set_title("Autoencoder Reconstruction Error", fontsize=14, weight="bold", pad=12)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean Squared Error (MSE)")
    axis.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False)


def _draw_mse_series(axis, epochs, train_mse, validation_mse):
    """Draw the training and validation MSE curves with their shared styles."""
    styles = ((train_mse, "#2563eb", "Training MSE"),
              (validation_mse, "#f97316", "Validation MSE"))
    for values, color_code, label in styles:
        axis.plot(epochs, values, color=color_code, marker="o", markersize=4,
                  linewidth=2, label=label)


def _mark_best_epoch(axis, best_epoch, validation_mse):
    """Mark and label the epoch with the lowest validation MSE."""
    best_loss = validation_mse[best_epoch - 1]
    axis.scatter(best_epoch, best_loss, color="#f97316", edgecolor="white",
                 linewidth=1.2, s=85, zorder=3)
    axis.annotate(f"Best validation: epoch {best_epoch}", (best_epoch, best_loss),
                  xytext=(8, 12), textcoords="offset points", fontsize=9)


def _style_history_axis(axis):
    """Hide the top and right axis spines for a cleaner training plot."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _finish_figure(fig, save_path=None):
    """Save a figure when requested and display it in the active notebook."""
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_classifier_history(history, save_path=None):
    """Plot supervised classifier loss and accuracy for training and validation."""
    epochs = np.arange(1, len(history.history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
    metrics = (("loss", "Cross-entropy loss"), ("accuracy", "Accuracy"))
    for axis, (metric, title) in zip(axes, metrics):
        _draw_classifier_metric(axis, history.history, epochs, metric, title)
    fig.suptitle("Supervised CNN training history", weight="bold")
    _finish_figure(fig, save_path)
    return fig, axes


def _draw_classifier_metric(axis, history, epochs, metric, title):
    """Draw one training and validation metric against epoch number."""
    axis.plot(epochs, history[metric], label="Training", linewidth=2)
    axis.plot(epochs, history[f"val_{metric}"], label="Validation", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    _style_history_axis(axis)


def plot_prediction_gallery(
    images,
    true_labels,
    predicted_labels,
    probabilities,
    class_names,
    split_name,
    output_dir,
    per_page=20,
    columns=5,
):
    """Display and save paginated image predictions with true labels and optional confidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = ceil(len(images) / per_page)
    rows = ceil(per_page / columns)
    paths = []
    for page in range(pages):
        paths.append(_save_prediction_page(images, true_labels, predicted_labels,
                                           probabilities, class_names, split_name,
                                           output_dir, per_page, columns, rows, page))
    return paths


def _save_prediction_page(images, true_labels, predicted_labels, probabilities,
                          class_names, split_name, output_dir, per_page, columns, rows, page):
    """Draw and save one page of the image prediction gallery."""
    start = page * per_page
    stop = min(start + per_page, len(images))
    fig, axes = plt.subplots(rows, columns, figsize=(2.5 * columns, 2.5 * rows), squeeze=False)
    _draw_prediction_page(axes.ravel(), images, true_labels, predicted_labels,
                          probabilities, class_names, start, stop)
    title = f"{split_name.title()} predictions ({start + 1}–{stop} of {len(images)})"
    fig.suptitle(title, weight="bold")
    fig.tight_layout()
    return _save_prediction_page_figure(fig, output_dir, split_name, page)


def _draw_prediction_page(axes, images, true_labels, predicted_labels,
                          probabilities, class_names, start, stop):
    """Draw labels and correctness colors for each image in one gallery page."""
    for offset, axis in enumerate(axes):
        index = start + offset
        if index >= stop:
            axis.axis("off")
            continue
        _draw_prediction_tile(axis, images[index], true_labels[index],
                              predicted_labels[index], probabilities, index, class_names)


def _draw_prediction_tile(axis, image, true_label, predicted_label,
                          probabilities, index, class_names):
    """Draw one image tile with its true label, predicted label, and correctness color."""
    true_name = class_names[int(true_label)]
    predicted_name = class_names[int(predicted_label)]
    confidence = _prediction_confidence(probabilities, index)
    color_code = "#15803d" if true_label == predicted_label else "#b91c1c"
    axis.imshow(image)
    axis.set_title(f"True: {true_name}\nPred: {predicted_name}{confidence}", fontsize=8, color=color_code)
    axis.axis("off")


def _save_prediction_page_figure(fig, output_dir, split_name, page):
    """Write, display, close, and return the path for one prediction gallery page."""
    save_path = output_dir / f"{split_name}_predictions_{page + 1:02d}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return str(save_path.resolve())


def _prediction_confidence(probabilities, image_index):
    """Format the predicted-class probability when probability scores are available."""
    if probabilities is None:
        return ""
    return f" ({float(probabilities[image_index].max()):.0%})"


def plot_reconstructions(autoencoder, images, count=8, save_path=None):
    """Display original images and their autoencoder reconstructions in paired rows."""
    images = images[:count]
    reconstructions = autoencoder.predict(images, verbose=0)
    count = len(images)
    fig, axes = plt.subplots(2, count, figsize=(2.5 * count, 5), squeeze=False)
    _draw_reconstruction_pairs(axes, images, reconstructions, count)
    _finish_figure(fig, save_path)
    return fig, axes


def _draw_reconstruction_pairs(axes, originals, reconstructions, count):
    """Draw originals and reconstructions into corresponding columns."""
    for index in range(count):
        axes[0, index].imshow(originals[index])
        axes[0, index].set_title("original")
        axes[1, index].imshow(np.clip(reconstructions[index], 0, 1))
        axes[1, index].set_title("reconstruction")
        axes[0, index].axis("off")
        axes[1, index].axis("off")


def plot_branch_results(branch, class_names, save_path=None):
    """Plot shared-validation and test metrics together for one branch."""
    evaluations = [("Shared validation set", branch["validation"]),
                   ("Held-out test set", branch["test"])]
    return plot_evaluation_grid(f"{branch['name']} branch", evaluations,
                                class_names, save_path=save_path)


def plot_classification_results(title, validation_results, test_results,
                                class_names, save_path=None):
    """Plot confusion and classification metrics for validation and test sets."""
    evaluations = [("Shared validation set", validation_results),
                   ("Held-out test set", test_results)]
    return plot_evaluation_grid(title, evaluations, class_names, save_path)


def plot_evaluation_grid(title, evaluations, class_names, save_path=None):
    """Plot confusion matrices and classwise metrics for a list of evaluations."""
    fig, axes = plt.subplots(len(evaluations), 2, figsize=(15, 5.2 * len(evaluations)),
                             dpi=120, squeeze=False)
    _draw_evaluations(axes, evaluations, class_names)
    fig.suptitle(title, fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_and_show(fig, save_path)
    return fig, axes


def _draw_evaluations(axes, evaluations, class_names):
    """Draw each evaluation row with its confusion matrix and metric heatmap."""
    for row, (name, results) in enumerate(evaluations):
        _draw_evaluation_row(axes[row], name, results, class_names)


def _draw_evaluation_row(axes, name, results, class_names):
    """Draw one row of confusion-matrix and classwise metric panels."""
    accuracy = results["accuracy"]
    baseline = results["baseline_accuracy"]
    title = f"{name}: confusion matrix\nAccuracy {accuracy:.3f} | majority baseline {baseline:.3f}"
    draw_confusion_matrix(axes[0], results["confusion_matrix"], class_names, title)
    draw_class_metrics(axes[1], results["classification_report"], class_names,
                       f"{name}: precision, recall, and F1")


def _save_and_show(fig, save_path):
    """Save a figure when requested and display it in the active notebook."""
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def draw_confusion_matrix(ax, matrix, class_names, title):
    """Draw an annotated confusion matrix with true and predicted class labels."""
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_title(title)
    _annotate_confusion_matrix(ax, matrix)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _annotate_confusion_matrix(axis, matrix):
    """Annotate matrix cells with counts and contrast-aware text colors."""
    threshold = np.max(matrix) / 2 if np.size(matrix) else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:g}", ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "#1f2937")


def draw_class_metrics(ax, report, class_names, title):
    """Draw a heatmap table for class precision, recall, and F1 scores."""
    row_names, values = _class_metric_table(report, class_names)
    _draw_metric_heatmap(ax, values, row_names, title)


def _class_metric_table(report, class_names):
    """Build labeled precision, recall, and F1 rows for classes and aggregate averages."""
    row_names = list(class_names)
    row_names.extend(name for name in ("macro avg", "weighted avg") if name in report)
    metrics = ("precision", "recall", "f1-score")
    values = np.array([[report[name][metric] for metric in metrics] for name in row_names])
    row_names = [f"{name} (n={int(report[name]['support'])})" if name in class_names
                 else name for name in row_names]
    return row_names, values


def _draw_metric_heatmap(axis, values, row_names, title):
    """Render metric values as a labeled heatmap with a zero-to-one scale."""
    image = axis.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(3), labels=("Precision", "Recall", "F1"))
    axis.set_yticks(np.arange(len(row_names)), labels=row_names)
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center")
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def plot_partition_search_summary(partition_rows, baseline_rows, save_path=None):
    """Plot cross-validation accuracy and macro-F1 for candidate class partitions."""
    labels, cv_scores, validation_scores, test_scores = _partition_metric_series(
        partition_rows, baseline_rows
    )
    fig, axis = plt.subplots(figsize=(19, 7), dpi=120)
    _draw_partition_metrics(axis, labels, cv_scores, validation_scores, test_scores)
    _finish_figure(fig, save_path)
    return fig, axis


def _partition_metric_series(partition_rows, baseline_rows):
    """Extract readable partition labels and cross-validation summary metrics."""
    labels = [row["label"] for row in partition_rows]
    labels.extend(row["label"] for row in baseline_rows)
    cv_scores = [row["cv_macro_f1"] for row in partition_rows]
    cv_scores.extend([np.nan] * len(baseline_rows))
    validation_scores = [row["validation_macro_f1"] for row in partition_rows]
    validation_scores.extend(row["validation_macro_f1"] for row in baseline_rows)
    test_scores = [row["test_macro_f1"] for row in partition_rows]
    test_scores.extend(row["test_macro_f1"] for row in baseline_rows)
    return labels, cv_scores, validation_scores, test_scores


def _draw_partition_metrics(axis, labels, cv_scores, validation_scores, test_scores):
    """Draw ranked accuracy and macro-F1 bars for candidate class partitions."""
    positions = np.arange(len(labels))
    axis.plot(positions, cv_scores, marker="o", linewidth=2, label="3-fold CV")
    axis.plot(positions, validation_scores, marker="o", linewidth=2, label="Validation")
    axis.plot(positions, test_scores, marker="o", linewidth=2, label="Test")
    axis.set_xticks(positions, labels=labels, rotation=65, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Macro F1")
    axis.set_title("All two-group hierarchy partitions vs flat ensembles", weight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def preview_handcrafted_features(image, label=None, hog_orientations=9,
                                 hog_pixels_per_cell=(8, 8), lbp_points=8,
                                 lbp_radius=1, lbp_grid=(1, 1), color_bins=16):
    """Visualize handcrafted LBP, HOG, HSV, and channel features for one image."""
    arrays = _handcrafted_preview_arrays(image, hog_orientations,
                                         hog_pixels_per_cell, lbp_points, lbp_radius)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), dpi=110)
    _draw_handcrafted_preview(axes, image, arrays, lbp_points, lbp_radius)
    title = f"Handcrafted feature preview: {label}" if label else "Handcrafted feature preview"
    fig.suptitle(title, weight="bold")
    fig.tight_layout()
    summary = _handcrafted_feature_summary(image, arrays, hog_orientations,
                                           hog_pixels_per_cell, lbp_points,
                                           lbp_radius, lbp_grid, color_bins)
    print(summary)
    plt.show()
    return fig, axes, summary


def _handcrafted_preview_arrays(image, hog_orientations, hog_pixels_per_cell,
                                lbp_points, lbp_radius):
    """Calculate display arrays for RGB, HSV, LBP, HOG, and histogram panels."""
    rgb = _to_rgb_float(image)
    gray = color.rgb2gray(rgb)
    hsv = color.rgb2hsv(rgb)
    lbp = local_binary_pattern(gray, lbp_points, lbp_radius, method="uniform")
    histogram = _normalized_lbp_histogram(lbp, lbp_points)
    hog_image, hog_count = _hog_preview(gray, hog_orientations, hog_pixels_per_cell)
    return {"rgb": rgb, "hsv": hsv, "lbp": lbp, "histogram": histogram,
            "hog": hog_image, "hog_count": hog_count}


def _normalized_lbp_histogram(lbp, lbp_points):
    """Return the normalized global histogram for uniform LBP codes."""
    bins = lbp_points + 2
    histogram, _ = np.histogram(lbp.ravel(), bins=np.arange(bins + 1), density=False)
    return histogram / max(histogram.sum(), 1)


def _hog_preview(gray, orientations, pixels_per_cell):
    """Compute a HOG visualization and the flattened HOG feature count."""
    features, hog_image = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        visualize=True,
        block_norm="L2-Hys",
    )
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, np.max(hog_image) or 1))
    return hog_image, features.size


def _draw_handcrafted_preview(axes, image, arrays, lbp_points, lbp_radius):
    """Draw image, texture, color-space, and channel-distribution panels."""
    _draw_handcrafted_top_row(axes[0], image, arrays, lbp_points, lbp_radius)
    _draw_handcrafted_bottom_row(axes[1], arrays)
    for axis in axes.flat:
        axis.axis("off")
    axes[0, 2].axis("on")
    axes[1, 2].axis("on")


def _draw_handcrafted_top_row(axes, image, arrays, lbp_points, lbp_radius):
    """Draw input image, uniform LBP map, and normalized LBP histogram."""
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[1].imshow(arrays["lbp"], cmap="viridis")
    axes[1].set_title(f"Uniform LBP map (P={lbp_points}, R={lbp_radius})")
    axes[2].bar(np.arange(len(arrays["histogram"])), arrays["histogram"], color="#4f83e3")
    axes[2].set_title("Normalized global LBP histogram")
    axes[2].set_xlabel("LBP code")


def _draw_handcrafted_bottom_row(axes, arrays):
    """Draw HOG orientation, HSV representation, and RGB and HSV distributions."""
    axes[0].imshow(arrays["hog"], cmap="gray")
    axes[0].set_title("HOG orientation visualization")
    axes[1].imshow(arrays["hsv"])
    axes[1].set_title("HSV representation")
    _draw_color_distributions(axes[2], arrays["rgb"], arrays["hsv"])


def _draw_color_distributions(axis, rgb, hsv):
    """Plot normalized RGB and HSV channel distributions on a shared axis."""
    colors = ("red", "green", "blue", "crimson", "purple", "saddlebrown")
    labels = ("Red", "Green", "Blue", "Hue", "Saturation", "Value")
    channels = [rgb[..., i] for i in range(3)] + [hsv[..., i] for i in range(3)]
    for channel, color_code, label in zip(channels, colors, labels):
        histogram, edges = np.histogram(channel, bins=16, range=(0, 1), density=True)
        axis.plot((edges[:-1] + edges[1:]) / 2, histogram, color=color_code, label=label)
    axis.set_title("RGB and HSV channel distributions")
    axis.set_xlabel("Channel value")
    axis.legend(fontsize=8, frameon=True)


def _handcrafted_feature_summary(image, arrays, hog_orientations,
                                 hog_pixels_per_cell, lbp_points, lbp_radius,
                                 lbp_grid, color_bins):
    """Collect image dimensions and handcrafted feature dimensions for display."""
    features = extract_image_features(
        image,
        hog_orientations=hog_orientations,
        hog_pixels_per_cell=hog_pixels_per_cell,
        lbp_points=lbp_points,
        lbp_radius=lbp_radius,
        lbp_grid=lbp_grid,
        color_bins=color_bins,
    )
    return {
        "image_shape": tuple(image.shape),
        "RGB mean": np.round(arrays["rgb"].mean(axis=(0, 1)), 4).tolist(),
        "RGB standard deviation": np.round(arrays["rgb"].std(axis=(0, 1)), 4).tolist(),
        "HSV mean": np.round(arrays["hsv"].mean(axis=(0, 1)), 4).tolist(),
        "HOG feature count": arrays["hog_count"],
        "LBP histogram bins": len(arrays["histogram"]),
        "feature dimensions": {name: int(features[name].size) for name in ("hog", "lbp", "hsv")},
    }
