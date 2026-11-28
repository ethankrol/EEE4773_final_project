"""Handcrafted computer-vision features and classical campus-scene classifiers."""

from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from src.preprocessing import load_dataset
from src.features.handcrafted_features import extract_feature_dataset


def load_data_splits(data_dir="data", seed=33541943, test_size=0.2, image_size=224):
    """Load images and make a reproducible stratified development/test split."""
    images, labels = load_dataset(data_dir=data_dir, image_size=image_size)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    return {
        "train": {"images": train_images, "labels": train_labels},
        "test": {"images": test_images, "labels": test_labels},
    }


def make_classifier(name, seed=42):
    """Create a supported classical classifier by its short name."""
    models = {
        "linear_svm": LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=10000),
        "rbf_svm": SVC(C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed),
        "logistic_regression": LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                                 class_weight="balanced", random_state=seed, n_jobs=-1),
    }
    return models[name]


def _feature_combinations():
    """Return the required single and combined feature group configurations."""
    groups = ("hog", "lbp", "rgb", "hsv")
    return [(name,) for name in groups] + [
        combo for combo in combinations(groups, 2) if combo in (
            ("hog", "lbp"), ("hog", "hsv"), ("lbp", "hsv")
        )
    ] + [("hog", "lbp", "hsv")]


def run_cv_experiments(train_images, train_labels, validation_images, validation_labels,
                       feature_options=None, classifier_names=("linear_svm", "rbf_svm", "logistic_regression", "random_forest"),
                       cv_splits=3, pca_components=(None, 64, 128), seed=42):
    """Compare handcrafted feature combinations and classifiers with stratified CV and a fixed validation set."""
    combinations_to_test = _feature_combinations()
    train_features, val_features = _extract_cv_features(
        train_images, validation_images, combinations_to_test, feature_options
    )
    folds = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    rows = []
    for groups in combinations_to_test:
        matrices = _combine_feature_matrices(groups, train_features, val_features)
        rows.extend(_evaluate_feature_group(
            groups, matrices, train_labels, validation_labels,
            classifier_names, pca_components, folds, seed,
        ))
    return sorted(rows, key=lambda row: row["cv_accuracy_mean"], reverse=True)


def _extract_cv_features(train_images, validation_images, combinations_to_test, options):
    """Extract each feature group once for the training and fixed validation images."""
    feature_options = options or {}
    needed = tuple(sorted({group for combo in combinations_to_test for group in combo}))
    train_features = extract_feature_dataset(train_images, needed, **feature_options)
    validation_features = extract_feature_dataset(validation_images, needed, **feature_options)
    return train_features, validation_features


def _combine_feature_matrices(groups, train_features, validation_features):
    """Concatenate requested feature arrays in the configured group order."""
    train_matrix = np.concatenate([train_features[group] for group in groups], axis=1)
    validation_matrix = np.concatenate([validation_features[group] for group in groups], axis=1)
    return train_matrix, validation_matrix


def _evaluate_feature_group(groups, matrices, train_labels, validation_labels,
                            classifier_names, pca_components, folds, seed):
    """Evaluate all classifier and PCA settings for one feature-group combination."""
    rows = []
    train_matrix, validation_matrix = matrices
    components = pca_components if "hog" in groups else (None,)
    for classifier_name in classifier_names:
        for component_count in components:
            rows.append(_evaluate_cv_model(
                groups, train_matrix, validation_matrix, train_labels,
                validation_labels, classifier_name, component_count, folds, seed,
            ))
    return rows


def _evaluate_cv_model(groups, train_matrix, validation_matrix, train_labels,
                       validation_labels, classifier_name, component_count, folds, seed):
    """Cross-validate, refit, and score one classifier and PCA configuration."""
    pipeline = _make_cv_pipeline(classifier_name, component_count, seed)
    started = time.perf_counter()
    scores = cross_validate(
        pipeline, train_matrix, train_labels, cv=folds,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        return_train_score=True, n_jobs=1,
    )
    fitted = clone(pipeline).fit(train_matrix, train_labels)
    predictions = fitted.predict(validation_matrix)
    row = _assemble_cv_result(
        groups, component_count, classifier_name, scores, predictions,
        validation_labels, fitted, started,
    )
    _print_cv_result(row)
    return row


def _make_cv_pipeline(classifier_name, component_count, seed):
    """Create the scaled classifier pipeline with optional PCA reduction."""
    steps = [("scale", StandardScaler())]
    if component_count is not None:
        steps.append(("pca", PCA(n_components=component_count, random_state=seed)))
    steps.append(("classifier", make_classifier(classifier_name, seed)))
    return Pipeline(steps)


def _assemble_cv_result(groups, component_count, classifier_name, scores,
                        predictions, validation_labels, fitted, started):
    """Collect cross-validation, fixed-validation, timing, and fitted-model results."""
    return {
        "feature_groups": "+".join(groups),
        "pca_components": component_count,
        "classifier": classifier_name,
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_macro_f1_mean": scores["test_f1_macro"].mean(),
        "cv_train_accuracy_mean": scores["train_accuracy"].mean(),
        "validation_accuracy": accuracy_score(validation_labels, predictions),
        "validation_macro_f1": f1_score(validation_labels, predictions, average="macro"),
        "seconds": time.perf_counter() - started,
        "model": fitted,
        "validation_predictions": predictions,
    }


def _print_cv_result(row):
    """Print the feature, classifier, PCA, CV, and validation summary for one model."""
    print(
        f"Completed {row['feature_groups']} | {row['classifier']} | "
        f"PCA={row['pca_components']}: CV acc={row['cv_accuracy_mean']:.3f} "
        f"Â± {row['cv_accuracy_std']:.3f}; val acc={row['validation_accuracy']:.3f}"
    )


def save_experiment_results(results, output_path="results/classical_cv_results.csv"):
    """Save experiment configuration and metrics to a CSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    columns = [key for key in results[0] if key not in ("model", "validation_predictions")]
    table = pd.DataFrame([{key: row[key] for key in columns} for row in results])
    table.to_csv(output_path, index=False)
    return table


def evaluate_classifier(model, images, labels, feature_groups=("hog", "lbp", "hsv"), feature_options=None,
                         class_names=None):
    """Evaluate a fitted pipeline on held-out images and return its metrics and confusion matrix."""
    options = feature_options or {}
    features = extract_feature_dataset(images, feature_groups, **options)
    matrix = np.concatenate([features[name] for name in feature_groups], axis=1)
    predictions = model.predict(matrix)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "report": classification_report(labels, predictions, target_names=class_names, output_dict=True),
        "confusion_matrix": confusion_matrix(labels, predictions),
        "predictions": predictions,
    }
