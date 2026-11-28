"""Run the compact handcrafted-feature and classifier cross-validation grid."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from src.preprocessing import augment_training_images
from src.features.handcrafted_features import extract_feature_dataset
from src.experiments.classical_cv_models import (
    HandcraftedFeaturePreprocessor,
    HandcraftedImageClassifier,
    _make_fixed_classifier,
)


def _compact_grid_feature_configs():
    """Return five feature profiles with a compact augmentation plan for CV comparison."""
    return [
        {"name": "global_lbp_hsv", "groups": ("lbp", "hsv"), "options": {},
         "pca": None, "augmentation_copies": (0, 3, 5)},
        {"name": "spatial4x4_lbp_hsv", "groups": ("lbp", "hsv"),
         "options": {"lbp_grid": (4, 4), "color_grid": (4, 4)},
         "pca": None, "augmentation_copies": (0, 3, 5)},
        {"name": "spatial4x4_lbp_hsv_hog_pca64", "groups": ("lbp", "hsv", "hog"),
         "options": {"lbp_grid": (4, 4), "color_grid": (4, 4)},
         "pca": 64, "augmentation_copies": (0, 3, 5)},
        {"name": "spatial4x4_lbp_hsv_hog_full", "groups": ("lbp", "hsv", "hog"),
         "options": {"lbp_grid": (4, 4), "color_grid": (4, 4)},
         "pca": None, "augmentation_copies": (0, 3, 5)},
        {"name": "spatial4x4_lbp_hsv_gabor", "groups": ("lbp", "hsv", "gabor"),
         "options": {"lbp_grid": (4, 4), "color_grid": (4, 4), "include_gabor": True},
         "pca": None, "augmentation_copies": (0, 3, 5)},
    ]


def run_compact_classical_grid(train_images, train_labels, cv_splits=3,
                               seed=42,
                               output_path="results/classical_cv_grid_search.csv"):
    """Compare compact feature, classifier, and augmentation candidates with stratified CV."""
    configs = _compact_grid_feature_configs()
    classifier_names = ("rbf_svm", "random_forest", "logistic_regression", "knn")
    folds = _make_stratified_folds(train_labels, cv_splits, seed)
    results_path = Path(output_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    total = _compact_candidate_count(configs, classifier_names)
    _print_compact_search_plan(total, cv_splits)
    results = _run_compact_candidates(
        train_images, train_labels, configs, classifier_names, folds,
        cv_splits, seed, total, results_path
    )
    results.sort(key=lambda row: (row["cv_accuracy_mean"], row["cv_macro_f1_mean"]),
                  reverse=True)
    _save_compact_results(results, results_path)
    best = _fit_compact_winner(train_images, train_labels, configs, results[0], seed)
    return results, best


def _compact_candidate_count(configs, classifier_names):
    """Count feature, augmentation, and classifier combinations in the compact grid."""
    return sum(len(config["augmentation_copies"]) for config in configs) * len(classifier_names)


def _print_compact_search_plan(candidate_total, cv_splits):
    """Print the number of candidates and fold-level classifier fits before search."""
    print(f"Search plan: {candidate_total} candidates; {cv_splits} stratified CV folds each; "
          f"{candidate_total * cv_splits} fold fits.")


def _run_compact_candidates(images, labels, configs, classifier_names, folds,
                            cv_splits, seed, candidate_total, results_path):
    """Run the compact CV candidates and save each completed result row."""
    results = []
    with tqdm(total=candidate_total, desc="Classical CV grid",
              unit="candidate", dynamic_ncols=True) as progress:
        for config in configs:
            for copies in config["augmentation_copies"]:
                scores = _score_compact_candidate(
                    images, labels, config, copies, classifier_names,
                    folds, cv_splits, seed, progress
                )
                for row in _make_compact_result_rows(config, copies, scores):
                    results.append(row)
                    _record_compact_candidate(results, config, copies, candidate_total,
                                              progress, results_path)
    return results


def _score_compact_candidate(images, labels, config, copies, classifier_names,
                             folds, cv_splits, seed, progress):
    """Accumulate held-out and original-training scores across folds for one candidate."""
    scores = _empty_compact_scores(classifier_names)
    for fold_number, indices in enumerate(folds):
        progress.set_postfix_str(
            f"{config['name']} | copies={copies} | fold {fold_number + 1}/{cv_splits}"
        )
        fold_scores = _score_compact_fold(
            images, labels, config, copies, classifier_names,
            indices, fold_number, seed
        )
        _accumulate_compact_scores(scores, fold_scores)
    return scores


def _empty_compact_scores(classifier_names):
    """Initialize per-classifier lists for CV accuracy, macro-F1, and train accuracy."""
    return {name: {"accuracy": [], "macro_f1": [], "train_accuracy": []}
            for name in classifier_names}


def _score_compact_fold(images, labels, config, copies, classifier_names,
                        indices, fold_number, seed):
    """Augment and score every classifier for one prepared feature fold."""
    train_indices, heldout_indices = indices
    matrices, fold_labels = _compact_fold_matrices(
        images, labels, config, copies, train_indices, heldout_indices,
        fold_number, seed
    )
    return _score_compact_classifiers(
        matrices, labels, train_indices, heldout_indices, fold_labels, classifier_names, seed
    )


def _compact_fold_matrices(images, labels, config, copies, train_indices,
                           heldout_indices, fold_number, seed):
    """Build scaled augmented-training, held-out, and original-training matrices."""
    fold_images, fold_labels = augment_training_images(
        images[train_indices], labels[train_indices],
        copies_per_image=copies, seed=seed + fold_number
    )
    train_features = extract_feature_dataset(fold_images, config["groups"], **config["options"])
    heldout_features = extract_feature_dataset(
        images[heldout_indices], config["groups"], **config["options"]
    )
    original_features = extract_feature_dataset(
        images[train_indices], config["groups"], **config["options"]
    )
    preprocessor = HandcraftedFeaturePreprocessor(config["groups"], config["pca"])
    matrices = _transform_compact_feature_sets(
        preprocessor, train_features, heldout_features, original_features
    )
    return matrices, fold_labels


def _transform_compact_feature_sets(preprocessor, train_features,
                                   heldout_features, original_features):
    """Fit preprocessing on augmented training features and transform both evaluation sets."""
    train_matrix = preprocessor.fit_transform(train_features)
    heldout_matrix = preprocessor.transform(heldout_features)
    original_matrix = preprocessor.transform(original_features)
    return train_matrix, heldout_matrix, original_matrix


def _score_compact_classifiers(matrices, labels, train_indices, heldout_indices,
                               fold_labels, classifier_names, seed):
    """Fit each fixed classifier and return its held-out and training scores."""
    train_matrix, heldout_matrix, original_matrix = matrices
    result = {}
    for name in classifier_names:
        result[name] = _score_one_compact_classifier(
            name, seed, train_matrix, heldout_matrix, original_matrix,
            fold_labels, labels[heldout_indices], labels[train_indices]
        )
    return result


def _score_one_compact_classifier(name, seed, train_matrix, heldout_matrix,
                                  original_matrix, train_labels, heldout_labels,
                                  original_labels):
    """Fit one compact-grid classifier and return its accuracy and macro-F1 scores."""
    classifier = _make_fixed_classifier(name, seed)
    classifier.fit(train_matrix, train_labels)
    heldout_prediction = classifier.predict(heldout_matrix)
    train_prediction = classifier.predict(original_matrix)
    return {
        "accuracy": accuracy_score(heldout_labels, heldout_prediction),
        "macro_f1": f1_score(heldout_labels, heldout_prediction, average="macro"),
        "train_accuracy": accuracy_score(original_labels, train_prediction),
    }


def _accumulate_compact_scores(scores, fold_scores):
    """Append one fold's metrics to the matching classifier score lists."""
    for name, metrics in fold_scores.items():
        for metric, value in metrics.items():
            scores[name][metric].append(value)


def _make_compact_result_rows(config, copies, scores):
    """Build one summarized CV result row for each classifier in a candidate."""
    rows = []
    for name, values in scores.items():
        rows.append(_compact_result_row(config, copies, name, values))
    return rows


def _compact_result_row(config, copies, classifier_name, scores):
    """Summarize one classifier's fold metrics with means and standard deviations."""
    return {
        "feature_config": config["name"],
        "feature_groups": "+".join(config["groups"]),
        "pca_components": config["pca"],
        "classifier": classifier_name,
        "augmentation_copies": copies,
        "cv_accuracy_mean": float(np.mean(scores["accuracy"])),
        "cv_accuracy_std": float(np.std(scores["accuracy"])),
        "cv_macro_f1_mean": float(np.mean(scores["macro_f1"])),
        "cv_macro_f1_std": float(np.std(scores["macro_f1"])),
        "cv_train_accuracy_mean": float(np.mean(scores["train_accuracy"])),
    }


def _record_compact_candidate(results, config, copies, candidate_total,
                              progress, results_path):
    """Advance progress, report candidate metrics, and checkpoint accumulated results."""
    row = results[-1]
    progress.update(1)
    tqdm.write(
        f"{progress.n}/{candidate_total} {config['name']} | {row['classifier']} | "
        f"copies={copies}: CV accuracy={row['cv_accuracy_mean']:.3f} "
        f"± {row['cv_accuracy_std']:.3f}; macro-F1="
        f"{row['cv_macro_f1_mean']:.3f} ± {row['cv_macro_f1_std']:.3f}"
    )
    _save_compact_results(results, results_path)


def _save_compact_results(results, results_path):
    """Save current compact-grid metrics to the configured CSV file."""
    pd.DataFrame(results).to_csv(results_path, index=False)


def _fit_compact_winner(images, labels, configs, best_row, seed):
    """Fit the top CV candidate on augmented development data and return its model."""
    config = next(item for item in configs if item["name"] == best_row["feature_config"])
    final_images, final_labels = augment_training_images(
        images, labels, copies_per_image=best_row["augmentation_copies"], seed=seed
    )
    model = HandcraftedImageClassifier(config, best_row["classifier"], seed=seed)
    model.fit(final_images, final_labels)
    best_row["model"] = model
    _print_compact_winner(best_row)
    return best_row


def _print_compact_winner(best_row):
    """Print the selected compact-grid candidate and its cross-validation score."""
    print(f"Best CV candidate: {best_row['feature_config']} | {best_row['classifier']} | "
          f"copies={best_row['augmentation_copies']} | accuracy="
          f"{best_row['cv_accuracy_mean']:.3f} ± {best_row['cv_accuracy_std']:.3f}")
