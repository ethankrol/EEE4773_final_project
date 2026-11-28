"""Tune RBF SVM hyperparameters for the selected handcrafted campus-image features."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm.auto import tqdm

from src.preprocessing import augment_training_images
from src.features.handcrafted_features import extract_feature_dataset
from src.experiments.classical_cv_models import HandcraftedFeaturePreprocessor


def _feature_config():
    """Return the selected spatial LBP, HSV, and full-HOG feature configuration."""
    return {
        "name": "spatial4x4_lbp_hsv_hog_full",
        "groups": ("lbp", "hsv", "hog"),
        "options": {"lbp_grid": (4, 4), "color_grid": (4, 4)},
        "pca": None,
    }


def _make_model_matrix(images, feature_config, preprocessor=None, fit=False):
    """Extract the fixed feature set and transform it using training-fold scaling."""
    features = extract_feature_dataset(
        images, feature_config["groups"], **feature_config["options"]
    )
    if fit:
        preprocessor = HandcraftedFeaturePreprocessor(
            feature_config["groups"], feature_config["pca"]
        )
        matrix = preprocessor.fit_transform(features)
    else:
        matrix = preprocessor.transform(features)
    return matrix, preprocessor


def _scale_gamma(matrix):
    """Calculate the RBF gamma value equivalent to scikit-learn gamma='scale'."""
    variance = float(matrix.var())
    return 1.0 / (matrix.shape[1] * variance) if variance > 0.0 else 1.0


class TunedHandcraftedRbfSvm:
    """Predict campus scene labels with the selected handcrafted features and tuned RBF SVM."""

    def __init__(self, feature_config, c_value, gamma_multiplier, seed=33541943):
        """Store the selected feature configuration and SVM hyperparameters."""
        self.feature_config = feature_config
        self.c_value = c_value
        self.gamma_multiplier = gamma_multiplier
        self.seed = seed

    def fit(self, images, labels):
        """Fit feature scaling and the selected RBF SVM on supplied training images."""
        matrix, self.preprocessor_ = _make_model_matrix(
            images, self.feature_config, fit=True
        )
        self.gamma_ = self.gamma_multiplier * _scale_gamma(matrix)
        self.classifier_ = SVC(
            C=self.c_value, gamma=self.gamma_, kernel="rbf", class_weight="balanced"
        )
        self.classifier_.fit(matrix, labels)
        return self

    def predict(self, images):
        """Predict scene labels for RGB images using the fitted SVM pipeline."""
        matrix, _ = _make_model_matrix(
            images, self.feature_config, preprocessor=self.preprocessor_
        )
        return self.classifier_.predict(matrix)


def run_svm_hyperparameter_search(train_images, train_labels, cv_splits=3,
                                  copies_per_image=5, c_values=(0.1, 1, 10, 100, 1000),
                                  gamma_multipliers=(0.1, 0.3, 1, 3, 10),
                                  seed=33541943,
                                  output_path="results/svm_tuning_results.csv"):
    """Tune C and gamma for the fixed five-copy spatial LBP, HSV, and full-HOG pipeline using stratified CV."""
    feature_config = _feature_config()
    candidates = _make_candidates(c_values, gamma_multipliers)
    folds = _make_folds(train_labels, cv_splits, seed)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    start_time = perf_counter()
    _print_search_setup(feature_config, copies_per_image, candidates, cv_splits, seed)
    fold_data = _prepare_fold_data(train_images, train_labels, folds,
                                   feature_config, copies_per_image, seed)
    results = _search_candidates(candidates, fold_data, feature_config,
                                 copies_per_image, output_file)
    results.sort(key=lambda row: (row["cv_accuracy_mean"], row["cv_macro_f1_mean"]), reverse=True)
    pd.DataFrame(results).to_csv(output_file, index=False)
    best = results[0]
    best["model"] = _fit_best_model(train_images, train_labels, feature_config,
                                    best, copies_per_image, seed)
    best["total_seconds"] = perf_counter() - start_time
    _print_selected_result(best)
    return results, best


def _make_candidates(c_values, gamma_multipliers):
    """Create every requested C and gamma multiplier pair in search order."""
    return [(c_value, gamma) for c_value in c_values for gamma in gamma_multipliers]


def _make_folds(train_labels, cv_splits, seed):
    """Create the seeded stratified folds used by all candidate settings."""
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(train_labels)), train_labels))


def _print_search_setup(feature_config, copies_per_image, candidates, cv_splits, seed):
    """Print the fixed feature setup and total number of cross-validation fits."""
    count = len(candidates)
    print(f"Fixed features: {feature_config['name']}; augmented copies per image: {copies_per_image}.")
    print(f"Search: {count} C/gamma settings x {cv_splits} folds = "
          f"{count * cv_splits} SVM fits; CV seed={seed}.")


def _prepare_fold_data(train_images, train_labels, folds, feature_config,
                       copies_per_image, seed):
    """Augment fold-training images and extract reusable train and held-out matrices."""
    fold_data = []
    for fold_number, indices in enumerate(folds):
        fold_data.append(_prepare_one_fold(
            train_images, train_labels, indices, fold_number, len(folds),
            feature_config, copies_per_image, seed,
        ))
    return fold_data


def _prepare_one_fold(train_images, train_labels, indices, fold_number, fold_count,
                      feature_config, copies_per_image, seed):
    """Prepare feature matrices and labels for one stratified cross-validation fold."""
    train_indices, heldout_indices = indices
    print(f"Preparing CV fold {fold_number + 1}/{fold_count}: augmentation and feature extraction.")
    augmented_images, augmented_labels = augment_training_images(
        train_images[train_indices], train_labels[train_indices],
        copies_per_image=copies_per_image, seed=seed + fold_number,
    )
    train_matrix, preprocessor = _make_model_matrix(augmented_images, feature_config, fit=True)
    heldout_matrix, _ = _make_model_matrix(
        train_images[heldout_indices], feature_config, preprocessor=preprocessor
    )
    return _assemble_fold_data(train_matrix, heldout_matrix, augmented_labels,
                               train_labels[heldout_indices])


def _assemble_fold_data(train_matrix, heldout_matrix, train_labels, heldout_labels):
    """Store matrices, labels, and scale gamma for reuse across SVM settings."""
    return {
        "train_matrix": train_matrix,
        "heldout_matrix": heldout_matrix,
        "train_labels": train_labels,
        "heldout_labels": heldout_labels,
        "gamma_scale": _scale_gamma(train_matrix),
    }


def _search_candidates(candidates, fold_data, feature_config, copies_per_image, output_file):
    """Evaluate all SVM candidates, report progress, and checkpoint results to CSV."""
    results = []
    with tqdm(total=len(candidates), desc="RBF SVM tuning", unit="setting",
              dynamic_ncols=True) as progress:
        for number, (c_value, gamma_multiplier) in enumerate(candidates, start=1):
            progress.set_postfix_str(f"C={c_value:g}, gamma={gamma_multiplier:g} x scale")
            row = _evaluate_candidate(c_value, gamma_multiplier, fold_data,
                                      feature_config, copies_per_image)
            results.append(row)
            progress.update(1)
            _report_candidate(number, len(candidates), row)
            _save_search_results(results, output_file)
    return results


def _evaluate_candidate(c_value, gamma_multiplier, fold_data, feature_config,
                        copies_per_image):
    """Fit one SVM setting on each prepared fold and aggregate its scores."""
    scores = [_score_svm_fold(c_value, gamma_multiplier, fold) for fold in fold_data]
    accuracies, macro_f1_scores, gamma_values = map(list, zip(*scores))
    return _candidate_result(feature_config, copies_per_image, c_value,
                             gamma_multiplier, accuracies, macro_f1_scores, gamma_values)


def _score_svm_fold(c_value, gamma_multiplier, fold):
    """Fit and score one SVM setting on a prepared training and held-out fold."""
    gamma_value = gamma_multiplier * fold["gamma_scale"]
    classifier = SVC(C=c_value, gamma=gamma_value, kernel="rbf", class_weight="balanced")
    classifier.fit(fold["train_matrix"], fold["train_labels"])
    predictions = classifier.predict(fold["heldout_matrix"])
    accuracy = accuracy_score(fold["heldout_labels"], predictions)
    macro_f1 = f1_score(fold["heldout_labels"], predictions, average="macro")
    return accuracy, macro_f1, gamma_value


def _candidate_result(feature_config, copies_per_image, c_value, gamma_multiplier,
                      accuracies, macro_f1_scores, gamma_values):
    """Build a result row with means, standard deviations, and per-fold scores."""
    return {
        "feature_config": feature_config["name"],
        "augmentation_copies": copies_per_image,
        "C": c_value,
        "gamma_multiplier_of_scale": gamma_multiplier,
        "fold_gamma_values": ", ".join(f"{value:.8g}" for value in gamma_values),
        "cv_accuracy_mean": float(np.mean(accuracies)),
        "cv_accuracy_std": float(np.std(accuracies)),
        "cv_macro_f1_mean": float(np.mean(macro_f1_scores)),
        "cv_macro_f1_std": float(np.std(macro_f1_scores)),
        "fold_accuracies": ", ".join(f"{value:.3f}" for value in accuracies),
        "fold_macro_f1": ", ".join(f"{value:.3f}" for value in macro_f1_scores),
    }


def _report_candidate(number, total, row):
    """Print accuracy and macro-F1 mean and standard deviation for one candidate."""
    tqdm.write(
        f"{number}/{total} C={row['C']:g}, "
        f"gamma={row['gamma_multiplier_of_scale']:g} x scale: "
        f"accuracy={row['cv_accuracy_mean']:.3f} +/- {row['cv_accuracy_std']:.3f}; "
        f"macro-F1={row['cv_macro_f1_mean']:.3f} +/- {row['cv_macro_f1_std']:.3f}"
    )


def _save_search_results(results, output_file):
    """Write the current candidate results to the search CSV checkpoint."""
    pd.DataFrame(results).to_csv(output_file, index=False)


def _fit_best_model(train_images, train_labels, feature_config, best,
                    copies_per_image, seed):
    """Augment all development images and fit the selected SVM pipeline."""
    augmented_images, augmented_labels = augment_training_images(
        train_images, train_labels, copies_per_image=copies_per_image, seed=seed
    )
    model = TunedHandcraftedRbfSvm(
        feature_config, best["C"], best["gamma_multiplier_of_scale"], seed=seed
    )
    return model.fit(augmented_images, augmented_labels)


def _print_selected_result(best):
    """Print the selected cross-validation setting and its summary scores."""
    print(
        f"Selected CV setting: C={best['C']:g}, "
        f"gamma={best['gamma_multiplier_of_scale']:g} x scale; "
        f"accuracy={best['cv_accuracy_mean']:.3f} +/- {best['cv_accuracy_std']:.3f}; "
        f"macro-F1={best['cv_macro_f1_mean']:.3f} +/- {best['cv_macro_f1_std']:.3f}"
    )
