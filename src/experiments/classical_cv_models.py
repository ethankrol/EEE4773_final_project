"""Reusable handcrafted-feature classifiers and their fitted preprocessors."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features.handcrafted_features import extract_feature_dataset


class HandcraftedRbfSvm:
    """Fit a group-scaled handcrafted-feature RBF SVM and use it on RGB images."""

    def __init__(self, feature_groups, feature_options, pca_components=None,
                 c_value=10.0, gamma="scale", group_scaling="balanced", seed=42):
        """Store feature groups, transform settings, and RBF SVM parameters."""
        self.feature_groups = tuple(feature_groups)
        self.feature_options = dict(feature_options)
        self.pca_components = pca_components
        self.c_value = c_value
        self.gamma = gamma
        self.group_scaling = group_scaling
        self.seed = seed

    def _transform(self, feature_arrays, fit=False):
        """Scale feature groups, optionally reduce HOG dimensions, and concatenate them."""
        blocks = []
        if fit:
            self.scalers_ = {}
            self.pcas_ = {}
        for group in self.feature_groups:
            values = feature_arrays[group]
            if fit:
                scaler = StandardScaler()
                values = scaler.fit_transform(values)
                self.scalers_[group] = scaler
                if group == "hog" and self.pca_components is not None:
                    reducer = PCA(n_components=self.pca_components, random_state=self.seed)
                    values = reducer.fit_transform(values)
                    self.pcas_[group] = reducer
            else:
                values = self.scalers_[group].transform(values)
                if group in self.pcas_:
                    values = self.pcas_[group].transform(values)
            if self.group_scaling == "balanced":
                values = values / np.sqrt(values.shape[1])
            blocks.append(values)
        return np.concatenate(blocks, axis=1)

    def fit(self, images, labels):
        """Extract and fit the feature transformations and RBF SVM on supplied training images."""
        features = extract_feature_dataset(
            images, self.feature_groups, **self.feature_options
        )
        matrix = self._transform(features, fit=True)
        self.classifier_ = SVC(
            kernel="rbf", C=self.c_value, gamma=self.gamma,
            class_weight="balanced", random_state=self.seed,
        )
        self.classifier_.fit(matrix, labels)
        return self

    def predict(self, images):
        """Predict labels for RGB images using the fitted handcrafted-feature SVM."""
        features = extract_feature_dataset(
            images, self.feature_groups, **self.feature_options
        )
        return self.classifier_.predict(self._transform(features))


class HandcraftedFeaturePreprocessor:
    """Fit group-wise scaling and optional HOG PCA on training-fold features."""

    def __init__(self, feature_groups, pca_components=None):
        """Store the feature groups and optional HOG PCA component count."""
        self.feature_groups = tuple(feature_groups)
        self.pca_components = pca_components
        self.scalers_ = {}
        self.pca_ = None

    def fit_transform(self, feature_arrays):
        """Fit feature scaling/PCA on a training fold and return its transformed matrix."""
        matrices = []
        for group in self.feature_groups:
            matrix = StandardScaler().fit(feature_arrays[group])
            values = matrix.transform(feature_arrays[group])
            self.scalers_[group] = matrix
            if group == "hog" and self.pca_components is not None:
                self.pca_ = PCA(n_components=self.pca_components, random_state=42)
                values = self.pca_.fit_transform(values)
            values = values / np.sqrt(values.shape[1])
            matrices.append(values)
        return np.concatenate(matrices, axis=1)

    def transform(self, feature_arrays):
        """Transform held-out-fold or test features using only fitted training statistics."""
        matrices = []
        for group in self.feature_groups:
            values = self.scalers_[group].transform(feature_arrays[group])
            if group == "hog" and self.pca_ is not None:
                values = self.pca_.transform(values)
            values = values / np.sqrt(values.shape[1])
            matrices.append(values)
        return np.concatenate(matrices, axis=1)


class HandcraftedImageClassifier:
    """Store a fitted handcrafted-feature preprocessor and classical image classifier."""

    def __init__(self, feature_config, classifier_name, seed=42):
        """Store the feature configuration and selected classifier settings."""
        self.feature_config = feature_config
        self.classifier_name = classifier_name
        self.seed = seed

    def fit(self, images, labels):
        """Fit feature extraction transforms and the selected classifier on RGB images."""
        features = extract_feature_dataset(
            images, self.feature_config["groups"], **self.feature_config["options"]
        )
        self.preprocessor_ = HandcraftedFeaturePreprocessor(
            self.feature_config["groups"], self.feature_config["pca"]
        )
        matrix = self.preprocessor_.fit_transform(features)
        self.classifier_ = _make_fixed_classifier(self.classifier_name, self.seed)
        self.classifier_.fit(matrix, labels)
        return self

    def predict(self, images):
        """Predict scene labels for RGB images using the fitted feature/classifier pipeline."""
        features = extract_feature_dataset(
            images, self.feature_config["groups"], **self.feature_config["options"]
        )
        return self.classifier_.predict(self.preprocessor_.transform(features))


def _make_fixed_classifier(classifier_name, seed):
    """Create one of the four fixed-setting classifiers used in the compact CV grid."""
    classifiers = {
        "rbf_svm": SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=seed,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance", p=2),
    }
    return classifiers[classifier_name]
