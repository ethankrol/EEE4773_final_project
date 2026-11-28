# EEE4773 Campus Image Classification

This project compares classical computer-vision features and classifiers on campus images.

## Files

```text
data/
|-- data_train.npy
`-- labels_train.npy
notebooks/
|-- utils.py
|-- preprocessing.ipynb
|-- classical_cv_experiment.ipynb
|-- svm_hyperparameter_search.ipynb
`-- results/                 # Search outputs, figures, and saved artifacts
src/
|-- __init__.py
|-- artifact_io.py
|-- preprocessing.py
|-- visualization.py
|-- features/
|   |-- __init__.py
|   `-- handcrafted_features.py
`-- experiments/
    |-- __init__.py
    |-- classical_cv_experiment.py
    |-- classical_cv_grid_search.py
    |-- classical_cv_models.py
    |-- classical_cv_refinement.py
    `-- svm_hyperparameter_search.py
```

## Notebooks

- **Preprocessing:** Load the saved image data and preview full and mild augmentations.
- **Classical CV experiment:** Preview LBP, HSV, and HOG features, compare feature and classifier settings with cross-validation, and evaluate the selected model on the held-out test set.
- **SVM hyperparameter search:** Tune the RBF SVM settings using the selected handcrafted features, then evaluate on the held-out test set.

## Code and data

The Python files in `src/` contain the reusable preprocessing, feature extraction, model, search, visualization, and artifact-saving code. The notebooks call these functions and display the results.

Both classification notebooks use `data/data_train.npy` and `data/labels_train.npy`. They use the same stratified 80/20 split seed, `33541943`; cross-validation uses only the training portion.

Search results are written under `notebooks/results/`. The fitted SVM can be saved there as `svm_tuned_model.joblib` using the save snippet in the SVM notebook. Load it later with `joblib.load`.
