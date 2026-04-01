# -------------------- Import Modules --------------------
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from Zeus.ML_BaseConfigBox.BaseModelConfig import BaseModelConfig

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOT_DIR = os.path.join(BASE_DIR, "ML_Report/KNNCla_Plot")
MODEL_DIR = os.path.join(BASE_DIR, "ML_Report/KNNCla_Trained_Model")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------- KNNClassifier Missioner --------------------
class KNNClassifier_Missioner(BaseModelConfig):
    """
    Mission-layer class for K-Nearest Neighbors classification workflows.

    This class extends :class:`BaseModelConfig` and provides shared classification
    utilities for KNN-based model trainers. It serves as the mission layer between
    the shared base configuration workflow and concrete KNN classifier model
    classes.

    This class does not define the final estimator construction or CV scoring
    dispatch itself. Concrete model-layer classes are responsible for building the
    final estimator and parameter grid, while the base layer handles shared
    training workflow logic, including model-selection scoring dispatch.

    Main Responsibilities
    ---------------------
    - Define the machine learning task type as classification.
    - Define the estimator step name used inside sklearn pipelines.
    - Evaluate trained KNN classification pipelines on train and test splits.
    - Generate confusion matrix plots for classification diagnostics.
    - Save and load trained model artifacts together with basic metadata.

    Supported Evaluation Modes
    --------------------------
    1. Single-output classification
    - Accuracy
    - Weighted F1 score
    - Confusion matrix
    - Classification report
    - Prediction preview
    - Optional probability preview when supported

    2. Multi-output classification
    - Per-target accuracy
    - Per-target weighted F1 score
    - Per-target confusion matrix
    - Per-target classification report
    - Macro-average metrics across targets
    - Prediction preview by rows
    - Optional probability preview per target when supported

    Diagnostic Plot Utilities
    -------------------------
    - Confusion matrix plot

    Notes
    -----
    - Model-selection scoring for both single-output and multi-output workflows is
    handled by the base layer.
    - KNN classifiers do not expose ``feature_importances_``, so no
    feature-importance utility is implemented here.
    - Plotting support in this class is intentionally focused on confusion
    matrices. For multi-output classification, confusion matrices are expected
    to be interpreted per target if needed.
    - Probability previews depend on whether the trained pipeline exposes
    ``predict_proba()``.
    - This mission layer focuses on reusable evaluation, plotting, and persistence
    logic shared across KNN-based classification workflows.

    Attributes Inherited from BaseModelConfig
    -----------------------------------------
    model_pipeline : sklearn Pipeline or compatible estimator, optional
        Trained classification pipeline.

    X_train, X_test : pd.DataFrame or np.ndarray
        Training and testing feature sets.

    Y_train, Y_test : pd.Series, pd.DataFrame, or np.ndarray
        Training and testing target sets.

    cleaned_Y_data : pd.Series or pd.DataFrame, optional
        Cleaned target data used during training preparation.

    feature_names : list[str] or None
        Feature names recorded after preprocessing.

    input_model_type : str, optional
        Model label used in reports, logging, and output filenames.

    y_train_pred, y_test_pred : optional
        Cached train/test predictions produced during evaluation.

    prediction_preview : optional
        Cached prediction preview for reporting or quick inspection.
    """

    # -------------------- Classification task --------------------
    @property
    def task(self) -> str:
        """
        Return the machine learning task type for this mission class.

        This property is required by :class:`BaseModelConfig` so that the base
        configuration layer can apply task-specific data validation, scoring,
        workflow selection, and evaluation behavior.

        Returns
        -------
        str
            Always returns ``"classification"``.
        """
        return "classification"

    # -------------------- Step name in piepline --------------------
    @property
    def step_name(self) -> str:
        """
        Return the estimator step name used inside the sklearn Pipeline.

        This value is used to retrieve the fitted classifier from
        ``self.model_pipeline.named_steps`` and to define parameter-grid keys
        when GridSearchCV is used.

        Returns
        -------
        str
            Always returns ``"classifier"``.

        Notes
        -----
        Hyperparameter grid keys for GridSearchCV must use this prefix, for example:
        - ``"classifier__n_neighbors"``
        - ``"classifier__weights"``
        - ``"classifier__p"``
        """
        return "classifier"

    # -------------------- Model evaluation --------------------
    def model_evaluation_engine(self) -> Dict[str, Any]:
        """
        Evaluate a trained KNN classification pipeline on train and test datasets.

        This method generates predictions from the fitted classification pipeline
        and computes performance metrics according to whether the task is
        single-output or multi-output classification.

        Evaluation Behavior
        -------------------
        Single-output classification
            Computes:
            - train accuracy
            - test accuracy
            - train weighted F1
            - test weighted F1
            - test confusion matrix
            - test classification report
            - first 10 predicted labels
            - optional first 10 predicted probabilities when supported

        Multi-output classification
            Computes:
            - per-target train accuracy
            - per-target test accuracy
            - per-target train weighted F1
            - per-target test weighted F1
            - per-target confusion matrix
            - per-target classification report
            - macro-average train/test accuracy across targets
            - macro-average train/test weighted F1 across targets
            - first 10 predicted rows
            - optional probability preview per target when supported

        Returns
        -------
        Dict[str, Any]
            Evaluation summary dictionary.

            For single-output classification, keys include:
            - ``"mode"``
            - ``"train_accuracy"``
            - ``"test_accuracy"``
            - ``"train_f1_weighted"``
            - ``"test_f1_weighted"``
            - ``"confusion_matrix"``
            - ``"classification_report"``
            - ``"prediction_preview_first10"``
            - ``"probability_preview_first10"``

            For multi-output classification, keys include:
            - ``"mode"``
            - ``"targets"``
            - ``"macro_train_accuracy"``
            - ``"macro_test_accuracy"``
            - ``"macro_train_f1_weighted"``
            - ``"macro_test_f1_weighted"``
            - ``"per_target"``
            - ``"prediction_preview_first10_rows"``
            - ``"probability_preview_first10_per_target"``

        Raises
        ------
        ValueError
            If `self.model_pipeline` is not trained or not available.

        Side Effects
        ------------
        Updates the following instance attributes:
        - ``self.y_train_pred``
        - ``self.y_test_pred``
        - ``self.prediction_preview``

        Notes
        -----
        - The method assumes train/test split data already exist in the object,
        typically prepared by base-layer logic.
        - For multi-output classification, predicted values are converted to
        DataFrame form so target-column names and indices stay aligned.
        - Macro metrics here are simple arithmetic means across targets.
        - Probability previews are only returned if the pipeline exposes
        `predict_proba()`.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ Model pipeline not trained yet ‼️")

        # ---------- Record predictions from train / test dataset ----------
        self.y_train_pred = self.model_pipeline.predict(self.X_train)
        self.y_test_pred = self.model_pipeline.predict(self.X_test)

        y_true_train = self.Y_train
        y_true_test = self.Y_test
        y_pred_train = self.y_train_pred
        y_pred_test = self.y_test_pred

        # ---------- Single-output ----------
        if not self._is_multi_output(y_true_test):
            # ---------- Accuracy ----------
            train_acc = accuracy_score(y_true_train, y_pred_train)
            test_acc = accuracy_score(y_true_test, y_pred_test)

            # ---------- F1 score ----------
            train_f1 = f1_score(y_true_train, y_pred_train, average="weighted")
            test_f1 = f1_score(y_true_test, y_pred_test, average="weighted")

            # ---------- Confusion matrix and classification report ----------
            cm = confusion_matrix(y_true_test, y_pred_test)
            report = classification_report(y_true_test, y_pred_test, digits=4)

            self.prediction_preview = np.array(y_pred_test[:10])  # Array

            # ---------- Probability preview ----------
            proba_preview = None
            if hasattr(self.model_pipeline, "predict_proba"):
                proba_preview = self.model_pipeline.predict_proba(self.X_test[:10])

            return {
                "mode": "single_output",
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "train_f1_weighted": float(train_f1),
                "test_f1_weighted": float(test_f1),
                "confusion_matrix": cm,
                "classification_report": report,
                "prediction_preview_first10": self.prediction_preview,
                "probability_preview_first10": proba_preview,
            }

        # ---------- Multi-output ----------
        # Help Numpy ndarray to be DataFrame format in order to get its column's val
        if not isinstance(y_true_test, pd.DataFrame):
            y_true_test = pd.DataFrame(y_true_test)
        if not isinstance(y_true_train, pd.DataFrame):
            y_true_train = pd.DataFrame(y_true_train)

        y_pred_train_df = pd.DataFrame(
            y_pred_train, columns=y_true_train.columns, index=y_true_train.index
        )
        y_pred_test_df = pd.DataFrame(
            y_pred_test, columns=y_true_test.columns, index=y_true_test.index
        )

        per_target = {}
        acc_list_train, acc_list_test, f1_list_train, f1_list_test = [], [], [], []

        for col in y_true_test.columns:
            # ---------- Predicted and true value from train / test dataset ----------
            yt_tr = y_true_train[col]
            yp_tr = y_pred_train_df[col]
            yt_te = y_true_test[col]
            yp_te = y_pred_test_df[col]

            # ---------- Acccuracy and F1 score for each target ----------
            acc_tr = accuracy_score(yt_tr, yp_tr)
            acc_te = accuracy_score(yt_te, yp_te)
            f1_tr = f1_score(yt_tr, yp_tr, average="weighted")
            f1_te = f1_score(yt_te, yp_te, average="weighted")

            acc_list_train.append(acc_tr)
            acc_list_test.append(acc_te)
            f1_list_train.append(f1_tr)
            f1_list_test.append(f1_te)

            # ---------- Record above results for each target ----------
            per_target[col] = {
                "train_accuracy": acc_tr,
                "test_accuracy": acc_te,
                "train_f1_weighted": f1_tr,
                "test_f1_weighted": f1_te,
                "classification_report": classification_report(yt_te, yp_te, digits=4),
                "confusion_matrix": confusion_matrix(yt_te, yp_te),
            }

        self.prediction_preview = y_pred_test_df.head(10)  # DataFrame

        # ---------- Probability preview for each target ----------
        proba_preview_per_target = None
        if hasattr(self.model_pipeline, "predict_proba"):
            try:
                proba_raw = self.model_pipeline.predict_proba(self.X_test[:10])

                # Probability in list format
                if isinstance(proba_raw, list):
                    proba_preview_per_target = {
                        col: proba_raw[idx]
                        for idx, col in enumerate(y_true_test.columns)
                    }
                else:
                    # fallback: if some estimator returns non-list format
                    proba_preview_per_target = {"all_targets": proba_raw}
            except Exception:
                proba_preview_per_target = None

        return {
            "mode": "multi_output",
            "targets": list(y_true_test.columns),
            "macro_train_accuracy": (
                float(np.mean(acc_list_train)) if acc_list_train else None
            ),
            "macro_test_accuracy": (
                float(np.mean(acc_list_test)) if acc_list_test else None
            ),
            "macro_train_f1_weighted": (
                float(np.mean(f1_list_train)) if f1_list_train else None
            ),
            "macro_test_f1_weighted": (
                float(np.mean(f1_list_test)) if f1_list_test else None
            ),
            "per_target": per_target,
            "prediction_preview_first10_rows": y_pred_test_df.head(10),
            "probability_preview_first10_per_target": proba_preview_per_target,
        }

    # -------------------- Confusion matrix plot --------------------
    def confusion_matrix_plot_engine(
        self,
        normalize: bool = False,
        filename: Optional[str] = None,
        target_col: Optional[str] = None,
    ):
        """
        Plot and save a confusion matrix for KNN classification results.

        This method supports both single-output and multi-output classification.

        Supported Modes
        ---------------
        Single-output classification
            The confusion matrix is generated directly from ``self.Y_test`` and
            predictions on ``self.X_test``.

        Multi-output classification
            A specific target column must be provided through ``target_col``.
            The confusion matrix is then generated only for that selected target.

        Parameters
        ----------
        normalize : bool, default=False
            Whether to normalize the confusion matrix row-wise.

            - If ``False``, raw counts are displayed.
            - If ``True``, each row is divided by its row sum and displayed as
            proportions.

        filename : str or None, default=None
            Output filename for the saved image.

            - If provided, that name is used directly.
            - If ``None``, an automatic filename is generated based on the current
            model type and optional target information.

        target_col : str or None, default=None
            Target column name used only for multi-output classification.

            - Required when ``self.Y_test`` is multi-output.
            - Ignored for single-output classification.

        Returns
        -------
        None
            This method saves and displays the plot but does not return a value.

        Raises
        ------
        ValueError
            If:

            - ``self.model_pipeline`` is not trained,
            - multi-output mode is detected but ``target_col`` is missing,
            - or ``target_col`` does not exist in ``self.Y_test``.

        Side Effects
        ------------
        - Saves the generated figure into ``PLOT_DIR``.
        - Displays the figure using matplotlib.
        - Prints the saved file path.

        Notes
        -----
        - The confusion matrix is computed from predictions on ``self.X_test``.
        - In multi-output classification, predictions are aligned with target-column
        names by converting prediction output into a DataFrame before selecting the
        requested ``target_col``.
        - Cell values are annotated directly on the matrix.
        - In normalization mode, cells are displayed with two decimal places.
        - In raw-count mode, cells are displayed as integers.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️  Model pipeline not trained yet ‼️")

        y_pred = self.model_pipeline.predict(self.X_test)

        # ---------- Single-output ----------
        if not self._is_multi_output(self.Y_test):
            y_true = self.Y_test
            title_suffix = "single_output"

        # ---------- Multi-output ----------
        else:
            if target_col is None:
                raise ValueError(
                    "⚠️  Multi-output confusion matrix requires target_col ‼️"
                )

            if not isinstance(self.Y_test, pd.DataFrame):
                y_true_df = pd.DataFrame(self.Y_test)
            else:
                y_true_df = self.Y_test.copy()

            y_pred_df = pd.DataFrame(
                y_pred,
                columns=y_true_df.columns,
                index=y_true_df.index,
            )

            if target_col not in y_true_df.columns:
                raise ValueError(
                    f"⚠️  target_col '{target_col}' not found in Y_test columns ‼️"
                )

            y_true = y_true_df[target_col]
            y_pred = y_pred_df[target_col]
            title_suffix = f"Target: {target_col}"

        # ---------- Confusion matrix ----------
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            row_sum = cm.sum(axis=1, keepdims=True)
            cm = np.divide(
                cm,
                row_sum,
                out=np.zeros_like(cm, dtype=float),
                where=row_sum != 0,
            )

        # ---------- Plot settings ----------
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix ({title_suffix})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()

        # ---------- Annotate settings ----------
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                txt = f"{val:.2f}" if normalize else str(int(val))
                plt.text(j, i, txt, ha="center", va="center")

        plt.tight_layout()

        # ---------- Save and show plot ----------
        if filename is None:
            if target_col is None:
                filename = f"{self.input_model_type}_cm.png"
            else:
                filename = f"{self.input_model_type}_{target_col}_cm.png"

        out = os.path.join(PLOT_DIR, filename)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"💾 Confusion matrix saved path: {out}")

        plt.show()
        plt.close()

        return out

    # -------------------- Save trained model --------------------
    def save_model_joblib(self, filename: str = "knn_model.joblib") -> str:
        """
        Save the trained KNN classification pipeline and related metadata as a joblib artifact.

        The saved artifact contains both the fitted model pipeline and selected
        metadata needed for later inference, reporting, or inspection.

        Saved Contents
        --------------
        The joblib file stores a dictionary containing:
        - ``"model"`` :
        the fitted pipeline object
        - ``"feature_names"`` :
        recorded feature names after preprocessing
        - ``"y_mode"`` :
        target-output mode, either ``"single_output"`` or ``"multi_output"``
        - ``"y_columns"`` :
        target column names when multi-output classification is used

        Parameters
        ----------
        filename : str, default="knn_model.joblib"
            Output filename saved under `MODEL_DIR`.

        Returns
        -------
        str
            Full output path of the saved model artifact.

        Raises
        ------
        ValueError
            If `self.model_pipeline` is not trained or not available.

        Side Effects
        ------------
        - Writes a `.joblib` file to disk
        - Prints the saved file path

        Notes
        -----
        - Saving metadata together with the model makes downstream loading more
        robust and self-descriptive.
        - `y_columns` is only stored when `cleaned_Y_data` is a DataFrame.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️  No trained model to save ‼️")

        out = os.path.join(MODEL_DIR, filename)

        dump(
            {
                "model": self.model_pipeline,
                "feature_names": self.feature_names,
                "y_mode": (
                    "multi_output"
                    if self._is_multi_output(self.cleaned_Y_data)
                    else "single_output"
                ),
                "y_columns": (
                    list(self.cleaned_Y_data.columns)
                    if isinstance(self.cleaned_Y_data, pd.DataFrame)
                    else None
                ),
            },
            out,
        )
        print(f"💾 Model saved path: {out}")
        return out

    # -------------------- Load trained model --------------------
    @classmethod
    def load_model_joblib(cls, filepath: str):
        """
        Load a previously saved KNN classification model artifact from a joblib
        file and restore it as a ``KNNClassifier_Missioner`` instance.

        This class method reconstructs a mission-layer object from a joblib
        artifact created by ``save_model_joblib()``. If the saved file contains
        a metadata dictionary, the method restores both the trained pipeline and
        associated metadata such as feature names and target-output mode. If the
        saved file contains only a model object, that object is assigned
        directly to ``model_pipeline`` and metadata fields are left as ``None``.

        Parameters
        ----------
        filepath : str
            Full path to the saved joblib file.

        Returns
        -------
        KNNClassifier_Missioner
            A restored mission-layer instance containing the loaded trained
            model pipeline and any available metadata.

        Raises
        ------
        FileNotFoundError
            If the specified file path does not exist.

        Restored Attributes
        -------------------
        When the loaded artifact is a dictionary containing a ``"model"`` key,
        the following attributes are restored when available:

        - ``model_pipeline`` :
          Fitted classification pipeline
        - ``feature_names`` :
          Feature names recorded during training
        - ``y_mode`` :
          Target-output mode, such as ``"single_output"`` or ``"multi_output"``
        - ``y_columns`` :
          Target column names for multi-output classification, if stored

        Notes
        -----
        - The restored object is initialized with ``cleaned_X_data=None`` and
          ``cleaned_Y_data=None`` because loading focuses on the trained model
          artifact rather than the original training dataset.
        - If the loaded content is not a dictionary with a ``"model"`` key,
          it is treated as the pipeline object itself.
        - This method prints the loaded file path after successful restoration.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"⚠️ Model file not found: {filepath} ‼️")

        loaded = load(filepath)

        obj = cls(
            cleaned_X_data=None, cleaned_Y_data=None
        )  # Setup object (Missioner) for recording message in this class

        # ---------- Get each properties from saved trained model ----------
        if isinstance(loaded, dict) and "model" in loaded:
            obj.model_pipeline = loaded["model"]
            obj.feature_names = loaded.get("feature_names", None)
            obj.y_mode = loaded.get("y_mode", None)
            obj.y_columns = loaded.get("y_columns", None)
        else:
            obj.model_pipeline = loaded
            obj.feature_names = None
            obj.y_mode = None
            obj.y_columns = None

        print(f"📦 Model loaded path: {filepath}")
        return obj


# =================================================
