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
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

from Zeus.ML_BaseConfigBox.BaseModelConfig import BaseModelConfig

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOT_DIR = os.path.join(BASE_DIR, "ML_Report/KNNCla_Plot")
ROC_DIR = os.path.join(BASE_DIR, "ML_Report/ROC_Plot")
MODEL_DIR = os.path.join(BASE_DIR, "ML_Report/KNNCla_Trained_Model")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------- KNNClassifier Missioner --------------------
class KNNClassifier_Missioner(BaseModelConfig):
    """
    Mission-layer class for K-Nearest Neighbors classification workflows.

    This class extends :class:`BaseModelConfig` and provides shared
    classification utilities for KNN-based model trainers. It serves as the
    mission layer between the shared base-configuration workflow and concrete
    KNN classifier model classes.

    This class does not define the final estimator construction or CV scoring
    dispatch itself. Concrete model-layer classes are responsible for building
    the final estimator and parameter grid, while the base layer handles shared
    training workflow logic, including model-selection scoring dispatch.

    Main Responsibilities
    ---------------------
    - Define the machine learning task type as classification.
    - Define the estimator step name used inside sklearn pipelines.
    - Evaluate trained KNN classification pipelines on train and test splits.
    - Generate classification diagnostic plots, including confusion matrix,
      ROC curve, and Precision-Recall curve.
    - Validate binary-classification plotting requirements for probability-based
      curves.
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
    - ROC curve plot
    - Precision-Recall curve plot

    Notes
    -----
    - Model-selection scoring for both single-output and multi-output
      workflows is handled by the base layer.
    - KNN classifiers do not expose ``feature_importances_``, so no
      feature-importance utility is implemented here.
    - Probability-based plots in this class rely on ``predict_proba()``.
    - ROC and Precision-Recall plotting utilities are intended for binary
      classification only.
    - In multi-output classification, ROC and Precision-Recall plots require
      the user to specify one binary ``target_col``.
    - If the outer pipeline includes preprocessing before the final classifier,
      the mission layer can transform the selected dataset before calling the
      fitted target estimator in multi-output workflows.
    - This mission layer focuses on reusable evaluation, plotting, validation,
      and persistence logic shared across KNN-based classification workflows.

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
        Evaluate the trained classification pipeline on both training and test sets.

        This method generates predictions from the fitted model pipeline and computes
        evaluation metrics according to whether the current classification problem is
        single-output or multi-output.

        Evaluation Behavior
        -------------------
        Single-output classification
            Computes:
            - training accuracy
            - test accuracy
            - training weighted F1 score
            - test weighted F1 score
            - test confusion matrix
            - test classification report
            - first 10 predicted labels, inverse transformed to original class names
            when label encoders are available
            - optional first 10 probability predictions when supported by the model

        Multi-output classification
            Computes:
            - per-target training accuracy
            - per-target test accuracy
            - per-target training weighted F1 score
            - per-target test weighted F1 score
            - per-target confusion matrix
            - per-target classification report
            - macro-average training accuracy across targets
            - macro-average test accuracy across targets
            - macro-average training weighted F1 across targets
            - macro-average test weighted F1 across targets
            - first 10 predicted rows, inverse transformed to original class names
            for encoded target columns when encoders are available
            - optional probability preview per target when supported

        Returns
        -------
        Dict[str, Any]
            Structured evaluation results.

            Single-output keys
            ------------------
            - ``"mode"``
            - ``"train_accuracy"``
            - ``"test_accuracy"``
            - ``"train_f1_weighted"``
            - ``"test_f1_weighted"``
            - ``"confusion_matrix"``
            - ``"classification_report"``
            - ``"prediction_preview_first10"``
            - ``"probability_preview_first10"``

            Multi-output keys
            -----------------
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
            If ``self.model_pipeline`` has not been trained.

        Side Effects
        ------------
        Updates the following instance attributes:
        - ``self.y_train_pred``
        - ``self.y_test_pred``
        - ``self.prediction_preview``

        Notes
        -----
        - In single-output classification, original class names are used in
        ``classification_report`` when available through stored label encoders.
        - In multi-output classification, class names are resolved independently for
        each target column.
        - Metric computation is performed on encoded or native numeric targets as
        stored internally; inverse transformation is used only for display-facing
        prediction previews.
        - Probability previews are returned only when the fitted pipeline exposes
        ``predict_proba()``.
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
            class_names = self._get_target_class_names()

            if class_names is not None:
                label_ids = list(range(len(class_names)))
                cm = confusion_matrix(y_true_test, y_pred_test, labels=label_ids)
                report = classification_report(
                    y_true_test,
                    y_pred_test,
                    labels=label_ids,
                    target_names=[str(x) for x in class_names],
                    digits=4,
                    zero_division=0,
                )
            else:
                cm = confusion_matrix(y_true_test, y_pred_test)
                report = classification_report(
                    y_true_test,
                    y_pred_test,
                    digits=4,
                    zero_division=0,
                )

            self.prediction_preview = self._inverse_transform_target_labels(
                pd.Series(y_pred_test[:10])
            )

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
            class_names = self._get_target_class_names(col)

            if class_names is not None:
                label_ids = list(range(len(class_names)))
                report = classification_report(
                    yt_te,
                    yp_te,
                    labels=label_ids,
                    target_names=[str(x) for x in class_names],
                    digits=4,
                    zero_division=0,
                )
                cm = confusion_matrix(yt_te, yp_te, labels=label_ids)
            else:
                report = classification_report(
                    yt_te,
                    yp_te,
                    digits=4,
                    zero_division=0,
                )
                cm = confusion_matrix(yt_te, yp_te)

            per_target[col] = {
                "train_accuracy": acc_tr,
                "test_accuracy": acc_te,
                "train_f1_weighted": f1_tr,
                "test_f1_weighted": f1_te,
                "classification_report": report,
                "confusion_matrix": cm,
            }
        self.prediction_preview = self._inverse_transform_target_labels(
            y_pred_test_df.head(10)
        )

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
            "prediction_preview_first10_rows": self.prediction_preview,
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

    # -------------------- Get binary KNN plot inputs --------------------
    def _get_binary_knn_plot_inputs(
        self,
        dataset: str = "test",
        preview_rows: int = 10,
        target_col: Optional[str] = None,
    ):
        """
        Validate and prepare binary-classification inputs for KNN probability-based plots.

        This helper prepares the selected dataset split, validates binary-target
        requirements, and returns the feature matrix, selected target values,
        preview subset, positive-class score array, and metadata needed by KNN
        ROC and Precision-Recall plotting methods.

        Supported Workflows
        -------------------
        1. Single-output binary classification
        The helper uses the selected train/test split directly and obtains class
        probabilities from ``self.model_pipeline.predict_proba(...)``.

        2. Multi-output classification with a selected binary target
        The helper requires ``target_col``. It selects the corresponding target
        column, validates that it is binary, and then extracts probabilities from
        the fitted estimator associated with that target.

        Dataset Selection
        -----------------
        - ``dataset="train"`` -> uses ``self.X_train`` and ``self.Y_train``
        - ``dataset="test"``  -> uses ``self.X_test`` and ``self.Y_test``

        Probability Source
        ------------------
        - Single-output:
        probabilities are produced directly by the fitted outer pipeline.
        - Multi-output:
        probabilities are produced by the selected fitted target estimator.

        If the outer pipeline contains preprocessing steps before the final
        classifier, the helper first applies:

        ``self.model_pipeline[:-1].transform(X_used)``

        and then calls ``predict_proba()`` on the selected target estimator.

        Parameters
        ----------
        dataset : str, default="test"
            Dataset split to use.

            Accepted values:
            - ``"train"``
            - ``"test"``

        preview_rows : int, default=10
            Number of rows included in the preview feature subset returned as
            ``X_preview``.

        target_col : str or None, default=None
            Target column name used only for multi-output classification.

            - Ignored for single-output classification
            - Required for multi-output ROC / PR workflows

        Returns
        -------
        tuple
            A tuple containing:

            - ``X_used`` : pd.DataFrame or np.ndarray
                Full selected feature matrix.
            - ``y_used`` : pd.Series, np.ndarray, or pd.DataFrame column
                Selected target values for the plotting workflow.
            - ``X_preview`` : pd.DataFrame or np.ndarray
                Preview subset of the selected feature matrix.
            - ``y_score`` : np.ndarray
                Positive-class probability scores used for ROC / PR plotting.
            - ``classes`` : np.ndarray
                Sorted unique class labels of the selected target.
            - ``dataset`` : str
                Echoed dataset split value.
            - ``target_col`` : str or None
                Echoed target-column value.

        Raises
        ------
        ValueError
            If:

            - ``self.model_pipeline`` is not trained,
            - ``dataset`` is not ``"train"`` or ``"test"``,
            - the requested dataset split is unavailable,
            - the pipeline or selected target estimator does not expose
            ``predict_proba()``,
            - the selected target is not binary,
            - multi-output classification is detected but ``target_col`` is missing,
            - or ``target_col`` does not exist in the current target data.

        Notes
        -----
        - This helper is intended for KNN probability-based binary plotting
        utilities such as ROC curves and Precision-Recall curves.
        - The returned positive-class score is taken from ``proba[:, 1]`` after
        confirming that the selected target is binary.
        - In multi-output classification, only the selected binary target is used.
        - ``classes[1]`` is expected to represent the positive class in downstream
        plotting methods.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ Model pipeline not trained yet ‼️")

        if dataset not in {"train", "test"}:
            raise ValueError("⚠️ dataset must be 'train' or 'test' ‼️")

        if dataset == "train":
            X_used = self.X_train
            y_used = self.Y_train
        else:
            X_used = self.X_test
            y_used = self.Y_test

        if X_used is None or y_used is None:
            raise ValueError("⚠️ Requested dataset split is not available ‼️")

        X_preview = X_used[:preview_rows]

        if not hasattr(self.model_pipeline, "predict_proba"):
            raise ValueError("⚠️ Current KNN pipeline has no predict_proba() ‼️")

        # ---------- Single-output ----------
        if not self._is_multi_output(y_used):
            classes = np.unique(y_used)
            if len(classes) != 2:
                raise ValueError(
                    "⚠️ Precision-Recall plot requires binary classification ‼️"
                )

            proba = self.model_pipeline.predict_proba(X_used)
            y_score = proba[:, 1]

            return X_used, y_used, X_preview, y_score, classes, dataset, target_col

        # ---------- Multi-output ----------
        if target_col is None:
            raise ValueError(
                "⚠️ target_col is required for multi-output classification ‼️"
            )

        if not isinstance(y_used, pd.DataFrame):
            y_used_df = pd.DataFrame(y_used)
        else:
            y_used_df = y_used.copy()

        if target_col not in y_used_df.columns:
            raise ValueError(f"⚠️ target_col '{target_col}' not found in Y data ‼️")

        target_idx = list(y_used_df.columns).index(target_col)
        y_target = y_used_df[target_col]

        classes = np.unique(y_target)
        if len(classes) != 2:
            raise ValueError(
                f"⚠️ target_col '{target_col}' is not binary classification ‼️"
            )

        clf = self.model_pipeline.named_steps.get(self.step_name, self.model_pipeline)

        if not hasattr(clf, "estimators_"):
            raise ValueError("⚠️ Multi-output classifier has no fitted estimators_ ‼️")

        base_est = clf.estimators_[target_idx]
        if not hasattr(base_est, "predict_proba"):
            raise ValueError("⚠️ Target estimator has no predict_proba() ‼️")

        if hasattr(self.model_pipeline, "steps") and len(self.model_pipeline.steps) > 1:
            X_trans = self.model_pipeline[:-1].transform(X_used)
            proba = base_est.predict_proba(X_trans)
        else:
            proba = base_est.predict_proba(X_used)

        y_score = proba[:, 1]

        return X_used, y_target, X_preview, y_score, classes, dataset, target_col

    # -------------------- ROC curve plot --------------------
    def roc_curve_plot_engine(
        self,
        dataset: str = "test",
        filename: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Plot and save the ROC curve for binary KNN classification.

        This method computes and plots the Receiver Operating Characteristic (ROC)
        curve using positive-class probabilities returned by ``predict_proba()``.
        It supports:

        - single-output binary classification, or
        - multi-output classification when a binary ``target_col`` is specified

        Parameters
        ----------
        dataset : str, default="test"
            Dataset split to use.

            Accepted values:
            - ``"train"``
            - ``"test"``

        filename : str or None, default=None
            Output filename for the saved image.

            - If provided, that name is used directly.
            - If ``None``, an automatic filename is generated based on model type,
            dataset split, and optional target column.

        target_col : str or None, default=None
            Target column name used only for multi-output classification.

            - Ignored for single-output classification
            - Required when plotting a ROC curve for a binary target from a
            multi-output classifier

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:

            - ``"saved_path"`` : str
                Full saved file path.
            - ``"roc_auc"`` : float
                Area under the ROC curve.

        Raises
        ------
        ValueError
            Propagated from ``_get_binary_knn_plot_inputs()`` when the task or
            selected target does not satisfy binary ROC plotting requirements.

        Side Effects
        ------------
        - Saves the figure into ``ROC_DIR``.
        - Displays the figure using matplotlib.
        - Prints the saved file path.

        Notes
        -----
        - The positive class is chosen as ``classes[1]`` after sorting the unique
        labels of the selected target.
        - The diagonal reference line represents random-guessing performance.
        - A higher ROC AUC indicates better ranking and binary class-separation
        ability.
        - In multi-output classification, the ROC curve is generated only for the
        selected binary target.
        """
        _, y_used, _, y_score, classes, dataset, target_col = (
            self._get_binary_knn_plot_inputs(
                dataset=dataset,
                preview_rows=10,
                target_col=target_col,
            )
        )

        y_true_arr = np.asarray(y_used)
        pos_label = classes[1]

        fpr, tpr, _ = roc_curve(
            y_true_arr,
            y_score,
            pos_label=pos_label,
        )
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        title_suffix = f" | target={target_col}" if target_col is not None else ""
        plt.title(f"KNN | {dataset} | ROC Curve{title_suffix}")
        plt.legend(loc="lower right")
        plt.tight_layout()

        if filename is None:
            if target_col is None:
                filename = f"{self.input_model_type}_{dataset}_roc_curve.png"
            else:
                filename = (
                    f"{self.input_model_type}_{dataset}_{target_col}_roc_curve.png"
                )

        out = os.path.join(ROC_DIR, filename)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"💾 ROC curve saved path: {out}")

        plt.show()
        plt.close()

        return {
            "saved_path": out,
            "roc_auc": float(roc_auc),
        }

    # -------------------- Precision recall curve plot --------------------
    def precision_recall_curve_plot_engine(
        self,
        dataset: str = "test",
        filename: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Plot and save the Precision-Recall curve for binary KNN classification.

        This method computes and plots the Precision-Recall (PR) curve using
        positive-class probabilities returned by ``predict_proba()``. It supports:

        - single-output binary classification, or
        - multi-output classification when a binary ``target_col`` is specified

        The PR curve is especially informative for imbalanced binary classification
        problems because it focuses directly on positive-class retrieval quality.

        Parameters
        ----------
        dataset : str, default="test"
            Dataset split to use.

            Accepted values:
            - ``"train"``
            - ``"test"``

        filename : str or None, default=None
            Output filename for the saved image.

            - If provided, that name is used directly.
            - If ``None``, an automatic filename is generated based on model type,
            dataset split, and optional target column.

        target_col : str or None, default=None
            Target column name used only for multi-output classification.

            - Ignored for single-output classification
            - Required when plotting a PR curve for a binary target from a
            multi-output classifier

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:

            - ``"saved_path"`` : str
                Full saved file path.
            - ``"pr_auc"`` : float
                Area under the Precision-Recall curve.

        Raises
        ------
        ValueError
            Propagated from ``_get_binary_knn_plot_inputs()`` when the task or
            selected target does not satisfy binary Precision-Recall plotting
            requirements.

        Side Effects
        ------------
        - Saves the figure into ``PLOT_DIR``.
        - Displays the figure using matplotlib.
        - Prints the saved file path.

        Notes
        -----
        - The positive class is chosen as ``classes[1]`` after sorting the unique
        labels of the selected target.
        - PR curves are often more sensitive than ROC curves under heavy class
        imbalance because they emphasize positive-class retrieval behavior.
        - The returned AUC is computed over the recall-precision curve.
        - In multi-output classification, the PR curve is generated only for the
        selected binary target.
        """
        _, y_used, _, y_score, classes, dataset, target_col = (
            self._get_binary_knn_plot_inputs(
                dataset=dataset,
                preview_rows=10,
                target_col=target_col,
            )
        )

        y_true_arr = np.asarray(y_used)
        pos_label = classes[1]

        precision, recall, _ = precision_recall_curve(
            y_true_arr,
            y_score,
            pos_label=pos_label,
        )
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        title_suffix = f" | target={target_col}" if target_col is not None else ""
        plt.title(f"KNN | {dataset} | Precision-Recall Curve{title_suffix}")
        plt.legend(loc="lower left")
        plt.tight_layout()

        if filename is None:
            if target_col is None:
                filename = f"{self.input_model_type}_{dataset}_pr_curve.png"
            else:
                filename = (
                    f"{self.input_model_type}_{dataset}_{target_col}_pr_curve.png"
                )

        out = os.path.join(PLOT_DIR, filename)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"💾 PR curve saved path: {out}")

        plt.show()
        plt.close()

        return {
            "saved_path": out,
            "pr_auc": float(pr_auc),
        }

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
