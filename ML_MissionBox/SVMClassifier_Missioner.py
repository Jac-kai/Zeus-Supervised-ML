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
PLOT_DIR = os.path.join(BASE_DIR, "ML_Report/SVC_Plot")
ROC_DIR = os.path.join(BASE_DIR, "ML_Report/ROC_Plot")
MODEL_DIR = os.path.join(BASE_DIR, "ML_Report/SVMCla_Trained_Model")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------- SVMClassifier Missioner --------------------
class SVMClassifier_Missioner(BaseModelConfig):
    """
    Mission-layer class for Support Vector Machine classification workflows.

    This class extends :class:`BaseModelConfig` and provides shared
    classification utilities for SVC-based model trainers. It serves as the
    mission layer between the shared base-configuration workflow and concrete
    SVC model classes.

    This class does not define the final estimator construction or CV scoring
    dispatch itself. Concrete model-layer classes are responsible for building
    the final estimator and parameter grid, while the base layer handles shared
    training workflow logic, including model-selection scoring dispatch.

    Main Responsibilities
    ---------------------
    - Define the machine learning task type as classification.
    - Define the estimator step name used inside sklearn pipelines.
    - Evaluate trained SVC pipelines on train and test data.
    - Generate classification diagnostic plots, including confusion matrix,
      decision-function distribution, ROC curve, and Precision-Recall curve.
    - Inspect SVC-specific learned attributes such as support vectors,
      coefficients, intercepts, and decision scores.
    - Validate binary-classification plotting requirements for
      decision-function-based diagnostic curves.
    - Save and load trained model artifacts with metadata.

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
    - Prediction preview
    - Optional probability preview per target when supported

    Diagnostic Plot Utilities
    -------------------------
    - Confusion matrix plot
    - Decision-function distribution plot
    - ROC curve plot
    - Precision-Recall curve plot

    Model Inspection Utilities
    --------------------------
    - SVC model insight summary
    - Support-vector metadata preview
    - Decision-function preview
    - Coefficient and intercept inspection when available

    Notes
    -----
    - Model-selection scoring for both single-output and multi-output
      workflows is handled by the base layer.
    - SVC does not provide ``feature_importances_``, so no native
      feature-importance utility is implemented here.
    - Decision-function-based diagnostic plots in this class are intended for
      binary classification only.
    - In multi-output classification, binary plots and SVC insight utilities
      require one selected ``target_col``.
    - When the outer pipeline includes preprocessing before the classifier,
      the mission layer can transform the selected dataset before calling the
      fitted target estimator in multi-output workflows.
    - Probability previews are returned only when the underlying pipeline or
      estimator exposes ``predict_proba()``. For standard SVC, this usually
      requires ``probability=True`` during model creation.
    - This mission layer focuses on reusable evaluation, plotting,
      model-inspection, validation, and persistence logic shared across
      SVC-based classification workflows.

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
        Model label used for filenames, reporting, and output records.

    y_train_pred, y_test_pred : optional
        Cached train/test predictions produced during evaluation.

    prediction_preview : optional
        Cached prediction preview for reporting or debugging.
    """

    # -------------------- Classification task --------------------
    @property
    def task(self) -> str:
        """
        Return the machine learning task type for this mission class.

        This property is required by :class:`BaseModelConfig` so that the base
        layer can apply task-specific validation, split logic, scoring choices,
        and workflow behavior.

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

        This step name is used to access the fitted classifier from
        ``self.model_pipeline.named_steps`` and to define parameter grid keys
        when GridSearchCV is used.

        Returns
        -------
        str
            Always returns ``"classifier"``.

        Notes
        -----
        Hyperparameter grid keys for GridSearchCV must use this prefix, for example:
        - ``"classifier__C"``
        - ``"classifier__kernel"``
        - ``"classifier__gamma"``
        """
        return "classifier"

    # -------------------- Model evaluation --------------------
    def model_evaluation_engine(self) -> Dict[str, Any]:
        """
        Evaluate a trained SVC classification pipeline on both train and test sets.

        This method generates predictions from the fitted pipeline and computes
        classification metrics according to whether the problem is single-output
        or multi-output.

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
            - macro average train/test accuracy across targets
            - macro average train/test weighted F1 across targets
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
            If `self.model_pipeline` is not fitted or not available.

        Side Effects
        ------------
        Updates the following instance attributes:
        - ``self.y_train_pred``
        - ``self.y_test_pred``
        - ``self.prediction_preview``

        Notes
        -----
        - The method assumes that train/test split data already exist in the
          object, typically prepared by base-layer logic.
        - Probability previews are only returned if the pipeline exposes
          ``predict_proba()``.
        - For multi-output classification, predicted outputs are converted to
          DataFrame form to preserve target-column alignment.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️  Model pipeline not trained yet ‼️")

        self.y_train_pred = self.model_pipeline.predict(self.X_train)
        self.y_test_pred = self.model_pipeline.predict(self.X_test)

        y_true_train = self.Y_train
        y_true_test = self.Y_test
        y_pred_train = self.y_train_pred
        y_pred_test = self.y_test_pred

        # ---------- Single-output ----------
        if not self._is_multi_output(y_true_test):
            # ---------- Acccuracy ----------
            train_acc = accuracy_score(y_true_train, y_pred_train)
            test_acc = accuracy_score(y_true_test, y_pred_test)

            # ---------- F1 score ----------
            train_f1 = f1_score(y_true_train, y_pred_train, average="weighted")
            test_f1 = f1_score(y_true_test, y_pred_test, average="weighted")

            # ---------- Confusion matrix and classification report ----------
            cm = confusion_matrix(y_true_test, y_pred_test)
            report = classification_report(y_true_test, y_pred_test, digits=4)

            # ---------- Probability preview ----------
            self.prediction_preview = np.array(y_pred_test[:10])  # Array

            proba_preview = None
            if hasattr(self.model_pipeline, "predict_proba"):
                proba_preview = self.model_pipeline.predict_proba(self.X_test[:10])

            return {
                "mode": "single_output",
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_f1_weighted": train_f1,
                "test_f1_weighted": test_f1,
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

        # ---------- Predicted value from train / test dataset ----------
        y_pred_train_df = pd.DataFrame(
            y_pred_train, columns=y_true_train.columns, index=y_true_train.index
        )
        y_pred_test_df = pd.DataFrame(
            y_pred_test, columns=y_true_test.columns, index=y_true_test.index
        )

        per_target = {}
        acc_list_train, acc_list_test, f1_list_train, f1_list_test = [], [], [], []

        # ---------- Predicted and true values from train / test dataset for each target ----------
        for col in y_true_test.columns:
            yt_tr = y_true_train[col]
            yp_tr = y_pred_train_df[col]
            yt_te = y_true_test[col]
            yp_te = y_pred_test_df[col]

            acc_tr = accuracy_score(yt_tr, yp_tr)
            acc_te = accuracy_score(yt_te, yp_te)
            f1_tr = f1_score(yt_tr, yp_tr, average="weighted")
            f1_te = f1_score(yt_te, yp_te, average="weighted")

            acc_list_train.append(acc_tr)
            acc_list_test.append(acc_te)
            f1_list_train.append(f1_tr)
            f1_list_test.append(f1_te)

            per_target[col] = {
                "train_accuracy": acc_tr,
                "test_accuracy": acc_te,
                "train_f1_weighted": f1_tr,
                "test_f1_weighted": f1_te,
                "classification_report": classification_report(yt_te, yp_te, digits=4),
                "confusion_matrix": confusion_matrix(yt_te, yp_te),
            }

        # ---------- Probability from train / test dataset for each target ----------
        self.prediction_preview = y_pred_test_df.head(10)  # DataFrame

        proba_preview_per_target = None
        if hasattr(self.model_pipeline, "predict_proba"):
            try:
                proba_raw = self.model_pipeline.predict_proba(self.X_test[:10])

                if isinstance(proba_raw, list):
                    proba_preview_per_target = {
                        col: proba_raw[idx]
                        for idx, col in enumerate(y_true_test.columns)
                    }
                else:
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

    # -------------------- SVC insight analysis pre-check (helper) --------------------
    def _get_svm_core_inputs(
        self,
        dataset: str = "test",
        preview_rows: int = 10,
        target_col: str | None = None,
    ):
        """
        Validate and prepare core inputs required for SVC insight methods.

        This helper supports:
        - single-output SVC insight directly
        - multi-output SVC insight for one selected target column

        Parameters
        ----------
        dataset : str, default="test"
            Dataset split to use. Accepted values: "train" or "test".
        preview_rows : int, default=10
            Number of rows to include in the preview subset returned as X_preview.
        target_col : str | None, optional
            Target column name required for multi-output classification insight.

        Returns
        -------
        tuple
            A tuple containing:

            - clf : fitted classifier object
            - X_used : full feature dataset
            - y_used : selected target dataset
            - X_preview : preview feature dataset
            - dataset : normalized dataset label
            - target_col : selected target column name or None
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️  Model pipeline not trained yet ‼️")

        dataset = dataset.lower().strip()
        if dataset not in ["test", "train"]:
            raise ValueError("⚠️  dataset must be 'test' or 'train' ‼️")

        clf = self.model_pipeline.named_steps.get(self.step_name)
        if clf is None:
            raise ValueError("⚠️  Pipeline has no classifier step ‼️")

        X_used = self.X_test if dataset == "test" else self.X_train
        y_used = self.Y_test if dataset == "test" else self.Y_train

        if hasattr(X_used, "iloc"):
            X_preview = X_used.iloc[:preview_rows]
        else:
            X_preview = X_used[:preview_rows]

        # ---------- Single-output ----------
        if not self._is_multi_output(y_used):
            return clf, X_used, y_used, X_preview, dataset, None

        # ---------- Multi-output ----------
        if target_col is None:
            raise ValueError("⚠️  Multi-output SVC insight requires target_col ‼️")

        if not isinstance(y_used, pd.DataFrame):
            y_used_df = pd.DataFrame(y_used)
        else:
            y_used_df = y_used.copy()

        if target_col not in y_used_df.columns:
            raise ValueError(
                f"⚠️  target_col '{target_col}' not found in selected target columns ‼️"
            )

        if not hasattr(clf, "estimators_"):
            raise ValueError("⚠️  Multi-output classifier has no fitted estimators_ ‼️")

        return clf, X_used, y_used_df[target_col], X_preview, dataset, target_col

    # -------------------- SVC insight analysis --------------------
    def svm_model_insight_engine(
        self,
        dataset: str = "test",
        preview_rows: int = 10,
        target_col: str | None = None,
    ):
        """
        Collect and return internal inspection information for the current SVC model.

        This method inspects the fitted SVC classifier stored in the current model
        pipeline and summarizes key learned attributes such as class labels, support
        vector counts, support-vector shapes, coefficients, intercept values, and a
        preview of decision-function scores.

        Supported Modes
        ---------------
        Single-output classification
            The fitted classifier step is inspected directly.

        Multi-output classification
            ``target_col`` must be provided. The method then selects the corresponding
            fitted estimator from the multi-output wrapper and returns insight for that
            specific target only.

        Parameters
        ----------
        dataset : str, default="test"
            Dataset split used when building the decision-function preview.

            Accepted values:
            - ``"train"``
            - ``"test"``

        preview_rows : int, default=10
            Number of leading rows used to generate the decision-function preview.

        target_col : str or None, default=None
            Target column name used only for multi-output classification.

            - Ignored for single-output classification
            - Required for multi-output classification insight
            - Must match one of the selected target columns

        Returns
        -------
        dict
            Dictionary containing SVC inspection results.

            Keys may include:
            - ``"dataset"``
            - ``"target_col"``
            - ``"classes_"``
            - ``"n_support_"``
            - ``"support_indices_preview_first20"``
            - ``"support_vector_shape"``
            - ``"coef_shape"``
            - ``"intercept_"``
            - ``"decision_function_preview_first10"``

        Raises
        ------
        ValueError
            Propagated from ``_get_svm_core_inputs()`` if:

            - the model pipeline is not trained,
            - the dataset label is invalid,
            - the classifier step cannot be found,
            - multi-output classification requires ``target_col`` but none is given,
            - ``target_col`` does not exist,
            - or the multi-output classifier does not expose fitted estimators.

        Notes
        -----
        - For multi-output classification, insight is generated for one selected target
        column at a time.
        - If the underlying estimator exposes ``coef_`` or ``intercept_``, their shapes
        or values are included in the returned summary.
        - Decision-function preview is attempted inside a ``try`` block. If preview
        generation fails, ``"decision_function_preview_first10"`` is returned as
        ``None``.
        - When preprocessing steps exist before the classifier in the pipeline, the
        preview data are transformed through ``self.model_pipeline[:-1]`` before
        calling the selected estimator's ``decision_function()`` in multi-output mode.
        """
        clf, _, _, X_preview, dataset, target_col = self._get_svm_core_inputs(
            dataset=dataset,
            preview_rows=preview_rows,
            target_col=target_col,
        )

        # ---------- Choose estimator ----------
        est = clf
        if target_col is not None:
            y_df = self.Y_test if dataset == "test" else self.Y_train
            if not isinstance(y_df, pd.DataFrame):
                y_df = pd.DataFrame(y_df)

            target_idx = list(y_df.columns).index(target_col)
            est = clf.estimators_[target_idx]

        result = {
            "dataset": dataset,
            "target_col": target_col,
            "classes_": getattr(est, "classes_", None),
            "n_support_": getattr(est, "n_support_", None),
            "support_indices_preview_first20": (
                est.support_[:20].tolist() if hasattr(est, "support_") else None
            ),
            "support_vector_shape": (
                tuple(est.support_vectors_.shape)
                if hasattr(est, "support_vectors_")
                else None
            ),
            "coef_shape": (tuple(est.coef_.shape) if hasattr(est, "coef_") else None),
            "intercept_": (
                est.intercept_.tolist() if hasattr(est, "intercept_") else None
            ),
        }

        # ---------- Decision function preview ----------
        try:
            if target_col is None:
                decision_preview = self.model_pipeline.decision_function(X_preview)
            else:
                if (
                    hasattr(self.model_pipeline, "steps")
                    and len(self.model_pipeline.steps) > 1
                ):
                    X_trans = self.model_pipeline[:-1].transform(X_preview)
                    decision_preview = est.decision_function(X_trans)
                else:
                    decision_preview = est.decision_function(X_preview)

            if hasattr(decision_preview, "tolist"):
                decision_preview = decision_preview.tolist()

            result["decision_function_preview_first10"] = decision_preview[:10]

        except Exception:
            result["decision_function_preview_first10"] = None

        return result

    # -------------------- Confusion matrix plot --------------------
    def confusion_matrix_plot_engine(
        self,
        normalize: bool = False,
        filename: Optional[str] = None,
        target_col: Optional[str] = None,
    ):
        """
        Plot and save a confusion matrix for SVC classification results.

        This method supports both single-output and multi-output classification.

        Supported Modes
        ---------------
        Single-output classification
            The confusion matrix is plotted directly from `self.Y_test`
            and predictions on `self.X_test`.

        Multi-output classification
            A specific target column must be provided through `target_col`.
            The confusion matrix is then generated only for that selected target.

        Parameters
        ----------
        normalize : bool, default=False
            Whether to normalize the confusion matrix row-wise.

            - If False, raw counts are displayed.
            - If True, each row is divided by its row sum and displayed as
              proportions.

        filename : str or None, default=None
            Output filename for the saved image.

            - If provided, that name is used directly.
            - If None, an automatic filename is generated based on the model type
              and target information.

        target_col : str or None, default=None
            Target column name used only for multi-output classification.

            - Required when `self.Y_test` is multi-output.
            - Ignored for single-output classification.

        Raises
        ------
        ValueError
            If:
            - `self.model_pipeline` is not trained,
            - multi-output mode is detected but `target_col` is missing,
            - `target_col` does not exist in `self.Y_test`.

        Side Effects
        ------------
        - Saves the generated figure into `PLOT_DIR`.
        - Displays the figure using matplotlib.
        - Prints the saved file path.

        Notes
        -----
        - The plot uses `plt.imshow()` for rendering.
        - Cell values are annotated directly on the matrix.
        - In normalization mode, cells are shown with 2 decimal places.
        - In raw-count mode, cells are shown as integers.
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

            # ---------- Y integrated into DataFrame ----------
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
        plt.title(f"Confusion Matrix | {title_suffix}")
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
            if self._is_multi_output(self.Y_test):
                filename = f"{self.input_model_type}_{target_col}_cm.png"
            else:
                filename = f"{self.input_model_type}_cm.png"

        out = os.path.join(PLOT_DIR, filename)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"💾 Confusion matrix saved path: {out}")

        plt.show()
        plt.close()

        return out

    # -------------------- Binary plot pre-check (helper) --------------------
    def _get_binary_svc_plot_inputs(
        self,
        dataset: str = "test",
        preview_rows: int = 10,
        target_col: Optional[str] = None,
    ):
        """
        Validate and prepare inputs required for binary SVC plotting methods.

        This helper centralizes the common pre-check and input-preparation logic used
        by binary SVC plotting utilities such as:

        - decision-function distribution plots
        - ROC curves
        - Precision-Recall curves

        It supports both:

        1. Single-output binary classification
        2. Multi-output classification when a binary target column is explicitly
        selected through ``target_col``

        Validation Behavior
        -------------------
        The method ensures that:

        - a trained model pipeline exists,
        - the requested dataset split is valid,
        - the pipeline contains the classifier step,
        - the selected classification target is binary,
        - the relevant estimator exposes ``decision_function()``.

        For multi-output classification, the method additionally:

        - requires ``target_col``,
        - validates that ``target_col`` exists in the selected target set,
        - selects the matching fitted estimator from the multi-output wrapper,
        - computes decision-function scores only for that chosen target.

        Parameters
        ----------
        dataset : str, default="test"
            Dataset split to use.

            Accepted values:
            - ``"train"``
            - ``"test"``

            The input is normalized using ``lower().strip()``.

        preview_rows : int, default=10
            Number of leading rows returned as preview feature data.

        target_col : str or None, default=None
            Target column name used only for multi-output classification.

            - Ignored for single-output classification
            - Required for multi-output binary plotting
            - Must refer to a valid binary target column in the selected dataset split

        Returns
        -------
        tuple
            A tuple containing:

            - X_used : pd.DataFrame or np.ndarray
                Full feature matrix for the selected dataset split.
            - y_used : pd.Series, pd.DataFrame, or np.ndarray
                Target labels used for plotting.

                - For single-output classification, this is the full single target.
                - For multi-output classification, this is the selected target column
                only.
            - X_preview : pd.DataFrame or np.ndarray
                First ``preview_rows`` rows of ``X_used``.
            - y_score : np.ndarray
                Decision-function scores corresponding to the selected target.
            - classes : np.ndarray
                Sorted unique class labels detected in the selected target.
            - dataset : str
                Normalized dataset label, either ``"train"`` or ``"test"``.
            - target_col : str or None
                Selected target column name for multi-output classification, otherwise
                ``None`` for single-output classification.

        Raises
        ------
        ValueError
            If:

            - ``self.model_pipeline`` is not trained,
            - ``dataset`` is not ``"train"`` or ``"test"``,
            - the pipeline does not contain the classifier step,
            - multi-output classification is detected but ``target_col`` is missing,
            - ``target_col`` does not exist in the selected target columns,
            - the selected target is not binary,
            - the relevant estimator does not expose ``decision_function()``.

        Notes
        -----
        - This helper is intended for internal use only.
        - In multi-output classification, binary plotting is performed for one target
        at a time.
        - For multi-output pipelines, the helper extracts the fitted estimator from
        ``clf.estimators_`` using the position of ``target_col``.
        - When the pipeline contains preprocessing steps before the classifier, the
        selected feature matrix is transformed through ``self.model_pipeline[:-1]``
        before computing the target-specific decision function.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️  Model pipeline not trained yet ‼️")

        # ---------- Record train / test dataset ----------
        dataset = dataset.lower().strip()
        if dataset not in ["train", "test"]:
            raise ValueError("⚠️  dataset must be 'train' or 'test' ‼️")

        if dataset == "train":
            X_used = self.X_train
            y_used = self.Y_train
        else:
            X_used = self.X_test
            y_used = self.Y_test

        # ---------- Check X format ----------
        if hasattr(X_used, "iloc"):
            X_preview = X_used.iloc[:preview_rows]
        else:
            X_preview = X_used[:preview_rows]

        # ---------- Get pipeline estimater ----------
        clf = self.model_pipeline.named_steps.get(self.step_name)
        if clf is None:
            raise ValueError("⚠️  Pipeline has no classifier step ‼️")

        # ---------- Single-output ----------
        if not self._is_multi_output(y_used):
            classes = np.unique(y_used)
            if len(classes) != 2:
                raise ValueError(
                    "⚠️  This plot currently supports binary classification only ‼️"
                )

            if not hasattr(self.model_pipeline, "decision_function"):
                raise ValueError("⚠️  Pipeline has no decision_function() ‼️")

            y_score = self.model_pipeline.decision_function(X_used)

            return X_used, y_used, X_preview, y_score, classes, dataset, None

        # ---------- Multi-output ----------
        if target_col is None:
            raise ValueError("⚠️  Multi-output binary plots require target_col ‼️")

        if not isinstance(y_used, pd.DataFrame):
            y_used_df = pd.DataFrame(y_used)
        else:
            y_used_df = y_used.copy()

        if target_col not in y_used_df.columns:
            raise ValueError(
                f"⚠️  target_col '{target_col}' not found in selected target columns ‼️"
            )

        target_idx = list(y_used_df.columns).index(
            target_col
        )  # Get target column index
        y_target = y_used_df[target_col]

        # ---------- Check binary values in target column ----------
        classes = np.unique(y_target)
        if len(classes) != 2:
            raise ValueError(
                f"⚠️  target_col '{target_col}' is not binary classification ‼️"
            )

        if not hasattr(clf, "estimators_"):
            raise ValueError("⚠️  Multi-output classifier has no fitted estimators_ ‼️")

        # ---------- Get estimator ----------
        base_est = clf.estimators_[target_idx]
        if not hasattr(base_est, "decision_function"):
            raise ValueError("⚠️  Target estimator has no decision_function() ‼️")

        # ---------- Get preprocessed X and record decision function ----------
        if hasattr(self.model_pipeline, "steps") and len(self.model_pipeline.steps) > 1:
            X_trans = self.model_pipeline[:-1].transform(
                X_used
            )  # Excluding estimator step in pipeline
            y_score = base_est.decision_function(X_trans)
        else:
            y_score = base_est.decision_function(X_used)

        return X_used, y_target, X_preview, y_score, classes, dataset, target_col

    # -------------------- Decision function distribution plot --------------------
    def decision_function_distribution_plot_engine(
        self,
        dataset: str = "test",
        bins: int = 30,
        alpha: float = 0.6,
        filename: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> str:
        """
        Plot and save decision-function score distributions for binary SVC classification.

        This method visualizes how decision-function scores are distributed for each
        true class in a binary classification problem. It can be used for:

        - single-output binary classification, or
        - multi-output classification when a binary ``target_col`` is specified

        The plot helps assess whether the classifier separates the two classes well in
        decision-score space. Greater separation between class-specific score
        distributions usually indicates stronger discrimination.

        Parameters
        ----------
        dataset : str, default="test"
            Dataset split to use.

            Accepted values:
            - ``"train"``
            - ``"test"``

        bins : int, default=30
            Number of histogram bins used for the score distributions.

        alpha : float, default=0.6
            Transparency level for the histograms.

        filename : str or None, default=None
            Output filename for the saved image.

            - If provided, that name is used directly.
            - If None, an automatic filename is generated based on model type,
            dataset split, and optional target column.

        target_col : str or None, default=None
            Target column name used only for multi-output classification.

            - Ignored for single-output classification
            - Required when plotting a binary target from a multi-output classifier

        Returns
        -------
        str
            Full saved file path of the generated plot image.

        Raises
        ------
        ValueError
            Propagated from ``_get_binary_svc_plot_inputs()`` when the task or
            selected target does not satisfy binary SVC plotting requirements.

        Side Effects
        ------------
        - Saves the figure into ``PLOT_DIR``.
        - Displays the figure using matplotlib.
        - Prints the saved file path.

        Notes
        -----
        - A vertical reference line is drawn at decision score ``0.0``, which is the
        standard separating threshold for binary SVC decision scores.
        - In multi-output classification, the plot is generated for the selected
        binary target only.
        - Histogram labels are based on the true class values in the selected target.
        """
        # ---------- Get binary dataset  ----------
        _, y_used, _, y_score, classes, dataset, target_col = (
            self._get_binary_svc_plot_inputs(
                dataset=dataset,
                preview_rows=10,
                target_col=target_col,
            )
        )

        # ---------- Turn true values into array format ----------
        y_true_arr = np.asarray(y_used)

        # ---------- Plot settings ----------
        plt.figure(figsize=(7, 5))

        for c in classes:
            mask = y_true_arr == c
            plt.hist(
                y_score[mask],
                bins=bins,
                alpha=alpha,
                label=f"True class = {c}",
            )

        plt.axvline(0.0)
        plt.xlabel("Decision score")
        plt.ylabel("Count")
        title_suffix = f" | target={target_col}" if target_col is not None else ""
        plt.title(f"SVC | {dataset} | Decision function distribution{title_suffix}")
        plt.legend()
        plt.tight_layout()

        # ---------- Save and show plot  ----------
        if filename is None:
            if target_col is None:
                filename = (
                    f"{self.input_model_type}_{dataset}_decision_distribution.png"
                )
            else:
                filename = f"{self.input_model_type}_{dataset}_{target_col}_decision_distribution.png"

        out = os.path.join(PLOT_DIR, filename)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"💾 Decision function distribution saved path: {out}")

        plt.show()
        plt.close()
        return out

    # -------------------- ROC plot --------------------
    def roc_curve_plot_engine(
        self,
        dataset: str = "test",
        filename: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Plot and save the ROC curve for binary SVC classification.

        This method computes and plots the Receiver Operating Characteristic (ROC)
        curve using decision-function scores rather than predicted class labels.
        It supports:

        - single-output binary classification, or
        - multi-output classification when a binary ``target_col`` is specified

        The ROC curve provides a threshold-independent view of classifier ranking and
        class-separation performance.

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
            - If None, an automatic filename is generated based on model type,
            dataset split, and optional target column.

        target_col : str or None, default=None
            Target column name used only for multi-output classification.

            - Ignored for single-output classification
            - Required when plotting ROC for a binary target from a multi-output
            classifier

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
            Propagated from ``_get_binary_svc_plot_inputs()`` when the task or
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
        - The diagonal reference line represents random guessing performance.
        - A higher ROC AUC indicates better ranking and class-separation ability.
        - In multi-output classification, the ROC curve is generated only for the
        selected binary target.
        """
        # ---------- Get binary dataset ----------
        _, y_used, _, y_score, classes, dataset, target_col = (
            self._get_binary_svc_plot_inputs(
                dataset=dataset,
                preview_rows=10,
                target_col=target_col,
            )
        )

        # ---------- Record ROC-AUC ----------
        y_true_arr = np.asarray(y_used)
        pos_label = classes[1]

        fpr, tpr, _ = roc_curve(y_true_arr, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        # ---------- Plot settings ----------
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        title_suffix = f" | target={target_col}" if target_col is not None else ""
        plt.title(f"SVC | {dataset} | ROC Curve{title_suffix}")
        plt.legend(loc="lower right")
        plt.tight_layout()

        # ---------- Save and show plot ----------
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
        Plot and save the Precision-Recall curve for binary SVC classification.

        This method computes and plots the Precision-Recall (PR) curve using
        decision-function scores. It supports:

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
            - If None, an automatic filename is generated based on model type,
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
            Propagated from ``_get_binary_svc_plot_inputs()`` when the task or
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
        # ---------- Get binary dataset ----------
        _, y_used, _, y_score, classes, dataset, target_col = (
            self._get_binary_svc_plot_inputs(
                dataset=dataset,
                preview_rows=10,
                target_col=target_col,
            )
        )

        # ---------- Get  ----------
        y_true_arr = np.asarray(y_used)
        pos_label = classes[1]

        # ---------- Record precision recall ----------
        precision, recall, _ = precision_recall_curve(
            y_true_arr,
            y_score,
            pos_label=pos_label,
        )
        pr_auc = auc(recall, precision)

        # ---------- Plot settings ----------
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        title_suffix = f" | target={target_col}" if target_col is not None else ""
        plt.title(f"SVC | {dataset} | Precision-Recall Curve{title_suffix}")
        plt.legend(loc="lower left")
        plt.tight_layout()

        # ---------- Save and show plot ----------
        if filename is None:
            if target_col is None:
                filename = f"{self.input_model_type}_{dataset}_pr_curve.png"
            else:
                filename = f"{self.input_model_type}_{dataset}_{target_col}_pr_curve.png"  # For multiple Y

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
    def save_model_joblib(self, filename: str = "svc_model.joblib") -> str:
        """
        Save the trained SVC pipeline and related metadata as a joblib artifact.

        The saved artifact includes both the fitted model pipeline and selected
        metadata required for later inference or inspection.

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
        filename : str, default="svc_model.joblib"
            Output filename to save under `MODEL_DIR`.

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
        - Writes a `.joblib` file to disk.
        - Prints the saved file path.

        Notes
        -----
        - This method saves metadata together with the model to make downstream
          loading more robust.
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
        Load a previously saved trained SVC model artifact from a joblib file
        and restore it as an ``SVMClassifier_Missioner`` instance.

        This class method reconstructs a mission-layer object from a joblib
        artifact created by ``save_model_joblib()``. If the saved file contains
        a metadata dictionary, the method restores both the trained pipeline and
        associated metadata such as feature names and target-output mode. If the
        saved file contains only a model object, that object is assigned
        directly to ``model_pipeline`` and metadata fields are left unset.

        Parameters
        ----------
        filepath : str
            Full path to the saved joblib file.

        Returns
        -------
        SVMClassifier_Missioner
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

        obj = cls(cleaned_X_data=None, cleaned_Y_data=None)  # Setup object (missioner)

        # -------------------- Get saved trained model properties --------------------
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
