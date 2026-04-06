# -------------------- Import Modules --------------------
from __future__ import annotations

import os
import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

from Zeus.ML_BaseConfigBox.BaseModelConfig import BaseModelConfig

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOT_DIR = os.path.join(BASE_DIR, "ML_Report/Tree_Plot")
MODEL_DIR = os.path.join(BASE_DIR, "ML_Report/TreeCla_Trained_Model")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------- TreeClassifier Missioner --------------------
class TreeClassifier_Missioner(BaseModelConfig):
    """
    Mission layer for tree-based classification models in the Zeus workflow.

    ``TreeClassifier_Missioner`` extends ``BaseModelConfig`` and provides reusable
    classification-oriented utilities shared across multiple tree-based classifier
    implementations. It serves as an intermediate layer between the shared base
    configuration workflow and concrete tree-based model-layer classes.

    This class does not define the final estimator construction or CV scoring
    dispatch itself. Concrete model classes are responsible for building the final
    estimators and parameter grids, while the base layer handles shared training
    workflow logic, including model-selection scoring dispatch.

    Responsibilities
    ----------------
    - Declare the machine learning task type as classification.
    - Provide the pipeline estimator step name for classifier models.
    - Evaluate trained models for both single-output and multi-output
    classification.
    - Extract feature importances from supported tree-based estimators.
    - Plot a fitted decision tree or one selected tree inside a random forest.
    - Save and load trained model artifacts with metadata.

    Supported Model Types
    ---------------------
    This mission layer is designed for tree-based classifiers such as:
    - ``DecisionTreeClassifier``
    - ``RandomForestClassifier``

    Inherited Capabilities
    ----------------------
    From ``BaseModelConfig``, this class relies on inherited methods and state,
    including but not limited to:
    - train/test splitting
    - preprocessing pipeline construction
    - grid-search training
    - feature-name extraction
    - multi-output scoring dispatch for model selection
    - cleaned input/output dataset storage

    Key Attributes Used
    -------------------
    cleaned_X_data : pandas.DataFrame or numpy.ndarray
        Prepared feature dataset.
    cleaned_Y_data : pandas.Series, pandas.DataFrame, or numpy.ndarray
        Prepared target dataset.
    X_train, X_test : pandas.DataFrame or numpy.ndarray
        Training and test feature sets.
    Y_train, Y_test : pandas.Series, pandas.DataFrame, or numpy.ndarray
        Training and test target sets.
    model_pipeline : sklearn.pipeline.Pipeline or None
        Fitted training pipeline.
    feature_names : list[str] or None
        Extracted feature names after preprocessing.
    y_train_pred, y_test_pred : numpy.ndarray or pandas.DataFrame or None
        Cached training and test predictions.
    prediction_preview : Any
        Short preview of test predictions for display.

    Notes
    -----
    - Feature importance extraction requires a fitted estimator that exposes
    ``feature_importances_``.
    - Multi-output classification evaluation is handled target by target because
    sklearn classification summaries are primarily designed for single-output use.
    - Random forest plotting visualizes only one estimator at a time for
    readability.
    - Model-selection scoring for both single-output and multi-output workflows is
    handled by the base layer.
    """

    # -------------------- Classification task --------------------
    @property
    def task(self) -> str:
        """
        Return the task type identifier used by the base configuration.

        This property fulfills the task contract required by
        ``BaseModelConfig`` and tells the inherited training workflow that this
        mission class handles classification problems.

        Returns
        -------
        str
            Always returns ``"classification"``.

        Notes
        -----
        The returned value may be used by the base layer to determine scoring
        behavior, estimator handling, and other task-dependent logic.
        """
        return "classification"

    # -------------------- Step name in piepline --------------------
    @property
    def step_name(self) -> str:
        """
        Return the estimator step name used inside the sklearn pipeline.

        This property defines the pipeline step key under which the classifier
        estimator is stored. It is used when accessing the fitted estimator from
        ``model_pipeline.named_steps`` and when building hyperparameter grid
        keys for grid search.

        Returns
        -------
        str
            Always returns ``"classifier"``.

        Notes
        -----
        This value must stay consistent with parameter-grid prefixes such as:
        - ``classifier__max_depth``
        - ``classifier__n_estimators``
        - ``classifier__min_samples_split``
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

        # ---------- Record predictions from train and test dataset ----------
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

            # ---------- Record above results ----------
            results = {
                "mode": "single_output",
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_f1_weighted": train_f1,
                "test_f1_weighted": test_f1,
                "confusion_matrix": cm,
                "classification_report": report,
                "prediction_preview_first10": self.prediction_preview,
            }

            try:
                results["feature_importance"] = self.feature_importance_engine()
            except Exception:
                results["feature_importance"] = None  # No feature names extracted

            return results

        # ---------- Multi-output ----------
        # Help Numpy ndarray to be DataFrame format in order to get its column's val
        if not isinstance(y_true_test, pd.DataFrame):
            y_true_test = pd.DataFrame(y_true_test)
        if not isinstance(y_true_train, pd.DataFrame):
            y_true_train = pd.DataFrame(y_true_train)

        y_pred_test_df = pd.DataFrame(
            y_pred_test, columns=y_true_test.columns, index=y_true_test.index
        )

        y_pred_train_df = pd.DataFrame(
            y_pred_train, columns=y_true_train.columns, index=y_true_train.index
        )

        # ---------- Initialization of target, accuracy and F1 lists ----------
        per_target = {}  # Record each target
        acc_list_train, acc_list_test = [], []
        f1_list_train, f1_list_test = [], []

        # ---------- Catch each target evaluation from test columns ----------
        for col in y_true_test.columns:
            # ---------- True values from train / test dataset ----------
            yt_tr = y_true_train[col]
            yt_te = y_true_test[col]

            # ---------- Predictions from train / test dataset ----------
            yp_tr = y_pred_train_df[col]
            yp_te = y_pred_test_df[col]

            acc_tr = accuracy_score(yt_tr, yp_tr)
            acc_te = accuracy_score(yt_te, yp_te)
            f1_tr = f1_score(yt_tr, yp_tr, average="weighted")
            f1_te = f1_score(yt_te, yp_te, average="weighted")

            # ---------- Record overall targets' accuracy and F1 ----------
            acc_list_train.append(acc_tr)
            acc_list_test.append(acc_te)
            f1_list_train.append(f1_tr)
            f1_list_test.append(f1_te)

            # ---------- Record each target's accuracy and F1 ----------
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
        )  # DataFrame

        # ---------- Record overall targets' and each target's accuracy and F1 ----------
        results = {
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
        }

        try:
            results["feature_importance"] = self.feature_importance_engine()
        except Exception:
            results["feature_importance"] = None

        return results

    # -------------------- Feature importance --------------------
    def feature_importance_engine(self) -> pd.DataFrame:
        """
        Extract feature importances from the fitted tree-based classifier.

        This method retrieves the classifier estimator from the fitted pipeline
        and reads its ``feature_importances_`` attribute. The result is returned
        as a sorted ``DataFrame`` containing raw importance values and
        percentage-style ratios.

        Returns
        -------
        pandas.DataFrame
            A DataFrame sorted by descending importance, containing:

            - ``feature`` :
              Feature name
            - ``importance`` :
              Raw feature importance value
            - ``importance_ratio`` :
              Importance expressed as ``importance * 100``

        Raises
        ------
        ValueError
            If ``self.model_pipeline`` is not available.
        ValueError
            If the fitted estimator cannot be found in the pipeline or does not
            expose ``feature_importances_``.

        Notes
        -----
        - The estimator is retrieved from ``self.model_pipeline.named_steps``
          using ``self.step_name``.
        - If ``self.feature_names`` is unavailable, default fallback names
          ``x0``, ``x1``, ..., ``xN`` are generated.
        - This method supports tree-based models with native feature-importance
          attributes, such as decision trees and random forests.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ Model pipeline not trained yet ‼️")

        clf = self.model_pipeline.named_steps.get(self.step_name)
        if clf is None or (not hasattr(clf, "feature_importances_")):
            raise ValueError("⚠️ Model has no feature_importances_ ‼️")

        # ---------- Record feature importance as DataFrame format ----------
        importances = clf.feature_importances_
        names = (
            self.feature_names
            if self.feature_names is not None
            else [
                f"x{i}" for i in range(len(importances))
            ]  # Give default name (x0, x1,......)
        )

        return (
            pd.DataFrame(
                {
                    "feature": names,
                    "importance": importances,
                    "importance_ratio": importances * 100,
                }
            )
            .sort_values(
                "importance", ascending=False
            )  # Sort by importance from highest to lowest
            .reset_index(drop=True)
        )

    # -------------------- Tree plot (DT or one RF tree) --------------------
    def tree_plot_engine(
        self,
        save_fig: bool = False,
        max_depth: int = 3,
        tree_index: int = 0,
    ):
        """
        Plot a fitted decision tree or a selected tree from a random forest.

        This method visualizes the trained classifier when the fitted estimator
        is either a ``DecisionTreeClassifier`` or a
        ``RandomForestClassifier``. For a random forest, only one internal tree
        is plotted at a time, selected by ``tree_index``.

        Parameters
        ----------
        save_fig : bool, default=False
            Whether to save the generated plot as a PNG file under ``PLOT_DIR``.
        max_depth : int, default=3
            Maximum visualization depth passed to ``sklearn.tree.plot_tree``.
            This affects only the displayed figure and does not modify the
            trained model itself.
        tree_index : int, default=0
            Index of the internal tree to plot when the fitted estimator is a
            ``RandomForestClassifier``.

        Returns
        -------
        None
            This method displays the plot and optionally saves it to disk.

        Raises
        ------
        ValueError
            If ``self.model_pipeline`` is ``None``.
        ValueError
            If the classifier step cannot be found in the pipeline.
        ValueError
            If the fitted random forest has no internal estimators available.
        ValueError
            If ``tree_index`` is outside the valid range.
        TypeError
            If the fitted estimator is neither ``DecisionTreeClassifier`` nor
            ``RandomForestClassifier``.

        Notes
        -----
        - For random forests, only one tree is shown because plotting the full
          ensemble is usually too large to read.
        - Feature names are taken from ``self.feature_names``.
        - Class names are automatically inferred from ``clf.classes_`` when
          available.
        - The plot is shown with ``matplotlib.pyplot.show()`` and then closed.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ Model pipeline not trained yet ‼️")

        # ---------- Get the pipeline estimator (classifier) ----------
        clf = self.model_pipeline.named_steps.get(self.step_name, None)
        if clf is None:
            raise ValueError("⚠️ Pipeline has no classifier step ‼️")

        if isinstance(clf, RandomForestClassifier):
            if not hasattr(clf, "estimators_") or len(clf.estimators_) == 0:
                raise ValueError("⚠️ RandomForest is not fitted yet ‼️")
            if tree_index < 0 or tree_index >= len(clf.estimators_):
                raise ValueError(
                    f"⚠️ tree_index out of range: 0 ~ {len(clf.estimators_) - 1} ‼️"
                )
            tree = clf.estimators_[tree_index]
            model_name = f"RandomForest_Tree{tree_index}"
        elif isinstance(clf, DecisionTreeClassifier):
            tree = clf
            model_name = "DecisionTree"
        else:
            raise TypeError(
                "⚠️ tree_plot_engine supports DecisionTreeClassifier / RandomForestClassifier only ‼️"
            )

        # ---------- Plot settings ----------
        plt.figure(figsize=(18, 8))

        auto_class_names = (
            [str(c) for c in clf.classes_] if hasattr(clf, "classes_") else None
        )

        plot_tree(
            tree,
            feature_names=self.feature_names,
            class_names=auto_class_names,
            filled=True,
            max_depth=max_depth,
            rounded=True,  # Set the format of nodes
            fontsize=6,
        )

        # ---------- Save and show plot ----------
        if save_fig:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = os.path.join(
                PLOT_DIR, f"{self.input_model_type}_{model_name}_{ts}.png"
            )
            plt.savefig(out, dpi=200, bbox_inches="tight")
            print(f"💾 Tree figure saved path >>>>> {out}")

        plt.show()
        plt.close()

    # -------------------- Save trained model --------------------
    def save_model_joblib(self, filename: str = "trained_model.joblib") -> str:
        """
        Save the trained model pipeline and related metadata as a joblib file.

        This method exports the fitted pipeline together with selected metadata
        so that the trained model can be restored later with structural context,
        including feature names and target-output mode.

        Parameters
        ----------
        filename : str, default="trained_model.joblib"
            Name of the output file to save under ``MODEL_DIR``.

        Returns
        -------
        str
            Full file path of the saved joblib artifact.

        Raises
        ------
        ValueError
            If ``self.model_pipeline`` is ``None``, meaning no trained model is
            available to save.

        Saved Content
        -------------
        The joblib file stores a dictionary containing:
        - ``model`` :
          The fitted sklearn pipeline
        - ``feature_names`` :
          Extracted feature names, if available
        - ``y_mode`` :
          Target structure, either ``"single_output"`` or ``"multi_output"``
        - ``y_columns`` :
          Target column names when the target is multi-output, otherwise ``None``

        Notes
        -----
        - The output directory is defined by ``MODEL_DIR``.
        - The method prints the saved path after successful export.
        - The saved metadata helps preserve target-structure information for
          later inference or inspection.
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
        print(f"💾 Model saved path >>>>> {out}")
        return out

    # -------------------- Load trained model --------------------
    @classmethod
    def load_model_joblib(cls, filepath: str):
        """
        Load a previously saved joblib model artifact and restore a model instance.

        This class method reconstructs a ``TreeClassifier_Missioner`` object from
        a saved joblib file. If the saved artifact contains a metadata dictionary,
        the method restores the fitted pipeline together with feature names and
        target-output metadata. If the saved artifact contains only a model
        object, the method restores that object as ``model_pipeline`` and leaves
        metadata attributes unset.

        Parameters
        ----------
        filepath : str
            Full path to the saved joblib file.

        Returns
        -------
        TreeClassifier_Missioner
            A restored model instance with ``model_pipeline`` loaded and metadata
            populated when available.

        Raises
        ------
        FileNotFoundError
            If the specified file path does not exist.

        Notes
        -----
        - The restored object is initialized with ``cleaned_X_data=None`` and
          ``cleaned_Y_data=None`` because this method focuses on reloading the
          trained artifact rather than the original training data.
        - If the loaded content is a dictionary containing a ``"model"`` key,
          the following attributes are restored when present:
            - ``model_pipeline``
            - ``feature_names``
            - ``y_mode``
            - ``y_columns``
        - If the loaded content is not a metadata dictionary, it is treated as
          the pipeline object itself.
        - The method prints the loaded file path after successful restoration.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"⚠️ Model file not found: {filepath} ‼️")

        loaded = load(filepath)

        obj = cls(cleaned_X_data=None, cleaned_Y_data=None)

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

        print(f"📦 Model loaded path >>>>> {filepath}")
        return obj


# =================================================
