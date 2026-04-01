# -------------------- Import Modules --------------------
from __future__ import annotations

import os
import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree

from Zeus.ML_BaseConfigBox.BaseModelConfig import BaseModelConfig

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOT_DIR = os.path.join(BASE_DIR, "ML_Report/Tree_Plot")
MODEL_DIR = os.path.join(BASE_DIR, "ML_Report/TreeReg_Trained_Model")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------- TreeRegressor Missioner --------------------
class TreeRegressor_Missioner(BaseModelConfig):
    """
    Mission layer for tree-based regression models in the Zeus workflow.

    ``TreeRegressor_Missioner`` extends ``BaseModelConfig`` and provides reusable
    regression-oriented utilities shared across multiple tree-based regressor
    implementations. It serves as an intermediate layer between the shared base
    configuration workflow and concrete tree-based model-layer classes.

    This class does not define the final estimator construction or CV scoring
    dispatch itself. Concrete model classes are responsible for building the final
    estimators and parameter grids, while the base layer handles shared training
    workflow logic, including model-selection scoring dispatch.

    Responsibilities
    ----------------
    - Declare the machine learning task type as regression.
    - Provide the pipeline estimator step name for regressor models.
    - Evaluate trained models for both single-output and multi-output regression.
    - Extract feature importances from supported tree-based estimators.
    - Plot a fitted decision tree or one selected tree inside a random forest.
    - Save and load trained model artifacts with metadata.

    Supported Model Types
    ---------------------
    This mission layer is designed for tree-based regressors such as:
    - ``DecisionTreeRegressor``
    - ``RandomForestRegressor``

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
    - Multi-output regression evaluation is handled target by target so each output
    remains transparent and independently inspectable.
    - Random forest plotting visualizes only one estimator at a time for
    readability.
    - Model-selection scoring for both single-output and multi-output workflows is
    handled by the base layer.
    """

    # -------------------- Classification or Regression task --------------------
    @property
    def task(self) -> str:
        """
        Return the task type identifier used by the base configuration.

        This property fulfills the task contract required by
        ``BaseModelConfig`` and tells the inherited training workflow that this
        mission class handles regression problems.

        Returns
        -------
        str
            Always returns ``"regression"``.

        Notes
        -----
        The returned value may be used by the base layer to determine scoring
        behavior, estimator handling, and other task-dependent logic.
        """
        return "regression"

    # -------------------- Step name in piepline --------------------
    @property
    def step_name(self) -> str:
        """
        Return the estimator step name used inside the sklearn pipeline.

        This property defines the pipeline step key under which the regressor
        estimator is stored. It is used when accessing the fitted estimator from
        ``model_pipeline.named_steps`` and when building hyperparameter grid
        keys for grid search.

        Returns
        -------
        str
            Always returns ``"regressor"``.

        Notes
        -----
        This value must stay consistent with parameter-grid prefixes such as:
        - ``regressor__max_depth``
        - ``regressor__n_estimators``
        - ``regressor__min_samples_split``
        """
        return "regressor"

    # -------------------- Model evaluation --------------------
    def model_evaluation_engine(self) -> Dict[str, Any]:
        """
        Evaluate the trained regression pipeline on both training and test sets.

        This method generates predictions from the fitted pipeline using
        ``X_train`` and ``X_test``, then computes evaluation metrics based on
        whether the target structure is single-output or multi-output.

        For single-output regression, the method returns overall train/test R2,
        MSE, MAE, and a short prediction preview.

        For multi-output regression, the method evaluates each target column
        independently and returns:
        - per-target train/test R2
        - per-target train/test MSE
        - per-target train/test MAE
        - macro mean train/test R2 across targets
        - macro mean train/test MSE across targets
        - macro mean train/test MAE across targets
        - first 10 rows of prediction preview

        Returns
        -------
        Dict[str, Any]
            Structured evaluation results.

            In single-output mode, the dictionary includes:
            - ``mode``
            - ``train_r2``
            - ``test_r2``
            - ``train_mse``
            - ``test_mse``
            - ``train_mae``
            - ``test_mae``
            - ``prediction_preview_first10``
            - ``feature_importance``

            In multi-output mode, the dictionary includes:
            - ``mode``
            - ``targets``
            - ``macro_train_r2``
            - ``macro_test_r2``
            - ``macro_train_mse``
            - ``macro_test_mse``
            - ``macro_train_mae``
            - ``macro_test_mae``
            - ``per_target``
            - ``prediction_preview_first10_rows``
            - ``feature_importance``

        Raises
        ------
        ValueError
            If ``self.model_pipeline`` is ``None``, meaning no trained model is
            available for evaluation.

        Side Effects
        ------------
        Updates the following instance attributes:
        - ``self.y_train_pred``
        - ``self.y_test_pred``
        - ``self.prediction_preview``

        Notes
        -----
        - Multi-output regression metrics are computed separately for each
          target to keep reporting explicit and easy to inspect.
        - Macro metrics are arithmetic means across target-level metric values.
        - Feature importance is added on a best-effort basis. If importance
          extraction fails, ``feature_importance`` is set to ``None``.
        - In single-output mode, prediction preview is stored as a NumPy array.
        - In multi-output mode, prediction preview is stored as a DataFrame.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ Model pipeline not trained yet ‼️")

        self.y_train_pred = self.model_pipeline.predict(self.X_train)
        self.y_test_pred = self.model_pipeline.predict(self.X_test)

        y_true_train = self.Y_train
        y_true_test = self.Y_test
        y_pred_train = self.y_train_pred
        y_pred_test = self.y_test_pred

        # ---------- Single-output ----------
        if not self._is_multi_output(y_true_test):
            # ---------- Train and test dataset R2 ----------
            r2_tr = r2_score(y_true_train, y_pred_train)
            r2_te = r2_score(y_true_test, y_pred_test)

            # ---------- Train and test dataset MSE ----------
            mse_tr = mean_squared_error(y_true_train, y_pred_train)
            mse_te = mean_squared_error(y_true_test, y_pred_test)

            # ---------- Train and test dataset MAE ----------
            mae_tr = mean_absolute_error(y_true_train, y_pred_train)
            mae_te = mean_absolute_error(y_true_test, y_pred_test)

            self.prediction_preview = np.array(
                y_pred_test[:10]
            )  # DataFrame turn into Numpy array

            # ---------- Record above results ----------
            results = {
                "mode": "single_output",
                "train_r2": float(r2_tr),
                "test_r2": float(r2_te),
                "train_mse": float(mse_tr),
                "test_mse": float(mse_te),
                "train_mae": float(mae_tr),
                "test_mae": float(mae_te),
                "prediction_preview_first10": self.prediction_preview,
            }

            try:
                results["feature_importance"] = (
                    self.feature_importance_engine()
                )  # Record feature importance
            except Exception:
                results["feature_importance"] = None

            return results

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

        # ---------- Initialization of target, R2, MSE and MAE lists ----------
        per_target = {}
        r2_list_train, r2_list_test = [], []
        mse_list_train, mse_list_test = [], []
        mae_list_train, mae_list_test = [], []

        # ---------- Catch each target evaluation from test columns ----------
        for col in y_true_test.columns:
            yt_tr, yp_tr = y_true_train[col], y_pred_train_df[col]
            yt_te, yp_te = y_true_test[col], y_pred_test_df[col]

            # ---------- Record each target's R2, MSE and MAE ----------
            r2_tr = r2_score(yt_tr, yp_tr)
            r2_te = r2_score(yt_te, yp_te)

            mse_tr = mean_squared_error(yt_tr, yp_tr)
            mse_te = mean_squared_error(yt_te, yp_te)

            mae_tr = mean_absolute_error(yt_tr, yp_tr)
            mae_te = mean_absolute_error(yt_te, yp_te)

            # ---------- Record overall targets' R2, MSE and MAE ----------
            r2_list_train.append(r2_tr)
            r2_list_test.append(r2_te)
            mse_list_train.append(mse_tr)
            mse_list_test.append(mse_te)
            mae_list_train.append(mae_tr)
            mae_list_test.append(mae_te)

            # ---------- Record each target's R2, MSE and MAE ----------
            per_target[col] = {
                "train_r2": float(r2_tr),
                "test_r2": float(r2_te),
                "train_mse": float(mse_tr),
                "test_mse": float(mse_te),
                "train_mae": float(mae_tr),
                "test_mae": float(mae_te),
            }

        self.prediction_preview = y_pred_test_df.head(10)  # DataFrame

        # ---------- Record above results ----------
        results = {
            "mode": "multi_output",
            "targets": list(y_true_test.columns),
            "macro_train_r2": float(np.mean(r2_list_train)) if r2_list_train else None,
            "macro_test_r2": float(np.mean(r2_list_test)) if r2_list_test else None,
            "macro_train_mse": (
                float(np.mean(mse_list_train)) if mse_list_train else None
            ),
            "macro_test_mse": float(np.mean(mse_list_test)) if mse_list_test else None,
            "macro_train_mae": (
                float(np.mean(mae_list_train)) if mae_list_train else None
            ),
            "macro_test_mae": float(np.mean(mae_list_test)) if mae_list_test else None,
            "per_target": per_target,
            "prediction_preview_first10_rows": self.prediction_preview,
        }

        try:
            results["feature_importance"] = (
                self.feature_importance_engine()
            )  # Record feature importance
        except Exception:
            results["feature_importance"] = None

        return results

    # -------------------- Feature importance --------------------
    def feature_importance_engine(self) -> pd.DataFrame:
        """
        Extract feature importances from the fitted tree-based regressor.

        This method retrieves the regressor estimator from the fitted pipeline
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
            raise ValueError("⚠️  Model pipeline not trained yet ‼️")

        # ---------- Record feature importance ----------
        reg = self.model_pipeline.named_steps.get(self.step_name)
        if reg is None or (not hasattr(reg, "feature_importances_")):
            raise ValueError("⚠️  Model has no feature_importances_ ‼️")

        # ---------- Record feature names ----------
        importances = reg.feature_importances_
        names = (
            self.feature_names
            if self.feature_names is not None
            else [f"x{i}" for i in range(len(importances))]
        )

        # ---------- Record feature importance as DataFrame ----------
        return (
            pd.DataFrame(
                {
                    "feature": names,
                    "importance": importances,
                    "importance_ratio": importances * 100,
                }
            )
            .sort_values("importance", ascending=False)
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
        Plot a fitted decision tree or a selected tree from a random forest regressor.

        This method visualizes the trained regressor when the fitted estimator
        is either a ``DecisionTreeRegressor`` or a
        ``RandomForestRegressor``. For a random forest, only one internal tree
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
            ``RandomForestRegressor``.

        Returns
        -------
        None
            This method displays the plot and optionally saves it to disk.

        Raises
        ------
        ValueError
            If ``self.model_pipeline`` is ``None``.
        ValueError
            If the regressor step cannot be found in the pipeline.
        ValueError
            If the fitted random forest has no internal estimators available.
        ValueError
            If ``tree_index`` is outside the valid range.
        TypeError
            If the fitted estimator is neither ``DecisionTreeRegressor`` nor
            ``RandomForestRegressor``.

        Notes
        -----
        - For random forests, only one tree is shown because plotting the full
          ensemble is usually too large to read.
        - Feature names are taken from ``self.feature_names``.
        - The plot is shown with ``matplotlib.pyplot.show()`` and then closed.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️  Model pipeline not trained yet ‼️")

        # ---------- Get pipeline estimator (regressor) ----------
        reg = self.model_pipeline.named_steps.get(self.step_name, None)
        if reg is None:
            raise ValueError("⚠️  Pipeline has no regressor step ‼️")

        if isinstance(reg, RandomForestRegressor):
            if not hasattr(reg, "estimators_") or len(reg.estimators_) == 0:
                raise ValueError("⚠️  RandomForest is not fitted yet ‼️")
            if tree_index < 0 or tree_index >= len(reg.estimators_):
                raise ValueError(
                    f"⚠️  tree_index out of range: 0 ~ {len(reg.estimators_) - 1} ‼️"
                )
            tree = reg.estimators_[tree_index]
            model_name = f"RandomForestRegressor_Tree{tree_index}"
        elif isinstance(reg, DecisionTreeRegressor):
            tree = reg
            model_name = "DecisionTreeRegressor"
        else:
            raise TypeError(
                "⚠️  tree_plot_engine supports DecisionTreeRegressor / RandomForestRegressor only ‼️"
            )

        # ---------- Plot settings ----------
        plt.figure(figsize=(18, 8))
        plot_tree(
            tree,
            feature_names=self.feature_names,
            filled=True,
            max_depth=max_depth,
            rounded=True,
            fontsize=6,
        )

        # ---------- Save and shoe plot ----------
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

        This class method reconstructs a ``TreeRegressor_Missioner`` object from
        a saved joblib file. If the saved artifact contains a metadata
        dictionary, the method restores the fitted pipeline together with
        feature names and target-output metadata. If the saved artifact contains
        only a model object, the method restores that object as
        ``model_pipeline`` and leaves metadata attributes unset.

        Parameters
        ----------
        filepath : str
            Full path to the saved joblib file.

        Returns
        -------
        TreeRegressor_Missioner
            A restored model instance with ``model_pipeline`` loaded and
            metadata populated when available.

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
