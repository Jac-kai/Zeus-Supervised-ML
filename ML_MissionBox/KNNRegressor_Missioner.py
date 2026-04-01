# -------------------- Import Modules --------------------
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Zeus.ML_BaseConfigBox.BaseModelConfig import BaseModelConfig

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOT_DIR = os.path.join(BASE_DIR, "ML_Report/KNNReg_Plot")
MODEL_DIR = os.path.join(BASE_DIR, "ML_Report/KNNReg_Trained_Model")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------- KNNRegressor Missioner --------------------
class KNNRegressor_Missioner(BaseModelConfig):
    """
    Mission-layer class for K-Nearest Neighbors regression workflows.

    This class extends :class:`BaseModelConfig` and provides shared regression
    utilities for KNN-based regressor trainers. It serves as the mission layer
    between the shared base configuration workflow and concrete KNN regressor
    model classes.

    This class does not define the final estimator construction or CV scoring
    dispatch itself. Concrete model-layer classes are responsible for building the
    final estimator and parameter grid, while the base layer handles shared
    training workflow logic, including model-selection scoring dispatch.

    Main Responsibilities
    ---------------------
    - Define the machine learning task type as regression.
    - Define the estimator step name used inside sklearn pipelines.
    - Evaluate trained KNN regression pipelines on train and test splits.
    - Generate regression diagnostic plots based on predictions and residuals.
    - Save and load trained model artifacts together with basic metadata.

    Supported Evaluation Modes
    --------------------------
    1. Single-output regression
    - R2
    - Mean squared error (MSE)
    - Mean absolute error (MAE)
    - Prediction preview

    2. Multi-output regression
    - Per-target R2
    - Per-target MSE
    - Per-target MAE
    - Macro-average metrics across targets
    - Prediction preview by rows

    Diagnostic Plot Utilities
    -------------------------
    - Predicted vs Actual scatter plot
    - Residual scatter plot
    - Residual histogram

    Notes
    -----
    - Model-selection scoring for both single-output and multi-output workflows is
    handled by the base layer.
    - KNN regressors do not expose ``feature_importances_``, so no
    feature-importance utility is implemented here.
    - Unlike tree-based models, KNN does not provide a structural visualization
    such as tree plots.
    - Diagnostic plotting relies on prediction results, so
    ``model_evaluation_engine()`` must be run first.
    - This mission layer focuses on reusable evaluation, plotting, and persistence
    logic shared across KNN-based regression workflows.

    Attributes Inherited from BaseModelConfig
    -----------------------------------------
    model_pipeline : sklearn Pipeline or compatible estimator, optional
        Trained regression pipeline.

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

    # -------------------- Task in piepline --------------------
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
            Always returns ``"regression"``.
        """
        return "regression"

    # -------------------- Step name in piepline --------------------
    @property
    def step_name(self) -> str:
        """
        Return the estimator step name used inside the sklearn Pipeline.

        This value is used to retrieve the fitted regressor from
        ``self.model_pipeline.named_steps`` and to define parameter-grid keys
        when GridSearchCV is used.

        Returns
        -------
        str
            Always returns ``"regressor"``.

        Notes
        -----
        Hyperparameter grid keys for GridSearchCV must use this prefix, for example:
        - ``"regressor__n_neighbors"``
        - ``"regressor__weights"``
        - ``"regressor__p"``
        """
        return "regressor"

    # -------------------- Model evaluation --------------------
    def model_evaluation_engine(self) -> Dict[str, Any]:
        """
        Evaluate a trained KNN regression pipeline on train and test datasets.

        This method generates predictions from the fitted regression pipeline
        and computes performance metrics according to whether the task is
        single-output or multi-output regression.

        Evaluation Behavior
        -------------------
        Single-output regression
            Computes:
            - train R2
            - test R2
            - train MSE
            - test MSE
            - train MAE
            - test MAE
            - first 10 predicted values

        Multi-output regression
            Computes:
            - per-target train R2
            - per-target test R2
            - per-target train MSE
            - per-target test MSE
            - per-target train MAE
            - per-target test MAE
            - macro-average train/test R2 across targets
            - macro-average train/test MSE across targets
            - macro-average train/test MAE across targets
            - first 10 predicted rows

        Returns
        -------
        Dict[str, Any]
            Evaluation summary dictionary.

            For single-output regression, keys include:
            - ``"mode"``
            - ``"train_r2"``
            - ``"test_r2"``
            - ``"train_mse"``
            - ``"test_mse"``
            - ``"train_mae"``
            - ``"test_mae"``
            - ``"prediction_preview_first10"``

            For multi-output regression, keys include:
            - ``"mode"``
            - ``"targets"``
            - ``"macro_train_r2"``
            - ``"macro_test_r2"``
            - ``"macro_train_mse"``
            - ``"macro_test_mse"``
            - ``"macro_train_mae"``
            - ``"macro_test_mae"``
            - ``"per_target"``
            - ``"prediction_preview_first10_rows"``

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
        - For multi-output regression, predicted values are converted to
        DataFrame form so target-column names and indices stay aligned.
        - Macro metrics here are simple arithmetic means across targets.
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

            self.prediction_preview = np.array(y_pred_test[:10])  # Array

            return {
                "mode": "single_output",
                "train_r2": float(r2_tr),
                "test_r2": float(r2_te),
                "train_mse": float(mse_tr),
                "test_mse": float(mse_te),
                "train_mae": float(mae_tr),
                "test_mae": float(mae_te),
                "prediction_preview_first10": self.prediction_preview,
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

        # ---------- Initialization of target, R2, MSE and MAE lists ----------
        per_target = {}
        r2_list_train, r2_list_test = [], []
        mse_list_train, mse_list_test = [], []
        mae_list_train, mae_list_test = [], []

        # ---------- Catch each target evaluation from test columns ----------
        for col in y_true_test.columns:
            yt_tr = y_true_train[col]
            yp_tr = y_pred_train_df[col]
            yt_te = y_true_test[col]
            yp_te = y_pred_test_df[col]

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

        return {
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

    # -------------------- Residual scatter plot --------------------
    def plot_knn_regression_diagnostics(
        self,
        dataset: str = "test",
        bins: int = 30,
        alpha: float = 0.6,
        max_points: Optional[int] = 5000,
    ) -> dict:
        """
        Generate and save KNN regression diagnostic plots.

        This method produces a diagnostic report based on already-generated
        predictions from `model_evaluation_engine()`. It supports both
        single-output and multi-output regression by normalizing target and
        prediction values into DataFrame form.

        Generated Charts
        ----------------
        For each target column, the method creates three plots:
        1. Predicted vs Actual scatter plot
        2. Residual scatter plot
        where residual = Actual - Predicted
        3. Residual histogram

        Parameters
        ----------
        dataset : str, default="test"
            Which split to plot.

            Accepted values:
            - ``"test"``
            - ``"train"``

            The value is normalized using ``lower().strip()``.

        bins : int, default=30
            Number of histogram bins used for each residual histogram.

        alpha : float, default=0.6
            Transparency applied to scatter plots.

        max_points : int or None, default=5000
            Maximum number of points used in scatter plots.

            - If None, all available samples are plotted.
            - If an integer is provided and the dataset size exceeds this value,
            a random subsample is drawn for scatter plots to reduce visual
            clutter and rendering cost.

        Returns
        -------
        dict
            Dictionary containing plotting outputs and summary statistics.

            Keys include:
            - ``"dataset"`` :
            selected dataset label
            - ``"n_samples"`` :
            total number of samples before optional subsampling
            - ``"subsampled"`` :
            whether random subsampling was applied
            - ``"saved_files"`` :
            list of saved plot file paths
            - ``"residual_stats"`` :
            per-target residual summary statistics

        Raises
        ------
        ValueError
            If:
            - `dataset` is not ``"train"`` or ``"test"``
            - prediction results are not available because
            `model_evaluation_engine()` has not been run first

        Side Effects
        ------------
        - Saves multiple diagnostic plot images into `PLOT_DIR`
        - Displays each figure using matplotlib
        - Closes each figure after display

        Notes
        -----
        - Residuals are defined as ``Actual - Predicted``.
        - To support both pandas and numpy target formats, `y_true` and `y_pred`
        are normalized into DataFrame form before plotting.
        - A fixed random seed is used for subsampling so repeated calls are
        reproducible when the same inputs are used.
        - In multi-output settings, one full set of three plots is generated
        for each target column.
        - The diagonal line in the Predicted vs Actual plot represents perfect
        predictions.
        - Residual summary statistics include:
            * mean residual
            * residual standard deviation
            * residual MAE
        """
        # ---------- Select training or testing dataset ----------
        dataset = dataset.lower().strip()
        if dataset not in ["test", "train"]:
            raise ValueError("⚠️ dataset must be 'test' or 'train' ‼️")

        if (
            self.y_test_pred is None or self.y_train_pred is None
        ):  # Need predictions ready
            raise ValueError(
                "⚠️  Run model_evaluation_engine() before plotting diagnostics ‼️"
            )

        # ---------- Record training or testing dataset after selecting dataset ----------
        if dataset == "test":
            y_true = self.Y_test
            y_pred = self.y_test_pred
        else:
            y_true = self.Y_train
            y_pred = self.y_train_pred

        # ---------- Normalize true values to DataFrame for unified multi/single handling ----------
        if isinstance(y_true, pd.Series):  # Series format
            y_true_df = y_true.to_frame(name=y_true.name or "y")
        elif isinstance(y_true, pd.DataFrame):  # DataFrame format
            y_true_df = y_true.copy()
        else:
            y_true_arr = np.asarray(y_true)  # Array format
            if y_true_arr.ndim == 1:  # 1D array
                y_true_df = pd.DataFrame({"y": y_true_arr})
            else:  # 2D array
                y_true_df = pd.DataFrame(
                    y_true_arr, columns=[f"y{i}" for i in range(y_true_arr.shape[1])]
                )

        # ---------- Normalize predicted values to DataFrame for unified multi/single handling ----------
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 1:
            y_pred_df = pd.DataFrame(
                {y_true_df.columns[0]: y_pred_arr}, index=y_true_df.index
            )
        else:
            y_pred_df = pd.DataFrame(
                y_pred_arr, columns=y_true_df.columns, index=y_true_df.index
            )

        # ---------- Subsampling and record tru and predticted values for plot data ----------
        n = len(
            y_true_df
        )  # Set true value amount as total (predicted values == tru values)
        idx = None
        if (max_points is not None) and (n > max_points):
            rng = np.random.default_rng(42)
            idx = rng.choice(np.arange(n), size=max_points, replace=False)
            y_true_plot = y_true_df.iloc[idx]
            y_pred_plot = y_pred_df.iloc[idx]
        else:
            y_true_plot = y_true_df
            y_pred_plot = y_pred_df

        # ---------- Initialization of saving contents ----------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        residual_stats = {}

        # ---------- Residual calculations for each columns ----------
        for col in y_true_df.columns:
            yt = y_true_plot[col].astype(float).to_numpy()  # Turn into array
            yp = y_pred_plot[col].astype(float).to_numpy()  # Turn into array
            resid = yt - yp

            # ---------- Record each columns' residual calculation results ----------
            residual_stats[col] = {
                "residual_mean": float(np.mean(resid)),  # Mean
                "residual_std": float(np.std(resid)),  # Standard deviation
                "residual_mae": float(np.mean(np.abs(resid))),  # MAE
            }

            # ---------- Prediction and actual plot ----------
            fig = plt.figure()
            plt.scatter(yt, yp, alpha=alpha)
            # diagonal line
            minv = float(np.min([yt.min(), yp.min()]))
            maxv = float(np.max([yt.max(), yp.max()]))
            plt.plot([minv, maxv], [minv, maxv])
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(f"knn_reg | {dataset} | Predicted vs Actual | target={col}")

            out = os.path.join(
                PLOT_DIR, f"knn_reg_{dataset}_pred_vs_actual_{col}_{timestamp}.png"
            )
            plt.savefig(out, dpi=150, bbox_inches="tight")
            saved_files.append(out)

            plt.show()
            plt.close(fig)

            # ---------- Residual scatter plot ----------
            fig = plt.figure()
            plt.scatter(yp, resid, alpha=alpha)
            plt.axhline(0.0)
            plt.xlabel("Predicted")
            plt.ylabel("Residual (Actual - Predicted)")
            plt.title(f"knn_reg | {dataset} | Residual scatter | target={col}")

            out = os.path.join(
                PLOT_DIR, f"knn_reg_{dataset}_residual_scatter_{col}_{timestamp}.png"
            )
            plt.savefig(out, dpi=150, bbox_inches="tight")
            saved_files.append(out)

            plt.show()
            plt.close(fig)

            # ---------- Residual histogram plot ----------
            fig = plt.figure()
            plt.hist(resid, bins=bins)
            plt.xlabel("Residual (Actual - Predicted)")
            plt.ylabel("Count")
            plt.title(f"knn_reg | {dataset} | Residual histogram | target={col}")

            # ---------- Save and show plot ----------
            out = os.path.join(
                PLOT_DIR, f"knn_reg_{dataset}_residual_hist_{col}_{timestamp}.png"
            )
            plt.savefig(out, dpi=150, bbox_inches="tight")
            saved_files.append(out)

            plt.show()
            plt.close(fig)

        return {
            "dataset": dataset,
            "n_samples": int(n),
            "subsampled": (idx is not None),
            "saved_files": saved_files,
            "residual_stats": residual_stats,
        }

    # -------------------- Save trained model --------------------
    def save_model_joblib(self, filename: str = "knn_regressor.joblib") -> str:
        """
        Save the trained KNN regression pipeline and related metadata as a joblib artifact.

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
        target column names when multi-output regression is used

        Parameters
        ----------
        filename : str, default="knn_regressor.joblib"
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
            raise ValueError("⚠️ No trained model to save ‼️")

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
        Load a previously saved KNN regression model artifact from a joblib file
        and restore it as a ``KNNRegressor_Missioner`` instance.

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
        KNNRegressor_Missioner
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
          Fitted regression pipeline
        - ``feature_names`` :
          Feature names recorded during training
        - ``y_mode`` :
          Target-output mode, such as ``"single_output"`` or ``"multi_output"``
        - ``y_columns`` :
          Target column names for multi-output regression, if stored

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

        # -------------------- Get model properties --------------------
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
