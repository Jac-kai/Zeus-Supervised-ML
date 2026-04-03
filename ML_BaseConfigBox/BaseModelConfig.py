# -------------------- Import Modules --------------------
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

BASED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CV_REPORT_DIR = os.path.join(BASED_PATH, "ML_Report", "CV_Report")
os.makedirs(CV_REPORT_DIR, exist_ok=True)


# -------------------- Base Model Configure --------------------
class BaseModelConfig(ABC):
    """
    BaseModelConfig
    ===============

    Abstract base class for sklearn-style model managers.

    This class centralizes the shared end-to-end workflow used across Zeus model
    families so that model-layer classes do not need to repeat common boilerplate.

    Main Responsibilities
    ---------------------
    - Store cleaned feature and target data, supporting both single-output and
    multi-output targets.
    - Split data into train and test sets, with optional stratification for
    single-output classification.
    - Build a preprocessing ``ColumnTransformer`` for numeric and categorical
    features.
    - Build a full sklearn ``Pipeline`` consisting of preprocessing, optional
    intermediate steps, and the final estimator.
    - Optionally run ``GridSearchCV`` for hyperparameter tuning.
    - Dispatch scoring logic for both single-output and multi-output workflows.
    - Extract transformed feature names after preprocessing.

    Subclass Contract
    -----------------
    Subclasses must implement:
    - ``task`` (property):
    Return ``"classification"`` or ``"regression"``.
    - ``step_name`` (property):
    Return ``"classifier"`` or ``"regressor"``.
    This value must match the estimator step prefix used in pipeline parameter
    grids, such as ``"classifier__max_depth"`` or
    ``"regressor__n_estimators"``.

    Notes
    -----
    - Stratification is applied only when:
        * ``task == "classification"``
        * ``stratify=True``
        * target is single-output
    - Multi-output targets are not used for stratification.
    - CV splitter policy:
        * classification + single-output -> ``StratifiedKFold``
        * otherwise -> ``KFold``
    - Multi-output scoring is handled centrally by base-layer scorer builders:
        * ``_build_multioutput_classification_scorer()``
        * ``_build_multioutput_regression_scorer()``
    - Preprocessing policy:
        * numeric features:
        ``SimpleImputer(strategy="median")``
        * categorical features:
        ``SimpleImputer(strategy="most_frequent")`` + encoder
            - ``OneHotEncoder(handle_unknown="ignore")`` for OHE
            - ``OrdinalEncoder(handle_unknown="use_encoded_value",
            unknown_value=-1)`` for ordinal encoding
    """

    # -------------------- Initialization --------------------
    def __init__(
        self,
        cleaned_X_data: pd.DataFrame,
        cleaned_Y_data: Union[pd.Series, pd.DataFrame],
    ):
        """
        Initialize the base manager with cleaned feature and target data.

        Parameters
        ----------
        cleaned_X_data : pd.DataFrame
            Cleaned feature matrix (X).

        cleaned_Y_data : Union[pd.Series, pd.DataFrame]
            Target(s) (Y). Normalization rules:
            - If Series: treated as single-output.
            - If DataFrame with 1 column: converted to Series (single-output).
            - If DataFrame with >= 2 columns: kept as DataFrame (multi-output).

        Side Effects
        ------------
        Initializes internal state fields used by the workflow:
        - Train/test split buffers: X_train, X_test, Y_train, Y_test
        - Prediction buffers: y_train_pred, y_test_pred
        - Trained pipeline: model_pipeline
        - Feature name cache: feature_names
        - Column caches for fallback naming: _numeric_cols, _categorical_cols
        - Run metadata: input_model_type, input_use_cv, input_cv_folds
        """
        self.cleaned_X_data = cleaned_X_data

        # ---------- Y exchanging format ----------
        if isinstance(cleaned_Y_data, pd.DataFrame):
            if cleaned_Y_data.shape[1] == 1:
                self.cleaned_Y_data = cleaned_Y_data.iloc[:, 0]
            else:
                self.cleaned_Y_data = cleaned_Y_data.copy()
        else:
            self.cleaned_Y_data = cleaned_Y_data

        # ---------- Test / train dataset ----------
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        # ---------- Predictions from test / train dataset ----------
        self.y_train_pred = None
        self.y_test_pred = None
        self.prediction_preview = None

        # ---------- Pipeline / feature names ----------
        self.model_pipeline = None
        self.feature_names = None

        # ---------- Record numeric- / categorical-type columns (internal used) ----------
        self._numeric_cols = None
        self._categorical_cols = None

        # ---------- Record CV report ----------
        self.cv_search_report = None

        # ---------- Record internal metadata ----------
        self.input_model_type = None
        self.input_use_cv = None
        self.input_cv_folds = None

    # -------------------- Required by subclass (task) --------------------
    @property
    @abstractmethod
    def task(self) -> str:
        """
        Task type of the manager.

        Returns
        -------
        str
            Must be one of:
            - "classification"
            - "regression"

        Notes
        -----
        Used to decide:
        - Whether stratify is allowed in train/test split.
        - Which CV splitter to use (StratifiedKFold vs KFold).
        """

    # -------------------- Required by subclass (step name) --------------------
    @property
    @abstractmethod
    def step_name(self) -> str:
        """
        Estimator step name used inside the sklearn Pipeline.

        Returns
        -------
        str
            Must be one of:
            - "classifier"
            - "regressor"

        Notes
        -----
        This value must match the parameter prefix in GridSearchCV param_grid, e.g.:
        - "classifier__max_depth"
        - "regressor__n_estimators"
        """

    # -------------------- Helper: Check Y is multiple output --------------------
    def _is_multi_output(self, y: Union[pd.Series, pd.DataFrame]) -> bool:
        """
        Check whether a target is multi-output.

        Parameters
        ----------
        y : Union[pd.Series, pd.DataFrame]
            Target data.

        Returns
        -------
        bool
            True if y is a DataFrame with >= 2 columns, otherwise False.
        """
        return isinstance(y, pd.DataFrame) and y.shape[1] > 1

    # -------------------- Helper: Scoring method for multiple output in classification --------------------
    def _build_multioutput_classification_scorer(self, scoring: str):
        """
        Build a scorer for multi-output classification model selection.

        This helper creates a ``make_scorer`` object for use in ``GridSearchCV`` when
        the current task is classification and the target is multi-output.

        Scoring Strategy
        ----------------
        For multi-output classification, each target column is scored independently,
        and the final CV score is the arithmetic mean across all target columns.

        Supported scoring names
        -----------------------
        - ``"accuracy"``
        - ``"f1"``
        - ``"f1_weighted"``
        - ``"precision_weighted"``
        - ``"recall_weighted"``

        Parameters
        ----------
        scoring : str
            Scoring name used to determine which per-target classification metric
            should be applied.

        Returns
        -------
        callable
            Scorer object created by ``make_scorer`` for use in ``GridSearchCV``.

        Raises
        ------
        ValueError
            If the scoring name is not supported for multi-output classification.

        Notes
        -----
        - Each target column contributes equally to the final aggregated score.
        - ``"accuracy"`` computes plain classification accuracy per target, then
        averages across targets.
        - ``"f1"`` and ``"f1_weighted"`` both use
        ``f1_score(..., average="weighted")`` for each target so that both binary
        and multiclass targets are supported in multi-output workflows.
        - ``"precision_weighted"`` and ``"recall_weighted"`` use weighted averaging
        per target with ``zero_division=0`` for stable metric computation.
        - Inputs are normalized to ``DataFrame`` form internally so predicted and
        true target columns remain aligned during per-target evaluation.
        """

        def _to_dataframes(y_true, y_pred):
            # Normalize true/pred targets to DataFrame form for aligned per-target scoring.
            if not isinstance(y_true, pd.DataFrame):
                y_true = pd.DataFrame(y_true)
            if not isinstance(y_pred, pd.DataFrame):
                y_pred = pd.DataFrame(y_pred, columns=y_true.columns)
            return y_true, y_pred

        def _mean_per_target_score(y_true, y_pred, scorer_func):
            # Compute one score per target column, then return the mean across targets.
            y_true, y_pred = _to_dataframes(y_true, y_pred)
            scores = [scorer_func(y_true[col], y_pred[col]) for col in y_true.columns]
            return float(np.mean(scores)) if scores else 0.0

        if scoring == "accuracy":
            return make_scorer(
                lambda y_true, y_pred: _mean_per_target_score(
                    y_true, y_pred, accuracy_score
                ),
                greater_is_better=True,
            )

        if scoring == "f1":
            return make_scorer(
                lambda y_true, y_pred: _mean_per_target_score(
                    y_true,
                    y_pred,
                    lambda yt, yp: f1_score(yt, yp, average="weighted"),
                ),
                greater_is_better=True,
            )

        if scoring == "f1_weighted":
            return make_scorer(
                lambda y_true, y_pred: _mean_per_target_score(
                    y_true,
                    y_pred,
                    lambda yt, yp: f1_score(yt, yp, average="weighted"),
                ),
                greater_is_better=True,
            )

        if scoring == "precision_weighted":
            return make_scorer(
                lambda y_true, y_pred: _mean_per_target_score(
                    y_true,
                    y_pred,
                    lambda yt, yp: precision_score(
                        yt, yp, average="weighted", zero_division=0
                    ),
                ),
                greater_is_better=True,
            )

        if scoring == "recall_weighted":
            return make_scorer(
                lambda y_true, y_pred: _mean_per_target_score(
                    y_true,
                    y_pred,
                    lambda yt, yp: recall_score(
                        yt, yp, average="weighted", zero_division=0
                    ),
                ),
                greater_is_better=True,
            )

        raise ValueError(
            f"⚠️ Unsupported multi-output classification scoring: {scoring} ‼️"
        )

    # -------------------- Helper: Scoring method for multiple output in regression --------------------
    def _build_multioutput_regression_scorer(self, scoring: str):
        """
        Build a scorer for multi-output regression model selection.

        This helper creates a ``make_scorer`` object for use in ``GridSearchCV`` when
        the current task is regression and the target is multi-output.

        Scoring Strategy
        ----------------
        For multi-output regression, a single aggregated score is produced using
        uniform-average aggregation across target columns.

        Supported scoring names
        -----------------------
        - ``"r2"``
        - ``"neg_mean_squared_error"``
        - ``"neg_mean_absolute_error"``

        Parameters
        ----------
        scoring : str
            Scoring name used to determine which regression metric should be applied.

        Returns
        -------
        callable
            Scorer object created by ``make_scorer`` for use in ``GridSearchCV``.

        Raises
        ------
        ValueError
            If the scoring name is not supported for multi-output regression.

        Notes
        -----
        - ``"r2"`` uses ``r2_score(..., multioutput="uniform_average")``.
        - ``"neg_mean_squared_error"`` uses
        ``mean_squared_error(..., multioutput="uniform_average")`` together with
        ``greater_is_better=False`` so that ``GridSearchCV`` can still maximize
        the resulting score correctly.
        - ``"neg_mean_absolute_error"`` uses
        ``mean_absolute_error(..., multioutput="uniform_average")`` together with
        ``greater_is_better=False`` for the same reason.
        - All target columns contribute equally to the final aggregated score.
        """
        if scoring == "r2":
            return make_scorer(
                lambda y_true, y_pred: r2_score(
                    y_true,
                    y_pred,
                    multioutput="uniform_average",
                ),
                greater_is_better=True,
            )

        if scoring == "neg_mean_squared_error":
            return make_scorer(
                lambda y_true, y_pred: mean_squared_error(
                    y_true,
                    y_pred,
                    multioutput="uniform_average",
                ),
                greater_is_better=False,
            )

        if scoring == "neg_mean_absolute_error":
            return make_scorer(
                lambda y_true, y_pred: mean_absolute_error(
                    y_true,
                    y_pred,
                    multioutput="uniform_average",
                ),
                greater_is_better=False,
            )

        raise ValueError(
            f"⚠️ Unsupported multi-output regression scoring: {scoring} ‼️"
        )

    # -------------------- Helper: Scaler selections --------------------
    def _build_scaler(self, scaler_type: str = "standard"):
        """
        Build a scaler object based on user selection.

        Parameters
        ----------
        scaler_type : str, default="standard"
            Supported scaler types:
            - "standard" / "std"   -> StandardScaler
            - "minmax" / "min_max" -> MinMaxScaler
            - "robust" / "rbst"    -> RobustScaler
            - "none" / "no" / "off" -> no scaler (returns None)

        Returns
        -------
        Any or None
            sklearn scaler instance if a valid scaler is selected,
            otherwise None when scaler_type indicates no scaling.

        Raises
        ------
        ValueError
            If scaler_type is not supported.

        Notes
        -----
        This helper is intended for model-layer usage together with
        ``fit_with_grid(..., extra_steps=...)``.

        Example
        -------
        A model layer may do:

        extra_steps = []
        scaler = self._build_scaler(scaler_type)
        if scaler is not None:
            extra_steps.append(("scaler", scaler))
        """
        scaler_type = scaler_type.lower().strip()

        if scaler_type in ["none", "no", "off"]:
            return None
        if scaler_type in ["standard", "std"]:
            return StandardScaler()
        if scaler_type in ["minmax", "min_max"]:
            return MinMaxScaler()
        if scaler_type in ["robust", "rbst"]:
            return RobustScaler()

        raise ValueError(
            "⚠️ scaler_type must be 'standard', 'minmax', 'robust', or 'none' ‼️"
        )

    # -------------------- Split train and test dataset --------------------
    def train_test_split_engine(
        self,
        test_size: float = 0.2,
        split_random_state: int = 42,
        stratify: bool = True,
    ):
        """
        Split X/Y into train and test sets, with optional stratification.

        Parameters
        ----------
        test_size : float, default=0.2
            Fraction of samples used for the test set.

        split_random_state : int, default=42
            Seed for reproducibility.

        stratify : bool, default=True
            Whether to stratify by Y labels. Applied only when:
            - task == "classification"
            - Y is single-output

        Returns
        -------
        tuple
            (X_train, X_test, Y_train, Y_test)

        Notes
        -----
        - Stratification requires 1D labels. Multi-output targets will ignore stratify.
        """
        y = self.cleaned_Y_data
        use_stratify = None

        # ---------- Stratify applied ----------
        if (
            self.task == "classification"
            and stratify
            and (not self._is_multi_output(y))
        ):
            use_stratify = y

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.cleaned_X_data,
            self.cleaned_Y_data,
            test_size=test_size,
            random_state=split_random_state,
            stratify=use_stratify,
        )
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    # -------------------- Build preprocess --------------------
    def build_preprocessor(
        self,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        cat_encoder: str = "ohe",
    ) -> ColumnTransformer:
        """
        Build a preprocessing ColumnTransformer for numeric and categorical features.

        Parameters
        ----------
        categorical_cols : Optional[List[str]], default=None
            Categorical feature columns. If None, inferred by dtype:
            object/category/bool.

        numeric_cols : Optional[List[str]], default=None
            Numeric feature columns. If None, inferred as all remaining columns
            not in categorical_cols.

        cat_encoder : str, default="ohe"
            Categorical encoding strategy:
            - "ohe" / "onehot" / "one_hot": OneHotEncoder(handle_unknown="ignore")
            - "ordinal" / "ord" / "ord_label":
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        Returns
        -------
        ColumnTransformer
            Transformer used in the preprocessing step of the Pipeline.

        Side Effects
        ------------
        Stores:
        - self._numeric_cols
        - self._categorical_cols
        for fallback feature name extraction.
        """
        df = self.cleaned_X_data

        if categorical_cols is None:
            categorical_cols = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

        if numeric_cols is None:
            numeric_cols = [c for c in df.columns if c not in categorical_cols]

        self._numeric_cols = numeric_cols
        self._categorical_cols = categorical_cols

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        cat_encoder = cat_encoder.lower().strip()
        if cat_encoder in ["ohe", "onehot", "one_hot"]:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

        elif cat_encoder in ["ordinal", "ord", "ord_label"]:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ord",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )
        else:
            raise ValueError("⚠️ cat_encoder must be 'ohe' or 'ordinal' ‼️")

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    # -------------------- Fit model (with optional CV) --------------------
    def fit_with_grid(
        self,
        base_model,
        param_grid: Optional[Dict[str, Any]],
        use_cv: bool,
        cv_folds: int | None,
        scoring: str,
        split_random_state: int = 42,
        cat_encoder: str = "ohe",
        extra_steps: Optional[List[Tuple[str, Any]]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """
        Fit a preprocessing + estimator Pipeline, optionally using GridSearchCV.

        This method builds a full sklearn workflow consisting of:

        1. a preprocessing step created by ``build_preprocessor()``
        2. optional intermediate extra steps such as scaling or PCA
        3. the final estimator step provided by the subclass

        If cross-validation is enabled, the method runs ``GridSearchCV`` and stores
        the best fitted pipeline. Otherwise, it fits the pipeline directly on the
        training split.

        Parameters
        ----------
        base_model : Any
            Sklearn estimator instance used as the final model step, such as
            ``DecisionTreeClassifier`` or ``RandomForestRegressor``.

        param_grid : Optional[Dict[str, Any]]
            Parameter grid used by ``GridSearchCV``. Keys must follow sklearn Pipeline
            parameter naming rules and include the estimator step prefix, for example:

            - ``{"classifier__max_depth": [3, 5, None]}``
            - ``{"regressor__n_estimators": [100, 300]}``

            If ``None`` or empty, cross-validation can still run, but no hyperparameter
            combinations will be searched beyond the default estimator configuration.

        use_cv : bool
            Whether to apply ``GridSearchCV``. If ``False``, the pipeline is fitted
            directly without cross-validation.

        cv_folds : int | None
            Number of folds used in cross-validation when ``use_cv=True``.
            If ``use_cv=False``, this value is ignored.

        scoring : str
            Scoring name used during model selection.

            Examples include:

            - ``"accuracy"``
            - ``"f1"``
            - ``"f1_weighted"``
            - ``"precision_weighted"``
            - ``"recall_weighted"``
            - ``"r2"``
            - ``"neg_mean_squared_error"``
            - ``"neg_mean_absolute_error"``

            For single-output tasks, this value is passed directly to sklearn
            scoring.

            For multi-output tasks, the base layer builds a task-specific scorer
            from this scoring name through:

            - ``_build_multioutput_classification_scorer()``
            - ``_build_multioutput_regression_scorer()``

        split_random_state : int, default=42
            Random seed used for reproducibility in the CV splitter.

        cat_encoder : str, default="ohe"
            Categorical encoding strategy passed to ``build_preprocessor()``.
            Supported values include:

            - ``"ohe"``, ``"onehot"``, ``"one_hot"``
            - ``"ordinal"``, ``"ord"``, ``"ord_label"``

        extra_steps : Optional[List[Tuple[str, Any]]], default=None
            Optional additional Pipeline steps inserted between preprocessing and the
            final estimator. This is typically used for components such as a scaler,
            dimensionality reduction, or feature selector.

        Returns
        -------
        Tuple[Optional[Dict[str, Any]], Optional[float]]
            A tuple ``(best_params, best_score)``.

            - ``best_params``:
            Best parameter combination found by ``GridSearchCV`` when
            ``use_cv=True``; otherwise ``None``.

            - ``best_score``:
            Best cross-validation score when ``use_cv=True``; otherwise ``None``.

        Raises
        ------
        ValueError
            If ``train_test_split_engine()`` has not been executed before model fitting.

        Side Effects
        ------------
        Sets and updates the following attributes:

        - ``self.model_pipeline`` :
        The fitted Pipeline. When CV is used, this is the ``best_estimator_`` from
        ``GridSearchCV``. Otherwise, it is the directly fitted pipeline.

        - ``self.feature_names`` :
        Extracted transformed feature names after fitting.

        - ``self.cv_search_report`` :
        A summary dictionary containing CV settings, best parameters, best CV score,
        and the top-ranked CV results when ``use_cv=True``.

        Also calls ``save_cv_search_report()`` automatically when cross-validation
        results are available.

        Notes
        -----
        CV splitter policy:

        - classification + single-output target -> ``StratifiedKFold``
        - otherwise -> ``KFold``

        Scoring policy:

        - single-output target -> use the provided ``scoring`` string
        - multi-output classification -> use a base-layer scorer built from the
        selected classification scoring name
        - multi-output regression -> use a base-layer scorer built from the
        selected regression scoring name
        """
        if self.X_train is None or self.Y_train is None:
            raise ValueError("⚠️ Run train_test_split_engine() before training ‼️")

        # ---------- Setup preprocess ----------
        preprocess = self.build_preprocessor(cat_encoder=cat_encoder)

        # ---------- Setup step to pipeline ----------
        steps = [("preprocess", preprocess)]

        if extra_steps:  # Extra steps
            steps.extend(extra_steps)

        steps.append((self.step_name, base_model))
        pipe = Pipeline(steps=steps)  # Final pipeline

        best_params = None
        best_score = None

        # ---------- CV application ----------
        if use_cv:
            is_multi = self._is_multi_output(self.Y_train)

            if self.task == "classification" and not is_multi:
                splitter = StratifiedKFold(
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=split_random_state,
                )
                scoring_for_cv = scoring
            else:
                splitter = KFold(
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=split_random_state,
                )

                # ---------- Multiple output scoring methods ----------
                if is_multi:
                    if self.task == "classification":
                        scoring_for_cv = self._build_multioutput_classification_scorer(
                            scoring
                        )
                    elif self.task == "regression":
                        scoring_for_cv = self._build_multioutput_regression_scorer(
                            scoring
                        )
                    else:
                        raise ValueError(f"⚠️ Unsupported task type: {self.task} ‼️")
                else:
                    scoring_for_cv = scoring

            # ---------- CV training ----------
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid or {},
                scoring=scoring_for_cv,
                cv=splitter,
                n_jobs=-1,
                refit=True,
            )
            gs.fit(self.X_train, self.Y_train)
            self.model_pipeline = gs.best_estimator_
            best_params = gs.best_params_
            best_score = gs.best_score_

            # ---------- CV report saved as CSV file ----------
            cv_results_df = pd.DataFrame(gs.cv_results_)
            top_cv_results = (
                cv_results_df[
                    ["rank_test_score", "mean_test_score", "std_test_score", "params"]
                ]
                .sort_values("rank_test_score")
                .head(5)
                .to_dict(orient="records")
            )

            self.cv_search_report = {
                "use_cv": use_cv,
                "cv_folds": cv_folds,
                "scoring": scoring,
                "best_params": best_params,
                "best_cv_score": best_score,
                "top_cv_results": top_cv_results,
            }

            self.save_cv_search_report()

        # ---------- Training original model only ----------
        else:
            pipe.fit(self.X_train, self.Y_train)
            self.model_pipeline = pipe

        self._extract_feature_names()  # Get feature name

        return best_params, best_score

    # -------------------- Save CV report --------------------
    def save_cv_search_report(self):
        """
        Save the stored cross-validation search summary to a CSV file.

        This method exports the top-ranked CV results stored in
        ``self.cv_search_report["top_cv_results"]`` into the predefined
        ``CV_REPORT_DIR`` folder. The output filename is automatically generated from
        the current model type and the current timestamp.

        Returns
        -------
        str | None
            Full saved file path if the report is successfully exported.
            Returns ``None`` if no CV report or no top CV results are available.

        Side Effects
        ------------
        - Creates a CSV file under ``CV_REPORT_DIR``.
        - Prints the saved file path when export succeeds.
        - Prints a warning message and returns ``None`` when export is skipped.

        Notes
        -----
        The saved CSV currently contains the top CV result rows with these fields:

        - ``rank_test_score``
        - ``mean_test_score``
        - ``std_test_score``
        - ``params``

        The filename format is:

        ``{model_name}_cv_report_{YYYYMMDD_HHMMSS}.csv``

        where ``model_name`` comes from ``self.input_model_type``. If that value is
        not available, ``"model"`` is used as the fallback name.
        """
        if not self.cv_search_report:
            print("⚠️ No CV search report available to save ‼️")
            return None

        top_cv_results = self.cv_search_report.get("top_cv_results")
        if not top_cv_results:
            print("⚠️ No top CV results available to save ‼️")
            return None

        model_name = self.input_model_type or "model"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file_name = f"{model_name}_cv_report_{stamp}.csv"
        save_path = os.path.join(CV_REPORT_DIR, final_file_name)

        cv_df = pd.DataFrame(top_cv_results)
        cv_df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"📦 CV report saved path: {save_path}")
        return save_path

    # -------------------- Feature names --------------------
    def _extract_feature_names(self):
        """
        Extract feature names from the fitted preprocessing step (best-effort).

        Behavior
        --------
        1) Preferred path:
        Use ColumnTransformer.get_feature_names_out() when available, and
        reject obviously generic names like x0, x1, ...

        2) Fallback path:
        Use cached numeric/categorical column lists. When the categorical
        pipeline contains OneHotEncoder, attempt to expand category names via
        ohe.get_feature_names_out(categorical_cols), so the final name list
        matches the transformed feature dimension.

        Side Effects
        ------------
        Sets:
        - self.feature_names : Optional[List[str]]
        """
        if self.model_pipeline is None:
            self.feature_names = None
            return

        # ---------- Check preprocess executed ----------
        pre = self.model_pipeline.named_steps.get("preprocess", None)
        if pre is None:
            self.feature_names = None
            return

        # ---------- Extract method ----------
        try:
            if hasattr(pre, "get_feature_names_out"):
                names = pre.get_feature_names_out()
                if names is not None:
                    names = list(names)
                    if all(
                        re.match(r"^[a-zA-Z]\d+$", str(n)) for n in names
                    ):  # Check generic feature name
                        raise ValueError("⚠️ generic feature names ‼️")

                    self.feature_names = names
                    return
            raise ValueError("⚠️ no usable feature names ‼️")

        # ---------- Backup method-1 ----------
        except Exception:
            num_cols = getattr(self, "_numeric_cols", None)
            cat_cols = getattr(self, "_categorical_cols", None)

            if num_cols is None or cat_cols is None:
                self.feature_names = None
                return

            names_out = list(num_cols)

            try:
                cat_pipe = pre.named_transformers_.get("cat", None)
                if (
                    cat_pipe is not None
                    and hasattr(cat_pipe, "named_steps")
                    and ("ohe" in cat_pipe.named_steps)
                ):
                    ohe = cat_pipe.named_steps["ohe"]
                    names_out.extend(ohe.get_feature_names_out(cat_cols).tolist())
                else:
                    names_out.extend(list(cat_cols))

                self.feature_names = names_out

            # ---------- Backup method-2 ----------
            except Exception:
                self.feature_names = list(num_cols) + list(cat_cols)


# =================================================
