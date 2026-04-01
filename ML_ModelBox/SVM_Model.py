"""
SVM model-layer trainers for classification and regression tasks.

This module defines the concrete model-layer trainer classes for SVM-based
pipelines:

- ``SVMClassifier_Model``
- ``SVMRegressor_Model``

These model-layer classes inherit from their corresponding mission-layer
classes:

- :class:`SVMClassifier_Missioner`
- :class:`SVMRegressor_Missioner`

Architecture Role
-----------------
This module belongs to the model layer of the project architecture.

The mission/base layers are responsible for shared workflow utilities such as:
- train/test split handling
- preprocessing pipeline construction
- optional GridSearchCV execution through shared fitting logic
- evaluation metrics and reporting
- persistence utilities for saving and loading trained models
- additional diagnostic and plotting helpers

The model layer is responsible for:
- constructing concrete sklearn estimator instances
- wrapping estimators for multi-output tasks when required
- defining fixed GridSearchCV parameter grids
- assembling optional preprocessing steps such as scaling and PCA
- calling the inherited shared fitting routine
- triggering mission-layer evaluation
- returning a compact training summary dictionary

Classes
-------
SVMClassifier_Model
    Model-layer trainer for SVC-based classification pipelines.

SVMRegressor_Model
    Model-layer trainer for SVR-based regression pipelines.

Notes
-----
- For multi-output classification, the classifier model is wrapped with
  :class:`sklearn.multioutput.MultiOutputClassifier`.
- For multi-output regression, the regressor model is wrapped with
  :class:`sklearn.multioutput.MultiOutputRegressor`.
- Optional PCA and scaler steps can be inserted before the estimator.
- When ``use_cv=True``, fixed parameter grids are defined in this module and
  passed to the inherited fitting workflow.
"""

# -------------------- Import Modules --------------------
from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR

from Zeus.ML_MissionBox.SVMClassifier_Missioner import SVMClassifier_Missioner
from Zeus.ML_MissionBox.SVMRegressor_Missioner import SVMRegressor_Missioner


# -------------------- SVMClassifier Model --------------------
class SVMClassifier_Model(SVMClassifier_Missioner):
    """
    Model-layer trainer for Support Vector Classification (SVC).

    This class extends :class:`SVMClassifier_Missioner` and provides the concrete
    training logic for SVM-based classification pipelines. It is responsible for
    constructing the estimator, configuring optional preprocessing steps, defining
    cross-validation parameter grids, invoking the shared fitting workflow, and
    returning a compact training summary.

    Main Responsibilities
    ---------------------
    - Construct an ``SVC`` estimator instance.
    - Automatically wrap the estimator with ``MultiOutputClassifier`` when the
    target is multi-output.
    - Optionally insert scaling and PCA steps into the pipeline.
    - Define a fixed parameter grid when cross-validation is enabled.
    - Call the inherited shared fitting routine from the base/mission layers.
    - Call mission-layer evaluation after training.
    - Return a compact dictionary containing training and evaluation results.

    Workflow Summary
    ----------------
    1. Validate that training data are available.
    2. Record training configuration into instance attributes.
    3. Build the base SVC estimator.
    4. Wrap with ``MultiOutputClassifier`` when needed.
    5. Build optional preprocessing steps such as scaler and PCA.
    6. Define a GridSearchCV parameter grid when ``use_cv=True``.
    7. Train the pipeline through inherited shared fitting logic.
    8. Evaluate the trained model through mission-layer utilities.
    9. Return a compact training summary dictionary.

    Notes
    -----
    - Single-output and multi-output classification are both supported.
    - When ``use_cv=True``, parameter grids are defined separately for single-output
    and multi-output pipelines because wrapped estimators require different
    parameter prefixes.
    - Probability output is enabled in the underlying SVC estimator so that
    probability-based evaluation and preview utilities remain available.
    """

    # -------------------- SVM classifier model training --------------------
    def train(
        self,
        use_cv: bool = True,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        shrinking: bool = True,
        tol: float = 1e-3,
        class_weight: Optional[dict | str] = None,
        split_random_state: int = 42,
        use_pca: bool = False,
        pca_n_components: Optional[int] = None,
        scaler_type: str = "standard",
        cat_encoder: str = "ohe",
    ) -> Dict[str, Any]:
        """
        Train an SVC-based classification pipeline with optional GridSearchCV.

        This method builds the classification estimator, optionally adds preprocessing
        steps such as scaling and PCA, performs model fitting through inherited shared
        logic, and then runs mission-layer evaluation.

        Training Behavior
        -----------------
        - If ``use_cv=False``:
        The method builds an SVC estimator using the provided fixed hyperparameters.

        - If ``use_cv=True``:
        The method defines a fixed parameter grid and delegates model selection to the
        inherited GridSearchCV workflow.

        - If the target is multi-output:
        The base estimator is automatically wrapped with
        :class:`sklearn.multioutput.MultiOutputClassifier`.

        Parameters
        ----------
        use_cv : bool, default=True
            Whether to use GridSearchCV for hyperparameter selection.

        cv_folds : int, default=5
            Number of cross-validation folds used when `use_cv=True`.

        scoring : str, default="f1_weighted"
            Scoring method used during cross-validation for single-output
            classification.

            For multi-output classification, the inherited mission/base layer may
            automatically substitute a custom scorer if needed.

        C : float, default=1.0
            Regularization strength used when `use_cv=False`.

        kernel : str, default="rbf"
            Kernel type used when `use_cv=False`.

        gamma : str or float, default="scale"
            Kernel coefficient for ``"rbf"``, ``"poly"``, and ``"sigmoid"`` kernels.

        degree : int, default=3
            Polynomial degree used when `kernel="poly"`.

        coef0 : float, default=0.0
            Independent term in the kernel function for ``"poly"`` and ``"sigmoid"``.

        shrinking : bool, default=True
            Whether to use the shrinking heuristic.

        tol : float, default=1e-3
            Tolerance for the stopping criterion.

        class_weight : dict, str, or None, default=None
            Class weighting strategy passed to ``SVC``.

            Common values include:
            - ``None``
            - ``"balanced"``
            - a user-defined class-weight dictionary

        split_random_state : int, default=42
            Random seed passed to shared training utilities where applicable.

        use_pca : bool, default=False
            Whether to insert a PCA step into the pipeline.

        pca_n_components : int or None, default=None
            Number of PCA components used when `use_pca=True` and `use_cv=False`.

            When `use_cv=True`, PCA component choices may instead be searched through
            the parameter grid.

        scaler_type : str, default="standard"
            Scaler type used when `use_cv=False`.

            Supported handling depends on inherited scaler-building utilities.

        Returns
        -------
        Dict[str, Any]
            Compact training summary dictionary containing:
            - ``"model"`` :
            model label
            - ``"use_cv"`` :
            whether cross-validation was used
            - ``"best_params"`` :
            best parameter set from shared fitting logic
            - ``"best_cv_score"`` :
            best cross-validation score
            - ``"feature_names_len"`` :
            number of feature names recorded after preprocessing
            - ``"evaluation"`` :
            evaluation dictionary returned by the mission layer

        Raises
        ------
        ValueError
            If training data are not ready, typically when
            `train_test_split_engine()` has not been run first.

        Side Effects
        ------------
        Updates shared instance state, including:
        - ``self.input_model_type``
        - ``self.input_use_cv``
        - ``self.input_cv_folds``
        - trained pipeline stored through inherited fitting logic
        - feature-name tracking from the shared workflow
        - evaluation-related cached predictions and previews through the mission layer

        Notes
        -----
        - When `use_cv=True`, scaler options are searched as part of the parameter grid.
        - When `use_cv=False`, scaler selection is built directly from `scaler_type`.
        - Parameter-grid key names differ between single-output and multi-output modes
        because wrapped estimators require nested prefixes such as
        ``classifier__estimator__...``.
        """
        if self.Y_train is None or self.X_train is None:
            raise ValueError("⚠️  Run train_test_split_engine() before training ‼️")

        # ---------- Record parameters input ----------
        self.input_model_type = "SVC"
        self.input_use_cv = use_cv
        self.input_cv_folds = cv_folds

        # ---------- Base estimator ----------
        svc_estimator = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            shrinking=shrinking,
            probability=True,
            tol=tol,
            class_weight=class_weight,
            random_state=split_random_state,
        )

        # ---------- Multi-output handling ----------
        if self._is_multi_output(self.cleaned_Y_data):
            base_model = MultiOutputClassifier(svc_estimator)
        else:
            base_model = svc_estimator

        # ---------- Extra processes in pipeline ----------
        extra_steps = []

        # ---------- Scaler ----------
        if use_cv:
            extra_steps.append(("scaler", StandardScaler()))
        else:
            scaler = self._build_scaler(scaler_type)
            if scaler is not None:
                extra_steps.append(("scaler", scaler))

        # ---------- PCA ----------
        if use_pca:
            max_components = self.X_train.shape[1]

            if pca_n_components is not None and pca_n_components > max_components:
                raise ValueError(
                    f"⚠️ pca_n_components={pca_n_components} exceeds current feature count={max_components} ‼️"
                )

            extra_steps.append(("pca", PCA(n_components=pca_n_components)))

        param_grid = None

        # ---------- CV parameters ----------
        if use_cv:
            if self._is_multi_output(self.cleaned_Y_data):
                param_grid = {
                    "scaler": [
                        StandardScaler(),
                        MinMaxScaler(),
                        RobustScaler(),
                        "passthrough",
                    ],
                    "classifier__estimator__C": [0.1, 1, 10],
                    "classifier__estimator__kernel": ["linear", "rbf"],
                    "classifier__estimator__gamma": ["scale", "auto"],
                    "classifier__estimator__class_weight": [None, "balanced"],
                }
            else:
                param_grid = {
                    "scaler": [
                        StandardScaler(),
                        MinMaxScaler(),
                        RobustScaler(),
                        "passthrough",
                    ],
                    "classifier__C": [0.1, 1, 10],
                    "classifier__kernel": ["linear", "rbf"],
                    "classifier__gamma": ["scale", "auto"],
                    "classifier__class_weight": [None, "balanced"],
                }

        # ---------- Set the range for PCA component and feature amount ----------
        if use_cv and use_pca:
            max_components = self.X_train.shape[1]
            valid_components = [n for n in [2, 3, 5] if n <= max_components]

            if valid_components:
                param_grid["pca__n_components"] = valid_components

        # ---------- CV or original model training ----------
        best_params, best_score = self.fit_with_grid(
            base_model=base_model,
            param_grid=param_grid,
            use_cv=use_cv,
            cv_folds=cv_folds,
            scoring=scoring,
            split_random_state=split_random_state,
            extra_steps=extra_steps,
            cat_encoder=cat_encoder,
        )

        # ---------- Model evaluation ----------
        eval_results = self.model_evaluation_engine()

        return {
            "model": self.input_model_type,
            "use_cv": use_cv,
            "best_params": best_params,
            "best_cv_score": best_score,
            "feature_names_len": (
                None if self.feature_names is None else len(self.feature_names)
            ),
            "evaluation": eval_results,
        }


# -------------------- SVMRegressor Model --------------------
class SVMRegressor_Model(SVMRegressor_Missioner):
    """
    Model-layer trainer for Support Vector Regression (SVR).

    This class extends :class:`SVMRegressor_Missioner` and provides the concrete
    training logic for SVM-based regression pipelines. It is responsible for
    constructing the estimator, configuring optional preprocessing steps, defining
    cross-validation parameter grids, invoking the shared fitting workflow, and
    returning a compact training summary.

    Main Responsibilities
    ---------------------
    - Construct an ``SVR`` estimator instance.
    - Automatically wrap the estimator with ``MultiOutputRegressor`` when the
    target is multi-output.
    - Optionally insert scaling and PCA steps into the pipeline.
    - Define a fixed parameter grid when cross-validation is enabled.
    - Call the inherited shared fitting routine from the base/mission layers.
    - Call mission-layer evaluation after training.
    - Return a compact dictionary containing training and evaluation results.

    Workflow Summary
    ----------------
    1. Validate that training data are available.
    2. Record training configuration into instance attributes.
    3. Build the base SVR estimator.
    4. Wrap with ``MultiOutputRegressor`` when needed.
    5. Build optional preprocessing steps such as scaler and PCA.
    6. Define a GridSearchCV parameter grid when ``use_cv=True``.
    7. Train the pipeline through inherited shared fitting logic.
    8. Evaluate the trained model through mission-layer utilities.
    9. Return a compact training summary dictionary.

    Notes
    -----
    - Single-output and multi-output regression are both supported.
    - When ``use_cv=True``, parameter grids are defined separately for single-output
    and multi-output pipelines because wrapped estimators require different
    parameter prefixes.
    - This class focuses on model construction and training orchestration, while
    regression metrics, diagnostic plotting, and persistence are handled by the
    inherited mission/base layers.
    """

    # -------------------- SVM regressor model training --------------------
    def train(
        self,
        use_cv: bool = True,
        cv_folds: int = 5,
        scoring: str = "r2",
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        epsilon: float = 0.1,
        shrinking: bool = True,
        tol: float = 1e-3,
        cache_size: float = 200,
        max_iter: int = -1,
        split_random_state: int = 42,
        use_pca: bool = False,
        pca_n_components: Optional[int] = None,
        scaler_type: str = "standard",
        cat_encoder: str = "ohe",
    ) -> Dict[str, Any]:
        """
        Train an SVR-based regression pipeline with optional GridSearchCV.

        This method builds the regression estimator, optionally adds preprocessing
        steps such as scaling and PCA, performs model fitting through inherited shared
        logic, and then runs mission-layer evaluation.

        Training Behavior
        -----------------
        - If ``use_cv=False``:
        The method builds an SVR estimator using the provided fixed hyperparameters.

        - If ``use_cv=True``:
        The method defines a fixed parameter grid and delegates model selection to the
        inherited GridSearchCV workflow.

        - If the target is multi-output:
        The base estimator is automatically wrapped with
        :class:`sklearn.multioutput.MultiOutputRegressor`.

        Parameters
        ----------
        use_cv : bool, default=True
            Whether to use GridSearchCV for hyperparameter selection.

        cv_folds : int, default=5
            Number of cross-validation folds used when `use_cv=True`.

        scoring : str, default="r2"
            Scoring method used during cross-validation for single-output regression.

            For multi-output regression, the inherited mission/base layer may
            automatically substitute a custom scorer if needed.

        C : float, default=1.0
            Regularization strength used when `use_cv=False`.

        kernel : str, default="rbf"
            Kernel type used when `use_cv=False`.

        gamma : str or float, default="scale"
            Kernel coefficient for ``"rbf"``, ``"poly"``, and ``"sigmoid"`` kernels.

        degree : int, default=3
            Polynomial degree used when `kernel="poly"`.

        coef0 : float, default=0.0
            Independent term in the kernel function for ``"poly"`` and ``"sigmoid"``.

        epsilon : float, default=0.1
            Epsilon value in the epsilon-SVR loss function.

        shrinking : bool, default=True
            Whether to use the shrinking heuristic.

        tol : float, default=1e-3
            Tolerance for the stopping criterion.

        cache_size : float, default=200
            Size of the kernel cache in megabytes.

        max_iter : int, default=-1
            Hard iteration limit for the solver.

            - ``-1`` means no explicit iteration limit.
            - Positive integers impose a maximum number of iterations.

        split_random_state : int, default=42
            Random seed passed to shared training utilities where applicable.

        use_pca : bool, default=False
            Whether to insert a PCA step into the pipeline.

        pca_n_components : int or None, default=None
            Number of PCA components used when `use_pca=True` and `use_cv=False`.

            When `use_cv=True`, PCA component choices may instead be searched through
            the parameter grid.

        scaler_type : str, default="standard"
            Scaler type used when `use_cv=False`.

            Supported handling depends on inherited scaler-building utilities.

        Returns
        -------
        Dict[str, Any]
            Compact training summary dictionary containing:
            - ``"model"`` :
            model label
            - ``"use_cv"`` :
            whether cross-validation was used
            - ``"best_params"`` :
            best parameter set from shared fitting logic
            - ``"best_cv_score"`` :
            best cross-validation score
            - ``"feature_names_len"`` :
            number of feature names recorded after preprocessing
            - ``"evaluation"`` :
            evaluation dictionary returned by the mission layer

        Raises
        ------
        ValueError
            If training data are not ready, typically when
            `train_test_split_engine()` has not been run first.

        Side Effects
        ------------
        Updates shared instance state, including:
        - ``self.input_model_type``
        - ``self.input_use_cv``
        - ``self.input_cv_folds``
        - trained pipeline stored through inherited fitting logic
        - feature-name tracking from the shared workflow
        - evaluation-related cached predictions and previews through the mission layer

        Notes
        -----
        - When `use_cv=True`, scaler options are searched as part of the parameter grid.
        - When `use_cv=False`, scaler selection is built directly from `scaler_type`.
        - Parameter-grid key names differ between single-output and multi-output modes
        because wrapped estimators require nested prefixes such as
        ``regressor__estimator__...``.
        - For multi-output regression, each target is modeled through the wrapped
        estimator interface provided by ``MultiOutputRegressor``.
        """
        if self.Y_train is None or self.X_train is None:
            raise ValueError("⚠️  Run train_test_split_engine() before training ‼️")

        # ---------- Record parameters input ----------
        self.input_model_type = "SVR"
        self.input_use_cv = use_cv
        self.input_cv_folds = cv_folds

        # ---------- Base estimator ----------
        svr_estimator = SVR(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            epsilon=epsilon,
            shrinking=shrinking,
            tol=tol,
            cache_size=cache_size,
            max_iter=max_iter,
        )

        # ---------- Multi-output handling ----------
        if self._is_multi_output(self.cleaned_Y_data):
            base_model = MultiOutputRegressor(svr_estimator)
        else:
            base_model = svr_estimator

        # ---------- Extra processes in pipeline ----------
        extra_steps = []

        # ---------- Scaler ----------
        if use_cv:
            extra_steps.append(("scaler", StandardScaler()))
        else:
            scaler = self._build_scaler(scaler_type)
            if scaler is not None:
                extra_steps.append(("scaler", scaler))

        # ---------- PCA ----------
        if use_pca:
            max_components = self.X_train.shape[1]

            if pca_n_components is not None and pca_n_components > max_components:
                raise ValueError(
                    f"⚠️ pca_n_components={pca_n_components} exceeds current feature count={max_components} ‼️"
                )

            extra_steps.append(("pca", PCA(n_components=pca_n_components)))

        param_grid = None

        # ---------- CV parameters ----------
        if use_cv:
            if self._is_multi_output(self.cleaned_Y_data):
                param_grid = {
                    "scaler": [
                        StandardScaler(),
                        MinMaxScaler(),
                        RobustScaler(),
                        "passthrough",
                    ],
                    "regressor__estimator__C": [0.1, 1, 10],
                    "regressor__estimator__kernel": ["linear", "rbf"],
                    "regressor__estimator__gamma": ["scale", "auto"],
                    "regressor__estimator__epsilon": [0.01, 0.1, 0.5],
                }
            else:
                param_grid = {
                    "scaler": [
                        StandardScaler(),
                        MinMaxScaler(),
                        RobustScaler(),
                        "passthrough",
                    ],
                    "regressor__C": [0.1, 1, 10],
                    "regressor__kernel": ["linear", "rbf"],
                    "regressor__gamma": ["scale", "auto"],
                    "regressor__epsilon": [0.01, 0.1, 0.5],
                }

        # ---------- Set the range for PCA component and feature amount ----------
        if use_cv and use_pca:
            max_components = self.X_train.shape[1]
            valid_components = [n for n in [2, 3, 5] if n <= max_components]

            if valid_components:
                param_grid["pca__n_components"] = valid_components

        # ---------- CV or original model training ----------
        best_params, best_score = self.fit_with_grid(
            base_model=base_model,
            param_grid=param_grid,
            use_cv=use_cv,
            cv_folds=cv_folds,
            scoring=scoring,
            split_random_state=split_random_state,
            extra_steps=extra_steps,
            cat_encoder=cat_encoder,
        )

        # ---------- Model evaluation ----------
        eval_results = self.model_evaluation_engine()

        return {
            "model": self.input_model_type,
            "use_cv": use_cv,
            "best_params": best_params,
            "best_cv_score": best_score,
            "feature_names_len": (
                None if self.feature_names is None else len(self.feature_names)
            ),
            "evaluation": eval_results,
        }


# =================================================
