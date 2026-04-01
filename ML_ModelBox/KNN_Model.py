"""
KNN model-layer trainers for classification and regression tasks.

This module defines the concrete model-layer trainer classes for KNN-based
pipelines:

- ``KNNClassifier_Model``
- ``KNNRegressor_Model``

These model-layer classes inherit from their corresponding mission-layer
classes:

- :class:`KNNClassifier_Missioner`
- :class:`KNNRegressor_Missioner`

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
- assembling optional preprocessing steps such as scaling and PCA
- defining fixed GridSearchCV parameter grids
- calling the inherited shared fitting routine
- triggering mission-layer evaluation
- returning a compact training summary dictionary

Classes
-------
KNNClassifier_Model
    Model-layer trainer for KNeighborsClassifier-based classification pipelines.

KNNRegressor_Model
    Model-layer trainer for KNeighborsRegressor-based regression pipelines.

Notes
-----
- Optional PCA and scaler steps can be inserted before the estimator.
- When ``use_cv=True``, fixed parameter grids are defined in this module and
  passed to the inherited fitting workflow.
- Unlike some other model families, KNN models in this module are not wrapped
  for multi-output handling at the model layer because sklearn KNN estimators
  can already support the required target formats in the surrounding workflow.
"""

# -------------------- Import Modules --------------------
from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from Zeus.ML_MissionBox.KNNClassifier_Missioner import KNNClassifier_Missioner
from Zeus.ML_MissionBox.KNNRegressor_Missioner import KNNRegressor_Missioner


# -------------------- KNNClassifier Model --------------------
class KNNClassifier_Model(KNNClassifier_Missioner):
    """
    Model-layer trainer for K-Nearest Neighbors classification.

    This class extends :class:`KNNClassifier_Missioner` and provides the concrete
    training logic for KNN-based classification pipelines. It is responsible for
    constructing the estimator, configuring optional preprocessing steps, defining
    cross-validation parameter grids, invoking the shared fitting workflow, and
    returning a compact training summary.

    Main Responsibilities
    ---------------------
    - Construct a ``KNeighborsClassifier`` estimator instance.
    - Optionally insert scaling and PCA steps into the pipeline.
    - Define a fixed parameter grid when cross-validation is enabled.
    - Call the inherited shared fitting routine from the base/mission layers.
    - Call mission-layer evaluation after training.
    - Return a compact dictionary containing training and evaluation results.

    Workflow Summary
    ----------------
    1. Validate that training data are available.
    2. Record training configuration into instance attributes.
    3. Build the base KNN classifier estimator.
    4. Build optional preprocessing steps such as scaler and PCA.
    5. Define a GridSearchCV parameter grid when ``use_cv=True``.
    6. Train the pipeline through inherited shared fitting logic.
    7. Evaluate the trained model through mission-layer utilities.
    8. Return a compact training summary dictionary.

    Training Strategy
    -----------------
    - ``use_cv=True``:
    Run GridSearchCV using the fixed search space defined inside `train()`.

    - ``use_cv=False``:
    Train a single KNeighborsClassifier using the user-provided fixed
    hyperparameters.

    Notes
    -----
    - This class focuses on model construction and training orchestration, while
    evaluation, confusion matrix plotting, and persistence are handled by the
    inherited mission/base layers.
    - The GridSearchCV search space is intentionally kept fixed inside the model
    layer for interface consistency across model families in the project.
    """

    # -------------------- KNN classifier model trainning --------------------
    def train(
        self,
        use_cv: bool = True,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
        n_neighbors: int = 5,
        algorithm: str = "auto",
        p: int = 2,
        weights: str = "uniform",
        split_random_state: int = 42,
        use_pca: bool = False,
        pca_n_components: Optional[int] = None,
        scaler_type: str = "standard",
        cat_encoder: str = "ohe",
    ) -> Dict[str, Any]:
        """
        Train a KNeighborsClassifier pipeline with optional GridSearchCV.

        This method builds the classifier estimator, optionally adds preprocessing
        steps such as scaling and PCA, performs model fitting through inherited shared
        logic, and then runs mission-layer evaluation.

        Training Behavior
        -----------------
        - If ``use_cv=False``:
        The method builds a KNeighborsClassifier using the provided fixed
        hyperparameters.

        - If ``use_cv=True``:
        The method defines a fixed parameter grid and delegates model selection to the
        inherited GridSearchCV workflow.

        Parameters
        ----------
        use_cv : bool, default=True
            Whether to use GridSearchCV for hyperparameter selection.

        cv_folds : int, default=5
            Number of cross-validation folds used when `use_cv=True`.

        scoring : str, default="f1_weighted"
            Scoring method used during cross-validation for single-output
            classification.

            Depending on the inherited mission/base implementation, multi-output
            classification may use a custom scorer automatically when needed.

        n_neighbors : int, default=5
            Number of nearest neighbors used when `use_cv=False`.

        algorithm : str, default="auto"
            Neighbor search algorithm used when `use_cv=False`.

            Common options include:
            - ``"auto"``
            - ``"ball_tree"``
            - ``"kd_tree"``
            - ``"brute"``

        p : int, default=2
            Power parameter for the Minkowski distance used when `use_cv=False`.

            Common values:
            - ``p=1`` : Manhattan distance
            - ``p=2`` : Euclidean distance

        weights : str, default="uniform"
            Weighting strategy used when `use_cv=False`.

            Common options:
            - ``"uniform"``
            - ``"distance"``

        split_random_state : int, default=42
            Random seed passed to shared fitting utilities where applicable,
            especially for splitter behavior in the base layer.

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
        - User-provided hyperparameters are applied only when `use_cv=False`.
        """
        if self.Y_train is None or self.X_train is None:
            raise ValueError("⚠️  Run train_test_split_engine() before training ‼️")

        # ---------- Record parameters input ----------
        self.input_model_type = "KNeighborsClassifier"
        self.input_use_cv = use_cv
        self.input_cv_folds = cv_folds

        # ---------- Basic KNNClassifier model ----------
        base_model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            p=p,
            weights=weights,
        )

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
            param_grid = {
                "scaler": [
                    StandardScaler(),
                    MinMaxScaler(),
                    RobustScaler(),
                    "passthrough",
                ],
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
                "classifier__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
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


# -------------------- KNNRegressor Model --------------------
class KNNRegressor_Model(KNNRegressor_Missioner):
    """
    Model-layer trainer for K-Nearest Neighbors regression.

    This class extends :class:`KNNRegressor_Missioner` and provides the concrete
    training logic for KNN-based regression pipelines. It is responsible for
    constructing the estimator, configuring optional preprocessing steps, defining
    cross-validation parameter grids, invoking the shared fitting workflow, and
    returning a compact training summary.

    Main Responsibilities
    ---------------------
    - Construct a ``KNeighborsRegressor`` estimator instance.
    - Optionally insert scaling and PCA steps into the pipeline.
    - Define a fixed parameter grid when cross-validation is enabled.
    - Call the inherited shared fitting routine from the base/mission layers.
    - Call mission-layer evaluation after training.
    - Return a compact dictionary containing training and evaluation results.

    Workflow Summary
    ----------------
    1. Validate that training data are available.
    2. Record training configuration into instance attributes.
    3. Build the base KNN regressor estimator.
    4. Build optional preprocessing steps such as scaler and PCA.
    5. Define a GridSearchCV parameter grid when ``use_cv=True``.
    6. Train the pipeline through inherited shared fitting logic.
    7. Evaluate the trained model through mission-layer utilities.
    8. Return a compact training summary dictionary.

    Training Strategy
    -----------------
    - ``use_cv=True``:
    Run GridSearchCV using the fixed search space defined inside `train()`.

    - ``use_cv=False``:
    Train a single KNeighborsRegressor using the user-provided fixed
    hyperparameters.

    Notes
    -----
    - This class focuses on model construction and training orchestration, while
    regression metrics, diagnostic plotting, and persistence are handled by the
    inherited mission/base layers.
    - Diagnostic plots are not generated automatically during training. They should
    be triggered later through higher-level workflow control or by directly
    calling mission-layer diagnostic methods.
    """

    # -------------------- KNN regressor model trainning --------------------
    def train(
        self,
        use_cv: bool = True,
        cv_folds: int = 5,
        scoring: str = "r2",
        n_neighbors: int = 5,
        algorithm: str = "auto",
        p: int = 2,
        weights: str = "uniform",
        split_random_state: int = 42,
        use_pca: bool = False,
        pca_n_components: Optional[int] = None,
        scaler_type: str = "standard",
        cat_encoder: str = "ohe",
    ) -> Dict[str, Any]:
        """
        Train a KNeighborsRegressor pipeline with optional GridSearchCV.

        This method builds the regressor estimator, optionally adds preprocessing
        steps such as scaling and PCA, performs model fitting through inherited shared
        logic, and then runs mission-layer evaluation.

        Training Behavior
        -----------------
        - If ``use_cv=False``:
        The method builds a KNeighborsRegressor using the provided fixed
        hyperparameters.

        - If ``use_cv=True``:
        The method defines a fixed parameter grid and delegates model selection to the
        inherited GridSearchCV workflow.

        Parameters
        ----------
        use_cv : bool, default=True
            Whether to use GridSearchCV for hyperparameter selection.

        cv_folds : int, default=5
            Number of cross-validation folds used when `use_cv=True`.

        scoring : str, default="r2"
            Scoring method used during cross-validation for single-output
            regression.

            Depending on the inherited mission/base implementation, multi-output
            regression may use a custom scorer automatically when needed.

        n_neighbors : int, default=5
            Number of nearest neighbors used when `use_cv=False`.

        algorithm : str, default="auto"
            Neighbor search algorithm used when `use_cv=False`.

            Common options include:
            - ``"auto"``
            - ``"ball_tree"``
            - ``"kd_tree"``
            - ``"brute"``

        p : int, default=2
            Power parameter for the Minkowski distance used when `use_cv=False`.

            Common values:
            - ``p=1`` : Manhattan distance
            - ``p=2`` : Euclidean distance

        weights : str, default="uniform"
            Weighting strategy used when `use_cv=False`.

            Common options:
            - ``"uniform"``
            - ``"distance"``

        split_random_state : int, default=42
            Random seed passed to shared fitting utilities where applicable,
            especially for splitter behavior in the base layer.

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
        - User-provided hyperparameters are applied only when `use_cv=False`.
        - This method does not automatically generate diagnostic plots during training.
        Plotting should be triggered separately after fitting and evaluation.
        """
        if self.Y_train is None or self.X_train is None:
            raise ValueError("⚠️  Run train_test_split_engine() before training ‼️")

        # ---------- Record parameters input ----------
        self.input_model_type = "KNeighborsRegressor"
        self.input_use_cv = use_cv
        self.input_cv_folds = cv_folds

        # ---------- Basic KNNRegressor model ----------
        base_model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            p=p,
            weights=weights,
        )

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
            param_grid = {
                "scaler": [
                    StandardScaler(),
                    MinMaxScaler(),
                    RobustScaler(),
                    "passthrough",
                ],
                "regressor__n_neighbors": [3, 5, 7, 9, 11],
                "regressor__weights": ["uniform", "distance"],
                "regressor__p": [1, 2],
                "regressor__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
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
