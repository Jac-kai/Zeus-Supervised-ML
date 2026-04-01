"""
Tree-Forest Classification Model Trainers
=========================================

This module defines the model-layer trainers for tree-based classification:

- DecisionTreeClassifier_Model
- RandomForestClassifier_Model

These classes inherit from TreeClassifier_Missioner, which provides shared
classification mission utilities, including:

- train/test split handling
- preprocessing pipeline construction
- optional GridSearchCV support via fit_with_grid()
- evaluation metrics and reports via model_evaluation_engine()
- confusion matrix, feature importance, and tree plotting helpers
- model persistence utilities

Shared responsibilities from mission/base layers
------------------------------------------------
The inherited mission/base layers handle reusable workflow logic such as:

- dataset split preparation
- preprocessing pipeline assembly
- cross-validation fitting workflow
- evaluation dispatch
- feature name tracking after preprocessing

Model-layer responsibility
--------------------------
This module focuses on model-specific training responsibilities:

- construct estimator instances
- define fixed GridSearchCV param grids when use_cv=True
- call the shared fitting routine
- call evaluation
- return a compact training summary dictionary

Notes
-----
This module keeps GridSearchCV search spaces fixed inside the model layer
for interface consistency across tree-based classification trainers.
Plotting and further workflow orchestration should typically be triggered
later by higher-level control layers such as Zeus.
"""

# -------------------- Import Modules --------------------
from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from Zeus.ML_MissionBox.TreeClassifier_Missioner import TreeClassifier_Missioner


# -------------------- DecisionTreeClassifier Model --------------------
class DecisionTreeClassifier_Model(TreeClassifier_Missioner):
    """
    Tree-Forest Classification Model Trainers
    =========================================

    This module defines the model-layer trainers for tree-based classification:

    - DecisionTreeClassifier_Model
    - RandomForestClassifier_Model

    These classes inherit from TreeClassifier_Missioner, which provides shared
    classification mission utilities, including:

    - train/test split handling
    - preprocessing pipeline construction
    - optional GridSearchCV support via fit_with_grid()
    - evaluation metrics and reports via model_evaluation_engine()
    - confusion matrix, feature importance, and tree plotting helpers
    - model persistence utilities

    Shared responsibilities from mission/base layers
    ------------------------------------------------
    The inherited mission/base layers handle reusable workflow logic such as:

    - dataset split preparation
    - preprocessing pipeline assembly
    - cross-validation fitting workflow
    - evaluation dispatch
    - feature name tracking after preprocessing

    Model-layer responsibility
    --------------------------
    This module focuses on model-specific training responsibilities:

    - construct estimator instances
    - define fixed GridSearchCV param grids when use_cv=True
    - call the shared fitting routine
    - call evaluation
    - return a compact training summary dictionary

    Notes
    -----
    This module keeps GridSearchCV search spaces fixed inside the model layer
    for interface consistency across tree-based classification trainers.
    Plotting and further workflow orchestration should typically be triggered
    later by higher-level control layers such as Zeus.
    """

    # -------------------- DecisionTree classifier model --------------------
    def train(
        self,
        use_cv: bool = True,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        split_random_state: int = 42,
        model_random_state: int | None = None,
        cat_encoder: str = "ohe",
    ) -> Dict[str, Any]:
        """
        Train a DecisionTreeClassifier pipeline with optional GridSearchCV.

        Parameters
        ----------
        use_cv : bool, default=True
            Whether to run GridSearchCV.

        cv_folds : int, default=5
            Number of cross-validation folds.

        scoring : str, default="f1_weighted"
            Scoring string used for single-output classification.
            For multi-output classification, a mission-layer custom scorer
            may be used automatically depending on the shared base implementation.

        criterion : str, default="gini"
            Split criterion used when use_cv=False.

            Common options include:
            - "gini"
            - "entropy"

        max_depth : Optional[int], default=None
            Maximum tree depth used when use_cv=False.
            If None, the tree expands until stopping conditions are met.

        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node
            when use_cv=False.

        min_samples_leaf : int, default=1
            Minimum number of samples required at a leaf node
            when use_cv=False.

        split_random_state : int, default=42
            Seed used for estimator reproducibility and shared fitting utilities
            when applicable.

        Returns
        -------
        Dict[str, Any]
            Training summary dictionary containing:

            - "model":
                Model type name.
            - "use_cv":
                Whether GridSearchCV was used.
            - "best_params":
                Best parameter set returned by GridSearchCV.
                Returns None when use_cv=False.
            - "best_cv_score":
                Best cross-validation score returned by GridSearchCV.
                Returns None when use_cv=False.
            - "feature_names_len":
                Number of transformed feature names after preprocessing.
                Returns None if feature names are unavailable.
            - "evaluation":
                Evaluation result dictionary returned by
                mission-layer model_evaluation_engine().

        Raises
        ------
        ValueError
            If train_test_split_engine() has not been called before training.

        Notes
        -----
        This method keeps the GridSearchCV search space fixed inside the model layer
        for interface consistency with other model trainers in the project.
        User-provided hyperparameters are applied only when use_cv=False.
        """
        if self.Y_train is None or self.X_train is None:
            raise ValueError("⚠️  Run train_test_split_engine() before training ‼️")

        # ---------- Record parameters input ----------
        self.input_model_type = "DecisionTreeClassifier"
        self.input_use_cv = use_cv
        self.input_cv_folds = cv_folds

        # ---------- Base model setting ----------
        base_model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=(
                model_random_state
                if model_random_state is not None
                else split_random_state
            ),
        )

        param_grid = None  # Record CV parameters

        # ---------- CV parameters ----------
        if use_cv:
            param_grid = {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [None, 3, 5, 8, 12],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            }

        # ---------- CV or original model training ----------
        best_params, best_score = self.fit_with_grid(
            base_model=base_model,
            param_grid=param_grid,
            use_cv=use_cv,
            cv_folds=cv_folds,
            scoring=scoring,
            split_random_state=split_random_state,
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


# -------------------- RandomForestClassifier Model --------------------
class RandomForestClassifier_Model(TreeClassifier_Missioner):
    """
    RandomForestClassifier_Model
    ============================

    Model-layer trainer for RandomForestClassifier.

    Responsibilities
    ----------------
    - Construct a RandomForestClassifier estimator.
    - Define a fixed param_grid for GridSearchCV when use_cv=True.
    - Call BaseModelConfig.fit_with_grid() through inherited shared logic.
    - Call mission-layer evaluation.
    - Return a compact training summary dictionary.

    Training Strategy
    -----------------
    - use_cv=True:
        Run GridSearchCV using the fixed search space defined inside train().
    - use_cv=False:
        Train a single RandomForestClassifier using the user-provided
        fixed hyperparameters.

    Notes
    -----
    This class only handles training-time estimator construction, fitting,
    and evaluation dispatch. Additional analysis such as confusion matrix
    inspection, feature importance extraction, or per-tree visualization
    should be triggered separately from workflow control layers such as Zeus
    when needed.
    """

    # -------------------- RandomForest classifier model --------------------
    def train(
        self,
        use_cv: bool = True,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[str] = None,
        split_random_state: int = 42,
        model_random_state: int | None = None,
        cat_encoder: str = "ohe",
    ) -> Dict[str, Any]:
        """
        Train a RandomForestClassifier pipeline with optional GridSearchCV.

        Parameters
        ----------
        use_cv : bool, default=True
            Whether to run GridSearchCV.

        cv_folds : int, default=5
            Number of cross-validation folds.

        scoring : str, default="f1_weighted"
            Scoring string used for single-output classification.
            For multi-output classification, a mission-layer custom scorer
            may be used automatically depending on the shared base implementation.

        n_estimators : int, default=100
            Number of trees used when use_cv=False.

        max_depth : Optional[int], default=None
            Maximum tree depth used when use_cv=False.
            If None, each tree expands until stopping conditions are met.

        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node
            when use_cv=False.

        min_samples_leaf : int, default=1
            Minimum number of samples required at a leaf node
            when use_cv=False.

        class_weight : Optional[str], default=None
            Class weighting strategy used when use_cv=False.

            Common options include:
            - None
            - "balanced"

        split_random_state : int, default=42
            Seed used for estimator reproducibility and shared fitting utilities
            when applicable.

        Returns
        -------
        Dict[str, Any]
            Training summary dictionary containing:

            - "model":
                Model type name.
            - "use_cv":
                Whether GridSearchCV was used.
            - "best_params":
                Best parameter set returned by GridSearchCV.
                Returns None when use_cv=False.
            - "best_cv_score":
                Best cross-validation score returned by GridSearchCV.
                Returns None when use_cv=False.
            - "feature_names_len":
                Number of transformed feature names after preprocessing.
                Returns None if feature names are unavailable.
            - "evaluation":
                Evaluation result dictionary returned by
                mission-layer model_evaluation_engine().

        Raises
        ------
        ValueError
            If train_test_split_engine() has not been called before training.

        Notes
        -----
        This method keeps the GridSearchCV search space fixed inside the model layer
        for interface consistency with other model trainers in the project.
        User-provided hyperparameters are applied only when use_cv=False.
        """
        if self.Y_train is None or self.X_train is None:
            raise ValueError("⚠️  Run train_test_split_engine() before training ‼️")

        # ---------- Record parameters input ----------
        self.input_model_type = "RandomForestClassifier"
        self.input_use_cv = use_cv
        self.input_cv_folds = cv_folds

        # ---------- Base model setting ----------
        base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=(
                model_random_state
                if model_random_state is not None
                else split_random_state
            ),
            n_jobs=-1,
        )

        param_grid = None

        # ---------- CV parameters ----------
        if use_cv:
            param_grid = {
                "classifier__n_estimators": [100, 300, 600],
                "classifier__max_depth": [None, 5, 10, 20],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__class_weight": [None, "balanced"],
            }

        # ---------- CV or original model training ----------
        best_params, best_score = self.fit_with_grid(
            base_model=base_model,
            param_grid=param_grid,
            use_cv=use_cv,
            cv_folds=cv_folds,
            scoring=scoring,
            split_random_state=split_random_state,
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
