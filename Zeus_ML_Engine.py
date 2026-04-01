# -------------------- Import Modules --------------------
import glob
import logging
import os
import time
from pprint import pformat

from Cornus.Data_Hunter.HuntingDataCore import HuntingDataCore
from Cornus.MetaUnits.VisionCore import VisionCore
from Zeus.ML_BaseConfigBox.FeatureCore import FeatureCore
from Zeus.ML_ModelBox.KNN_Model import KNNClassifier_Model, KNNRegressor_Model
from Zeus.ML_ModelBox.SVM_Model import SVMClassifier_Model, SVMRegressor_Model
from Zeus.ML_ModelBox.Tree_Forest_Model_Classifier import (
    DecisionTreeClassifier_Model,
    RandomForestClassifier_Model,
)
from Zeus.ML_ModelBox.Tree_Forest_Model_Regressor import (
    DecisionTreeRegressor_Model,
    RandomForestRegressor_Model,
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("Zeus")
# -------------------- Model categories --------------------
MODEL_REGISTRY = {
    "DecisionTreeClassifier": {
        "class": DecisionTreeClassifier_Model,
        "task_type": "classifier",
        "default_scoring": "f1_weighted",
    },
    "RandomForestClassifier": {
        "class": RandomForestClassifier_Model,
        "task_type": "classifier",
        "default_scoring": "f1_weighted",
    },
    "DecisionTreeRegressor": {
        "class": DecisionTreeRegressor_Model,
        "task_type": "regressor",
        "default_scoring": "r2",
    },
    "RandomForestRegressor": {
        "class": RandomForestRegressor_Model,
        "task_type": "regressor",
        "default_scoring": "r2",
    },
    "SVMClassifier": {
        "class": SVMClassifier_Model,
        "task_type": "classifier",
        "default_scoring": "f1_weighted",
    },
    "SVMRegressor": {
        "class": SVMRegressor_Model,
        "task_type": "regressor",
        "default_scoring": "r2",
    },
    "KNNClassifier": {
        "class": KNNClassifier_Model,
        "task_type": "classifier",
        "default_scoring": "f1_weighted",
    },
    "KNNRegressor": {
        "class": KNNRegressor_Model,
        "task_type": "regressor",
        "default_scoring": "r2",
    },
}


# -------------------- Zeus Engine --------------------
class ZeusEngine:
    """
    Top-level orchestration engine for Zeus machine-learning workflows.

    `ZeusEngine` serves as the central controller for dataset loading, preview,
    feature/target preparation, model construction, model training, evaluation
    lookup, and trained-model persistence workflows. It coordinates the major
    data and model components used throughout the Zeus project so that menu
    functions or higher-level scripts can interact with a unified engine
    interface.

    Responsibilities
    ----------------
    - load ML-ready datasets through `HuntingDataCore`
    - preview dataset structure through `VisionCore`
    - prepare feature and target data through `FeatureCore`
    - build registered model instances from `MODEL_REGISTRY`
    - train classifier and regressor models through a unified interface
    - expose the current model and latest training result
    - save trained models through each model's own persistence workflow
    - locate and load previously saved trained-model files

    Attributes
    ----------
    hunter_core : HuntingDataCore
        Core responsible for folder search, file search, and dataset opening.
    vision_core : VisionCore or None
        Core responsible for dataset preview and report-style inspection.
    feature_core : FeatureCore or None
        Core responsible for target selection, feature selection, and `X` / `y`
        construction.
    current_model : object or None
        Active model instance currently managed by the engine.
    current_model_name : str or None
        Registered name of the current active model.
    current_model_result : dict or None
        Latest training summary returned by the current model's training workflow.
    """

    # -------------------- Initialization --------------------
    def __init__(self):
        """
        Initialize the Zeus engine.

        The engine starts with the data-loading core already available and leaves
        downstream workflow cores unbuilt until a dataset is successfully loaded.
        Model-related runtime state is also initialized so that later build, train,
        evaluation, save, and load workflows can reuse a shared engine state.

        Attributes initialized here include:

        - ``hunter_core`` for dataset loading
        - ``vision_core`` for preview/report workflows
        - ``feature_core`` for feature/target preparation
        - ``current_model`` for the active model instance
        - ``current_model_name`` for the active model name
        - ``current_model_result`` for the latest training summary
        """
        # ---------- Import cores ----------
        self.hunter_core = HuntingDataCore()
        self.vision_core = None
        self.feature_core = None

        # ---------- Record current model and its properties ----------
        self.current_model = None
        self.current_model_name = None
        self.current_model_result = None
        logger.info("ZeusEngine initialized")

    # -------------------- Source data property --------------------
    @property
    def source_data(self):
        """
        Return the currently loaded source dataset.

        Returns
        -------
        pandas.DataFrame or None
            Dataset currently stored in `hunter_core.target_data`.
            Returns `None` if no dataset has been loaded yet.
        """
        return self.hunter_core.target_data

    # -------------------- Build pipeline cores --------------------
    def build_cores(self):
        """
        Build all downstream pipeline cores from the current source dataset.

        Raises
        ------
        ValueError
            If no source dataset is currently loaded.

        Notes
        -----
        This method initializes:
        - `vision_core` for preview/report operations
        - `feature_core` for target/feature selection and `X` / `y` creation
        """
        if self.source_data is None:
            logger.warning("build_cores failed: no source data available")
            raise ValueError("⚠️ No source data available. Please load data first ‼️")

        self.vision_core = VisionCore(self.hunter_core)
        self.feature_core = FeatureCore(self.source_data)
        logger.info("Pipeline cores built successfully")

    # -------------------- Helper: refresh all pipeline cores (Changing another dataset) --------------------
    def _refresh_cores(self):
        """
        Rebuild all downstream pipeline cores from the currently loaded dataset.

        This method refreshes the preview/report core and feature-preparation core so
        that both reference the latest currently loaded source dataset.

        Returns
        -------
        None

        Notes
        -----
        This method is typically used after loading a different dataset and is
        implemented as a lightweight wrapper around ``build_cores()``.
        """
        self.build_cores()

    # -------------------- ML dataset searching --------------------
    def ml_dataset_search(
        self,
        selected_folder_num: int,
        selected_file_num: int,
        opener_param_dict: dict | None = None,
    ):
        """
        Search for and load an ML dataset from the working place.

        Parameters
        ----------
        selected_folder_num : int
            Folder index selected from the working place.
        selected_file_num : int
            File index selected from the target folder.
        opener_param_dict : dict, optional
            Optional keyword arguments passed to `hunter_core.opener()`.

        Returns
        -------
        pandas.DataFrame or None
            Loaded dataset if successful, otherwise `None`.

        Notes
        -----
        On successful loading, this method rebuilds downstream cores so that
        `vision_core` and `feature_core` both reference the latest dataset.
        """
        logger.info(
            "Start dataset search | folder_num=%s | file_num=%s",
            selected_folder_num,
            selected_file_num,
        )
        # ---------- Searching folder and file ----------
        self.hunter_core.working_place_searcher()  # Get directories from working place
        self.hunter_core.files_searcher_from_folders(  # Select folder and file and get file's path
            selected_folder_num=selected_folder_num,
            selected_file_num=selected_file_num,
        )

        # ---------- Opening target data ----------
        opener_param_dict = opener_param_dict or {}
        loaded_data = self.hunter_core.opener(
            **opener_param_dict
        )  # Opening selected file by HuntingDataCore

        if loaded_data is not None:
            logger.info("Dataset loaded successfully")
            self.build_cores()
        else:
            logger.warning("Dataset loading failed")

        return loaded_data

    # -------------------- Set target column --------------------
    def set_target_column(self, target_column: str | list[str]):
        """
        Set one or multiple target columns through `FeatureCore`.

        Parameters
        ----------
        target_column : str or list[str]
            Column name or column names to use as the target / dependent variable(s).

        Returns
        -------
        pandas.DataFrame or None
            Selected target DataFrame if successful, otherwise `None`.

        Notes
        -----
        This method delegates target-column validation and target-data construction
        to `FeatureCore.set_target_column()`.

        Examples
        --------
        Single target:

        >>> zeus.set_target_column("Outcome")

        Multiple targets:

        >>> zeus.set_target_column(["Outcome", "RiskScore"])
        """
        if self.feature_core is None:
            logger.warning("set_target_column failed: FeatureCore not built")
            print("⚠️ FeatureCore has not been built ‼️")
            return None

        logger.info("Setting target column: %s", target_column)
        return self.feature_core.set_target_column(
            target_column
        )  # Return target columns from FeatureCore

    # -------------------- Set feature columns --------------------
    def set_feature_columns(self, feature_columns: list[str]):
        """
        Set feature columns through `FeatureCore`.

        Parameters
        ----------
        feature_columns : list[str]
            Column names to use as features / independent variables.

        Returns
        -------
        pandas.DataFrame or None
            Selected feature DataFrame if successful, otherwise `None`.

        Notes
        -----
        This method delegates feature-column validation and feature-data construction
        to `FeatureCore.set_feature_columns()`.

        Examples
        --------
        Set feature columns explicitly::

            zeus.set_feature_columns(["Glucose", "BMI", "Age"])
        """
        if self.feature_core is None:
            logger.warning("set_feature_columns failed: FeatureCore not built")
            print("⚠️ FeatureCore has not been built ‼️")
            return None

        logger.info("Setting feature columns: %s", feature_columns)
        return self.feature_core.set_feature_columns(
            feature_columns
        )  # Return feature columns from FeatureCore

    # -------------------- Build X and y --------------------
    def build_xy_data(self):
        """
        Build and return `X` and `y` through `FeatureCore`.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame] or None
            Tuple of `(X, y)` if successful, otherwise `None`.

        Notes
        -----
        This method delegates the final feature/target matrix construction to
        `FeatureCore.build_xy_data()`. If feature columns have not been explicitly
        selected, `FeatureCore` may automatically use all non-target columns as
        features depending on its internal workflow.
        """
        if self.feature_core is None:
            logger.warning("build_xy_data failed: FeatureCore not built")
            print("⚠️ FeatureCore has not been built ‼️")
            return None

        result = (
            self.feature_core.build_xy_data()
        )  # Build X/Y variables from FeatureCore (build_xy_data method)
        if result is None:
            logger.warning("build_xy_data failed during FeatureCore X/y build")
            return None

        logger.info("X and y built successfully")
        return result  # Return self.X and self.Y

    # -------------------- Reset feature selection --------------------
    def reset_feature_selection(self):
        """
        Reset the current feature-selection state through `FeatureCore`.

        This method clears the currently stored target columns, feature columns,
        and the built `X` / `y` objects managed by `FeatureCore`, so that a new
        feature/target selection workflow can start from a clean state.

        Returns
        -------
        None

        Notes
        -----
        This method does not modify the original source dataset. It only resets
        the feature-selection state stored in `FeatureCore`.

        If `FeatureCore` has not been built yet, the method prints a warning and
        returns `None`.
        """
        if self.feature_core is None:
            logger.warning("reset_feature_selection failed: FeatureCore not built")
            print("⚠️ FeatureCore has not been built ‼️")
            return None

        self.feature_core.reset_feature_state()  # Reset feature by FeatureCore (reset_feature_state method)
        logger.info("Feature selection state reset")

    # -------------------- Feature and target selection --------------------
    def select_feature_target(
        self,
        target_column: list[str],
        feature_columns: list[str] | None = None,
    ):
        """
        Select target and feature columns, then build `X` and `y`.

        Parameters
        ----------
        target_column : list[str]
            Column names to use as the target / dependent variable(s).
        feature_columns : list[str] or None, optional
            Column names to use as features / independent variables.
            If `None`, all non-target columns are used automatically.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame] or None
            Tuple of `(X, y)` if successful, otherwise `None`.

        Workflow
        --------
        1. Validate that `FeatureCore` has been built.
        2. Set target column(s).
        3. Optionally set feature columns.
        4. Build final `X` and `y`.

        Notes
        -----
        This method serves as the main high-level entry point for feature/target
        preparation in menu-driven workflows.
        """
        if self.feature_core is None:
            logger.warning("select_feature_target failed: FeatureCore not built")
            print("⚠️ FeatureCore has not been built ‼️")
            return None

        logger.info(
            "Selecting feature/target columns | target=%s | feature=%s",
            target_column,
            feature_columns,
        )

        target_result = self.set_target_column(target_column)
        if target_result is None:
            logger.warning(
                "select_feature_target failed during target selection | target=%s",
                target_column,
            )
            return None

        if feature_columns is not None:
            feature_result = self.set_feature_columns(feature_columns)
            if feature_result is None:
                logger.warning(
                    "select_feature_target failed during feature selection | feature=%s",
                    feature_columns,
                )
                return None

        result = self.build_xy_data()
        if result is None:
            logger.warning("select_feature_target failed during X/y build")
            return None

        logger.info("Feature/target selection completed successfully")
        return result

    # -------------------- Current feature and target selections --------------------
    def show_current_feature_selection(self):
        """
        Print the current feature/target selection summary.

        This method displays the currently selected target column(s), feature columns,
        and the shapes of the built `X` and `y` objects stored in `FeatureCore`.

        Returns
        -------
        None

        Notes
        -----
        This method is intended for workflow confirmation after feature/target
        selection, especially in interactive menu systems before model training.
        """
        if self.feature_core is None:
            logger.warning(
                "show_current_feature_selection failed: FeatureCore not available"
            )
            print("⚠️ FeatureCore is not available ‼️")
            return None

        logger.info("Showing current feature selection summary")

        print(f"\n🔥 Target column : {self.feature_core.target_column}\n{'-'*100}")

        if self.feature_core.feature_columns is None:
            print("🔥 Feature count : 0")
            print(f"🔥 Feature columns: None\n{'-'*100}")
        else:
            print(f"🔥 Feature count : {len(self.feature_core.feature_columns)}")
            print(f"🔥 Feature columns: {self.feature_core.feature_columns}\n{'-'*100}")

        if self.feature_core.X is None:
            print(f"🔥 X shape: None\n{'-'*100}")
        else:
            print(f"🔥 X shape: {self.feature_core.X.shape}\n{'-'*100}")

        if self.feature_core.y is None:
            print(f"🔥 y shape: None\n{'-'*100}")
        else:
            print(f"🔥 y shape: {self.feature_core.y.shape}\n{'-'*100}")

        logger.info(
            "Current feature selection | target=%s | feature_count=%s | X_shape=%s | y_shape=%s",
            self.feature_core.target_column,
            (
                0
                if self.feature_core.feature_columns is None
                else len(self.feature_core.feature_columns)
            ),
            None if self.feature_core.X is None else self.feature_core.X.shape,
            None if self.feature_core.y is None else self.feature_core.y.shape,
        )

    # -------------------- Get models --------------------
    def get_available_models(self, task_type: str | None = None) -> list[str]:
        """
        Return available model names from the model registry.

        Parameters
        ----------
        task_type : str or None, default=None
            Optional task-type filter. Typical values are `"classifier"` and
            `"regressor"`. If `None`, all registered model names are returned.

        Returns
        -------
        list[str]
            Available model names, optionally filtered by task type.

        Notes
        -----
        This method reads from `MODEL_REGISTRY` and returns only registry keys.
        It does not validate whether feature/target data has already been built.
        """
        if task_type is None:
            model_list = list(MODEL_REGISTRY.keys())
            logger.info("Getting all available models: %s", model_list)
            return list(MODEL_REGISTRY.keys())  # List available models

        # ---------- Get model list ----------
        model_list = [
            model_name
            for model_name, meta in MODEL_REGISTRY.items()
            if meta["task_type"] == task_type
        ]
        logger.info(
            "Getting available models for task_type=%s: %s", task_type, model_list
        )
        return model_list

    # -------------------- Build model instance --------------------
    def build_model(self, model_name: str):
        """
        Build and store a model trainer instance from the current feature data.

        Parameters
        ----------
        model_name : str
            Registered model name defined in `MODEL_REGISTRY`.

        Returns
        -------
        object or None
            Built model trainer instance if successful, otherwise `None`.

        Notes
        -----
        This method validates that feature/target data has already been prepared,
        looks up the requested model class from `MODEL_REGISTRY`, constructs the
        model instance with the current `X` and `y`, and stores it in
        `self.current_model`.

        The built model instance becomes the active model used by later operations
        such as training, evaluation lookup, plotting, and model persistence.
        """
        if self.feature_core is None:
            logger.warning("build_model failed: FeatureCore not built")
            print("⚠️ FeatureCore has not been built ‼️")
            return None

        if self.feature_core.X is None or self.feature_core.y is None:
            logger.warning("build_model failed: X or y not built yet")
            print(
                "⚠️ X and y have not been built yet. Please select feature/target first ‼️"
            )
            return None

        # ---------- Get model properties ----------
        model_meta = MODEL_REGISTRY.get(model_name)
        if model_meta is None:
            logger.warning("build_model failed: unsupported model type %s", model_name)
            print(f"⚠️ Unsupported model type: {model_name} ‼️")
            return None

        model_cls = model_meta["class"]  # Get model class

        # ---------- Convey X/Y to model ----------
        self.current_model = model_cls(
            cleaned_X_data=self.feature_core.X,
            cleaned_Y_data=self.feature_core.y,
        )
        self.current_model_name = model_name  # Record current model
        self.current_model_result = None  # Not training yet

        logger.info("Model instance built successfully: %s", model_name)
        print(f"🔥 Model instance built successfully: {model_name}\n{'-'*100}")
        return self.current_model

    # -------------------- Train current model --------------------
    def train_model(
        self,
        model_name: str,
        test_size: float = 0.2,
        split_random_state: int = 42,
        use_cv: bool = True,
        cv_folds: int = 5,
        scoring: str | None = None,
        cat_encoder: str = "ohe",
        **train_kwargs,
    ):
        """
        Build, split, and train a registered model.

        Parameters
        ----------
        model_name : str
            Registered model name defined in `MODEL_REGISTRY`.
        test_size : float, default=0.2
            Test split ratio passed to the model's train/test split workflow.
        split_random_state : int, default=42
            Random seed used for dataset splitting and model training where
            applicable.
        use_cv : bool, default=True
            Whether to use cross-validation / grid search during training.
        cv_folds : int, default=5
            Number of folds used for cross-validation when `use_cv=True`.
        scoring : str or None, default=None
            Scoring method used during training. If `None`, the default scoring
            value from `MODEL_REGISTRY` is used.
        cat_encoder : str, default="ohe"
            Categorical-encoding strategy passed to the model's `train()` method.
            Typical values include one-hot encoding or ordinal-style encoding,
            depending on the downstream model implementation.
        **train_kwargs
            Additional keyword arguments passed directly to the model's `train()`
            method.

        Returns
        -------
        dict or None
            Training summary dictionary if successful, otherwise `None`.

        Workflow
        --------
        1. Build the requested model from the current feature/target data.
        2. Split the dataset into training and test sets.
        3. Run model training.
        4. Store the latest training result in `self.current_model_result`.
        5. Return the training summary dictionary.

        Notes
        -----
        This method is the main unified training entry point for all registered
        classifier and regressor models managed by `ZeusEngine`.
        """
        logger.info(
            "Start training | model=%s | test_size=%s | use_cv=%s | cv_folds=%s",
            model_name,
            test_size,
            use_cv,
            cv_folds,
        )
        try:
            model = self.build_model(
                model_name
            )  # Get built model from build_model method
            if model is None:
                return None

            # ---------- Get model properties ----------
            model_meta = MODEL_REGISTRY.get(model_name)
            if model_meta is None:
                logger.warning(
                    "train_model failed: unsupported model type %s", model_name
                )
                print(f"⚠️ Unsupported model type: {model_name} ‼️")
                return None

            # ---------- Scoring method (using default) ----------
            if scoring is None:
                scoring = model_meta.get("default_scoring")

            # ---------- Train and test split ----------
            model.train_test_split_engine(
                test_size=test_size,
                split_random_state=split_random_state,
            )

            # ---------- Model results ----------
            result = model.train(
                use_cv=use_cv,
                cv_folds=cv_folds,
                scoring=scoring,
                split_random_state=split_random_state,
                cat_encoder=cat_encoder,
                **train_kwargs,
            )

            self.current_model_result = result

            # ---------- Auto save evaluation report ----------
            try:
                self._save_current_model_evaluation_txt()
            except Exception:
                logger.exception("Auto save evaluation report failed")

            logger.info("Model training completed successfully: %s", model_name)
            print(f"🔥 Model training completed: {model_name}\n{'-'*100}")
            return result

        except Exception:
            logger.exception("Model training failed: %s", model_name)
            raise

    # -------------------- Get current model evaluation --------------------
    def get_model_evaluation(self):
        """
        Return the most recent evaluation result from the current trained model.

        Returns
        -------
        dict or None
            Evaluation result dictionary stored under the ``"evaluation"`` key in
            ``self.current_model_result`` if available; otherwise ``None``.

        Notes
        -----
        This method reads from the latest cached training summary created by
        ``train_model()``. If no training result is available yet, the method prints
        a warning and returns ``None``.
        """
        if self.current_model_result is None:
            print("⚠️ No model training result available yet ‼️")
            return None

        return self.current_model_result.get(
            "evaluation"
        )  # Get model evaluation (model class ---> missioner)

    # -------------------- Run current model method --------------------
    def run_current_model_method(self, method_name: str, **kwargs):
        """
        Run a supported method from the current model instance.

        Parameters
        ----------
        method_name : str
            Method name to execute on the current model.
        **kwargs
            Keyword arguments passed to the target method.

        Returns
        -------
        Any | None
            Return value of the invoked model method if the current model exists,
            the attribute is supported, and the target attribute is callable.
            Returns `None` if no current model is available, the requested method
            is unsupported, or the attribute exists but is not callable.

        Notes
        -----
        This helper provides a unified dispatch layer for menu-driven model tools
        such as evaluation display, plotting utilities, feature-importance access,
        and other model-specific helper methods.
        """
        if self.current_model is None:
            print("⚠️ No current model available ‼️")
            return None

        if not hasattr(self.current_model, method_name):
            print(f"⚠️ Current model does not support method: {method_name} ‼️")
            return None

        # -------------------- Get current model's method --------------------
        method = getattr(self.current_model, method_name)

        if not callable(method):
            print(f"⚠️ Attribute '{method_name}' is not callable ‼️")
            return None

        return method(**kwargs)  # Return with parameters (**kwargs)

    # -------------------- Show current model summary --------------------
    def show_current_model_summary(self):
        """
        Print the summary of the current trained model.

        This method displays key information from the most recent training result,
        including the active model name, whether cross-validation was used, the
        best hyperparameter configuration, the best cross-validation score, and
        the number of feature names recorded in the training summary.

        Returns
        -------
        None

        Notes
        -----
        This method reads from `self.current_model_result`, which is populated by
        `train_model()` after successful training.

        If no training result is currently available, the method prints a warning
        and returns without displaying a summary.
        """
        if self.current_model_result is None:
            print("⚠️ No trained model result available yet ‼️")
            return

        print(f"🪔 Current model : {self.current_model_name}")
        print(f"🪔 Use CV : {self.current_model_result.get('use_cv')}")
        print(f"🪔 Best params : {self.current_model_result.get('best_params')}")
        print(f"🪔 Best CV score : {self.current_model_result.get('best_cv_score')}")
        print(
            f"🪔 Feature names length : {self.current_model_result.get('feature_names_len')}\n{'-'*100}"
        )

    # -------------------- Save current evaluation report --------------------
    def _save_current_model_evaluation_txt(
        self,
        folder_name: str = "Evaluation_Report",
        file_name: str | None = None,
    ):
        """
        Save the current model evaluation result as a text report.

        This method retrieves the latest evaluation result stored in
        ``self.current_model_result`` through ``self.get_model_evaluation()``,
        formats it into a readable plain-text report, creates the target report
        folder if needed, and saves the report as a ``.txt`` file.

        The method is intended for unified evaluation-result persistence at the
        engine layer so that evaluation reports from different model missioners
        can be saved through a single Zeus interface.

        Parameters
        ----------
        folder_name : str, default="Evaluation_Report"
            Folder name created under ``ML_Report`` in the Zeus project root to
            store evaluation text reports.
        file_name : str or None, optional
            Custom output filename. If ``None``, an automatic filename is generated
            from the current model name and the current timestamp.

        Returns
        -------
        str or None
            Full saved file path if successful; otherwise ``None`` when no
            evaluation result is available.

        Workflow
        --------
        1. Retrieve the latest evaluation result through ``self.get_model_evaluation()``.
        2. Validate that an evaluation result is currently available.
        3. Create the target report folder if it does not already exist.
        4. Generate a default filename when ``file_name`` is not provided.
        5. Format the evaluation result into a readable text report.
        6. Save the report as a ``.txt`` file.
        7. Print and return the saved file path.

        Notes
        -----
        The saved content is based on ``self.get_model_evaluation()`` and is written
        using ``pprint.pformat`` so nested dictionaries, lists, NumPy arrays, pandas
        objects, and other evaluation content can be preserved in a readable text
        layout.

        If ``self.current_model_name`` is unavailable, ``"UnknownModel"`` is used in
        the auto-generated filename.

        Examples
        --------
        Save the current evaluation report with an automatic filename::

            zeus._save_current_model_evaluation_txt()

        Save the report with a custom filename::

            zeus._save_current_model_evaluation_txt(
                file_name="svc_evaluation_report.txt"
            )
        """
        logger.info("Start saving current model evaluation as text")

        evaluation = self.get_model_evaluation()
        if evaluation is None:
            logger.warning("Save evaluation failed: no evaluation result available")
            print("⚠️ No evaluation result available to save ‼️")
            return None

        report_root = os.path.join(project_root, "ML_Report", folder_name)
        os.makedirs(report_root, exist_ok=True)

        if file_name is None:
            model_label = self.current_model_name or "UnknownModel"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"{model_label}_evaluation_{timestamp}.txt"

        if not file_name.lower().endswith(".txt"):
            file_name += ".txt"

        save_path = os.path.join(report_root, file_name)

        report_lines = [
            "==========📝 Zeus Evaluation Report 📝==========",
            f"Model Name : {self.current_model_name}",
            f"Saved Time : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "-" * 100,
            pformat(evaluation, sort_dicts=False),
            "-" * 100,
        ]

        with open(save_path, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(report_lines))

        logger.info("Current model evaluation saved successfully: %s", save_path)
        print(f"🔥 Evaluation report saved successfully: {save_path}")
        return save_path

    # -------------------- Save current SVM insight report --------------------
    def _save_current_svm_insight_txt(
        self,
        insight_result: dict,
        target_col: str | None = None,
        folder_name: str = "SVM_Insight_Report",
        file_name: str | None = None,
    ):
        """
        Save the current SVM insight result as a text report.

        This method formats an SVM insight dictionary into a readable plain-text report,
        creates the target report folder if needed, generates a default filename when
        one is not provided, and saves the report as a ``.txt`` file.

        The method is intended for insight results returned by
        ``svm_model_insight_engine`` and is suitable for both single-output and
        multi-output SVM inspection workflows.

        Parameters
        ----------
        insight_result : dict
            Insight dictionary returned by ``svm_model_insight_engine``.
        target_col : str or None, optional
            Target column name used for multi-output SVM insight.

            - ``None`` indicates single-output inspection
            - a string value indicates the selected target column in a multi-output
            workflow
        folder_name : str, default="SVM_Insight_Report"
            Folder name created under ``ML_Report`` in the Zeus project root to
            store SVM insight text reports.
        file_name : str or None, optional
            Custom output filename. If ``None``, an automatic filename is generated
            from the current model name, target column, and timestamp.

        Returns
        -------
        str
            Full saved file path of the generated text report.

        Workflow
        --------
        1. Create the target report folder if it does not already exist.
        2. Generate a default filename when ``file_name`` is not provided.
        3. Format the insight result into a readable plain-text report.
        4. Save the report as a ``.txt`` file.
        5. Print the saved file path and return it.

        Notes
        -----
        The saved content is based on ``pprint.pformat`` so nested dictionaries, lists,
        arrays, and other insight content can be preserved in a readable text layout.

        For multi-output SVM workflows, ``target_col`` is included in both the saved
        report header and the auto-generated filename so that reports for different
        targets can be distinguished easily.

        Examples
        --------
        Save the current SVM insight report with an automatic filename::

            zeus._save_current_svm_insight_txt(result, target_col="Outcome")

        Save the report with a custom filename::

            zeus._save_current_svm_insight_txt(
                result,
                target_col="BloodPressure",
                file_name="svc_bp_insight.txt",
            )
        """
        logger.info("Start saving current SVM insight as text")

        report_root = os.path.join(project_root, "ML_Report", folder_name)
        os.makedirs(report_root, exist_ok=True)

        if file_name is None:
            model_label = self.current_model_name or "UnknownModel"
            target_label = "single_output" if target_col is None else str(target_col)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"{model_label}_{target_label}_svm_insight_{timestamp}.txt"

        if not file_name.lower().endswith(".txt"):
            file_name += ".txt"

        save_path = os.path.join(report_root, file_name)

        report_lines = [
            "==========📝 Zeus SVM Insight Report 📝==========",
            f"Model Name   : {self.current_model_name}",
            f"Target Column: {target_col}",
            f"Saved Time   : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "-" * 100,
            pformat(insight_result, sort_dicts=False),
            "-" * 100,
        ]

        with open(save_path, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(report_lines))

        logger.info("Current SVM insight saved successfully: %s", save_path)
        print(f"🔥 SVM insight report saved successfully: {save_path}")
        return save_path

    # -------------------- Save trained model --------------------
    def save_current_model(self):
        """
        Save the current trained model through the active model instance.

        This method delegates model persistence to the currently active model object
        through its ``save_model_joblib()`` method. The actual output directory,
        filename, and serialization details are managed by the model's missioner or
        model class, not by the engine.

        Returns
        -------
        str | None
            Full saved file path if the current model exists and supports
            ``save_model_joblib()``; otherwise ``None``.

        Notes
        -----
        This method does not ask for or manage a user-defined save directory.
        Instead, each model type is expected to save itself into its own predefined
        trained-model folder, such as a model-specific directory under
        ``ML_Report``.

        If no current model is available, or if the current model does not implement
        ``save_model_joblib()``, the method prints a warning message and returns
        ``None``.
        """
        if self.current_model is None:
            logger.warning("save_current_model failed: no current model")
            print("⚠️ No current model available to save ‼️")
            return None

        if not hasattr(self.current_model, "save_model_joblib"):
            logger.warning("save_current_model failed: model does not support saving")
            print("⚠️ Current model does not support model saving ‼️")
            return None

        logger.info("Saving current model using model default MODEL_DIR")
        return (
            self.current_model.save_model_joblib()
        )  # Recall missioner saveing model method

    # -------------------- Helper: Get Saved trained model folders --------------------
    def _get_model_save_folder(self, model_name: str) -> str | None:
        """
        Return the default trained-model folder path for a registered model name.

        This helper maps a model name defined in ``MODEL_REGISTRY`` to its
        corresponding default trained-model folder under the project's
        ``ML_Report`` directory.

        Parameters
        ----------
        model_name : str
            Registered model name used by Zeus, such as ``"SVMClassifier"``,
            ``"KNNRegressor"``, ``"DecisionTreeClassifier"``, or
            ``"RandomForestRegressor"``.

        Returns
        -------
        str | None
            Absolute folder path for the model's default trained-model directory
            if the model name is supported; otherwise ``None``.

        Notes
        -----
        This method is used by engine- and menu-level loading workflows to locate
        the folder that stores serialized ``.joblib`` model files for a given
        model type.

        Multiple model names may intentionally map to the same folder when they
        share a common missioner or trained-model storage convention. For example,
        tree-based classifier models may share one classifier folder, while
        tree-based regressor models may share one regressor folder.
        """
        folder_map = {
            "SVMClassifier": "SVMCla_Trained_Model",
            "SVMRegressor": "SVMReg_Trained_Model",
            "KNNClassifier": "KNNCla_Trained_Model",
            "KNNRegressor": "KNNReg_Trained_Model",
            "DecisionTreeClassifier": "TreeCla_Trained_Model",
            "RandomForestClassifier": "TreeCla_Trained_Model",
            "DecisionTreeRegressor": "TreeReg_Trained_Model",
            "RandomForestRegressor": "TreeReg_Trained_Model",
        }

        folder_name = folder_map.get(
            model_name
        )  # Using folder map to enter to certain trained model folder
        if folder_name is None:
            logger.warning(
                "_get_model_save_folder failed: unsupported model type %s", model_name
            )
            return None

        return os.path.join(project_root, "ML_Report", folder_name)

    # -------------------- Helper: Get Saved trained model files --------------------
    def _get_saved_model_files(self, model_name: str) -> list[str]:
        """
        Return all saved joblib files for a registered model type.

        This method resolves the default trained-model folder for the given
        ``model_name`` and searches that folder for serialized model artifacts with
        the ``.joblib`` extension.

        Parameters
        ----------
        model_name : str
            Registered model name defined in ``MODEL_REGISTRY``.

        Returns
        -------
        list[str]
            List of absolute file paths for matching ``.joblib`` files. The returned
            list is sorted by file modification time in descending order, so newer
            saved models appear first.

        Notes
        -----
        If the model name is unsupported, if the folder cannot be resolved, or if
        the resolved folder does not exist, the method returns an empty list.

        This helper is typically used by menu-level loading workflows to display
        available saved-model candidates for user selection.
        """
        folder_path = self._get_model_save_folder(model_name)
        if folder_path is None:
            logger.warning(
                "get_saved_model_files failed: unsupported model type %s", model_name
            )
            return []

        if not os.path.isdir(folder_path):
            logger.warning(
                "get_saved_model_files failed: folder not found | %s", folder_path
            )
            return []

        file_pattern = os.path.join(folder_path, "*.joblib")
        model_files = glob.glob(file_pattern)
        model_files.sort(key=os.path.getmtime, reverse=True)

        logger.info(
            "Saved model files found | model=%s | count=%s | folder=%s",
            model_name,
            len(model_files),
            folder_path,
        )
        return model_files  # List trained model folder

    # -------------------- Load trained model --------------------
    def load_trained_model(self, model_name: str, filepath: str):
        """
        Load a trained model from disk and store it as the current active model.

        This method resolves the requested model class from ``MODEL_REGISTRY`` and
        delegates model restoration to that class through ``load_model_joblib()``.
        The corresponding model or missioner class is expected to implement
        ``load_model_joblib()`` as a class method that accepts ``filepath`` and
        returns a restored model object.

        Parameters
        ----------
        model_name : str
            Registered model name defined in ``MODEL_REGISTRY``.
        filepath : str
            Full file path to the serialized trained model artifact.

        Returns
        -------
        object | None
            Restored model object if loading succeeds; otherwise ``None``.

        Notes
        -----
        After successful loading, the restored model object is stored in
        ``self.current_model``, the active model name is updated in
        ``self.current_model_name``, and any previously cached training summary in
        ``self.current_model_result`` is cleared.

        If the model name is unsupported, if the resolved model class does not
        implement ``load_model_joblib()``, or if loading fails, the method prints a
        warning message and returns ``None``.
        """
        logger.info("Loading trained model | model=%s | path=%s", model_name, filepath)

        model_meta = MODEL_REGISTRY.get(model_name)
        if model_meta is None:
            logger.warning(
                "load_trained_model failed: unsupported model type %s", model_name
            )
            print(f"⚠️ Unsupported model type: {model_name} ‼️")
            return None

        model_cls = model_meta["class"]

        if not hasattr(model_cls, "load_model_joblib"):
            logger.warning(
                "load_trained_model failed: model class does not support loading"
            )
            print("⚠️ This model class does not support loading ‼️")
            return None

        try:
            loaded_model = model_cls.load_model_joblib(filepath=filepath)
        except Exception:
            logger.exception(
                "load_trained_model failed | model=%s | path=%s",
                model_name,
                filepath,
            )
            print("⚠️ Failed to load model ‼️")
            return None

        if loaded_model is None:
            logger.warning("load_trained_model failed: returned None")
            print("⚠️ Failed to load model ‼️")
            return None

        self.current_model = loaded_model
        self.current_model_name = model_name
        self.current_model_result = None

        logger.info("Model loaded successfully: %s", model_name)
        print(f"🔥 Model loaded successfully: {model_name}")
        return loaded_model

    # -------------------- Predict new target by trained model --------------------
    def predict_with_current_model(self, new_data):
        """
        Predict target values for new input data using the current active model.

        This method performs an engine-level prediction workflow using the model
        currently stored in ``self.current_model``. It validates that an active model
        is available, confirms that the model exposes a trained pipeline, checks that
        stored feature names are available, and verifies that the provided prediction
        dataset contains all required feature columns.

        If validation succeeds, the input dataset is aligned to the stored feature
        order used during training and then passed to the model pipeline's
        ``predict()`` method.

        Parameters
        ----------
        new_data : pandas.DataFrame
            Input dataset used for prediction. This dataset must contain all feature
            columns required by the active model.

        Returns
        -------
        numpy.ndarray or None
            Prediction result if the workflow succeeds; otherwise ``None``.

        Workflow
        --------
        1. Validate that a current model exists.
        2. Validate that the current model has a trained ``model_pipeline``.
        3. Validate that stored feature names are available.
        4. Check whether all required feature columns exist in ``new_data``.
        5. Reorder the input dataset according to the stored feature order.
        6. Run ``model_pipeline.predict()`` on the aligned dataset.
        7. Return the prediction result.

        Notes
        -----
        This method assumes that the active model was previously trained or loaded and
        that its ``feature_names`` attribute reflects the correct feature schema for
        prediction.

        This method does not retrain the model, rebuild features, or apply external
        dataset validation beyond checking required column presence and feature order
        alignment.
        """
        if self.current_model is None:
            logger.warning("predict_with_current_model failed: no current model")
            print("⚠️ No current model available ‼️")
            return None

        model_pipeline = getattr(self.current_model, "model_pipeline", None)
        if model_pipeline is None:
            logger.warning("predict_with_current_model failed: model pipeline is None")
            print("⚠️ Current model pipeline is not available ‼️")
            return None

        feature_names = getattr(self.current_model, "feature_names", None)
        if feature_names is None:
            logger.warning(
                "predict_with_current_model failed: feature names are missing"
            )
            print("⚠️ Feature names are missing from current model ‼️")
            return None

        missing_cols = [col for col in feature_names if col not in new_data.columns]
        if missing_cols:
            logger.warning(
                "predict_with_current_model failed: missing columns %s",
                missing_cols,
            )
            print(f"⚠️ Missing required columns: {missing_cols} ‼️")
            return None

        aligned_data = new_data.loc[:, feature_names]  # Alignment new input data

        logger.info(
            "Running prediction with current model | rows=%s | cols=%s",
            len(aligned_data),
            len(aligned_data.columns),
        )
        return model_pipeline.predict(aligned_data)  # Predict result by trained model


# =================================================
