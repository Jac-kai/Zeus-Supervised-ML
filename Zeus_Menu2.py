# -------------------- Import Modules --------------------
import logging
import os

import pandas as pd

from Zeus.Menu_Config import COMMON_PARAM_CONFIG
from Zeus.Menu_Helper_Decorator import input_int, input_yesno, menu_wrapper
from Zeus.Zeus_ML_Engine import ZeusEngine
from Zeus.Zeus_Model_Menu_Helper import (
    collect_common_training_params,
    collect_model_train_kwargs,
    select_from_options,
    select_model_name,
)

logger = logging.getLogger("Zeus")


# -------------------- Helper: Encoded columns type check --------------------
def _has_categorical_features(data) -> bool:
    """
    Check whether the given feature dataset contains categorical-like columns.

    This helper inspects the provided feature table and determines whether at
    least one column should be treated as categorical in the Zeus menu workflow.

    In the current workflow, categorical-like columns are defined as columns
    whose dtype is one of:

    - ``object``
    - ``category``
    - ``bool``

    The helper is mainly used by training menus to decide whether the user
    should be prompted to choose a categorical feature encoder before model
    training begins.

    Parameters
    ----------
    data : pandas.DataFrame or None
        Feature dataset to inspect.

    Returns
    -------
    bool
        ``True`` if at least one categorical-like column exists in the given
        dataset.

        ``False`` if:
        - the dataset is ``None``, or
        - no categorical-like columns are found.

    Notes
    -----
    - This helper only checks feature-column dtypes.
    - It does not transform data and does not validate target columns.
    - The categorical-encoder selection step in the training menu is triggered
      only when this helper returns ``True``.
    """
    if data is None:
        return False

    # ---------- Check categorical-type columns ----------
    categorical_cols = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    return len(categorical_cols) > 0  # Return True/False


# -------------------- Helper: block invalid sparse-scaler combination --------------------
def _is_invalid_sparse_scaler_combo(
    model_name: str,
    cat_encoder: str | None,
    train_kwargs: dict,
) -> bool:
    """
    Check whether the selected model / categorical-encoder / scaler combination
    should be blocked before training.

    This helper is used by the Zeus training-menu workflow to prevent parameter
    combinations that are known to fail during pipeline fitting.

    In the current workflow, the combination is treated as invalid when all of
    the following conditions are satisfied:

    - the selected model belongs to the KNN or SVM family
    - the selected categorical encoder is ``"ohe"``
    - the selected scaler type is ``"standard"``

    Why this validation exists
    --------------------------
    ``OneHotEncoder`` commonly produces sparse matrix output for categorical
    features. In this project workflow, applying ``StandardScaler`` with its
    default centering behavior to sparse output may raise a runtime error during
    training because sparse matrices cannot be mean-centered in that form.

    This helper allows the menu layer to detect the invalid combination early
    and stop the workflow before the training engine is called.

    Parameters
    ----------
    model_name : str
        Registered model name selected by the user.

        Typical supported values include:

        - ``"KNNClassifier"``
        - ``"KNNRegressor"``
        - ``"SVMClassifier"``
        - ``"SVMRegressor"``

    cat_encoder : str or None
        Selected categorical encoder type.

        Common values in the current Zeus workflow include:

        - ``"ohe"``
        - ``"ordinal"``

        ``None`` may also be passed defensively when no categorical encoder
        selection is involved.

    train_kwargs : dict
        Dictionary of model-specific training keyword arguments collected from
        the model-parameter menu.

        This helper reads ``train_kwargs.get("scaler_type")`` to determine the
        currently selected scaler behavior.

    Returns
    -------
    bool
        ``True`` if the selected parameter combination is invalid and should be
        blocked before training.

        ``False`` if the combination is allowed.

    Notes
    -----
    - This helper performs menu-layer validation only.
    - It does not modify user selections.
    - The blocking rule is intentionally limited to KNN / SVM family models in
      this workflow because those models expose scaler selection and are more
      sensitive to sparse-output scaling combinations.
    - Tree-based models are not included in this validation rule.
    """
    scaler_type = train_kwargs.get("scaler_type")

    scaler_sensitive_models = {
        "KNNClassifier",
        "KNNRegressor",
        "SVMClassifier",
        "SVMRegressor",
    }

    return (
        model_name in scaler_sensitive_models
        and cat_encoder == "ohe"
        and scaler_type == "standard"
    )


# -------------------- Train classifier menu --------------------
@menu_wrapper("Train Classifier")
def train_classifier_menu(zeus: ZeusEngine):
    """
    Train a classifier model through the Zeus menu workflow.

    This menu function coordinates the full terminal-based workflow for training
    a classification model. It verifies that feature preparation has already
    been completed, collects shared and model-specific training parameters,
    optionally collects a categorical-feature encoding strategy, validates
    encoder / scaler compatibility, and finally dispatches the training request
    to ``zeus.train_model()``.

    Workflow
    --------
    1. Check that ``feature_core`` has been built.
    2. Check that both ``X`` and ``y`` have been prepared.
    3. Let the user select a classifier model.
    4. Collect common training parameters such as:
       - test size
       - split random state
       - CV usage
       - CV folds
       - scoring
    5. Collect model-specific training keyword arguments.
    6. If categorical feature columns are present, ask the user to choose a
       categorical encoder.
    7. Validate whether the selected model / encoder / scaler combination is
       allowed in the current workflow.
    8. If valid, dispatch training through ``zeus.train_model()``.
    9. Display success or failure information.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the prepared feature workflow and
        training interface.

    Returns
    -------
    None
        This function performs an interactive menu workflow and does not return
        a value.

    Validation Behavior
    -------------------
    The function exits early without training if any of the following occurs:

    - feature data or target data have not been prepared
    - the user cancels model selection
    - the user cancels common-parameter selection
    - the user cancels model-specific parameter selection
    - the user cancels categorical-encoder selection
    - the selected encoder / scaler combination is invalid for the chosen model
    - the underlying training call fails

    Notes
    -----
    - This function is intended for classification workflows only.
    - When available, the current feature count is passed to
      ``collect_model_train_kwargs()`` so PCA-related options can be validated
      before training.
    - If categorical features are detected, the selected encoder is passed to
      the training layer through the ``cat_encoder`` argument.
    - The menu layer also blocks known-invalid preprocessing combinations such
      as sparse OHE output used together with standard scaling for certain
      model families.
    """
    logger.info("Entered menu: Train Classifier")
    if zeus.feature_core is None:  # Check data be loaded
        logger.warning("Train Classifier failed: feature_core is None")
        print("⚠️ Please load data first ‼️")
        return

    if (
        getattr(zeus.feature_core, "X", None)
        is None  # Check feature and target selected
        or getattr(zeus.feature_core, "y", None) is None
    ):
        logger.warning("Train Classifier failed: X or y not prepared")
        print("⚠️ Please complete feature/target selection first ‼️")
        return

    # ---------- Classification task ----------
    model_name = select_model_name(zeus, task_type="classifier")
    if model_name is None:
        logger.info("Train Classifier cancelled at model selection")
        return

    logger.info("Classifier model selected: %s", model_name)

    # ---------- Classification training parameters ----------
    common_params = collect_common_training_params(task_type="classifier")
    if common_params is None:
        logger.info("Train Classifier cancelled at common parameter selection")
        return

    # ---------- Count feature amount ----------
    feature_count = (
        len(zeus.feature_core.feature_columns)
        if zeus.feature_core is not None
        and zeus.feature_core.feature_columns is not None
        else None
    )

    # ---------- Collect model training parameters ----------
    train_kwargs = collect_model_train_kwargs(
        model_name,
        feature_count=feature_count,
    )
    if train_kwargs is None:
        logger.info("Train Classifier cancelled at model-specific parameter selection")
        return

    # ---------- Feature is categorical-type ----------
    cat_encoder = "ohe"
    feature_data = getattr(zeus.feature_core, "X", None)

    if _has_categorical_features(feature_data):
        logger.info("Categorical features detected; encoder selection required")
        config = COMMON_PARAM_CONFIG["cat_encoder"]
        selected_encoder = select_from_options(
            label=config["label"],
            options=config["options"],
            default=config["default"],
        )
        if selected_encoder is None:
            logger.info("Train Classifier cancelled at categorical encoder selection")
            return

        cat_encoder = selected_encoder
        logger.info("Categorical encoder selected: %s", cat_encoder)

    # ---------- Block invalid encoder-scaler combination ----------
    if _is_invalid_sparse_scaler_combo(model_name, cat_encoder, train_kwargs):
        logger.warning(
            "Blocked invalid classifier combo | model=%s | cat_encoder=%s | scaler_type=%s",
            model_name,
            cat_encoder,
            train_kwargs.get("scaler_type"),
        )
        print("⚠️ Invalid parameter combination ‼️")
        print("🔔 OHE usually produces sparse output.")
        print("🔔 StandardScaler cannot center sparse matrices in this workflow.")
        print("🔔 Please change one of the following:")
        print("   - use cat_encoder = ordinal")
        print("   - use scaler_type = None")
        return

    logger.info(
        "Start classifier training | model=%s | common_params=%s | train_kwargs=%s | cat_encoder=%s",
        model_name,
        common_params,
        train_kwargs,
        cat_encoder,
    )

    # ---------- Classification parameter dispatched ----------
    result = zeus.train_model(
        model_name=model_name,
        test_size=common_params["test_size"],
        split_random_state=common_params["split_random_state"],
        use_cv=common_params["use_cv"],
        cv_folds=common_params["cv_folds"],
        scoring=common_params["scoring"],
        cat_encoder=cat_encoder,
        **train_kwargs,
    )

    if result is None:
        logger.warning("Classifier training failed: %s", model_name)
        print("⚠️ Classifier training failed ‼️")
        return

    logger.info("Classifier training completed successfully: %s", model_name)
    print(f"🍁 Classifier training completed: {model_name}")


# -------------------- Train regressor menu --------------------
@menu_wrapper("Train Regressor")
def train_regressor_menu(zeus: ZeusEngine):
    """
    Train a regressor model through the Zeus menu workflow.

    This menu function coordinates the full terminal-based workflow for training
    a regression model. It verifies that feature preparation has already been
    completed, collects shared and model-specific training parameters,
    optionally collects a categorical-feature encoding strategy, validates
    encoder / scaler compatibility, and finally dispatches the training request
    to ``zeus.train_model()``.

    Workflow
    --------
    1. Check that ``feature_core`` has been built.
    2. Check that both ``X`` and ``y`` have been prepared.
    3. Let the user select a regressor model.
    4. Collect common training parameters such as:
       - test size
       - split random state
       - CV usage
       - CV folds
       - scoring
    5. Collect model-specific training keyword arguments.
    6. If categorical feature columns are present, ask the user to choose a
       categorical encoder.
    7. Validate whether the selected model / encoder / scaler combination is
       allowed in the current workflow.
    8. If valid, dispatch training through ``zeus.train_model()``.
    9. Display success or failure information.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the prepared feature workflow and
        training interface.

    Returns
    -------
    None
        This function performs an interactive menu workflow and does not return
        a value.

    Validation Behavior
    -------------------
    The function exits early without training if any of the following occurs:

    - feature data or target data have not been prepared
    - the user cancels model selection
    - the user cancels common-parameter selection
    - the user cancels model-specific parameter selection
    - the user cancels categorical-encoder selection
    - the selected encoder / scaler combination is invalid for the chosen model
    - the underlying training call fails

    Notes
    -----
    - This function is intended for regression workflows only.
    - When available, the current feature count is passed to
      ``collect_model_train_kwargs()`` so PCA-related options can be validated
      before training.
    - If categorical features are detected, the selected encoder is passed to
      the training layer through the ``cat_encoder`` argument.
    - The menu layer also blocks known-invalid preprocessing combinations such
      as sparse OHE output used together with standard scaling for certain
      model families.
    """
    logger.info("Entered menu: Train Regressor")
    if zeus.feature_core is None:
        logger.warning("Train Regressor failed: feature_core is None")
        print("⚠️ Please load data first ‼️")
        return

    if (
        getattr(zeus.feature_core, "X", None) is None
        or getattr(zeus.feature_core, "y", None) is None
    ):
        logger.warning("Train Regressor failed: X or y not prepared")
        print("⚠️ Please complete feature/target selection first ‼️")
        return

    # ---------- Regression task ----------
    model_name = select_model_name(zeus, task_type="regressor")
    if model_name is None:
        logger.info("Train Regressor cancelled at model selection")
        return

    logger.info("Regressor model selected: %s", model_name)

    # ---------- Regression training parameters ----------
    common_params = collect_common_training_params(task_type="regressor")
    if common_params is None:
        logger.info("Train Regressor cancelled at common parameter selection")
        return

    feature_count = (
        len(zeus.feature_core.feature_columns)
        if zeus.feature_core is not None
        and zeus.feature_core.feature_columns is not None
        else None
    )

    # ---------- Collect model training parameters ----------
    train_kwargs = collect_model_train_kwargs(
        model_name,
        feature_count=feature_count,
    )
    if train_kwargs is None:
        logger.info("Train Regressor cancelled at model-specific parameter selection")
        return

    # ---------- Feature is categorical-type ----------
    cat_encoder = "ohe"
    feature_data = getattr(zeus.feature_core, "X", None)

    if _has_categorical_features(feature_data):
        logger.info("Categorical features detected; encoder selection required")
        config = COMMON_PARAM_CONFIG["cat_encoder"]
        selected_encoder = select_from_options(
            label=config["label"],
            options=config["options"],
            default=config["default"],
        )
        if selected_encoder is None:
            logger.info("Train Regressor cancelled at categorical encoder selection")
            return

        cat_encoder = selected_encoder
        logger.info("Categorical encoder selected: %s", cat_encoder)

    # ---------- Block invalid encoder-scaler combination ----------
    if _is_invalid_sparse_scaler_combo(model_name, cat_encoder, train_kwargs):
        logger.warning(
            "Blocked invalid regressor combo | model=%s | cat_encoder=%s | scaler_type=%s",
            model_name,
            cat_encoder,
            train_kwargs.get("scaler_type"),
        )
        print("⚠️ Invalid parameter combination ‼️")
        print("🔔 OHE usually produces sparse output.")
        print("🔔 StandardScaler cannot center sparse matrices in this workflow.")
        print("🔔 Please change one of the following:")
        print("   - use cat_encoder = ordinal")
        print("   - use scaler_type = None")
        return

    logger.info(
        "Start regressor training | model=%s | common_params=%s | train_kwargs=%s | cat_encoder=%s",
        model_name,
        common_params,
        train_kwargs,
        cat_encoder,
    )

    # ---------- Regression parameters dispatched ----------
    result = zeus.train_model(
        model_name=model_name,
        test_size=common_params["test_size"],
        split_random_state=common_params["split_random_state"],
        use_cv=common_params["use_cv"],
        cv_folds=common_params["cv_folds"],
        scoring=common_params["scoring"],
        cat_encoder=cat_encoder,
        **train_kwargs,
    )

    if result is None:
        logger.warning("Regressor training failed: %s", model_name)
        print("⚠️ Regressor training failed ‼️")
        return

    logger.info("Regressor training completed successfully: %s", model_name)
    print(f"🍁 Regressor training completed: {model_name}")


# -------------------- Current model summary menu --------------------
@menu_wrapper("Current Model Summary")
def current_model_summary_menu(zeus: ZeusEngine):
    """
    Display the summary of the current active model through the Zeus engine.

    This menu function serves as a terminal entry point for requesting the summary
    of the model currently stored in the active Zeus engine. It delegates the
    actual summary generation and display behavior to
    ``zeus.show_current_model_summary()``.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance that manages the current model object and the
        summary-display workflow.

    Returns
    -------
    None
        This function triggers an engine-level summary-display workflow and does
        not return a value.

    Notes
    -----
    This function does not construct or format the model summary itself. All
    availability checks, summary generation, and display behavior are handled by
    the Zeus engine.
    """
    logger.info("Entered menu: Current Model Summary")
    zeus.show_current_model_summary()  # Reveal model summary (from Zeus Engine)
    logger.info("Current model summary displayed")


# -------------------- Save current model menu --------------------
@menu_wrapper("Save Current Model")
def save_current_model_menu(zeus: ZeusEngine):
    """
    Save the current trained model using the model's default trained-model folder.

    This menu function checks whether a trained model is currently active in the
    Zeus engine and then delegates the save operation to
    ``zeus.save_current_model()``. The actual save directory, filename, and
    serialization details are determined by the active model's own missioner or
    model class.

    If no current model is available, the function prints a warning message and
    exits without calling the engine-level save workflow.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current trained model to be
        saved.

    Returns
    -------
    None
        This function performs an interactive menu workflow and does not return a
        value.

    Notes
    -----
    This function does not request a manual save path from the user. Model files
    are saved using the default trained-model folder defined by the active model's
    save implementation.
    """
    logger.info("Entered menu: Save Current Model")

    if zeus.current_model is None:
        logger.warning("Save Current Model failed: no current model")
        print("⚠️ No trained model to save ‼️")
        return

    logger.info("Start saving current model using model default MODEL_DIR")
    result = zeus.save_current_model()  # Save model into joblib

    if result is None:
        logger.warning("Failed to save current model")
        print("⚠️ Failed to save current model ‼️")
        return

    logger.info("Current model saved successfully: %s", result)
    print(f"🔥 Model saved successfully: {result}")


# -------------------- Load trained model menu --------------------
@menu_wrapper("Load Trained Model")
def load_trained_model_menu(zeus: ZeusEngine):
    """
    Load a previously saved trained model into the active Zeus engine.

    This menu function provides a terminal-based workflow for loading a serialized
    trained model. The workflow is divided into two interactive selection steps:

    1. select a registered model type from the Zeus model registry,
    2. select one saved ``.joblib`` file from that model type's default
    trained-model folder.

    After the user completes both selections, the function delegates the actual
    loading process to ``zeus.load_trained_model()``.

    If no registered models are available, if no saved model files are found for
    the selected model type, if the user cancels the workflow, or if loading
    fails, the function prints an appropriate warning message and exits without
    continuing.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance into which the trained model will be loaded.

    Returns
    -------
    None
        This function performs an interactive menu workflow and does not return a
        value.

    Notes
    -----
    Saved model files are resolved through the engine-level saved-model lookup
    workflow, and the displayed file list is generated from the default
    trained-model folder associated with the selected model type.

    Only the selected file's basename is displayed in the terminal menu, while the
    full file path is preserved internally and passed to the engine during the
    final load step.
    """
    logger.info("Entered menu: Load Trained Model")

    all_models = zeus.get_available_models()  # List avaliable models

    if not all_models:
        logger.warning("Load Trained Model failed: no model registry found")
        print("⚠️ No model registry found ‼️")
        return

    # ---------- Show the available models ----------
    model_map = {i: name for i, name in enumerate(all_models, 1)}

    print("\n----- 🔥 Registered Models 🔥 -----")
    for i, name in model_map.items():
        print(f"🗄️ {i}. {name}")
    print("-" * 50)

    # ---------- Select saved trained model ----------
    selected_num = input_int("🕯️ Select trained model type", default=-1)
    if selected_num is None:
        logger.info("Load Trained Model cancelled at model type selection")
        return

    if selected_num not in model_map:
        logger.warning(
            "Load Trained Model failed: model selection out of range | selected=%s",
            selected_num,
        )
        print("⚠️ Model selection is out of range ‼️")
        return

    # ---------- Get the model name ----------
    model_name = model_map[selected_num]
    logger.info("Load model type selected: %s", model_name)

    # ---------- List the folder saving model file ----------
    saved_model_files = zeus._get_saved_model_files(model_name)
    if not saved_model_files:
        logger.warning("No saved model files found for model: %s", model_name)
        print("⚠️ No saved model files found for this model type ‼️")
        return

    file_map = {i: path for i, path in enumerate(saved_model_files, 1)}

    print("\n----- 🔥 Saved Model Files 🔥 -----")
    for i, path in file_map.items():
        print(f"📦 {i}. {os.path.basename(path)}")
    print("-" * 50)

    # ---------- Select saved model file ----------
    selected_file_num = input_int("🕯️ Select saved model file", default=-1)
    if selected_file_num is None:
        logger.info("Load Trained Model cancelled at saved-file selection")
        return

    if selected_file_num not in file_map:
        logger.warning(
            "Load Trained Model failed: saved-file selection out of range | selected=%s",
            selected_file_num,
        )
        print("⚠️ Saved model selection is out of range ‼️")
        return

    filepath = file_map[selected_file_num]  # Get the saved trained model path
    logger.info("Selected saved model file: %s", filepath)

    logger.info(
        "Start loading trained model | model=%s | path=%s", model_name, filepath
    )

    # ---------- Dispath to engine ----------
    result = zeus.load_trained_model(
        model_name=model_name,
        filepath=filepath,
    )

    if result is None:
        logger.warning(
            "Failed to load trained model | model=%s | path=%s", model_name, filepath
        )
        print("⚠️ Failed to load trained model ‼️")
        return

    logger.info("Trained model loaded successfully: %s", model_name)
    print(f"🔥 Trained model loaded successfully: {model_name}")


# -------------------- Predict new target by trained model --------------------
@menu_wrapper("Predict with Current Model")
def predict_with_current_model_menu(zeus: ZeusEngine):
    """
    Predict target values for the currently loaded dataset using the active model.

    This menu function provides a terminal-based workflow for generating
    predictions from the model currently managed by the active Zeus engine. It
    first checks whether a trained or loaded model is available, then verifies
    that prediction input data has already been loaded into the current Zeus
    workflow.

    If the active model stores feature names from its training stage, the function
    displays those required feature columns so the user can confirm that the
    current dataset is compatible before continuing. Once confirmed, the function
    delegates the actual prediction step to ``zeus.predict_with_current_model()``
    and displays a preview of the prediction output.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model and the currently
        loaded source dataset.

    Returns
    -------
    None
        This function performs an interactive menu workflow and does not return a
        value.

    Workflow
    --------
    1. Validate that a current model exists.
    2. Validate that ``FeatureCore`` has been built.
    3. Retrieve the currently loaded source dataset from ``zeus.source_data``.
    4. Display required feature names if available.
    5. Ask the user to confirm prediction.
    6. Run prediction through ``zeus.predict_with_current_model(new_data)``.
    7. Display a preview of the prediction results.

    Notes
    -----
    This function assumes that the currently loaded dataset is intended to be used
    as prediction input and that its column names are compatible with the feature
    names expected by the active model.

    The function does not perform model training, feature selection, or dataset
    preprocessing. It only uses the already loaded source dataset and the current
    active model to generate predictions.
    """
    logger.info("Entered menu: Predict with Current Model")

    # ---------- Check current model ----------
    if zeus.current_model is None:
        logger.warning("Predict with Current Model failed: no current model")
        print("⚠️ No current model available for prediction ‼️")
        return

    # ---------- Check feature core ----------
    if zeus.feature_core is None:
        logger.warning("Predict with Current Model failed: FeatureCore not built")
        print("⚠️ FeatureCore has not been built ‼️")
        return

    # ---------- Get prediction dataset ----------
    new_data = zeus.source_data
    if new_data is None:
        logger.warning("Predict with Current Model failed: no source data available")
        print("⚠️ No source data available for prediction ‼️")
        return

    if not isinstance(new_data, pd.DataFrame):
        logger.warning(
            "Predict with Current Model failed: source data is not a DataFrame"
        )
        print("⚠️ Prediction data must be a pandas DataFrame ‼️")
        return

    # ---------- Show required feature names ----------
    feature_names = getattr(zeus.current_model, "feature_names", None)
    if feature_names:
        print("----- 🔥 Required Feature Columns 🔥 -----")
        for i, col in enumerate(feature_names, 1):
            print(f"🍒 {i}. {col}")
        print("-" * 50)

    # ---------- Confirm prediction ----------
    confirm = input_yesno("🕯️ Continue prediction with current dataset")
    if confirm is None or confirm is False:
        logger.info("Predict with Current Model cancelled at confirmation step")
        return

    # ---------- Run prediction ----------
    logger.info(
        "Start prediction with current model | rows=%s | cols=%s",
        len(new_data),
        len(new_data.columns),
    )
    predictions = zeus.predict_with_current_model(new_data)

    if predictions is None:
        logger.warning("Predict with Current Model failed")
        print("⚠️ Prediction failed ‼️")
        return

    logger.info("Prediction completed successfully | output_size=%s", len(predictions))

    # ---------- Display result preview ----------
    print("----- 🔥 Prediction Result Preview 🔥 -----")
    preview_count = min(10, len(predictions))
    for i in range(preview_count):
        print(f"{i+1}. {predictions[i]}")
    print("-" * 50)
    print(
        f"🔥 Prediction completed successfully. Total predictions: {len(predictions)}"
    )


# -------------------- Overall Menu2 Entering spot --------------------
@menu_wrapper("Model Management Menu")
def model_management_menu(zeus: ZeusEngine):
    """
    Display the top-level model management menu for Zeus workflows.

    This menu function provides the terminal-based entry point for all major
    model-related workflows in Zeus, including:

    - classifier training,
    - regressor training,
    - current model summary display,
    - current model saving,
    - trained model loading.

    The menu runs in a loop until the user selects the back option or cancels the
    selection input. Each valid menu item is mapped to a dedicated sub-menu
    function, and the selected workflow is executed using the active Zeus engine
    instance.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance used by all model-management workflows.

    Returns
    -------
    None
        This function manages an interactive terminal menu and does not return a
        value.

    Notes
    -----
    This function acts as a workflow dispatcher only. Detailed logic for training,
    saving, loading, and summary display is handled by the respective sub-menu
    functions.
    """
    logger.info("Entered menu: Model Management Menu")
    menu = [
        (1, "👁️ Training Classification", train_classifier_menu),
        (2, "🪢 Training Regression", train_regressor_menu),
        (3, "🔥 Current Model Summary", current_model_summary_menu),
        (4, "💾 Save Current trained Model", save_current_model_menu),
        (5, "📥 Load Trained Model", load_trained_model_menu),
        (6, "🔮 Predict with Current Model", predict_with_current_model_menu),
        (0, "↩️ Back", None),
    ]
    menu_width = 50

    while True:
        print("🏮  Model Management Menu 🏮 ".center(menu_width, "━"))
        for opt, label, _ in menu:
            print(f"{opt}. {label}")
        print("━" * menu_width)

        choice = input_int("🕯️ Select Model Training Services", default=-1)
        if choice is None:
            logger.info("Exited Model Management Menu by cancel/back")
            return

        matched = False
        for opt, _, func in menu:
            if choice == opt:
                logger.info("Model Management Menu selection: %s", choice)
                matched = True
                if func is None:
                    logger.info("Exited Model Management Menu")
                    return
                func(zeus)
                break

        if not matched:
            logger.warning("Invalid selection in Model Management Menu: %s", choice)
            print("⚠️ Invalid selection ‼️")


# =================================================
