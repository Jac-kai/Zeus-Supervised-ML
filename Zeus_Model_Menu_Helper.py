# -------------------- Import Modules --------------------
from Zeus.Menu_Config import COMMON_PARAM_CONFIG, MODEL_PARAM_CONFIG, SCORING_CONFIG
from Zeus.Menu_Helper_Decorator import input_int
from Zeus.Zeus_ML_Engine import MODEL_REGISTRY, ZeusEngine


# -------------------- Helper: select model name --------------------
def select_model_name(zeus: ZeusEngine, task_type: str) -> str | None:
    """
    Display available model names for a given task type and return the selected model.

    This helper queries the active ``ZeusEngine`` for all registered model names that
    match the requested task type, builds a numbered terminal menu, and lets the user
    choose one model by numeric input.

    Behavior
    --------
    - If no models are available for the requested task type, the function prints a
    warning and returns ``None``.
    - If the user cancels the input, the function returns ``None``.
    - If the selected menu number is outside the valid range, the function prints a
    warning and returns ``None``.
    - If a valid menu number is selected, the corresponding model name is returned.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance used to retrieve available model names from the
        registered model list.
    task_type : str
        Task category used to filter candidate models, typically ``"classifier"``
        or ``"regressor"``.

    Returns
    -------
    str | None
        The selected model name if a valid menu option is chosen; otherwise ``None``.

    Workflow
    --------
    1. Query the engine for all available model names matching ``task_type``
    2. Build a 1-based menu mapping from menu number to model name
    3. Print the available model list in the terminal
    4. Read the user's numeric selection through ``input_int()``
    5. Validate the selected menu number
    6. Return the mapped model name if valid

    Notes
    -----
    This helper only handles model selection. It does not verify whether data has
    already been loaded, features and targets have been selected, or training can
    start successfully.

    Examples
    --------
    Select a classifier model::

        model_name = select_model_name(zeus, "classifier")

    Possible result::

        "RandomForestClassifier"

    If the user cancels or selects an invalid option, the function returns ``None``.
    """
    model_names = zeus.get_available_models(
        task_type=task_type
    )  # Get model from Zeus Engine

    if not model_names:
        print(f"⚠️ No available {task_type} models found ‼️")
        return None

    # ---------- List models ----------
    model_map = {i: name for i, name in enumerate(model_names, 1)}

    print(f"\n----- 🔥 Available {task_type.title()} Models 🔥 -----")
    for i, name in model_map.items():
        print(f"🧠 {i}. {name}")
    print("-" * 50)

    # ---------- Select model ----------
    selected_num = input_int("🕯️ Select model", default=-1)
    if selected_num is None:
        return None

    if selected_num not in model_map:
        print("⚠️ Model selection is out of range ‼️")
        return None

    return model_map[
        selected_num
    ]  # Return selected model from get_available_models method (from Zeus engine)


# -------------------- Helper: select from options --------------------
def select_from_options(
    label: str, options: dict[int, object], default: int | None = None
):
    """
    Display a numbered option menu and return the mapped option value.

    This helper is a reusable terminal-menu utility for parameter selection. It
    prints a descriptive label, shows each available numeric menu option together
    with its mapped value, and reads the user's choice through ``input_int()``.

    Behavior
    --------
    - If the user cancels the numeric input, the function returns the sentinel
      string ``"__CANCELLED__"``.
    - If the selected menu number is not found in the provided ``options``
      mapping, the function prints a warning and returns ``"__CANCELLED__"``.
    - If the selected menu number is valid, the mapped option value is returned.

    Parameters
    ----------
    label : str
        Descriptive menu title shown above the option list.
    options : dict[int, object]
        Mapping from numeric menu choices to actual parameter values.
    default : int | None, optional
        Default numeric menu selection passed to ``input_int()``.

    Returns
    -------
    object | str
        The mapped option value corresponding to the selected numeric key if
        valid.

        Returns ``"__CANCELLED__"`` when:
        - the user cancels input,
        - or the selected numeric menu key is invalid.

    Workflow
    --------
    1. Print the menu label
    2. Print each numeric menu key and its mapped value
    3. Read the user's numeric selection through ``input_int()``
    4. Validate that the selected key exists in ``options``
    5. Return the mapped value for the chosen option, or the cancellation
       sentinel when the workflow is cancelled

    Notes
    -----
    This helper returns the mapped value stored in ``options``, not the numeric
    menu key entered by the user.

    The sentinel string is used so that legitimate mapped values such as
    ``None`` can still be returned safely without being confused with menu
    cancellation.

    Examples
    --------
    Select a scaler type from a menu::

        value = select_from_options(
            label="Scaler Type",
            options={1: "standard", 2: "minmax", 3: "robust", 4: None},
            default=1,
        )

    Possible results::

        "standard"
        None
        "__CANCELLED__"
    """
    # ---------- List parameters ----------
    print(f"\n----- {label} -----")
    for num, value in options.items():
        print(f"📌 {num}. {value}")
    print("-" * 50)

    # ---------- Select parameters ----------
    selected_num = input_int("🕯️ Select option", default=default)
    if selected_num is None:
        return "__CANCELLED__"

    if selected_num not in options:
        print("⚠️ Selection is out of range ‼️")
        return "__CANCELLED__"

    return options[selected_num]  # Return selected option


# -------------------- Helper: get task type --------------------
def get_model_task_type(zeus: ZeusEngine, model_name: str) -> str | None:
    """
    Return the registered task type for a given model name.

    This helper looks up the specified model name in the Zeus model registry and
    returns its associated task type, such as ``"classifier"`` or ``"regressor"``.
    It first checks whether the provided engine instance exposes ``MODEL_REGISTRY``.
    If not found there, it falls back to the module-level ``MODEL_REGISTRY``.

    Behavior
    --------
    - If the engine instance exposes ``MODEL_REGISTRY`` and the model name exists
    there, the task type from that registry entry is returned.
    - Otherwise, the helper checks the module-level ``MODEL_REGISTRY``.
    - If the model name is not found in either registry, the function returns
    ``None``.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance that may expose a model registry attribute.
    model_name : str
        Registered model name whose task type should be retrieved.

    Returns
    -------
    str | None
        The associated task type if the model name exists in the registry;
        otherwise ``None``.

    Workflow
    --------
    1. Check whether the engine instance exposes ``MODEL_REGISTRY``
    2. Attempt to retrieve the model metadata from the engine-level registry
    3. If not found, fall back to the module-level ``MODEL_REGISTRY``
    4. Return the ``task_type`` field if available
    5. Return ``None`` if the model is not registered

    Notes
    -----
    This helper only inspects model registry metadata. It does not verify whether a
    model has been trained, loaded, or set as the current active model.

    Examples
    --------
    Get the task type of a registered model::

        task_type = get_model_task_type(zeus, "RandomForestClassifier")

    Possible result::

        "classifier"
    """
    # ----------- Get MODEL REGISTRY List -----------
    model_info = (
        zeus.MODEL_REGISTRY.get(model_name) if hasattr(zeus, "MODEL_REGISTRY") else None
    )
    if model_info:
        return model_info.get("task_type")

    info = MODEL_REGISTRY.get(model_name)
    return info.get("task_type") if info else None


# -------------------- Helper: collect common training params --------------------
def collect_common_training_params(task_type: str) -> dict | None:
    """
    Collect common training parameters shared across Zeus model-training workflows.

    This helper interactively gathers general training settings that are not tied
    to a specific model implementation. It reads parameter definitions from
    ``COMMON_PARAM_CONFIG`` and scoring choices from ``SCORING_CONFIG``, then
    collects the selected values through repeated terminal menus.

    The collected parameters include test split ratio, split random seed,
    cross-validation usage, optional CV fold count, and the scoring metric for
    the specified task type.

    Behavior
    --------
    - The helper always collects ``test_size``, ``split_random_state``, and
      ``use_cv`` first.
    - If ``use_cv`` is ``True``, the helper additionally collects ``cv_folds``.
    - If ``use_cv`` is ``False``, ``cv_folds`` is recorded as ``None``.
    - The helper then collects the scoring method using the scoring
      configuration for the provided task type.
    - If any required selection is cancelled or invalid, the function returns
      ``None``.

    Parameters
    ----------
    task_type : str
        Task category used to choose the scoring menu, typically
        ``"classifier"`` or ``"regressor"``.

    Returns
    -------
    dict | None
        Dictionary containing the collected common training parameters if all
        selections succeed; otherwise ``None``.

    Returned Keys
    -------------
    The returned dictionary contains:

    - ``test_size``
    - ``split_random_state``
    - ``use_cv``
    - ``cv_folds``
    - ``scoring``

    Workflow
    --------
    1. Collect ``test_size`` from ``COMMON_PARAM_CONFIG``
    2. Collect ``split_random_state`` from ``COMMON_PARAM_CONFIG``
    3. Collect ``use_cv`` from ``COMMON_PARAM_CONFIG``
    4. If CV is enabled, collect ``cv_folds``
    5. Select the scoring method from ``SCORING_CONFIG[task_type]``
    6. Return the collected parameter dictionary

    Notes
    -----
    This helper only handles shared training parameters. Model-specific keyword
    arguments are collected separately by ``collect_model_train_kwargs()``.

    This workflow treats ``"__CANCELLED__"`` as the menu-cancellation signal so
    that legitimate parameter values such as ``None`` can still be preserved
    safely when they are valid configuration values.

    Examples
    --------
    Collect common classifier training parameters::

        params = collect_common_training_params("classifier")

    Possible returned result::

        {
            "test_size": 0.2,
            "split_random_state": 42,
            "use_cv": True,
            "cv_folds": 5,
            "scoring": "accuracy",
        }
    """
    params = {}

    # ---------- Record common parameters to model ----------
    for param_name in ("test_size", "split_random_state", "use_cv"):
        config = COMMON_PARAM_CONFIG[param_name]
        selected_value = select_from_options(
            label=config["label"],
            options=config["options"],
            default=config["default"],
        )
        if selected_value == "__CANCELLED__":
            return None
        params[param_name] = selected_value

    # ---------- Record CV fold when using CV ----------
    if params["use_cv"]:
        config = COMMON_PARAM_CONFIG["cv_folds"]
        selected_value = select_from_options(
            label=config["label"],
            options=config["options"],
            default=config["default"],
        )
        if selected_value == "__CANCELLED__":
            return None
        params["cv_folds"] = selected_value
    else:
        params["cv_folds"] = None

    # ---------- Record scoring method ----------
    scoring_config = SCORING_CONFIG[task_type]
    scoring_value = select_from_options(
        label=scoring_config["label"],
        options=scoring_config["options"],
        default=scoring_config["default"],
    )
    if scoring_value == "__CANCELLED__":
        return None
    params["scoring"] = scoring_value

    return params


# -------------------- Helper: should skip dependent param --------------------
def should_skip_param(param_config: dict, current_params: dict) -> bool:
    """
    Determine whether a model-specific parameter should be skipped in the current menu flow.

    This helper supports conditional parameter menus during model-specific argument
    collection. If a parameter configuration defines a dependency rule through the
    ``depends_on`` field, the function checks whether the required controlling
    parameter has already been selected with the expected value.

    Behavior
    --------
    - If no ``depends_on`` rule is defined, the function returns ``False``.
    - If a dependency rule exists and the required condition is not satisfied, the
    function returns ``True`` to indicate that the parameter should be skipped.
    - If the dependency condition is satisfied, the function returns ``False``.

    Expected Dependency Format
    --------------------------
    The ``depends_on`` field is expected to contain a two-element sequence:

    1. controlling parameter name
    2. expected parameter value

    For example::

        ("criterion", "entropy")

    means that the parameter should only be shown when ``current_params["criterion"]``
    equals ``"entropy"``.

    Parameters
    ----------
    param_config : dict
        Parameter configuration dictionary that may include a ``depends_on`` rule.
    current_params : dict
        Dictionary containing parameters that have already been selected in the
        current workflow.

    Returns
    -------
    bool
        ``True`` if the parameter should be skipped;
        ``False`` if the parameter should still be shown.

    Workflow
    --------
    1. Read the optional ``depends_on`` setting from the parameter configuration
    2. Return ``False`` immediately if no dependency is defined
    3. Compare the current selected value of the controlling parameter against the
    expected dependency value
    4. Return whether the parameter should be skipped

    Notes
    -----
    This helper assumes that dependent parameters appear after their controlling
    parameters in the configuration order.

    Examples
    --------
    Check whether a dependent parameter should be skipped::

        skip = should_skip_param(
            {"name": "max_features", "depends_on": ("use_feature_limit", True)},
            {"use_feature_limit": False},
        )

    Possible result::

        True
    """
    depends_on = param_config.get("depends_on")
    if not depends_on:
        return False

    dep_name, dep_expected_value = depends_on
    return current_params.get(dep_name) != dep_expected_value


# -------------------- Helper: collect model-specific kwargs --------------------
def collect_model_train_kwargs(
    model_name: str,
    feature_count: int | None = None,
) -> dict | None:
    """
    Collect model-specific training keyword arguments for a selected model.

    This helper reads the configuration for the given model name from
    ``MODEL_PARAM_CONFIG`` and interactively gathers model-specific training
    arguments through numbered terminal menus. Parameters are processed in the
    defined configuration order, and dependency-based parameters may be skipped
    automatically through ``should_skip_param()``.

    The helper also applies special menu logic for PCA-related parameters based
    on the current number of selected feature columns.

    Behavior
    --------
    - If the selected model has no model-specific parameter configuration, the
      function returns an empty dictionary.
    - Parameters with unmet dependency conditions are skipped automatically.
    - If ``use_pca`` is encountered and ``feature_count < 2``, PCA is disabled
      automatically by setting ``use_pca`` to ``False``.
    - If ``pca_n_components`` is shown and ``feature_count`` is provided, menu
      options greater than the current feature count are removed.
    - If the user cancels any required selection, the function returns ``None``.
    - Otherwise, the collected model-specific keyword arguments are returned.

    Parameters
    ----------
    model_name : str
        Registered model name whose model-specific training parameters should be
        collected.
    feature_count : int | None, optional
        Number of currently selected feature columns. This value is used to
        adjust PCA-related menu behavior.

    Returns
    -------
    dict | None
        Dictionary containing the collected model-specific keyword arguments if
        collection succeeds; otherwise ``None``.

    Workflow
    --------
    1. Read the parameter configuration list for ``model_name`` from
       ``MODEL_PARAM_CONFIG``
    2. Iterate through parameters in configuration order
    3. Skip dependency-controlled parameters when their conditions are not met
    4. Automatically disable PCA if ``feature_count < 2``
    5. Filter invalid ``pca_n_components`` options when needed
    6. Collect each remaining parameter value through ``select_from_options()``
    7. Store the selected value in the keyword-argument dictionary
    8. Return the collected keyword arguments

    Notes
    -----
    This helper only collects model-specific parameters. Shared training
    parameters such as ``test_size``, ``split_random_state``, ``use_cv``,
    ``cv_folds``, and ``scoring`` are collected separately by
    ``collect_common_training_params()``.

    PCA validation is handled here to prevent invalid menu selections before the
    training layer is called.

    Menu cancellation is detected through the sentinel value
    ``"__CANCELLED__"`` so that valid parameter values such as ``None`` can be
    preserved correctly. This is important for options like:

    - ``scaler_type = None``
    - ``pca_n_components = None``
    - other model parameters whose valid mapped value may be ``None``

    Examples
    --------
    Collect model-specific arguments for a model with PCA-related settings::

        kwargs = collect_model_train_kwargs(
            model_name="SVMClassifier",
            feature_count=4,
        )

    Possible returned result::

        {
            "scaler_type": "standard",
            "use_pca": True,
            "pca_n_components": 3,
            "kernel": "rbf",
            "C": 1.0,
        }
    """
    param_list = MODEL_PARAM_CONFIG.get(model_name, [])
    kwargs = {}

    # ---------- Get model's specific parameters ----------
    for param_config in param_list:
        if should_skip_param(param_config, kwargs):  # Skip certain parameters
            continue

        if (
            param_config["name"] == "use_pca"
            and feature_count is not None
            and feature_count < 2
        ):
            print(
                "🔔 Current feature count is < 2, PCA will be disabled automatically 🔔"
            )
            kwargs["use_pca"] = False
            continue

        # ---------- Check PCA number and feature number ----------
        if param_config["name"] == "pca_n_components" and feature_count is not None:
            raw_options = param_config["options"]

            filtered_options = {
                k: v for k, v in raw_options.items() if v is None or v <= feature_count
            }

            param_config = {
                **param_config,
                "options": filtered_options,
            }

        selected_value = select_from_options(
            label=param_config["label"],
            options=param_config["options"],
            default=param_config.get("default"),
        )

        if selected_value == "__CANCELLED__":
            return None

        kwargs[param_config["name"]] = selected_value

    return kwargs


# =================================================
