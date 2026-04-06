# -------------------- Import Modules --------------------
import logging
from pprint import pprint

from Zeus.Menu_Helper_Decorator import input_int, input_yesno, menu_wrapper
from Zeus.Zeus_ML_Engine import ZeusEngine

logger = logging.getLogger("Zeus")


# -------------------- Helper: select target column for multi-output plot --------------------
def _select_multioutput_target_col(zeus: ZeusEngine) -> tuple[str | None, bool]:
    """
    Select a target column for multi-output classifier plotting workflows.

    This helper is used by classifier plotting menus that may require a
    ``target_col`` argument only when the active model behaves as a multi-output
    classifier.

    The helper inspects the current model stored in ``ZeusEngine`` and determines
    whether target-column selection is required for the current workflow.

    Behavior
    --------
    - If no current model is available, the helper prints a warning and returns
    ``(None, False)``.
    - If the current model does not expose a target object with ``columns``,
    the helper treats the workflow as effectively single-output and returns
    ``(None, False)``.
    - If the current target set contains one or zero columns, the helper treats
    the workflow as effectively single-output and returns ``(None, False)``.
    - If multiple target columns are detected, the helper displays a numbered
    terminal menu, prompts the user to choose one target column, validates the
    selection, and returns the selected column name together with a flag
    indicating that target selection was required.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model object.

    Returns
    -------
    tuple[str | None, bool]
        A tuple containing:

        - target_col : str or None
            Selected target column name when chosen successfully.

            Returns ``None`` when:
            - no target selection is needed,
            - the user cancels the selection,
            - or an invalid target number is entered.

        - selection_required : bool
            Whether the current workflow required target-column selection.

            - ``False`` means the workflow is effectively single-output, so no
            target selection is needed.
            - ``True`` means the workflow is multi-output and target selection was
            required.

    Side Effects
    ------------
    - Prints a target-column selection menu in the terminal when multiple target
    columns are available.
    - Logs selection, cancellation, and invalid-input events.

    Notes
    -----
    This helper is intended for classifier plotting menus such as:

    - SVC confusion matrix plots
    - KNN confusion matrix plots
    - ROC curve plots
    - Precision-Recall curve plots
    - Decision-function distribution plots

    Returning both ``target_col`` and ``selection_required`` allows menu functions
    to distinguish between:

    - single-output workflows where no target selection is needed, and
    - multi-output workflows where target selection was required but not completed.
    """
    if zeus.current_model is None:
        logger.warning("Target-column selection failed: no current model")
        print("⚠️ No current model available ‼️")
        return None, False

    # ---------- Get target variable Y from current model ----------
    y_test = getattr(zeus.current_model, "Y_test", None)

    if y_test is None or not hasattr(y_test, "columns"):
        return None, False

    # ---------- List target variable Y columns ----------
    col_map = {i: col for i, col in enumerate(y_test.columns, 1)}

    if len(col_map) <= 1:  # No target variable Y or single target variable Y
        return None, False

    print("---------- 🔥 Target Column List 🔥 ----------")
    for idx, col in col_map.items():
        print(f"🍒 {idx}. {col}")

    # ---------- Select multiple target variable Y columns ----------
    selected_num = input_int("🎯 Select target column number for multi-output")
    if selected_num is None:
        logger.info("Target-column selection cancelled")
        return None, True

    if selected_num not in col_map:
        logger.warning(
            "Target-column selection failed: invalid selection %s",
            selected_num,
        )
        print("⚠️ Invalid target column selection ‼️")
        return None, True

    target_col = col_map[selected_num]  # Get target columns by index
    logger.info("Selected target column: %s", target_col)
    return target_col, True


# -------------------- Helper: select dataset split for plot --------------------
def _select_plot_dataset() -> str | None:
    """
    Select the dataset split used for classifier evaluation plots.

    This helper displays a small terminal menu that lets the user choose
    whether a plot should be generated from the training split or the testing
    split. The selected value is returned as a string and can be passed
    directly to model-layer plotting methods such as ROC or Precision-Recall
    curve engines.

    Supported dataset values are:

    - ``"train"``
    - ``"test"``

    If the user cancels the selection or enters an invalid menu number, the
    helper returns ``None`` so the caller can stop the plotting workflow
    gracefully.

    Returns
    -------
    str or None
        Selected dataset split.

        - Returns ``"train"`` when the training split is selected.
        - Returns ``"test"`` when the testing split is selected.
        - Returns ``None`` when the selection is cancelled or invalid.

    Side Effects
    ------------
    - Prints a dataset-selection menu in the terminal.
    - Logs selection, cancellation, and invalid-input events.

    Notes
    -----
    This helper is intended for classifier plotting menus that support both
    training-set and testing-set visualization, such as:

    - ROC curve plots
    - Precision-Recall curve plots

    The default selection is the testing split because evaluation plots are
    most commonly interpreted on unseen data.
    """
    dataset_menu = {
        1: "train",
        2: "test",
    }

    print("---------- 🔥 Dataset Split 🔥 ----------")
    print("🍒 1. train")
    print("🍒 2. test")

    selected_num = input_int("🎯 Select dataset split", default=2)
    if selected_num is None:
        logger.info("Dataset split selection cancelled")
        return None

    if selected_num not in dataset_menu:
        logger.warning(
            "Dataset split selection failed: invalid selection %s", selected_num
        )
        print("⚠️ Invalid dataset split selection ‼️")
        return None

    dataset = dataset_menu[selected_num]
    logger.info("Selected dataset split: %s", dataset)
    return dataset


# -------------------- Show evaluation result menu --------------------
@menu_wrapper("Show Evaluation Result")
def show_evaluation_result_menu(zeus: ZeusEngine):
    """
    Display the current model evaluation result.

    This menu function retrieves the evaluation output associated with the active
    model from ``ZeusEngine`` and prints it in a formatted terminal view using
    ``pprint`` for readability. If no evaluation result is available, the
    function prints a warning message and exits without raising an exception.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance that stores the current model evaluation
        result and exposes the method used to retrieve it.

    Returns
    -------
    None
        This function only displays evaluation content in the terminal and does
        not return a value.

    Notes
    -----
    The evaluation structure is printed exactly as returned by
    ``zeus.get_model_evaluation()``. This function does not calculate any
    metrics itself and does not modify the evaluation state stored in the engine.
    """
    logger.info("Entered menu: Show Evaluation Result")

    evaluation = (
        zeus.get_model_evaluation()
    )  # Get evaluation from get_model_evaluation method
    if evaluation is None:
        logger.warning("No evaluation result available")
        print("⚠️ No evaluation result available ‼️")
        return

    print("\n---------- 🔥 Evaluation Result 🔥 ----------")
    pprint(evaluation)
    print("-" * 100)


# -------------------- Feature importance menu --------------------
@menu_wrapper("Feature Importance")
def feature_importance_menu(zeus: ZeusEngine):
    """
    Run and display feature-importance analysis for the current model.

    This menu function dispatches ``feature_importance_engine`` through the active
    ``ZeusEngine`` instance to obtain feature-importance information from the
    currently selected model. If the active model does not support feature-
    importance analysis, the function prints a warning message and exits.
    Otherwise, the returned result is displayed in the terminal using ``pprint``.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model object and the
        model-method dispatch interface.

    Returns
    -------
    None
        This function performs an analysis-and-display workflow and does not
        return a value.

    Notes
    -----
    Feature importance is only available for models whose implementation exposes
    ``feature_importance_engine``. Availability therefore depends on the type of
    the currently active model rather than on the menu itself.
    """
    logger.info("Entered menu: Feature Importance")
    logger.info("Running feature importance analysis")

    result = zeus.run_current_model_method(
        "feature_importance_engine"
    )  # Get feature importance from current model's method
    if result is None:
        logger.warning("Feature importance is not available for the current model")
        print("⚠️ Feature importance is not available for the current model ‼️")
        return

    logger.info("Feature importance analysis completed")
    pprint(result)


# -------------------- Tree plot menu --------------------
@menu_wrapper("Tree Plot")
def tree_plot_menu(zeus: ZeusEngine):
    """
    Plot the current tree-based model if tree visualization is supported.

    This menu function displays a numbered list of predefined tree-depth options,
    reads the user's selection, validates the input, optionally asks whether the
    generated figure should be saved, and then dispatches
    ``tree_plot_engine`` through the active ``ZeusEngine`` instance.

    If the user cancels the workflow, enters an invalid depth option, or the
    current model does not support tree plotting, the function exits after
    printing an appropriate warning message.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model object and the
        model-method dispatch interface.

    Returns
    -------
    None
        This function performs an interactive plotting workflow and does not
        return a value.

    Notes
    -----
    This menu is intended for models that support tree visualization, such as
    decision trees or tree-based ensembles with compatible plotting logic. The
    selected menu value is converted to the actual ``max_depth`` argument passed
    to the underlying plotting method, and the save choice is passed as
    ``save_fig``.
    """
    logger.info("Entered menu: Tree Plot")

    max_depth_menu = {
        1: 3,
        2: 5,
        3: 10,
        4: None,
    }
    print("\n----- 🌳 Tree Plot Max Depth -----")
    for i, value in max_depth_menu.items():
        print(f"🍀 {i}. {value}")
    print("-" * 50)

    # -------------------- Select max depth --------------------
    selected_num = input_int("🕯️ Select max depth", default=4)
    if selected_num is None:
        logger.info("Tree Plot cancelled at max-depth selection")
        return

    if selected_num not in max_depth_menu:
        logger.warning("Tree Plot failed: invalid max-depth selection %s", selected_num)
        print("⚠️ Invalid selection ‼️")
        return

    logger.info(
        "Tree Plot max-depth selected | menu_choice=%s | max_depth=%s",
        selected_num,
        max_depth_menu[selected_num],
    )

    # -------------------- Save tree plot --------------------
    save_fig = input_yesno("💾 Save plot", default=False)
    if save_fig is None:
        logger.info("Tree Plot cancelled at save option")
        return

    logger.info(
        "Running tree plot | max_depth=%s | save_fig=%s",
        max_depth_menu[selected_num],
        save_fig,
    )

    # -------------------- Recall to current model method --------------------
    zeus.run_current_model_method(
        "tree_plot_engine",
        max_depth=max_depth_menu[selected_num],
        save_fig=save_fig,
    )


# -------------------- SVM model insight menu --------------------
@menu_wrapper("SVM Model Insight")
def svm_model_insight_menu(zeus: ZeusEngine):
    """
    Display internal inspection information for the current SVM classifier model.

    This menu function dispatches ``svm_model_insight_engine`` through the active
    ``ZeusEngine`` model-method interface using the current SVM classifier model.

    For single-output classification, the workflow runs directly without requesting
    a target column.

    For multi-output classification, the function first calls
    ``_select_multioutput_target_col(zeus)`` to prompt the user to choose which
    target column should be inspected. The selected column is then passed to the
    underlying insight method as ``target_col``.

    If no current model is available, or if multi-output target selection is
    required but not completed, the function exits gracefully without calling the
    insight method.

    If the underlying insight method raises an exception, the function logs the
    failure, prints a warning message, and exits. When successful, the returned
    inspection result is displayed in the terminal using ``pprint``.

    After displaying the insight result, the menu asks whether the insight report
    should be saved as a text file. If the user chooses to save it, the function
    dispatches ``zeus._save_current_svm_insight_txt(...)`` and stores the returned
    inspection result as a plain-text report.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model object and the
        model-method dispatch interface.

    Returns
    -------
    None
        This function performs an interactive inspection-and-display workflow and
        does not return a value.

    Notes
    -----
    This menu is intended primarily for SVM classifier models that expose
    ``svm_model_insight_engine``.

    The inspection workflow is dispatched with:

    - ``target_col=<selected column or None>``

    For single-output classification, ``target_col`` remains ``None``.
    For multi-output classification, inspection is performed only for the selected
    target column.

    If the save option is enabled by the user, the displayed insight result is
    saved through the engine-layer SVM insight text-report interface.
    """
    logger.info("Entered menu: SVM Model Insight")

    if zeus.current_model is None:
        logger.warning("SVM model insight failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    # ---------- Multiple target columns confirmation ----------
    target_col, selection_required = _select_multioutput_target_col(zeus)

    if selection_required and target_col is None:
        logger.info("SVM Model Insight cancelled: no target column selected")
        print("⚠️ No target column selected ‼️")
        return

    logger.info("Running SVM model insight | target_col=%s", target_col)

    # -------------------- SVM insight results --------------------
    try:
        result = zeus.run_current_model_method(
            "svm_model_insight_engine",
            target_col=target_col,
        )

    except Exception:
        logger.exception("SVM model insight failed | target_col=%s", target_col)
        print("⚠️ SVM model insight failed ‼️")
        return

    print("---------- 🔥 SVM Model Insight 🔥 ----------")
    logger.info("SVM model insight displayed | target_col=%s", target_col)
    pprint(result)

    # -------------------- Save SVM insight results --------------------
    save_report = input_yesno("💾 Save SVM insight report", default=False)
    if save_report is None:
        logger.info("SVM Model Insight save cancelled at save option")
        return

    if save_report:
        try:
            saved_path = zeus._save_current_svm_insight_txt(
                result,
                target_col=target_col,
            )
        except Exception:
            logger.exception(
                "SVM insight report save failed | target_col=%s", target_col
            )
            print("⚠️ Failed to save SVM insight report ‼️")
            return

        logger.info(
            "SVM insight report saved successfully | target_col=%s | saved_path=%s",
            target_col,
            saved_path,
        )


# -------------------- SVC confusion matrix menu --------------------
@menu_wrapper("SVC Confusion Matrix Plot")
def svc_confusion_matrix_plot_menu(zeus: ZeusEngine):
    """
    Plot a confusion matrix for the current SVC classifier model.

    This menu function dispatches ``confusion_matrix_plot_engine`` through the
    active ``ZeusEngine`` model-method interface using the current SVC classifier
    model.

    The menu first asks whether the confusion matrix should be normalized. It then
    checks whether the active model requires target-column selection for a
    multi-output workflow by calling ``_select_multioutput_target_col(zeus)``.

    Workflow
    --------
    Single-output classification
        The plot runs directly with ``target_col=None``.

    Multi-output classification
        The helper prompts the user to choose a target column. The selected column
        name is passed to the underlying confusion-matrix plotting method as
        ``target_col``.

    If the current model is unavailable, if normalization input is cancelled, or
    if multi-output target selection is required but not completed, the function
    exits gracefully without calling the plotting method.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model object and the
        model-method dispatch interface.

    Returns
    -------
    None
        This function performs an interactive plotting workflow and does not return
        a value.

    Notes
    -----
    This menu is intended for SVC classifier models or compatible classifier
    implementations that expose ``confusion_matrix_plot_engine``.

    The plotting workflow is dispatched with:

    - ``normalize=<user selection>``
    - ``filename=None``
    - ``target_col=<selected column or None>``

    For single-output classification, ``target_col`` remains ``None``.
    For multi-output classification, only the selected target column is used to
    generate the confusion matrix.

    When the plotting method completes successfully, the saved plot path returned by
    the underlying method is displayed in the terminal.
    """
    logger.info("Entered menu: SVC Confusion Matrix Plot")

    if zeus.current_model is None:
        logger.warning("SVC confusion matrix plot failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    # ---------- Normalize ----------
    normalize = input_yesno("🕯️ Normalize confusion matrix", default=False)
    if normalize is None:
        logger.info("SVC Confusion Matrix Plot cancelled at normalize selection")
        return

    logger.info("SVC confusion matrix normalize option: %s", normalize)

    # ---------- Multiple target columns confirmation ----------
    target_col, selection_required = _select_multioutput_target_col(zeus)

    if selection_required and target_col is None:
        logger.info("Plot cancelled: no target column selected")
        print("⚠️ No target column selected ‼️")
        return

    logger.info(
        "Running SVC confusion matrix plot | normalize=%s | target_col=%s",
        normalize,
        target_col,
    )

    # ---------- Confusion matrix result ----------
    try:
        result = zeus.run_current_model_method(
            "confusion_matrix_plot_engine",
            normalize=normalize,
            filename=None,
            target_col=target_col,
        )

    except Exception:
        logger.exception(
            "SVC confusion matrix plot failed | normalize=%s | target_col=%s",
            normalize,
            target_col,
        )
        print("⚠️ SVC confusion matrix plot failed ‼️")
        return

    logger.info(
        "SVC confusion matrix plot completed | normalize=%s | target_col=%s",
        normalize,
        target_col,
    )
    print(f"🔥 SVC confusion matrix plot completed 🔥\n💾 Saved path ---> {result}")


# -------------------- KNN confusion matrix plot menu --------------------
@menu_wrapper("KNN Confusion Matrix Plot")
def knn_confusion_matrix_plot_menu(zeus: ZeusEngine):
    """
    Plot a confusion matrix for the current KNN classifier model.

    This menu function dispatches ``confusion_matrix_plot_engine`` through the
    active ``ZeusEngine`` model-method interface using the current KNN classifier
    model.

    The menu first asks whether the confusion matrix should be normalized. It then
    checks whether the active model requires target-column selection for a
    multi-output workflow by calling ``_select_multioutput_target_col(zeus)``.

    Workflow
    --------
    Single-output classification
        The plot runs directly with ``target_col=None``.

    Multi-output classification
        The helper prompts the user to choose a target column. The selected column
        name is passed to the underlying confusion-matrix plotting method as
        ``target_col``.

    If the current model is unavailable, if normalization input is cancelled, or
    if multi-output target selection is required but not completed, the function
    exits gracefully without calling the plotting method.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model object and the
        model-method dispatch interface.

    Returns
    -------
    None
        This function performs an interactive plotting workflow and does not return
        a value.

    Notes
    -----
    This menu is intended for KNN classifier models or compatible classifier
    implementations that expose ``confusion_matrix_plot_engine``.

    The plotting workflow is dispatched with:

    - ``normalize=<user selection>``
    - ``filename=None``
    - ``target_col=<selected column or None>``

    For single-output classification, ``target_col`` remains ``None``.
    For multi-output classification, only the selected target column is used to
    generate the confusion matrix.
    """
    logger.info("Entered menu: KNN Confusion Matrix Plot")

    if zeus.current_model is None:
        logger.warning("KNN confusion matrix plot failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    # ---------- Normalize ----------
    normalize = input_yesno("🕯️ Normalize confusion matrix", default=False)
    if normalize is None:
        logger.info("KNN Confusion Matrix Plot cancelled at normalize selection")
        return

    logger.info("KNN confusion matrix normalize option: %s", normalize)

    # ---------- Multiple target columns confirmation ----------
    target_col, selection_required = _select_multioutput_target_col(zeus)

    if selection_required and target_col is None:
        logger.info("Plot cancelled: no target column selected")
        print("⚠️ No target column selected ‼️")
        return

    logger.info(
        "Running KNN confusion matrix plot | normalize=%s | target_col=%s",
        normalize,
        target_col,
    )

    # ---------- Confusion matrix result ----------
    try:
        result = zeus.run_current_model_method(
            "confusion_matrix_plot_engine",
            normalize=normalize,
            filename=None,
            target_col=target_col,
        )

    except Exception:
        logger.exception(
            "KNN confusion matrix plot failed | normalize=%s | target_col=%s",
            normalize,
            target_col,
        )
        print("⚠️ KNN confusion matrix plot failed ‼️")

    logger.info(
        "Running KNN confusion matrix plot | normalize=%s | target_col=%s",
        normalize,
        target_col,
    )
    print(f"🔥 KNN confusion matrix plot completed 🔥\n💾 Saved path ---> {result}")


# -------------------- ROC curve plot menu --------------------
@menu_wrapper("ROC Curve Plot")
def roc_curve_plot_menu(zeus: ZeusEngine):
    """
    Plot the ROC curve for the current classifier model.

    This menu function dispatches ``roc_curve_plot_engine`` through the active
    ``ZeusEngine`` model-method interface using the dataset split selected by
    the user.

    Workflow
    --------
    1. Confirm that a current model is available.
    2. Prompt the user to choose the dataset split through
       ``_select_plot_dataset()``.
    3. If the active workflow is multi-output classification, prompt the user
       to select a target column through
       ``_select_multioutput_target_col(zeus)``.
    4. Dispatch ``roc_curve_plot_engine`` with:

       - ``dataset=<selected split>``
       - ``target_col=<selected column or None>``

    Single-output classification
        The ROC workflow runs directly with ``target_col=None``.

    Multi-output classification
        The user must choose one target column first. The ROC curve is then
        generated only for that selected target.

    If no current model is available, if dataset selection is cancelled, if a
    required target column is not selected, or if the active model does not
    support ROC plotting, the function exits gracefully after printing a
    warning message.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model object and the
        model-method dispatch interface.

    Returns
    -------
    None
        This function performs an interactive plotting-and-display workflow and
        does not return a value.

    Side Effects
    ------------
    - Prompts the user to choose a dataset split.
    - Prompts for target-column selection when multi-output classification is
      detected.
    - Calls the current model's ROC plotting method through the engine.
    - Prints the returned ROC summary in the terminal using ``pprint``.

    Notes
    -----
    This menu is intended for classifier models that expose
    ``roc_curve_plot_engine``. Actual availability depends on the currently
    active model implementation.

    ROC plotting is typically meaningful only for binary classification, so
    model-layer validation may raise an exception if the selected workflow does
    not satisfy binary ROC requirements.
    """
    logger.info("Entered menu: ROC Curve Plot")

    if zeus.current_model is None:
        logger.warning("ROC Curve Plot failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    # ---------- Select dataset ----------
    dataset = _select_plot_dataset()
    if dataset is None:
        logger.info("ROC Curve Plot cancelled at dataset selection")
        return

    # ---------- Multiple target columns confirmation ----------
    target_col, selection_required = _select_multioutput_target_col(zeus)

    if selection_required and target_col is None:
        logger.info("Plot cancelled: no target column selected")
        print("⚠️ No target column selected ‼️")
        return

    logger.info(
        "Running ROC curve plot | dataset=%s | target_col=%s",
        dataset,
        target_col,
    )

    result = zeus.run_current_model_method(
        "roc_curve_plot_engine",
        dataset=dataset,
        target_col=target_col,
    )

    if result is None:
        logger.warning("ROC curve plot is not available for the current model")
        print("⚠️ ROC curve plot is not available for the current model ‼️")
        return

    logger.info(
        "ROC curve plot completed | dataset=%s | target_col=%s",
        dataset,
        target_col,
    )
    print("---------- 🔥 ROC Curve Result 🔥 ----------")
    pprint(result)


# -------------------- Prcision recall curve plot menu --------------------
@menu_wrapper("Precision-Recall Curve Plot")
def precision_recall_curve_plot_menu(zeus: ZeusEngine):
    """
    Plot the Precision-Recall curve for the current classifier model.

    This menu function dispatches ``precision_recall_curve_plot_engine``
    through the active ``ZeusEngine`` model-method interface using the dataset
    split selected by the user.

    Workflow
    --------
    1. Confirm that a current model is available.
    2. Prompt the user to choose the dataset split through
       ``_select_plot_dataset()``.
    3. If the active workflow is multi-output classification, prompt the user
       to select a target column through
       ``_select_multioutput_target_col(zeus)``.
    4. Dispatch ``precision_recall_curve_plot_engine`` with:

       - ``dataset=<selected split>``
       - ``target_col=<selected column or None>``

    Single-output classification
        The PR workflow runs directly with ``target_col=None``.

    Multi-output classification
        The user must choose one target column first. The PR curve is then
        generated only for that selected target.

    If no current model is available, if dataset selection is cancelled, if a
    required target column is not selected, or if the active model does not
    support Precision-Recall plotting, the function exits gracefully after
    printing a warning message.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model object and the
        model-method dispatch interface.

    Returns
    -------
    None
        This function performs an interactive plotting-and-display workflow and
        does not return a value.

    Side Effects
    ------------
    - Prompts the user to choose a dataset split.
    - Prompts for target-column selection when multi-output classification is
      detected.
    - Calls the current model's Precision-Recall plotting method through the
      engine.
    - Prints the returned Precision-Recall summary in the terminal using
      ``pprint``.

    Notes
    -----
    This menu is intended for classifier models that expose
    ``precision_recall_curve_plot_engine``. Actual availability depends on the
    currently active model implementation.

    Precision-Recall plots are especially useful for imbalanced binary
    classification because they focus directly on positive-class retrieval
    quality. Model-layer validation may raise an exception if the selected
    workflow does not satisfy binary PR-curve requirements.
    """
    logger.info("Entered menu: Precision-Recall Curve Plot")

    if zeus.current_model is None:
        logger.warning("Precision-Recall Curve Plot failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    # ---------- Select dataset ----------
    dataset = _select_plot_dataset()
    if dataset is None:
        logger.info("Precision-Recall Curve Plot cancelled at dataset selection")
        return

    # ---------- Multiple target columns confirmation ----------
    target_col, selection_required = _select_multioutput_target_col(zeus)

    if selection_required and target_col is None:
        logger.info("Plot cancelled: no target column selected")
        print("⚠️ No target column selected ‼️")
        return

    logger.info(
        "Running Precision-Recall curve plot | dataset=%s | target_col=%s",
        dataset,
        target_col,
    )

    result = zeus.run_current_model_method(
        "precision_recall_curve_plot_engine",
        dataset=dataset,
        target_col=target_col,
    )

    if result is None:
        logger.warning(
            "Precision-Recall curve plot is not available for the current model"
        )
        print(
            "⚠️ Precision-Recall curve plot is not available for the current model ‼️"
        )
        return

    logger.info(
        "Precision-Recall curve plot completed | dataset=%s | target_col=%s",
        dataset,
        target_col,
    )
    print("---------- 🔥 Precision-Recall Curve Result 🔥 ----------")
    pprint(result)


# -------------------- SVM (Regression) diagonstics menu --------------------
@menu_wrapper("SVR Regression Diagnostics")
def svr_regression_diagnostics_menu(zeus: ZeusEngine):
    """
    Plot and summarize SVR regression diagnostic charts.

    This menu function dispatches ``plot_svr_regression_diagnostics`` through the
    active ``ZeusEngine`` model-method interface using the test dataset and
    prints the returned diagnostic summary.

    If the current model does not support SVR regression diagnostics, the
    function prints a warning message and exits.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model object and the
        model-method dispatch interface.

    Returns
    -------
    None
        This function performs a plotting-and-display workflow and does not
        return a value.

    Notes
    -----
    This menu is intended for SVR regressor models only. Availability depends on
    whether the active model implements
    ``plot_svr_regression_diagnostics``. The underlying diagnostic workflow may
    include predicted-versus-actual plots, residual plots, and error-distribution
    visualizations.
    """
    logger.info("Entered menu: SVR Regression Diagnostics")

    # ---------- Regression diagnotistics plots ----------
    logger.info("Running SVR regression diagnostics on test dataset")
    result = zeus.run_current_model_method(
        "plot_svr_regression_diagnostics",
        dataset="test",
    )

    if result is None:
        logger.warning(
            "SVR regression diagnostics is not available for the current model"
        )
        print("⚠️ SVR regression diagnostics is not available for the current model ‼️")
        return

    logger.info("SVR regression diagnostics completed")
    print("---------- 🔥 SVR Regression Diagnostics 🔥 ----------")
    pprint(result)


# -------------------- KNN (Regression) diagonstics menu --------------------
@menu_wrapper("KNN Regression Diagnostics")
def knn_regression_diagnostics_menu(zeus: ZeusEngine):
    """
    Plot and summarize KNN regression diagnostic charts.

    This menu function dispatches ``plot_knn_regression_diagnostics`` through the
    active ``ZeusEngine`` model-method interface and displays the returned
    diagnostic summary. The underlying regression diagnostic workflow typically
    includes plots such as predicted-versus-actual comparisons, residual scatter
    plots, and residual-distribution visualizations.

    If the current model does not support KNN regression diagnostics, the
    function prints a warning message and exits.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model object and the
        model-method dispatch interface.

    Returns
    -------
    None
        This function performs a plotting-and-display workflow and does not
        return a value.

    Notes
    -----
    This menu is intended for KNN regressor models only. Availability depends on
    whether the active model implements ``plot_knn_regression_diagnostics``.
    """
    logger.info("Entered menu: KNN Regression Diagnostics")

    # ---------- Regression diagnotistics plots ----------
    logger.info("Running KNN regression diagnostics on test dataset")
    result = zeus.run_current_model_method(
        "plot_knn_regression_diagnostics",
        dataset="test",
    )

    if result is None:
        logger.warning(
            "KNN regression diagnostics is not available for the current model"
        )
        print("⚠️ KNN regression diagnostics is not available for the current model ‼️")
        return

    logger.info("KNN regression diagnostics completed")
    print("---------- 🔥 KNN Regression Diagnostics 🔥 ----------")
    pprint(result)


# -------------------- Discision function distribution plot menu --------------------
@menu_wrapper("Decision Function Distribution Plot")
def decision_function_distribution_plot_menu(zeus: ZeusEngine):
    """
    Plot the decision-function score distribution for the current SVC model.

    This menu function dispatches
    ``decision_function_distribution_plot_engine`` through the active
    ``ZeusEngine`` model-method interface using the test dataset.

    For single-output classification, the workflow runs directly without requesting
    a target column.

    For multi-output classification, the function first calls
    ``_select_multioutput_target_col(zeus)`` to prompt the user to choose which
    target column should be analyzed. The selected column is then passed to the
    underlying plotting method as ``target_col``.

    If no current model is available, or if the active model does not support
    decision-function distribution plotting, the function prints a warning message
    and exits gracefully. When successful, the function prints the saved plot path.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance containing the current model object and the
        model-method dispatch interface.

    Returns
    -------
    None
        This function performs an interactive plotting workflow and does not return
        a value.

    Notes
    -----
    This menu is intended primarily for SVC classifier models that expose
    ``decision_function_distribution_plot_engine``.

    The plotting workflow is dispatched with:

    - ``dataset="test"``
    - ``target_col=<selected column or None>``

    For single-output classification, ``target_col`` remains ``None``.
    For multi-output classification, the decision-function distribution is plotted
    only for the selected target column.
    """
    logger.info("Entered menu: Decision Function Distribution Plot")

    if zeus.current_model is None:
        logger.warning("Decision function distribution plot failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    # ---------- Multiple target columns confirmation ----------
    target_col, selection_required = _select_multioutput_target_col(zeus)

    if selection_required and target_col is None:
        logger.info("Plot cancelled: no target column selected")
        print("⚠️ No target column selected ‼️")
        return

    logger.info(
        "Running decision-function distribution plot on test dataset | target_col=%s",
        target_col,
    )

    # ---------- Decision function distribution results ----------
    result = zeus.run_current_model_method(
        "decision_function_distribution_plot_engine",
        dataset="test",
        target_col=target_col,
    )

    if result is None:
        logger.warning(
            "Decision function distribution plot is not available for the current model"
        )
        print(
            "⚠️ Decision function distribution plot is not available for the current model ‼️"
        )
        return

    logger.info(
        "Decision function distribution plot completed | target_col=%s | saved_path=%s",
        target_col,
        result,
    )
    print("🔥 Decision function distribution plot completed 🔥")
    print(f"💾 Saved path ---> {result}")


# -------------------- Evaluation menu --------------------
@menu_wrapper("Evaluation Menu")
def evaluation_menu(zeus: ZeusEngine):
    """
    Display evaluation and model-analysis tools for the current trained model.

    This menu function provides a unified terminal-based evaluation interface for
    the currently active model stored in ``ZeusEngine``. Available menu entries
    include general evaluation display, tree-model utilities, SVM classifier
    analysis, SVM regressor diagnostics, confusion-matrix plots, curve-based
    classification metrics, and KNN-specific diagnostics.

    The menu configuration itself is static, but actual method availability
    depends on the type and implementation of the active current model. If a
    selected tool is unsupported by the current model, the called submenu will
    display a warning message and exit gracefully.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance that stores the current model and provides
        method-dispatch support for evaluation workflows.

    Returns
    -------
    None
        This function runs an interactive menu loop and does not return a value.

    Notes
    -----
    This menu combines tools for multiple model families in a single entry point,
    including common evaluation display, tree-model analysis, SVC-specific
    analysis, SVR-specific diagnostics, and KNN classifier/regressor utilities.
    Support for each item is determined dynamically at runtime through
    ``zeus.run_current_model_method(...)`` or other engine-level access methods.
    """
    logger.info("Entered menu: Evaluation Menu")
    menu_config = {
        1: (
            "📜 Show Evaluation Result (‼️ New training model only)",
            show_evaluation_result_menu,
        ),
        2: (
            "🎀 Feature Importance (‼️ SVC/SVR/KNN no support)",
            feature_importance_menu,
        ),
        3: ("🌳 Tree Plot", tree_plot_menu),
        4: (
            "👀 SVM Model Insight (🔔 Only Classification applied)",
            svm_model_insight_menu,
        ),
        5: ("📋 SVC Confusion Matrix Plot", svc_confusion_matrix_plot_menu),
        6: ("🏹 ROC Curve Plot (Classification)", roc_curve_plot_menu),
        7: (
            "🔬 Precision-Recall Curve Plot (Classification)",
            precision_recall_curve_plot_menu,
        ),
        8: ("📈 SVR Regression Diagnostics", svr_regression_diagnostics_menu),
        9: (
            "🧮 SVC Decision Function Distribution Plot",
            decision_function_distribution_plot_menu,
        ),
        10: ("📋 KNN Confusion Matrix Plot ", knn_confusion_matrix_plot_menu),
        11: ("📈 KNN Regression Diagnostics", knn_regression_diagnostics_menu),
    }

    while True:
        print("━━━━━━━━🏮  Evaluation Menu 🏮 ━━━━━━━")
        for key, (label, _) in menu_config.items():
            print(f"{key}. {label}")
        print("0. ↩️ Back")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        selected_service = input_int("🕯️ Select Services")
        if selected_service is None or selected_service == 0:
            logger.info("Exited Evaluation Menu")
            return

        selected_item = menu_config.get(selected_service)
        if selected_item is None:
            logger.warning("Invalid selection in Evaluation Menu: %s", selected_service)
            print("⚠️ Invalid selection ‼️")
            continue

        label, action = selected_item
        logger.info("Evaluation Menu selection: %s - %s", selected_service, label)
        action(zeus)


# -----------------------------------------
