# -------------------- Import Modules --------------------
import logging

from Zeus.Menu_Helper_Decorator import column_list, input_int, input_list, menu_wrapper
from Zeus.Zeus_ML_Engine import ZeusEngine

logger = logging.getLogger("Zeus")


# -------------------- Loaded ML Data Menu --------------------
@menu_wrapper("Loaded ML Data")
def loaded_ml_data_menu(zeus: ZeusEngine):
    """
    Load an ML dataset into the Zeus workflow.

    This menu guides the user through selecting a folder and file from the current
    working place, then delegates the actual dataset-loading process to
    ``ZeusEngine``. If loading succeeds, the loaded dataset becomes the active
    source data for the Zeus workflow and downstream cores are initialized.

    The menu repeats when dataset loading fails, allowing the user to choose a
    different file without leaving the workflow.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance.

    Returns
    -------
    None
        Returns ``None`` when the user exits the menu or after a dataset has been
        loaded successfully.

    Workflow
    --------
    1. Show working-place folders.
    2. Ask the user to choose a folder number.
    3. Show files inside the selected folder.
    4. Ask the user to choose a file number.
    5. Load the dataset through ``zeus.ml_dataset_search()``.
    6. If loading fails, restart the selection loop.
    7. If loading succeeds, print a success message and exit the menu.

    Notes
    -----
    If loading succeeds, downstream cores such as ``vision_core`` and
    ``feature_core`` are built automatically by ``ZeusEngine``.
    """
    logger.info("Entered menu: Loaded ML Data")
    while True:
        # ---------- Show working-place folders ----------
        folders = zeus.hunter_core.working_place_searcher()
        logger.info("Working-place folders loaded")

        print(f"\n----- 🔥 Folder Lists 🔥-----\n{'-'*50}")
        for i, folder in folders.items():
            print(f"📂 {i}. {folder}")

        # ---------- Select folder ----------
        selected_folder_num = input_int("🕯️ Select folder")
        if selected_folder_num is None:
            logger.info("Loaded ML Data menu exited at folder selection")
            return None

        logger.info("Selected folder number: %s", selected_folder_num)

        # ---------- Show files inside selected folder ----------
        files = zeus.hunter_core.files_searcher_from_folders(
            selected_folder_num=selected_folder_num,
        )
        logger.info("Files loaded from folder number: %s", selected_folder_num)

        print(f"\n----- 🔥 File Lists 🔥-----\n{'-'*50}")
        for i, file in files.items():
            print(f"📄 {i}. {file}")

        # ---------- Select file ----------
        selected_file_num = input_int("🕯️ Select file")
        if selected_file_num is None:
            logger.info("Loaded ML Data menu exited at file selection")
            return None

        logger.info("Selected file number: %s", selected_file_num)

        # ---------- Load dataset to Zeus Engine ----------
        logger.info(
            "Start loading dataset | folder_num=%s | file_num=%s",
            selected_folder_num,
            selected_file_num,
        )

        loaded_data = zeus.ml_dataset_search(
            selected_folder_num=selected_folder_num,
            selected_file_num=selected_file_num,
        )

        if loaded_data is None:
            logger.warning(
                "Failed to load ML dataset | folder_num=%s | file_num=%s",
                selected_folder_num,
                selected_file_num,
            )
            print("⚠️ Failed to load ML dataset ‼️")
            continue

        logger.info(
            "ML dataset loaded successfully | folder_num=%s | file_num=%s",
            selected_folder_num,
            selected_file_num,
        )
        print(f"🔥 ML dataset loaded successfully.\n{'-' * 100}")
        return


# -------------------- Select Feature Menu --------------------
@menu_wrapper("Select Feature")
def select_feature_target_menu(zeus: ZeusEngine):
    """
    Interactively select target and feature columns for model training.

    This menu allows the user to define the supervised learning input-output
    structure from the currently loaded dataset in ``ZeusEngine``. The user first
    selects one or more target columns, then optionally selects one or more
    feature columns by their displayed numeric indices.

    If no feature columns are entered, all non-target columns are automatically
    used as features. After the selection is built, the current feature-target
    mapping is displayed and the user can either confirm it, reselect it, or go
    back.

    Parameters
    ----------
    zeus : ZeusEngine
        Active Zeus engine instance that stores the source dataset and handles
        feature-target selection through ``feature_core``.

    Returns
    -------
    None
        Returns ``None`` when the user exits the menu, when required engine/data
        state is not available, or after the selection flow finishes.

    Workflow
    --------
    1. Check whether source data has been loaded.
    2. Check whether ``feature_core`` has been built.
    3. Show all available source-data columns with numeric indices.
    4. Ask the user to choose one or more target columns.
    5. Ask the user to choose feature columns by numeric indices.
    6. If feature input is skipped, use all non-target columns automatically.
    7. Build ``X`` and ``y`` through ``zeus.select_feature_target()``.
    8. Show the current feature-target selection summary.
    9. Let the user confirm, reselect, or exit.

    Validation Rules
    ----------------
    - A target column selection is required.
    - The target selection must contain one or more numeric indices.
    - Feature selections, if provided, must all be numeric indices.
    - All selected indices must exist in the displayed column mapping.
    - Feature columns must not include any selected target column.

    Notes
    -----
    If the user chooses to reselect, the existing feature selection is cleared by
    calling ``zeus.reset_feature_selection()`` before starting the selection flow
    again.

    This menu depends on:

    - ``zeus.source_data`` being available
    - ``zeus.feature_core`` already being initialized
    """
    logger.info("Entered menu: Select Feature")
    if zeus.source_data is None:
        logger.warning("Select Feature menu failed: no source data available")
        print("⚠️ No source data available. Please load data first ‼️")
        return

    if zeus.feature_core is None:
        logger.warning("Select Feature menu failed: FeatureCore not built")
        print("⚠️ FeatureCore has not been built. Please load data first ‼️")
        return

    while True:
        # ---------- Show available columns ----------
        col_map = column_list(zeus.source_data)
        if not col_map:
            logger.warning("No columns available for feature selection")
            print("⚠️ No columns available for selection ‼️")
            return

        # ---------- Select target column by index ----------
        target_input = input_list("🕯️ Select TARGET column index")
        if target_input == "__BACK__":
            logger.info("Select Feature menu exited at target selection")
            return

        if not target_input:
            print("⚠️ TARGET column selection is required ‼️")
            continue

        if not all(
            str(item).isdigit() for item in target_input
        ):  # Check input must be a digit
            print("⚠️ TARGET selections must all be numeric indices ‼️")
            continue

        target_indices = [
            int(item) for item in target_input
        ]  # Collect selected target columns' index

        if any(idx not in col_map for idx in target_indices):
            print("⚠️ One or more TARGET indices are out of range ‼️")
            continue

        target_column = [
            col_map[idx] for idx in target_indices
        ]  # Get target columns from column maps by using index
        logger.info("Selected target columns: %s", target_column)

        # ---------- Select feature columns by index ----------
        feature_input = input_list("🕯️ Select FEATURE column index(es)")
        if feature_input == "__BACK__":
            logger.info("Select Feature menu exited at feature selection")
            return

        if feature_input is None:
            logger.info(
                "No feature columns entered; using all non-target columns automatically | target=%s",
                target_column,
            )
            print(
                "🔔 No feature columns entered. Using all non-target columns automatically."
            )
            xy_data = zeus.select_feature_target(
                target_column=target_column
            )  # Record to select_feature_target method (Zeus engine)

        else:
            if not all(str(item).isdigit() for item in feature_input):
                print("⚠️ FEATURE selections must all be numeric indices ‼️")
                continue

            feature_indices = [
                int(item) for item in feature_input
            ]  # Collect selected feature columns' index

            if any(idx not in col_map for idx in feature_indices):
                print("⚠️ One or more FEATURE indices are out of range ‼️")
                continue

            feature_columns = [
                col_map[idx] for idx in feature_indices
            ]  # Get feature columns from column maps by using index

            if any(col in target_column for col in feature_columns):
                print("⚠️ FEATURE columns cannot include TARGET columns ‼️")
                continue

            logger.info(
                "Selected feature columns | target=%s | features=%s",
                target_column,
                feature_columns,
            )

            # ---------- Convey to Zeus engine's method ----------
            xy_data = zeus.select_feature_target(
                target_column=target_column,
                feature_columns=feature_columns,
            )

        if xy_data is None:
            logger.warning(
                "Failed to build X and y | target=%s",
                target_column,
            )
            print("⚠️ Failed to build X and y ‼️")
            continue

        logger.info("Feature and target selection completed successfully")
        print("🔥 Feature and target selection completed.")
        print("👓 Show the selected features and targets.")
        zeus.show_current_feature_selection()  # Show out the selected feature and target columns

        # ---------- Confirm the selection ----------
        while True:
            confirm = input_int("🕯️ (1) Confirm selection | (2) Reselect | (0) Back")

            if confirm == 1:
                logger.info("Feature selection confirmed")
                print("🔥 Current selection confirmed.")
                return

            elif confirm == 2:
                logger.info("Feature selection reset requested")
                zeus.reset_feature_selection()
                print("♻️ Feature selection has been reset. Please select again.")
                break

            elif confirm == 0 or confirm is None:
                logger.info("Exited Select Feature menu at confirmation step")
                return

            else:
                print("⚠️ Invalid selection ‼️")


# =================================================
