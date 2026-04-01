# -------------------- Modules Import --------------------
import logging
import time

from Zeus.Menu_Helper_Decorator import input_int
from Zeus.Zeus_Logging import zeus_init_logging
from Zeus.Zeus_Menu1 import loaded_ml_data_menu, select_feature_target_menu
from Zeus.Zeus_Menu2 import model_management_menu
from Zeus.Zeus_Menu3 import evaluation_menu
from Zeus.Zeus_ML_Engine import ZeusEngine

logger = logging.getLogger("Zeus")


# -------------------- cornus_control --------------------
def zeus_control():
    """
    Run the main terminal control loop for the Zeus system.

    This function creates a single ``ZeusEngine`` instance and launches the
    top-level interactive menu for the Zeus workflow. Through this menu, the user
    can:

    1. load a dataset,
    2. select feature and target columns,
    3. manage model training and model I/O,
    4. open model evaluation tools.

    The function keeps the same engine instance alive across menu selections so
    that loaded data, prepared features, trained models, and evaluation results
    can be reused throughout the session.

    Returns
    -------
    None
        This function runs an interactive terminal loop and does not return a
        value.

    Workflow
    --------
    1. Create a ``ZeusEngine`` instance.
    2. Build the main menu configuration.
    3. Display the main menu repeatedly until the user exits.
    4. Dispatch the selected menu action using the shared engine instance.
    5. Print a goodbye message when the user leaves the system.

    Notes
    -----
    A single shared ``ZeusEngine`` object is used for the full session so that
    state is preserved between submenu operations.
    """
    logger.info("Starting Zeus main control loop")

    zeus_engine = ZeusEngine()
    logger.info("ZeusEngine instance created")

    menu = [
        (1, "📨 Upload Data", loaded_ml_data_menu),
        (2, "🔎 Features and Targets", select_feature_target_menu),
        (3, "🧠 Models", model_management_menu),
        (4, "🪶 Evaluations", evaluation_menu),
        (0, "🍂 Leave System", None),
    ]
    menu_width = 35

    while True:
        logger.info("Displaying Zeus main menu")
        print("🏮  Zeus Main Menu 🏮 ".center(menu_width, "━"))

        for opt, action, _ in menu:
            print(f"{opt}. {action:<{menu_width-6}}")
        print("━" * menu_width)

        choice = input_int(f"🕯️  Select Services (🔅 {time.asctime()})⚡ ", default=-1)

        if choice is None:
            logger.info("Main menu exited by user cancel/back")
            print("🎶🎶🎶 Leaving Zeus Engine... Goodbye 🍁 Zack King")
            break

        if choice == -1:
            logger.info("Displaying Zeus main menu")
            print("⚠️ Invalid selection ‼️")
            continue

        logger.info("Main menu selection: %s", choice)

        for opt, label, func in menu:
            if choice == opt and func:
                logger.info("Dispatching main menu action: %s - %s", opt, label)
                func(zeus_engine)
                break
            if choice == 0 and opt == 0:
                logger.info("User selected Leave System")
                print("🎶🎶🎶 Leaving Zeus Engine... Goodbye 🍁 Zack King")
                return
        else:
            logger.warning("Main menu selection out of range: %s", choice)
            print("⚠️ Invalid selection ‼️")


# -------------------- Execute --------------------
if __name__ == "__main__":
    logger = zeus_init_logging()
    logger.info("Zeus logging initialized from main entry")
    zeus_control()


# -----------------------------------------
