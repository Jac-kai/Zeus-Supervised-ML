"""
Logging initialization utilities for the Zeus project.

This module provides the logging bootstrap function used by Zeus to create and
configure a project-level logger. The logging setup includes both a file
handler and a console stream handler so that runtime messages can be recorded
persistently while also being displayed in the terminal during execution.

The log file is stored under the ``Zeus_Logs`` folder located in the same
directory as this module. If the log folder does not already exist, it is
created automatically.

Main Features
-------------
- create a dedicated ``Zeus`` logger
- write logs to a persistent project log file
- print logs to the console during runtime
- avoid duplicate handlers when logging is initialized multiple times
- use a unified timestamped log format for file and console output

Notes
-----
This module is intended to be imported and called during program startup so
that all Zeus components can share the same logger instance.
"""

import logging

# -------------------- Imported Modules -------------------
import os


# -------------------- Logging Setup --------------------
def zeus_init_logging() -> logging.Logger:
    """
    Initialize and return the project-level Zeus logger.

    This function creates and configures a dedicated logger named ``"Zeus"`` for
    the Zeus project. It ensures that a log directory named ``Zeus_Logs`` exists
    in the same directory as this module, creates a log file named
    ``Zeus_Log.log`` inside that folder, and attaches both file and console
    handlers to the logger.

    The logger uses a unified format containing timestamp, log level, logger name,
    and message text. To prevent duplicate log records, the function checks
    whether equivalent handlers have already been attached before adding new ones.

    Parameters
    ----------
    None

    Returns
    -------
    logging.Logger
        The configured Zeus logger instance.

    Workflow
    --------
    1. Resolve the current module directory as the project logging root
    2. Build the ``Zeus_Logs`` folder path
    3. Create the folder if it does not already exist
    4. Build the full log file path
    5. Create or retrieve the logger named ``"Zeus"``
    6. Set logger level to ``logging.INFO``
    7. Disable propagation to parent loggers
    8. Create a shared log formatter
    9. Add a file handler if the target log file handler is not already attached
    10. Add a console stream handler if a stream handler is not already attached
    11. Write an initialization message to the log
    12. Return the configured logger instance

    Notes
    -----
    The file handler uses ``utf-8-sig`` encoding so that log files can be opened
    more reliably in editors that expect UTF-8 with BOM.

    This function is designed to be safe for repeated calls during the same
    program session because it avoids reattaching duplicate handlers.

    Examples
    --------
    Initialize the Zeus logger during application startup::

        logger = zeus_init_logging()
        logger.info("Zeus system started.")

    The returned logger can then be reused throughout the project with::

        import logging
        logger = logging.getLogger("Zeus")
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    log_folder = os.path.join(project_root, "Zeus_Logs")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "Zeus_Log.log")

    logger = logging.getLogger("Zeus")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "") == log_file
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_file, encoding="utf-8-sig")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.info("Logging initialized to: %s", log_file)
    return logger


# -------------------- Execute --------------------
if __name__ == "__main__":
    logger = zeus_init_logging()


# --------------------------------------------------------
