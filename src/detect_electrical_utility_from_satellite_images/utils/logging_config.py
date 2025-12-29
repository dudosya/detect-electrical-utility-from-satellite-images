import datetime
import logging
from pathlib import Path


def setup_logger(
    log_dir: str | Path,
    logger_lvl: str,
    console_handler_lvl: str,
    file_handler_lvl: str,
):
    """sets up the logger

    Args:
        log_dir (str | Path): absolute path to the log directory. it will get normalized using idempotent Path() constructor
        logger_lvl (str): logger level
        console_handler_lvl (str): console handler level
        file_handler_lvl (str): file handler level
    """

    # get logger
    logger = logging.getLogger()

    # clear the previous loggers i guess
    if logger.hasHandlers():
        logger.handlers.clear()

    # set logger lvl
    logger.setLevel(logger_lvl.upper())

    # define formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # define filename
    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # idempotent constructor to converge str and Path classes to Path class
    log_dir = Path(log_dir)

    # create the folder
    log_dir.mkdir(parents=True, exist_ok=True)

    # define log file path
    LOG_FP = log_dir / f"{filename}.log"

    # define handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(LOG_FP)

    # set formatter to the handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # set different lvls to the handlers
    console_handler.setLevel(console_handler_lvl.upper())
    file_handler.setLevel(file_handler_lvl.upper())

    # add  handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


if __name__ == "__main__":
    # TODO: improve this
    # hardcoded arguments
    LOG_DIR_NAME = "testing"
    LOG_DIR = Path.cwd() / "logs" / LOG_DIR_NAME
    LOGGER_LVL = "debug"
    CONSOLE_HANDLER_LVL = "warning"
    FILE_HANDLER_LVL = "debug"

    # call the func
    setup_logger(LOG_DIR, LOGGER_LVL, CONSOLE_HANDLER_LVL, FILE_HANDLER_LVL)

    # get the logger
    logger = logging.getLogger(__name__)

    # use the  logger
    logger.debug("lvl 1 whaat")
    logger.warning("there is a warning dude")
