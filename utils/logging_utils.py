import logging
import jinja2
import time

from typing import Literal
from enum import Enum

try:
    # Enable ANSI colors on Windows terminals
    import colorama  # type: ignore
    colorama.init(autoreset=True)
except Exception:  # pragma: no cover - optional dependency
    colorama = None  # type: ignore


# internal
from utils.utils import UTIL_BASE


LOG_TEMPLATE_FILE = (UTIL_BASE / 'log_ansi_template.j2')
LOGGING_LEVEL = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


class ANSI_CODE(Enum):
    UNDERLINE = "\x1b[4m"

    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"

    RED_BG = "\x1b[41m"
    GREEN_BG = "\x1b[42m"
    BLUE_BG = "\x1b[44m"
    YELLOW_BG = "\x1b[43m"
    MAGENTA_BG = "\x1b[45m"

    RESET = "\x1b[0m"


class _Level_Color_Dict(dict):
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return ANSI_CODE.RESET


class ProjectFormatter(logging.Formatter):
    LEVEL_COLORS: dict[LOGGING_LEVEL, ANSI_CODE] = _Level_Color_Dict({
        'DEBUG': ANSI_CODE.GREEN_BG,
        'INFO': ANSI_CODE.BLUE_BG,
        'WARNING': ANSI_CODE.YELLOW_BG,
        'ERROR': ANSI_CODE.MAGENTA_BG,
        'CRITICAL': ANSI_CODE.RED_BG
    })
    LOG_TEMPLATE: jinja2.Template = jinja2.Template(LOG_TEMPLATE_FILE.read_text())

    def __init__(self,
                 name: str,
                 ansi_style: bool = True,
                 concise_time: bool = False) -> None:
        super().__init__()
        self.name = name
        self.ansi_style = ansi_style
        self.concise_time = concise_time

    def format(self, record: logging.LogRecord) -> str:
        now = time.time_ns() * 1e-9
        structed_time = time.localtime(now)
        if self.concise_time:
            mini_time_res = f'{now:.03f}'.split('.')[-1]
            timestamp = (f'{structed_time.tm_hour:02d}:{structed_time.tm_min:02d}:'
                         f'{structed_time.tm_sec:02d}.{mini_time_res}')
        else:
            nano_time_res = f'{now:.06f}'.split('.')[-1]
            timestamp = (f'{structed_time.tm_year}-{structed_time.tm_mon:02d}-{structed_time.tm_mday:02d} '
                         f'{structed_time.tm_hour:02d}:{structed_time.tm_min:02d}:'
                         f'{structed_time.tm_sec:02d}.{nano_time_res}')

        return self.LOG_TEMPLATE.render(
            name=self.name,
            ansi_style=self.ansi_style,
            selected_ansi=ANSI_CODE,
            level_colors=self.LEVEL_COLORS,
            record=record,
            timestamp=timestamp
        )


def init_logging(root_name: str,
                 level: LOGGING_LEVEL, *,
                 log_file: str,
                 always_ansi: bool = True) -> logging.Logger:
    console_formatter = ProjectFormatter(name=root_name, ansi_style=True, concise_time=True)
    file_formatter = (ProjectFormatter(name=root_name, ansi_style=True, concise_time=False)
                      if always_ansi else
                      ProjectFormatter(name=root_name, ansi_style=False, concise_time=False))

    file_handler = logging.FileHandler(filename=log_file)
    console_handler = logging.StreamHandler()

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler]
    )

    # Filter out OpenAI client INFO logs (e.g., retry messages)
    logging.getLogger('openai').setLevel(logging.WARNING)
    # Filter out httpx INFO logs (e.g., HTTP request messages)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    return


def get_logger(root_name: str,
               name: str,
               level: LOGGING_LEVEL, *,
               log_file: str | None = None,
               always_ansi: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    logger.propagate = False

    console_formatter = ProjectFormatter(name=root_name, ansi_style=True, concise_time=True)
    file_formatter = (ProjectFormatter(name=root_name, ansi_style=True, concise_time=False)
                      if always_ansi else
                      ProjectFormatter(name=root_name, ansi_style=False, concise_time=False))

    file_handler = logging.FileHandler(filename=log_file)
    console_handler = logging.StreamHandler()

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Filter out OpenAI client INFO logs (e.g., retry messages)
    logging.getLogger('openai').setLevel(logging.WARNING)
    # Filter out httpx INFO logs (e.g., HTTP request messages)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    return logger


def get_existing_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name. Raises ValueError if the logger doesn't exist.

    Args:
        name: The name of the logger to retrieve

    Returns:
        The existing logger instance

    Raises:
        ValueError: If no logger with the given name exists
    """
    # Check if logger exists in the manager's loggerDict
    if name not in logging.Logger.manager.loggerDict:
        raise ValueError(f"Logger '{name}' has not been instantiated. "
                        f"Please create it first using get_logger() or logging.getLogger() before using it.")

    # Get the logger object from the manager's dict directly to avoid creating a new one
    logger_obj = logging.Logger.manager.loggerDict[name]

    # Verify it's actually a Logger instance (not a PlaceHolder)
    if not isinstance(logger_obj, logging.Logger):
        raise ValueError(f"Logger '{name}' exists but is not a proper Logger instance. "
                        f"Please create it first using get_logger() or logging.getLogger() before using it.")

    return logger_obj
