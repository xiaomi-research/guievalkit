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
    LEVEL_COLORS: dict[str, str] = {
        'DEBUG': ANSI_CODE.GREEN_BG.value,
        'INFO': ANSI_CODE.BLUE_BG.value,
        'WARNING': ANSI_CODE.YELLOW_BG.value,
        'ERROR': ANSI_CODE.MAGENTA_BG.value,
        'CRITICAL': ANSI_CODE.RED_BG.value
    }

    def __init__(self,
                 name: str,
                 ansi_style: bool = True,
                 concise_time: bool = False) -> None:
        super().__init__()
        self.project_name = name
        self.ansi_style = ansi_style
        self.concise_time = concise_time

    def format(self, record: logging.LogRecord) -> str:
        # Get timestamp
        ct = self.converter(record.created)
        if self.concise_time:
            t = time.strftime("%H:%M:%S", ct)
            timestamp = f"{t}.{int(record.msecs):03d}"
        else:
            t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
            timestamp = f"{t}.{int(record.msecs):03d}"

        # Get level string with optional ANSI coloring
        level_name = record.levelname
        if self.ansi_style:
            color = self.LEVEL_COLORS.get(level_name, ANSI_CODE.RESET.value)
            level_str = f"{color} {level_name:<8} {ANSI_CODE.RESET.value}"
            name_str = f"{ANSI_CODE.CYAN.value}{self.project_name}{ANSI_CODE.RESET.value}"
            msg_str = f"{ANSI_CODE.WHITE.value if level_name == 'INFO' else ''}{record.getMessage()}{ANSI_CODE.RESET.value}"
        else:
            level_str = f"[{level_name:<8}]"
            name_str = self.project_name
            msg_str = record.getMessage()

        # Format final output: [Timestamp] [PROJECT] [LEVEL] Message
        return f"[{timestamp}] [{name_str}] {level_str} {msg_str}"


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
