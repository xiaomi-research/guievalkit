import enum

from typing import Literal, TypedDict
from typing_extensions import NotRequired

UNIFIED_ACTION = Literal['CLICK', 'TYPE', 'SCROLL', 'PRESS', 'STOP', 'LONG_POINT', 'OPEN', 'WAIT']
UNIFIED_ACTIONS: set[UNIFIED_ACTION] = {'CLICK', 'TYPE', 'SCROLL', 'PRESS', 'STOP', 'LONG_POINT', 'OPEN', 'WAIT'}

UNIFIED_FIELD = Literal["POINT", "to", "PRESS", "TYPE", "OPEN_APP", "INPUT", "duration"]
UNIFIED_FIELDS: set[UNIFIED_FIELD] = {"POINT", "to", "PRESS", "TYPE", "OPEN_APP", "INPUT", "duration"}

POINT = tuple[int | float, int | float]
DIRECTION = Literal["down", "up", "right", "left"]
DIRECTION_CHOICES: set[DIRECTION] = {"down", "up", "right", "left"}
DURATION = int | float
BUTTON = Literal["BACK", "HOME", "ENTER"]


class PREDICTION(TypedDict):
    POINT: NotRequired[POINT]
    to: NotRequired[DIRECTION]
    duration: NotRequired[DURATION]
    PRESS: NotRequired[BUTTON]
    TYPE: NotRequired[str]
    OPEN_APP: NotRequired[str]
    INPUT: NotRequired[str]


class ActionType(enum.IntEnum):

    UNUSED_2 = 2
    UNUSED_8 = 8
    UNUSED_9 = 9

    LONG_POINT = 0  # long point
    NO_ACTION = 1  # no action

    # A type action that sends text to the emulator. Note that this simply sends text and does not perform any clicks
    # for element focus or enter presses for submitting text.
    TYPE = 3

    # The dual point action used to represent all gestures.
    DUAL_POINT = 4

    # These actions differentiate pressing the home and back button from touches. They represent explicit presses of
    # back and home performed using ADB.
    PRESS_BACK = 5
    PRESS_HOME = 6

    # An action representing that ADB command for hitting enter was performed.
    PRESS_ENTER = 7

    # An action used to indicate the desired task has been completed and resets the environment. This action should
    # also be used in the case that the task has already been completed and there is nothing to do.  e.g. The task is
    # to turn on the Wi-Fi when it is already on.
    STATUS_TASK_COMPLETE = 10

    # An action used to indicate that desired task is impossible to complete and resets the environment. This can be a
    # result of many different things including UI changes, Android version differences, etc.
    STATUS_TASK_IMPOSSIBLE = 11

    # The action used to open app.
    OPEN_APP = 12

    @classmethod
    def action_map(cls, arguments: dict) -> UNIFIED_ACTION | None:
        if "STATUS" in arguments:
            return None

        elif "TYPE" in arguments:
            return "TYPE"

        elif "OPEN_APP" in arguments:
            return "OPEN"

        elif "POINT" in arguments:
            if "to" in arguments and arguments.get("duration", 0) <= 200:
                return "SCROLL"
            elif "duration" in arguments and arguments.get("duration", 1000) > 200:
                return "LONG_POINT"
            else:
                return "CLICK"

        elif "PRESS" in arguments:
            return "PRESS"

        elif "duration" in arguments:
            return "WAIT"

        else:
            raise ValueError("Unknown action type!")
