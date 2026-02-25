import numpy as np
import Levenshtein

# subsec internal
from guieval.utils.action_space import PREDICTION, PREDICTIONS, UNIFIED_ACTION


# section struc
DISTANCE = float


# section main
class ActionContinuousMetric:
    @staticmethod
    def click(click1: PREDICTIONS.CLICK, click2: PREDICTIONS.CLICK) -> DISTANCE:
        vec = [click1["POINT"][0] - click2["POINT"][0],
               click1["POINT"][1] - click2["POINT"][1]]
        return float(np.linalg.norm(vec))

    @staticmethod
    def type(type1: PREDICTIONS.TYPE, type2: PREDICTIONS.TYPE) -> DISTANCE:
        return Levenshtein.ratio(
            type1["TYPE"],
            type2["TYPE"]
        )

    @staticmethod
    def scroll(scroll1: PREDICTIONS.SCROLL, scroll2: PREDICTIONS.SCROLL) -> DISTANCE:
        direction_map = {
            'up': 0.0,
            'right': 90.0,
            'down': 180.0,
            'left': 270.0
        }
        dist = abs(direction_map[scroll1["to"]] - direction_map[scroll2["to"]])
        return min(dist, (360.0 - dist))

    @staticmethod
    def press(press1: PREDICTIONS.PRESS, press2: PREDICTIONS.PRESS) -> DISTANCE:
        button1 = press1["PRESS"] if isinstance(press1["PRESS"], str) else ''
        button2 = press2["PRESS"] if isinstance(press2["PRESS"], str) else ''
        return Levenshtein.ratio(button1, button2)

    @staticmethod
    def stop(stop1: PREDICTIONS.STOP, stop2: PREDICTIONS.STOP) -> DISTANCE:
        return Levenshtein.ratio(
            stop1["STATUS"],
            stop2["STATUS"]
        )

    @staticmethod
    def long_point(long_point1: PREDICTIONS.LONG_POINT, long_point2: PREDICTIONS.LONG_POINT) -> DISTANCE:
        return np.linalg.norm([long_point1["POINT"][0] - long_point2["POINT"][0],
                               long_point1["POINT"][1] - long_point2["POINT"][1]])

    @staticmethod
    def open(open1: PREDICTIONS.OPEN, open2: PREDICTIONS.OPEN) -> DISTANCE:
        return Levenshtein.ratio(
            open1["OPEN_APP"],
            open2["OPEN_APP"]
        )

    @staticmethod
    def wait(wait1: PREDICTIONS.WAIT, wait2: PREDICTIONS.WAIT) -> DISTANCE:
        return abs(wait1["duration"] - wait2["duration"])


class ActionMetric(ActionContinuousMetric):
    @staticmethod
    def scroll(scroll1: PREDICTIONS.SCROLL, scroll2: PREDICTIONS.SCROLL) -> DISTANCE:
        return 0 if scroll1.get('to', 'null1') == scroll2.get('to', 'null2') else float('inf')

    @staticmethod
    def press(press1: PREDICTIONS.PRESS, press2: PREDICTIONS.PRESS) -> DISTANCE:
        button1 = press1["PRESS"] if isinstance(press1["PRESS"], str) else ''
        button2 = press2["PRESS"] if isinstance(press2["PRESS"], str) else ''
        return 0 if button1 == button2 else float('inf')

    @staticmethod
    def stop(stop1: PREDICTIONS.STOP, stop2: PREDICTIONS.STOP) -> DISTANCE:
        return 0 if stop1["STATUS"] == stop2["STATUS"] else float('inf')


class ActionOperators:
    @staticmethod
    def eq(action_type: UNIFIED_ACTION, action1: PREDICTION, action2: PREDICTION) -> bool:
        if action_type == 'CLICK':
            return ActionMetric.click(action1, action2) <= 140
        elif action_type == 'LONG_POINT':
            return ActionMetric.long_point(action1, action2) <= 140
        elif action_type == 'OPEN':
            return not ActionMetric.open(action1, action2)
        elif action_type == 'PRESS':
            return not ActionMetric.press(action1, action2)
        elif action_type == 'SCROLL':
            return not ActionMetric.scroll(action1, action2)
        elif action_type == 'STOP':
            return not ActionMetric.stop(action1, action2)
        elif action_type == 'TYPE':
            return ActionMetric.type(action1, action2) <= 0.1
        elif action_type == 'WAIT':
            return True
        else:
            raise ValueError()
