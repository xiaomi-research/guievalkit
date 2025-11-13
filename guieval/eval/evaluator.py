import json
from typing import Literal
import numpy as np
import os
import Levenshtein

from guieval.utils.action_space import ActionType
from guieval.utils.action_utils import (get_direction, is_tap_action, obtain_gt_bbox, _get_direction,
                                        _resize_annotation_bounding_boxes)


EXTRACT_SCHEMA = json.load(
    open(os.path.join('guieval/utils/schema', 'schema_for_extraction.json'), encoding="utf-8"))

_TAP_DISTANCE_THRESHOLD = 0.14
_TAP_DISTANCE_THRESHOLD_AC = 0.04

ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.2
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.2

default_duration = EXTRACT_SCHEMA["properties"]["duration"]["default"]  # default 200


def process_step_data(step_data, evaluator, save_dir):

    subset = step_data.get('subset')
    episode_id = step_data.get('episode_id')
    step_id = step_data.get('step_id')

    if subset is None or episode_id is None or step_id is None:
        raise ValueError(f"Missing subset/episode_id/step_id in the test data: {step_data}")

    save_dir_ep = os.path.join(save_dir, f"{subset}-{episode_id}")
    cur_save_path = os.path.join(save_dir_ep, f"{subset}-{episode_id}_{step_id}.json")

    try:
        if not os.path.exists(cur_save_path):
            return None

        with open(cur_save_path, "r") as file:
            try:
                pred = json.load(file)
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed; File: {cur_save_path}; Error: {e}")
                pred = {
                    'action_predict': {
                        'COA': {
                            'txt': {
                                'ACTION': None,
                                'ARGS': None,
                                'STATUS': None
                            }
                        }
                    }
                }
        assert pred is not None

        result = evaluator(step_data, pred)
        return result

    except Exception:
        raise print(f"An error occurred, indicating unhandled edge case.")


class ActionEvaluator(object):

    def __init__(self, eval_android_control=False):
        self.demo_mode = "COA"
        self.screen_mode = "txt"
        self._aitz_action_type_ = ActionType
        self._stop_status = [
          "finish",
          "satisfied",
          "impossible",
          "interrupt",
          "need_feedback"
        ]
        self.eval_android_control = eval_android_control

    def action_map(self, action_api: dict):
        action = action_api.get('ACTION', None)
        args = action_api.get('ARGS', None)
        status = action_api.get('STATUS', None)
        duration = args.get('duration', default_duration) if args else None

        if action is None and args is None and status is None:
            print('Schema Error!')
            return None, {}

        elif status in self._stop_status:
            return "stop", {}

        elif "INPUT" in action:
            input_text = action['INPUT']
            return "input_text", {"text": input_text['text'], "point": input_text['point']}

        elif "TYPE" in action:
            return "type", action['TYPE']

        elif "OPEN_APP" in action:
            return "open", action['OPEN_APP']

        elif "POINT" in action and "to" not in args and duration == default_duration:  # click
            return "click", action['POINT']

        elif "POINT" in action and "to" in args and duration == default_duration:  # swipe
            return "scroll", {"start": action['POINT'], "end": args['to']}

        elif "POINT" in action and "duration" in args and duration > default_duration:  # long press
            return "long_point", {"coordinate": action['POINT'], "duration": args['duration']}

        elif "PRESS" in action:
            return "press", action['PRESS']

        elif "duration" in args:  # pause and wait
            return "wait", args['duration']

        else:
            raise ValueError("Unknown action type!")

    def get_uipositions(self, gt):

        if gt.get('ui_positions'):
            gt_cand_nodes = json.loads(gt.get('ui_positions'))
            return gt_cand_nodes

        elif gt.get('bbox'):
            gt_cand_nodes = json.loads(gt.get('bbox'))
            return gt_cand_nodes

        return []

    def _parse_action_(self, pred):

        pr = pred.get('action_predict', {})
        if self.demo_mode not in pr:
            return (None, ) * 7

        action = pr[self.demo_mode].get(self.screen_mode, {})
        if not action:
            return (None, ) * 7

        pd_action_type, pd_action_args = self.action_map(action)
        if pd_action_type is None:
            print('Unknown action: ', action)

        pd_action_direction = get_direction(
            pd_action_args["start"], pd_action_args["end"]) if pd_action_type == "scroll" else None

        pd_action_text = pd_action_args if pd_action_type == "type" or pd_action_type == "open" else None

        pd_action_button = pd_action_args.lower() if pd_action_type == "press" else None

        pd_duration = pd_action_args["duration"] if pd_action_type == "long_point" else None

        scale_x = 1000
        scale_y = 1000

        if pd_action_type == "click":
            try:
                pd_action_yx = {"x": pd_action_args[0] / scale_x, "y": pd_action_args[1] / scale_y}
            except Exception:
                pd_action_yx = {"x": 0.0, "y": 0.0}

        elif pd_action_type == "long_point":
            try:
                pd_action_yx = {"x": pd_action_args["coordinate"][0] / scale_x,
                                "y": pd_action_args["coordinate"][1] / scale_y}
            except Exception:
                pd_action_yx = {"x": 0.0, "y": 0.0}

        elif pd_action_type == "input_text":
            try:
                pd_action_text = pd_action_args['text']
                pd_action_yx = {"x": pd_action_args['point'][0] / scale_x, "y": pd_action_args['point'][1] / scale_y}
            except Exception:
                pd_action_text = None
                pd_action_yx = {"x": 0.0, "y": 0.0}

        else:
            pd_action_yx = None

        return pd_action_type, pd_action_yx, pd_action_text, pd_action_button, pd_action_direction, pd_duration

    def _parse_answer_(self, gt):
        gt_cand_nodes = None
        gt_action_text = None
        gt_action_yx = None
        gt_action_direction = None
        gt_action_button = None
        gt_duration = None  # This one is not used for evaluation?

        if gt['result_action_type'] == self._aitz_action_type_.TYPE:
            gt_action_type = "type"
            gt_action_text = gt['result_action_text']

        elif gt['result_action_type'] == self._aitz_action_type_.DUAL_POINT:
            normalized_start_yx = gt['result_touch_yx']
            normalized_start_yx = json.loads(normalized_start_yx)
            normalized_end_yx = gt['result_lift_yx']
            normalized_end_yx = json.loads(normalized_end_yx)

            if is_tap_action(normalized_start_yx, normalized_end_yx):
                gt_cand_nodes = self.get_uipositions(gt)
                gt_action_type = "click"
                gt_action_yx = {"y": normalized_start_yx[0], "x": normalized_start_yx[1]}
            else:
                point1 = {"y": normalized_start_yx[0], "x": normalized_start_yx[1]}
                point2 = {"y": normalized_end_yx[0], "x": normalized_end_yx[1]}
                gt_action_type = "scroll"
                gt_action_direction = _get_direction(point1, point2)

        elif gt['result_action_type'] == self._aitz_action_type_.LONG_POINT:
            normalized_start_yx = gt['result_touch_yx']
            normalized_start_yx = json.loads(normalized_start_yx)
            gt_cand_nodes = self.get_uipositions(gt)
            gt_action_type = "long_point"
            gt_action_yx = {"y": normalized_start_yx[0], "x": normalized_start_yx[1]}
            gt_duration = gt['duration']

        elif gt['result_action_type'] == self._aitz_action_type_.PRESS_BACK:
            gt_action_type = "press"
            gt_action_button = "back"

        elif gt['result_action_type'] == self._aitz_action_type_.PRESS_HOME:
            gt_action_type = "press"
            gt_action_button = "home"

        elif gt['result_action_type'] == self._aitz_action_type_.PRESS_ENTER:
            gt_action_type = "press"
            gt_action_button = "enter"

        elif (gt['result_action_type'] == self._aitz_action_type_.STATUS_TASK_COMPLETE or
              gt['result_action_type'] == self._aitz_action_type_.STATUS_TASK_IMPOSSIBLE):
            gt_action_type = "stop"
            gt_action_text = gt['result_action_text']

        elif gt['result_action_type'] == self._aitz_action_type_.NO_ACTION:
            gt_action_type = "wait"
            gt_duration = gt['duration']

        elif gt['result_action_type'] == self._aitz_action_type_.OPEN_APP:
            gt_action_type = "open"
            gt_action_text = gt['result_action_app_name']

        else:
            raise ValueError("Unknown action type.")

        return (gt_action_type, gt_action_yx, gt_cand_nodes,
                gt_action_text, gt_action_button, gt_action_direction, gt_duration)

    def __call__(self, gt, pred):

        pixel_distance = None

        subset, episode_id, step_id, _ = gt['subset'], gt['episode_id'], gt['step_id'], gt['instruction']

        (gt_action_type, gt_action_yx, gt_cand_nodes, gt_action_text, gt_action_button, gt_action_direction,
         gt_duration) = self._parse_answer_(gt)

        (pd_action_type, pd_action_yx, pd_action_text, pd_action_button, pd_action_direction,
         pd_duration) = self._parse_action_(pred)

        if pd_action_type == "input_text" and gt_action_type in ["click", "type"]:
            pd_action_type = gt_action_type

        # compute metrics
        hit_format = True if pd_action_type is not None else False
        type_match = (pd_action_type is not None and gt_action_type == pd_action_type)
        exact_match = False
        text_dist = None

        if type_match and (pd_action_type == "click" or pd_action_type == "long_point"):
            gt_cand_nodes = _resize_annotation_bounding_boxes(
                gt_cand_nodes, ANNOTATION_WIDTH_AUGMENT_FRACTION, ANNOTATION_HEIGHT_AUGMENT_FRACTION)
            gt_bbox = obtain_gt_bbox(gt_action_yx, gt_cand_nodes, self.eval_android_control)

            if gt_bbox == []:
                y_gt, x_gt = gt_action_yx["y"], gt_action_yx["x"]
                y_pd, x_pd = pd_action_yx["y"], pd_action_yx["x"]
                distance = np.linalg.norm(np.array([x_gt, y_gt]) - np.array([x_pd, y_pd]))
                exact_match = bool(
                    distance <= (_TAP_DISTANCE_THRESHOLD_AC if self.eval_android_control else _TAP_DISTANCE_THRESHOLD))
                reference_point = gt_action_yx["x"], gt_action_yx["y"]
            else:
                reference_point = gt_action_yx["x"], gt_action_yx["y"]
                for bbox in gt_bbox:
                    ymin, xmin, height, width = bbox
                    ymax, xmax = ymin + height, xmin + width
                    exact_match = ((ymin <= pd_action_yx["y"] <= ymax) and (xmin <= pd_action_yx["x"] <= xmax))
                    if exact_match:
                        reference_point = (xmax + xmin) / 2, (ymax + ymin) / 2
                        break
                if not exact_match:
                    y_gt, x_gt = gt_action_yx["y"], gt_action_yx["x"]
                    y_pd, x_pd = pd_action_yx["y"], pd_action_yx["x"]
                    distance = np.linalg.norm(np.array([x_gt, y_gt]) - np.array([x_pd, y_pd]))
                    exact_match = bool(distance <= (
                        _TAP_DISTANCE_THRESHOLD_AC if self.eval_android_control else _TAP_DISTANCE_THRESHOLD))

            pixel_distance = np.linalg.norm(
                np.array([pd_action_yx["x"], pd_action_yx["y"]]) * 1000 - np.array(reference_point) * 1000)

        if type_match and pd_action_type == "scroll":
            exact_match = (pd_action_direction == gt_action_direction)

        if type_match and (pd_action_type == "type" or pd_action_type == "open"):
            pd_text_norm = pd_action_text.lower().strip()
            gt_text_norm = gt_action_text.lower().strip()
            text_dist = Levenshtein.ratio(pd_text_norm, gt_text_norm)
            exact_match = (pd_text_norm in gt_text_norm or gt_text_norm in pd_text_norm)

        if type_match and pd_action_type == "press":
            exact_match = (pd_action_button == gt_action_button)

        if type_match and pd_action_type in ["stop", "wait"]:
            exact_match = True

        gt_action_detail = {
            "click": gt_action_yx,
            "scroll": gt_action_direction,
            "type": gt_action_text,
            "press": gt_action_button,
            "long_point": gt_action_yx,
            "stop": "stop",
            "open": gt_action_text
        }.get(gt_action_type, None)

        pd_action_detail = {
            "click": pd_action_yx,
            "scroll": pd_action_direction,
            "type": pd_action_text,
            "press": pd_action_button,
            "long_point": pd_action_yx,
            "stop": "stop",
            "open": pd_action_text
        }.get(pd_action_type, None)

        return {
            "subset": subset,
            "episode_id": episode_id,
            "step_id": step_id,
            "answer": {
                "action_type": gt_action_type,
                "action_detail": gt_action_detail
            },
            "pred": {
                "action_type": pd_action_type,
                "action_detail": pd_action_detail
            },
            "type_match": type_match,
            "exact_match": exact_match,
            "text_dist": text_dist,
            "format_hit": hit_format,
            "pixel_distance": pixel_distance,
        }


EVALUATOR_NAMES = Literal['androidcontrol', 'common']
EVALUATORS: dict[EVALUATOR_NAMES, ActionEvaluator] = dict(androidcontrol=ActionEvaluator(eval_android_control=True),
                                                          common=ActionEvaluator(eval_android_control=False))
