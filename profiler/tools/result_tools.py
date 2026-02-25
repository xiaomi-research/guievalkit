import os
import json
import re
import logging

from typing import overload
from typing_extensions import TypedDict

# subsec internal
from guieval import EvalTaskConfig
from guieval.main import StepTaskModel
from guieval.main.data import LOADED_STEP_TASKS
from profiler.main.step_task_result import StepTaskResult


# section struct
logger = logging.getLogger(__name__)


class SymmetricDifference(TypedDict):
    result_difference: list[StepTaskResult]
    task_difference: list[StepTaskModel]


# section tools
@overload  # noqa: E302
def load_results(prediction_output_dir: str | None = None) -> list[StepTaskResult]:
    ...
@overload  # noqa: E302
def load_results(setup: EvalTaskConfig | None = None) -> list[StepTaskResult]:
    ...
def load_results(prediction_output_dir: str | None = None,  # noqa: E302
                 setup: EvalTaskConfig | None = None) -> list[StepTaskResult]:
    if prediction_output_dir is None and setup is None:
        raise ValueError("Either prediction_output_dir or setup must be provided.")
    elif prediction_output_dir is not None and setup is not None:
        raise ValueError("Only one of prediction_output_dir or setup must be provided.")

    if prediction_output_dir is not None:
        prediction_output_paths = [os.path.join(prediction_output_dir, _path)
                                   for _path in os.listdir(prediction_output_dir)
                                   if re.match(r'^.*\.jsonl$', _path)]
    else:
        prediction_output_paths = setup.prediction_output_paths

    if not prediction_output_paths:
        logger.warning("No valid prediction output paths found in "
                       f"{prediction_output_dir if prediction_output_dir is not None else setup.prediction_output_dir}")
        return list()

    results: list[StepTaskResult] = list()
    for prediction_path in prediction_output_paths:
        try:
            with open(prediction_path, 'r', encoding='utf-8') as f:
                results.extend(StepTaskResult.model_validate(result)
                               for result in map(json.loads, f.readlines()))
        except Exception as err:
            logger.error(f"Failed to load {prediction_path}: {err}")

    if not results:
        logger.warning(f"No valid results found in {setup.prediction_output_dir}")

    return results


def result_task_difference(loaded_results: list[StepTaskResult],
                           loaded_tasks: list[StepTaskModel], *,
                           meta: bool = False) -> SymmetricDifference:
    result_dict = dict((result.hash_encode(meta=meta), result) for result in loaded_results)
    task_dict = dict((step_task.hash_encode(meta=meta), step_task) for step_task in loaded_tasks)

    hash_code_intersection = result_dict.keys() & task_dict.keys()
    result_difference = result_dict.keys() - hash_code_intersection
    task_difference = task_dict.keys() - hash_code_intersection

    return {
        "result_difference": [result_dict[_hash_code] for _hash_code in result_difference],
        "task_difference": [task_dict[_hash_code] for _hash_code in task_difference]
    }


def dataset_difference_result_update(loaded_dataset: LOADED_STEP_TASKS,
                                     loaded_results: list[StepTaskResult]):
    result_hash_codes = set(result.hash_encode() for result in loaded_results)

    filtered_grouped_step_tasks = list()
    for _step_tasks in loaded_dataset["step_tasks"]:
        task_dict = dict((_step_task.hash_encode(), _step_task) for _step_task in _step_tasks)
        task_difference = task_dict.keys() - result_hash_codes
        filtered_grouped_step_tasks.append(tuple(task_dict[_hash_code] for _hash_code in task_difference))

    loaded_dataset["step_tasks"] = filtered_grouped_step_tasks


__all__ = [
    'load_results',
    'step_task_result_dataset_difference',
]
