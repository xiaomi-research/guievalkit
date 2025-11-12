import os
import re
import json
import logging
from collections import defaultdict, OrderedDict
from typing import Any, Literal

# subsec internal
from guieval.config import EvalTaskConfig
from guieval.utils import UNIFIED_ACTION, UNIFIED_ACTIONS

# todo: unify from samples and from saved

logger = logging.getLogger(__name__)


def _compute_samples(samples: list[dict], total_count: int):
    task_count = len(samples)
    exact_match = sum(sample['evaluation']['exact_match'] for sample in samples)
    type_match = sum(sample['evaluation']['type_match'] for sample in samples)

    trajectories: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        trajectories[sample['episode_id']].append(sample)

    prog = 0
    for _, trajectory in trajectories.items():
        trajectory.sort(key=lambda item: item["step_id"])
        for i, step in enumerate(trajectory, start=1):
            if (i - 1) != step['step_id']:
                break
            if not step['evaluation']['exact_match']:
                prog += (i - 1)
                break
        else:
            prog += i

    return dict(type_match=(type_match / task_count),
                exact_match=(exact_match / task_count),
                progress=(prog / task_count),
                ratio=(task_count / total_count))


PRIORITY = float


def _order_map(item: tuple[str, Any]) -> PRIORITY:
    if item[0] == "unified_action_ratio":
        return -float('inf')
    elif item[0] == "online_src_ratio":
        return -1000
    elif item[0] == "IN_UNIFIED":
        return -100
    elif item[0] == "TARGET_NOT_OPEN":
        return -10
    else:
        try:
            return -item[1]["ratio"]
        except Exception:
            return float('inf')


def compute_saved_results(setup: EvalTaskConfig, *,
                          flush: bool = False,
                          write: bool = True) -> str | None:
    pred_output_dir = setup.predictions_output_dir
    prediction_files, datasets = list(), list()
    for _prediction_file in os.listdir(pred_output_dir):
        try:
            dataset = re.search(r'^(.*).jsonl', _prediction_file).group(1)
            prediction_files.append(_prediction_file)
            datasets.append(dataset)
        except AttributeError:
            pass
    else:
        if not dataset:
            raise ValueError('No valid file matching pattern `*.jsonl` found under:\n'
                             f'\t{pred_output_dir}')
    try:
        with open(setup.evaluation_file, 'r', encoding='utf-8') as f:
            results: dict[str, dict] = json.load(f)
            results = defaultdict(results)
    except Exception:
        results = defaultdict(dict)

    for prediction_file, dataset in zip(prediction_files, datasets):
        with open(os.path.join(pred_output_dir, prediction_file), 'r', encoding='utf-8') as f:
            samples: list[dict] = list(map(json.loads, f.readlines()))
        original_task_count = len(samples)

        # filter out samples not in unified action space <save>
        categorized_samples: dict[UNIFIED_ACTION | Literal['IN_UNIFIED', 'TARGET_NOT_OPEN'],
                                  list] = defaultdict(list)
        for sample in samples:
            target_action = sample.get('target_action')
            pred_action = sample.get('pred_action')
            if pred_action in UNIFIED_ACTIONS:
                categorized_samples['IN_UNIFIED'].append(sample)
                if sample['target_action'] != 'OPEN':
                    categorized_samples['TARGET_NOT_OPEN'].append(sample)
                categorized_samples[target_action].append(sample)

        if not categorized_samples['IN_UNIFIED']:
            logger.info(f'No computable samples found in results of Dataset {dataset} '
                        f'under `{pred_output_dir}`.\n'
                        f'Gonna ignore it. For more details, check the results at:\n'
                        f'\t{os.path.join(pred_output_dir, prediction_file)}')
            continue

        if not categorized_samples['TARGET_NOT_OPEN']:
            logger.info(f'No `target_not_open` samples found in results of Dataset {dataset} '
                        f'under `{pred_output_dir}`.\n'
                        f'Gonna ignore it. For more details, check the results at:\n'
                        f'\t{os.path.join(pred_output_dir, prediction_file)}')
            continue

        online_src_count, src_count = zip(*((sample['history_content_srcs'].count('online'),
                                             len(sample['history_content_srcs']))
                                            for sample in categorized_samples['IN_UNIFIED']))

        results[dataset]['online_src_ratio'] = sum(online_src_count) / sum(src_count)
        sample_in_unified_count = len(categorized_samples['IN_UNIFIED'])
        results[dataset]['unified_action_ratio'] = sample_in_unified_count / original_task_count
        results[dataset].update((_category, _compute_samples(_samples, total_count=sample_in_unified_count))
                                for _category, _samples in categorized_samples.items())
        # IN_UNIFIED and TARGET_NO_OPEN shall have highest priority
        # actions would be ordered in task ratio
        results[dataset] = OrderedDict(sorted(results[dataset].items(), key=_order_map))

    evaluation = json.dumps(results, indent=4, ensure_ascii=False)
    if flush:
        logger.info('Final evaluation:\n'
                    f'{evaluation}')

    if write:
        with open(setup.evaluation_file, 'w', encoding='utf-8') as f:
            f.write(evaluation)
        logger.info(f'Final evaluation saved at: {setup.evaluation_file}')
        return setup.evaluation_file
    else:
        return None
