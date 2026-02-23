import itertools
import os
import re
import json
import logging
import time
import jinja2
from collections import defaultdict, OrderedDict, Counter
from typing import Any, Literal, NotRequired, overload
from typing_extensions import TypedDict, NotRequired
from numbers import Number

# subsec internal
from guieval.main.config import EvalTaskConfig
from guieval.utils import UNIFIED_ACTION, UNIFIED_ACTIONS


logger = logging.getLogger(__name__)


# section struc
class Evaluation(TypedDict):
    type_match: float
    exact_match: float
    progress: float
    ratio: float


class FinalEvaluationResult(TypedDict):
    dataset: str
    timestamp: str

    unified_action_ratio: float
    history_src_distribution: float

    IN_UNIFIED: Evaluation
    TARGET_NOT_OPEN: Evaluation

    CLICK: NotRequired[Evaluation]
    SCROLL: NotRequired[Evaluation]
    TYPE: NotRequired[Evaluation]
    STOP: NotRequired[Evaluation]
    PRESS: NotRequired[Evaluation]
    LONG_POINT: NotRequired[Evaluation]
    WAIT: NotRequired[Evaluation]
    OPEN: NotRequired[Evaluation]


PRIORITY = Number


def _priority_map(item: tuple[str, Evaluation | Any]) -> PRIORITY:
    # IN_UNIFIED and TARGET_NO_OPEN shall have high priority
    priority_map = {
        "dataset": -float('inf'),
        "timestamp": -2000,
        "unified_action_ratio": -1000,
        "history_src_distribution": -500,
        "IN_UNIFIED": -200,
        "TARGET_NOT_OPEN": -100,
    }
    try:
        return priority_map[item[0]]
    except KeyError:
        try:
            # actions would be ordered in task ratio
            return -item[1]["ratio"]
        except Exception:
            return float('inf')

# section main
def _compute_samples(samples: list[dict], total_count: int):  # noqa: E302
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


_MSG_TEMPLATE1: jinja2.Template = jinja2.Template(
'''No computable samples found in results of Dataset `{{dataset}}`
{%- if pred_output_dir is none %} in predictions
{%- else %} under `{{pred_output_dir}}`
{%- endif -%}.
Gonna ignore it.
{%- if pred_output_dir is not none %} For more details, check the results at:
{{pred_output_file}}
{%- endif -%}''')
_MSG_TEMPLATE2: jinja2.Template = jinja2.Template(
'''No `target_not_open` samples found in results of Dataset `{{dataset}}`
{%- if pred_output_dir is none %} in predictions
{%- else %} under `{{pred_output_dir}}`
{%- endif -%}.
Gonna ignore it.
{%- if pred_output_dir is not none %} For more details, check the results at:
{{pred_output_file}}
{%- endif -%}''')


@overload  # noqa: E302
def compute_saved_results(setup: EvalTaskConfig | None = None, *,
                          flush: bool = False,
                          write: bool = True) -> dict[str, FinalEvaluationResult | dict[str, FinalEvaluationResult]]:
    '''
    Compute the evaluation results with setup.prediction_output_dir, and save the results to setup.evaluation_file.
    '''
    ...
@overload  # noqa: E302
def compute_saved_results(pred_dir: str | None = None, *,
                          flush: bool = False,
                          write: bool = False) -> dict[str, FinalEvaluationResult | dict[str, FinalEvaluationResult]]:
    '''
    Compute the evaluation results with predictions saved
    under directory `pred_dir`, and return the results without saving.
    '''
    ...
@overload  # noqa: E302
def compute_saved_results(predictions: dict[str, list[dict]] | None = None, *,
                          flush: bool = False,
                          write: bool = False) -> dict[str, FinalEvaluationResult | dict[str, FinalEvaluationResult]]:
    '''
    Compute the evaluation results with loaded predictions.
    Predictions shall be organized as {dataset: [prediction1, prediction2, ...]}, e.g.
    {
        'dataset1': [prediction1, prediction2, ...],
        'dataset2': [prediction1, prediction2, ...],
        ...
    }
    '''
    ...
def compute_saved_results(setup: EvalTaskConfig | None = None,  # noqa: E302
                          pred_dir: str | None = None,
                          predictions: dict[str, list[dict]] | None = None, *,
                          flush: bool = False,
                          write: bool = True) -> dict[str, FinalEvaluationResult | dict[str, FinalEvaluationResult]]:
    timestamp = time.strftime("%Y%m%d_%H%M%S") if setup is None else setup.model.timestamp

    if setup is None and pred_dir is None and predictions is None:
        raise ValueError('Either setup or pred_dir or predictions must be provided.')
    elif len(list(filter(bool, [setup, pred_dir, predictions]))) > 1:
        raise ValueError('Only one of setup or pred_dir or predictions shall be provided. '
                         'When you provide multiple of them, there would be priority confusion.')
    elif setup is not None:
        pred_output_dir = setup.prediction_output_dir
        evaluation_output_path = setup.evaluation_file
    elif pred_dir is not None:
        pred_output_dir = pred_dir
        evaluation_output_path = f'evaluation_{timestamp}.json'
    elif predictions is not None:
        pred_output_dir = None
        evaluation_output_path = None
        write = False
    else:
        raise ValueError('Either setup or pred_dir or predictions must be provided.')

    try:
        with open(evaluation_output_path, 'r', encoding='utf-8') as f:
            results: dict[str, FinalEvaluationResult | dict[str, FinalEvaluationResult]] = json.load(f)
            results = defaultdict(results)
    except Exception:
        results = defaultdict(dict)

    located_datasets = list()
    if pred_output_dir is not None:
        predictions = dict()
        for _prediction_file in os.listdir(pred_output_dir):
            try:
                dataset = re.search(r'^(.*).jsonl$', _prediction_file).group(1)
                if dataset in setup.datasets:
                    with open(os.path.join(pred_output_dir, _prediction_file), 'r', encoding='utf-8') as f:
                        samples: list[dict] = list(map(json.loads, f.readlines()))
                    predictions[dataset] = samples
                    located_datasets.append(dataset)
            except AttributeError:
                pass
        if set(located_datasets) != set(setup.datasets):
            raise ValueError('Valid files matching pattern `*.jsonl` found under:\n'
                             f'\t{pred_output_dir}\n'
                             'not exactly same as the datasets in setup.')
    else:
        located_datasets = list(predictions.keys())

    for dataset, samples in predictions.items():
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

        pred_output_file = (None
                            if pred_output_dir is None else
                            os.path.join(pred_output_dir, f'{dataset}.jsonl'))

        if not categorized_samples['IN_UNIFIED']:
            logger.info(_MSG_TEMPLATE1.render(
                dataset=dataset,
                pred_output_dir=pred_output_dir,
                pred_output_file=pred_output_file
            ))
            continue

        if not categorized_samples['TARGET_NOT_OPEN']:
            logger.info(_MSG_TEMPLATE2.render(
                dataset=dataset,
                pred_output_dir=pred_output_dir,
                pred_output_file=pred_output_file
            ))
            continue

        history_content_source_counter = Counter()
        for sample in categorized_samples['IN_UNIFIED']:
            history_content_source_counter.update(
                Counter(sample['history_content_srcs'])
            )
        history_content_source_total = history_content_source_counter.total()
        # implicitly length weighted distribution,
        # i.e., ignoring the dumplication of history for steps in one episode
        history_content_source_distribution = [(item[0], item[1] / history_content_source_total)
                                               for item in history_content_source_counter.most_common()]
        results[dataset]['history_src_distribution'] = dict(history_content_source_distribution)

        sample_in_unified_count = len(categorized_samples['IN_UNIFIED'])
        results[dataset]['unified_action_ratio'] = sample_in_unified_count / original_task_count
        results[dataset].update((_category, _compute_samples(_samples, total_count=sample_in_unified_count))
                                for _category, _samples in categorized_samples.items())
        results[dataset]["dataset"] = dataset
        results[dataset]["timestamp"] = timestamp
        results[dataset] = OrderedDict(sorted(results[dataset].items(), key=_priority_map))

    if len(located_datasets) > 1:
        summary_results = {','.join(sorted(located_datasets)):
                           list(itertools.chain.from_iterable(predictions.values()))}
        summary = compute_saved_results(predictions=summary_results, flush=False, write=False)
        results["summary"].update(
            summary
        )

    if flush or write:
        evaluation = json.dumps(results, indent=4, ensure_ascii=False)
        if flush:
            logger.info('Final evaluation:\n'
                        f'{evaluation}')
        if write:
            with open(evaluation_output_path, 'w', encoding='utf-8') as f:
                f.write(evaluation)
            logger.info(f'Final evaluation saved at: {evaluation_output_path}')

    return results
