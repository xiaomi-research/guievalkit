import os
import re
import itertools

from collections import defaultdict

# subsec internal
from profiler.main import StepTaskResult


def merge_result_samples(results: list[StepTaskResult]):
    '''
    merge the result samples of the single step task
    '''
    categorized_results = defaultdict(list)
    for result in results:
        categorized_results[(result.dataset, result.episode_id, result.step_id)].append(result)

    merged_results = []

    for results in categorized_results.values():

        pos_results = [result for result in results if result.evaluation.exact_match]
        neg_results = [result for result in results
                       if (not result.evaluation.exact_match) and result.pred_action is not None]
        if pos_results:
            merged_result = pos_results[0].model_copy()
        elif neg_results:
            merged_result = neg_results[0].model_copy()
        else:
            merged_result = results[0].model_copy()

        result_samples = list(itertools.chain.from_iterable(result.result_samples
                                                            for result in results))

        merged_result.result_samples = result_samples
        merged_results.append(merged_result)

    return merged_results


def load_sampled_results(result_base: str, *,
                         fix_thought: str | None = None):
    '''
    merge sampled results with sample size 64
    '''
    experiment_pattern = re.compile(r'thinking_(\w+)_mode_(\w+_\w+)_size_64_seed_\d+')

    merged_results = defaultdict(list)

    for experiment_dir in os.listdir(result_base):
        if experiment_pattern.match(experiment_dir):
            thinking, mode = experiment_pattern.match(experiment_dir).groups()
            experiment_path = os.path.join(result_base, experiment_dir)
            experiment_result_dir = os.path.join(experiment_path, mode, f'predictions_thinking_{thinking.capitalize()}')
            for result_file in os.listdir(experiment_result_dir):
                if result_file.endswith('.jsonl'):
                    result_path = os.path.join(experiment_result_dir, result_file)
                    with open(result_path, 'r') as f:
                        results = list(map(StepTaskResult.model_validate_json, f.readlines()))
                        merged_results[(f'thinking_{thinking}', mode)].extend(results)

    if fix_thought is not None:
        for experiment_dir in os.listdir(fix_thought):
            thinking, mode = 'true', 'fix_thought'
            experiment_path = os.path.join(fix_thought, experiment_dir)
            experiment_result_dir = os.path.join(experiment_path,
                                                 'semi_online',
                                                 f'predictions_thinking_{thinking.capitalize()}')
            for result_file in os.listdir(experiment_result_dir):
                if result_file.endswith('.jsonl'):
                    result_path = os.path.join(experiment_result_dir, result_file)
                    with open(result_path, 'r') as f:
                        results = list(map(StepTaskResult.model_validate_json, f.readlines()))
                        merged_results[(f'thinking_{thinking}', mode)].extend(results)

    for (thinking, mode), results in merged_results.items():
        merged_results[(thinking, mode)] = merge_result_samples(results)

    return merged_results
