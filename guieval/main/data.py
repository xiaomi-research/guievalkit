# section import
import json
import os
import re
import itertools
import functools
from typing import Literal, Iterator
from typing_extensions import TypedDict

# subsec internal
from config import CONFIG_BASE
from guieval.main.step_task import StepTaskModel, MODE
from guieval.main.history_sampler import HistorySamplerConfig
from guieval.main.utils import DATA_BASE, DATASET


# section struct
try:
    DATA_CONFIG = json.loads(CONFIG_BASE.joinpath('dataset_info.json').read_bytes())
except json.JSONDecodeError as e:
    raise ValueError(f"Failed to load dataset_info.json: {e}")

SUBSET = str
SUBSET_DIR = str
DATASET_SUBSET = dict[tuple[DATASET, Literal['test']], tuple[SUBSET, SUBSET_DIR]]
STEP_TASKS = tuple[StepTaskModel]
GROUPED_STEP_TASKS = list[STEP_TASKS]
EPISODE = list[StepTaskModel]
EPISODES = list[EPISODE]


class LOADED_STEP_TASKS(TypedDict):
    dataset_info: DATASET_SUBSET
    step_tasks: GROUPED_STEP_TASKS


class LOADED_EPISODES(TypedDict):
    dataset_info: DATASET_SUBSET
    episodes: EPISODES


def load_episode(dataset: str, split: Literal['test'], subset: str, subset_dir: str, episode_file_name: str,
                 model_name: str | None = None, model_alias: str | None = None,
                 enable_think: bool = True,
                 enable_conclude: bool = True,
                 fixed_memory: bool = False,
                 fixed_thought: bool = False,
                 history_sampler: HistorySamplerConfig = HistorySamplerConfig(),
                 eval_mode: MODE = 'offline_rule',
                 vllm_mode: Literal['online', 'offline', False] = 'online') -> EPISODE:
    episode_dir = os.path.join(subset_dir, episode_file_name)
    episode_json = os.path.join(episode_dir, f"{episode_file_name}.json")
    if not os.path.exists(episode_json):
        episode_files = os.listdir(episode_dir)
        episode_json_files = [file for file in episode_files if re.match(r'^.*\.json$', file)]
        assert len(episode_json_files) == 1, ("Multiple/None episode json files found in "
                                              f"{episode_dir}: {episode_json_files}")
        episode_json = os.path.join(episode_dir, episode_json_files[0])
    with open(episode_json, 'r', encoding='utf-8') as f:
        try:
            episode_data: list[dict] = json.load(f)
            extra_info = dict(dataset=dataset,
                              split=split,
                              subset=subset,
                              subset_dir=subset_dir,
                              episode_file_name=episode_file_name,
                              model=model_name,
                              model_alias=model_alias,
                              enable_think=enable_think,
                              enable_conclude=enable_conclude,
                              fixed_memory=fixed_memory,
                              fixed_thought=fixed_thought,
                              history_sampler=history_sampler,
                              mode=eval_mode,
                              vllm_mode=vllm_mode)
            for _i, _step_task in enumerate(episode_data):
                _step_task.update(extra_info)
                _step_task['episode_step_id'] = _step_task['step_id']  # store the original step index
                _step_task['step_id'] = _i  # align all step id to 0-start index i in episode <save>
        except Exception as e:
            raise ValueError(f"Failed to load {episode_json}: {e}")
    return list(map(StepTaskModel.model_validate, episode_data))


# section load_data
def load_datasets(dataset: str | list[str],
                  model_name: str | None = None,
                  model_alias: str | None = None,
                  enable_think: bool = True,
                  enable_conclude: bool = True,
                  fixed_memory: bool = False,
                  fixed_thought: bool = False,
                  history_sampler: HistorySamplerConfig = HistorySamplerConfig(),
                  eval_mode: MODE = 'offline_rule',
                  vllm_mode: Literal['online', 'offline', False] = 'online', *,
                  group_episodes: bool = True) -> LOADED_STEP_TASKS | LOADED_EPISODES:
    '''
    Tasks are firstly grouped by dataset,
    then by subset,
    finally grouped by their step id in ascending order corresponding to the index of the list.

    e.g.,
    ```python
    {
        'subset1': [
            [('subset1', 'step1_episode1'), ('subset1', 'step2_episode1'), ('subset1', 'step3_episode1')],
            [('subset1', 'step1_episode2'), ('subset1', 'step2_episode2')],
        ]
    }
    ```
    will be transformed to:
    ```python
    {
        'subset1': [
            [('subset1', 'step1_episode1'), ('subset1', 'step1_episode2')],
            [('subset1', 'step2_episode1'), ('subset1', 'step2_episode2')],
            [('subset1', 'step3_episode1')],
        ]
    }
    ```

    Finally they will be merged, and the complete `dataset, split, subset`
    info are distributed to every single step task.
    '''
    if model_alias is None:
        model_alias = model_name

    datasets = ([dataset]
                if isinstance(dataset, str) else
                dataset)
    dataset_subset: dict[tuple[DATASET, Literal['test']], list[tuple[SUBSET, str]]] = dict()

    for _dataset in datasets:
        if _dataset not in DATA_CONFIG:
            raise ValueError(f"Dataset {_dataset} not found in ./config/dataset_info.json")

        data_config = DATA_CONFIG[_dataset]
        data_dir = DATA_BASE.joinpath(data_config['folder_name'])
        split: Literal['test'] = data_config['split']  # currently only test split is supported <save>
        split_dir = data_dir.joinpath(split)
        _subsets: list[str] = data_config['subset']
        _subset_items = [(_subset, os.fspath(split_dir.joinpath(_subset)))
                        for _subset in _subsets]
        _filtered_subset_items = filter(lambda _item: os.path.exists(_item[1]), _subset_items)
        subsets, subset_dirs = zip(*_filtered_subset_items)
        dataset_subset[(_dataset, split)] = list(zip(subsets, subset_dirs))

    def load_subset(dataset: str,
                    split: Literal['test'],
                    subset: str,
                    subset_dir: str) -> GROUPED_STEP_TASKS | EPISODES:
        load_dataset_episode = functools.partial(load_episode,
                                                 dataset=dataset,
                                                 split=split,
                                                 subset=subset,
                                                 subset_dir=subset_dir,
                                                 model_name=model_name,
                                                 model_alias=model_alias,
                                                 enable_think=enable_think,
                                                 enable_conclude=enable_conclude,
                                                 fixed_memory=fixed_memory,
                                                 fixed_thought=fixed_thought,
                                                 history_sampler=history_sampler,
                                                 eval_mode=eval_mode,
                                                 vllm_mode=vllm_mode)
        if group_episodes:
            # grouped_step_tasks = itertools.zip_longest(*map(load_dataset_episode, os.listdir(subset_dir)))
            grouped_step_tasks = itertools.zip_longest(*(load_dataset_episode(episode_file_name=episode_file_name)
                                                         for episode_file_name in os.listdir(subset_dir)))
            return [tuple(filter(bool, group)) for group in grouped_step_tasks]
        else:
            return [load_dataset_episode(episode_file_name=episode_file_name)
                    for episode_file_name in os.listdir(subset_dir)]

    loaded_datasets: LOADED_STEP_TASKS | LOADED_EPISODES = dict()
    loaded_datasets['dataset_info'] = dataset_subset

    if group_episodes:
        _step_tasks: list[Iterator[GROUPED_STEP_TASKS]] = itertools.chain.from_iterable(
            (load_subset(dataset, split, _subset, _subset_dir)
             for _subset, _subset_dir in subset_items)
            for (dataset, split), subset_items in dataset_subset.items())

        loaded_datasets['step_tasks'] = [
            tuple(itertools.chain.from_iterable(_step_i_tasks))
            for _step_i_tasks in itertools.zip_longest(*_step_tasks, fillvalue=tuple())]
    else:
        episodes: EPISODES = list(itertools.chain.from_iterable(
            load_subset(dataset, split, _subset, _subset_dir)
            for (dataset, split), subset_items in dataset_subset.items()
            for _subset, _subset_dir in subset_items))
        loaded_datasets['episodes'] = episodes

    return loaded_datasets
