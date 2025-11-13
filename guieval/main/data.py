import json
import os
import itertools

from typing import (Literal, Iterator)

# subsec internal
from config import CONFIG_BASE
from guieval.config import DATASET
from guieval.main.step_task import StepTaskModel, MODE
from guieval.main.utils import DATA_BASE


# section struct
try:
    DATA_CONFIG = json.loads(CONFIG_BASE.joinpath('dataset_info.json').read_bytes())
except json.JSONDecodeError as e:
    raise ValueError(f"Failed to load dataset_info.json: {e}")

SUBSET = str
DATASET_SUBSET = dict[tuple[DATASET, Literal['test']], tuple[SUBSET, str]]
STEP_TASKS = list[tuple[StepTaskModel]]
LOADED_DATASET = dict[Literal['dataset_info', 'step_tasks'],
                      DATASET_SUBSET | STEP_TASKS]


# section load_data
def load_tasks(dataset: str | list[str], *,
               model_name: str | None = None,
               model_alias: str | None = None,
               enable_think: bool = True,
               enable_conclude: bool = True,
               eval_mode: MODE = 'offline_rule',
               vllm_mode: Literal['online', 'offline', False] = 'online') -> LOADED_DATASET:
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

    Finally they will be merged, and the complete
    `dataset, split, subset` info are distributed to every single step task.
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

    def load_episode(dataset: str,
                     split: Literal['test'],
                     subset: str,
                     subset_dir: str, episode_file_name: str) -> list[StepTaskModel]:
        episode_path = os.path.join(subset_dir, episode_file_name, f"{episode_file_name}.json")
        with open(episode_path, 'r', encoding='utf-8') as f:
            try:
                episode_data: list[dict] = json.load(f)
                episode_start_index = episode_data[0].get('step_id', 0)
                extra_info = dict(dataset=dataset,
                                  split=split,
                                  subset=subset,
                                  subset_dir=subset_dir,
                                  episode_file_name=episode_file_name,
                                  episode_start_index=episode_start_index,  # the original start step index
                                  model=model_name,
                                  model_alias=model_alias,
                                  enable_think=enable_think,
                                  enable_conclude=enable_conclude,
                                  mode=eval_mode,
                                  vllm_mode=vllm_mode)
                for _i, _step_task in enumerate(episode_data):
                    _step_task.update(extra_info)
                    _step_task['step_id'] = _i  # align all step id to 0-start index i in episode <save>
            except Exception as e:
                raise ValueError(f"Failed to load {episode_path}: {e}")
        return list(map(StepTaskModel.model_validate, episode_data))

    def group_step_tasks(dataset: str,
                         split: Literal['test'],
                         subset: str,
                         subset_dir: str) -> list[tuple[StepTaskModel]]:
        grouped_step_tasks = itertools.zip_longest(*map(load_episode,
                                                        *zip(*itertools.product(
                                                            [dataset, ], [split,],
                                                            [subset, ], [subset_dir, ],
                                                            os.listdir(subset_dir)))))
        return [tuple(filter(bool, group)) for group in grouped_step_tasks]

    loaded_datasets: LOADED_DATASET = dict()
    loaded_datasets['dataset_info'] = dataset_subset

    _step_tasks: list[Iterator[STEP_TASKS]] = itertools.chain.from_iterable((group_step_tasks(dataset, split, _subset,
                                                                                              _subset_dir)
                                                                             for (_subset,
                                                                                  _subset_dir) in subset_itms)
                                                                            for ((dataset, split),
                                                                                 subset_itms) in dataset_subset.items())

    loaded_datasets['step_tasks'] = [
        tuple(itertools.chain.from_iterable(_step_i_tasks))
        for _step_i_tasks in itertools.zip_longest(*_step_tasks, fillvalue=tuple())]

    return loaded_datasets

# section load results
# memo todo
