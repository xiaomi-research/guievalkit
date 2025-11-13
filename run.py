import multiprocessing
import os
import logging

from contextlib import ExitStack
from jsonargparse import auto_cli
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Literal, Annotated
from tqdm import tqdm

# subsec internal
from utils import init_logging, Timer
from guieval import EvalTaskConfig

TASK = Annotated[
    Literal["all", "infer", "eval"],
    "Task phase: all=full pipeline, infer=inference only, eval=evaluate saved results"
]


def main(task: TASK, setup: EvalTaskConfig):
    init_logging(
        root_name='GUIEval',
        level='INFO',
        log_file=setup.log_file,
        always_ansi=False
    )
    logger = logging.getLogger(__name__)
    timer = Timer()

    # subsec delayed import
    from guieval.utils import batched, get_simplified_traceback
    from guieval.main import (load_tasks,
                              DeployedModel, ModelProcessor, StepTaskModel,
                              compute_saved_results)

    # subsec main
    if task == "all" or task == "infer":
        logger.info("Loading datasets and distributing task configs...")
        loaded_dataset = load_tasks(dataset=setup.datasets,
                                    model_name=setup.model.model_name,
                                    model_alias=setup.model.model_alias,
                                    enable_think=setup.enable_thinking,
                                    enable_conclude=setup.enable_conclusion,
                                    eval_mode=setup.eval_mode,
                                    vllm_mode=setup.vllm_mode)

        _paths = '\n- '.join(_subset_item[1]
                             for _subset_items in loaded_dataset['dataset_info'].values()
                             for _subset_item in _subset_items)
        logger.info(f"Processing Plan:\n"
                    f"- {_paths}\n"
                    f"<split_line>")  # the formmater would auto process <split_line> token

        # subsec deploy
        DeployedModel.init_worker(setup=setup)
        processor = ModelProcessor.init(model_name=setup.model.model_name,
                                        sampling_params=None)
        logger.info(f"Model processor initialized in {timer.elapsed}")

        # subsec generate
        with ExitStack() as io_stack:
            main_io = dict((dataset,
                            io_stack.enter_context(
                                open(
                                    os.path.join(setup.predictions_output_dir, f'{dataset}.jsonl'),
                                    mode='w',
                                    encoding='utf-8')))
                           for (dataset, _), _ in loaded_dataset['dataset_info'].items())

            with ThreadPoolExecutor(max_workers=setup.max_concurrent_tasks) as tpexecutor:
                step_task_count = 0
                for _step_idx, _step_i_tasks in enumerate(loaded_dataset['step_tasks']):
                    step_i_task_count = len(_step_i_tasks)
                    next_step_task_count = step_task_count + step_i_task_count

                    step_i_tasks = list()

                    step_i_preprocess_futures = [tpexecutor.submit(processor.prepare_task_input, _task)
                                                 for _task in _step_i_tasks]
                    with tqdm(total=next_step_task_count, desc=f'Preprocessing Tasks Step {_step_idx}',
                              leave=False, initial=step_task_count) as pbar:
                        for _f in as_completed(step_i_preprocess_futures):
                            step_i_tasks.append(_f.result())
                            pbar.set_postfix_str(f"Elapsed: {timer.elapsed}")
                            pbar.update(1)

                    if setup.vllm_mode == 'offline':
                        step_i_tasks = batched(step_i_tasks, n=setup.batch_size)

                    step_i_task_futures = [tpexecutor.submit(processor.run_task, _task)
                                           for _task in step_i_tasks]
                    with tqdm(total=next_step_task_count, desc=f'Processing Tasks Step {_step_idx}',
                              leave=False, initial=step_task_count) as pbar:
                        for _f in as_completed(step_i_task_futures):
                            try:
                                task_results: list[StepTaskModel] = _f.result()
                            except Exception as e:
                                logger.error(f"Processor Error: {repr(e)}\n"
                                             f"Trackback:\n{get_simplified_traceback()}")
                            else:
                                for step in task_results:
                                    try:
                                        main_io[step.dataset].write(step.model_dump_json() + "\n")
                                        main_io[step.dataset].flush()
                                    except Exception as io_error:
                                        logger.error(f'IO broken: {repr(io_error)}\n'
                                                     f'Trackback:\n'
                                                     f'{get_simplified_traceback()}')
                            pbar.set_postfix_str(f"Elapsed: {timer.elapsed}")
                            pbar.update(min(setup.batch_size, next_step_task_count - pbar.n))

                    step_task_count = next_step_task_count

            DeployedModel.deprecate_worker(alias=setup.model.model_alias)

        logger.info(f"Predictions saved under: {setup.predictions_output_dir}")
        compute_saved_results(setup=setup, flush=True, write=True)
        logger.info(f"Task elapsed: {timer.elapsed}")

    if task == "eval":
        raise NotImplementedError()

    return


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    auto_cli(main)
