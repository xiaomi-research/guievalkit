from contextlib import ExitStack
import itertools
import multiprocessing
import logging

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
    from guieval.main import (load_datasets,
                              DeployedModel, ModelProcessor, StepTaskModel,
                              compute_saved_results)
    from profiler.tools.result_tools import load_results, result_task_difference, dataset_difference_result_update

    # subsec main
    if task == "all" or task == "infer":
        logger.info("Loading datasets and distributing task configs...")
        loaded_dataset = load_datasets(dataset=setup.datasets,
                                       model_name=setup.model.model_name,
                                       model_alias=setup.model.model_alias,
                                       enable_think=setup.enable_thinking,
                                       enable_conclude=setup.enable_conclusion,
                                       fixed_memory=(setup.fix_memory is not None),
                                       fixed_thought=(setup.fix_thought is not None),
                                       history_sampler=setup.history_sampler,
                                       eval_mode=setup.eval_mode,
                                       vllm_mode=setup.vllm_mode)

        _paths = '\n- '.join(_subset_item[1]
                             for _subset_items in loaded_dataset['dataset_info'].values()
                             for _subset_item in _subset_items)
        logger.info(f"Processing Plan:\n"
                    f"- {_paths}\n"
                    f"<split_line>")  # the formmater would auto process <split_line> token

        # subsec instantiate processor with core identifier `model_name` and all other configs
        processor = ModelProcessor.init(model_name=setup.model.model_name,
                                        sampling_params=None,
                                        sample_size=setup.sample_size,
                                        sample_seed=setup.sample_seed,
                                        temperature=setup.temperature,
                                        top_p=setup.top_p,
                                        top_k=setup.top_k,
                                        repetition_penalty=setup.repetition_penalty,
                                        presence_penalty=setup.presence_penalty,
                                        max_tokens=setup.max_tokens)

        # subsec fill processor memory with existing results
        if setup.fix_memory is not None:
            fixed_memory = load_results(prediction_output_dir=setup.fix_memory)
            memory_dataset_difference = result_task_difference(
                loaded_results=fixed_memory,
                loaded_tasks=itertools.chain.from_iterable(loaded_dataset["step_tasks"]),
                meta=True  # a more tolerant way to handle the dataset difference
            )
            if memory_dataset_difference["task_difference"]:
                logger.error(f"The fixed memory can't fully cover the dataset. "
                             f"Missing tasks count: {len(memory_dataset_difference['task_difference'])}")
                raise ValueError("The fixed memory can't fully cover the dataset.")

            logger.info("Filling processor memory with fixed memory. "
                        "The memory unit would be frozen for runtime and "
                        "any other prefilling would be blocked.")
            for _result in tqdm(fixed_memory):
                processor.fill_memory(_result)

        if setup.fix_thought is not None:
            fixed_thoughts = load_results(prediction_output_dir=setup.fix_thought)
            logger.info("Filling processor with fixed thoughts. ")
            for _result in tqdm(fixed_thoughts):
                processor.fill_fixed_thoughts(_result)

        if setup.restart:
            loaded_results = [_result for _result in load_results(setup=setup)
                              if _result.response is not None and _result.response != '']
            # filter out fake results without actual generated response
            dataset_difference_result_update(loaded_dataset=loaded_dataset,
                                             loaded_results=loaded_results)
            if sum(len(_step_tasks) for _step_tasks in loaded_dataset['step_tasks']) == 0:
                final_evaluation = compute_saved_results(setup=setup, flush=True, write=True)
                # recompute all the metrics for all the results
                if final_evaluation:
                    logger.info(f"Task has already completed. Predictions under: {setup.prediction_output_dir}")
                    logger.info(f"Task elapsed: {timer.elapsed}")
                    return
                else:
                    # logically impossible case. Reserved for completeness.
                    logger.info(f"No valid task results found. Gonna rerun this whole eval task...")
            elif setup.fix_memory is None:
                logger.info(f"Filling processor memory with processed results...")
                for _result in tqdm(loaded_results):
                    processor.fill_memory(_result)
            else:
                logger.info(f"Would restart with prefilled memory from {setup.fix_memory}...")

        # subsec deploy
        DeployedModel.init_worker(setup=setup)
        logger.info(f"Model processor initialized and service deployed in {timer.elapsed}")

        # subsec generate
        with ExitStack() as io_stack:
            # open i/o for for each group (dataset, split, subset)
            # io = dict(((dataset, split, subset),
            #             open(os.path.join(pred_output_dir,
            #                               f'{dataset}_{split}_{subset}.jsonl'),
            #                  mode='w',
            #                  encoding='utf-8'))
            #           for (dataset, split), subset_items in loaded_dataset['dataset_info'].items()
            #           for subset, _ in subset_items)  # currently it seems there is no need to apply this io
            # open all_in_one i/0 for each dataset
            main_io = dict((_dataset,
                            io_stack.enter_context(
                                open(
                                    _prediction_output_path,
                                    mode='w',
                                    encoding='utf-8')))
                           for _dataset, _prediction_output_path in zip(setup.datasets, setup.prediction_output_paths))

            # serialize loaded_results to file
            # after filtering out fake results without actual generated response
            # those tasks with invalid results would get rerun for new valid results
            # thus we need to overwrite the saved results with all validated samples
            # to avoid duplication that would affect final evaluation
            if setup.restart:
                for _result in loaded_results:
                    main_io[_result.dataset].write(
                        _result.model_dump_json() + '\n'
                    )

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

        logger.info(f"Predictions saved under: {setup.prediction_output_dir}")

        # subsec compute all the metrics for all the results
        compute_saved_results(setup=setup, flush=True, write=True)

        logger.info(f"Task elapsed: {timer.elapsed}")

    if task == "eval":
        raise NotImplementedError()

    return


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    auto_cli(main)
