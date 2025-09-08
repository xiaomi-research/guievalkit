import argparse
import json
import multiprocessing
import os
import traceback

from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

import guieval.utils.global_tokenizer as global_tokenizer

from guieval.eval.compute_metric import compute_atomic_metrics, compute_episode_metrics
from guieval.eval.convert_output import convert2aitz
from guieval.eval.dataset import EvalDataset
from guieval.eval.evaluator import process_step_data, ActionEvaluator
from guieval.models.execute import init_llm, dummy_task, prepare_inputs, run_step_batch, shut_llm
from guieval.utils.utils import get_gpu_list, load_info
from guieval.utils.config_utils import model_config_handler


_datasets = load_info("./config", "dataset_info.json")
_models = load_info("./config", "model_info.json")

DEVICES = get_gpu_list()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="name of the dataset/benchmark")
    parser.add_argument("--model", type=str, required=True, help="name of the model")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "infer", "eval"])
    parser.add_argument("--no-think", action="store_true", help="disable the thinking mode if applicable")
    parser.add_argument("--outputs", type=str, default="./outputs", help="output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size used for inference")
    parser.add_argument("--use-vllm", action="store_true", help="use vllm for inference")
    parser.add_argument("--over-size", action="store_true", help="use four gpus for inferring large model with vllm")
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    if args.dataset not in _datasets:
        raise ValueError(f"{args.dataset} is not supported!")

    if args.model not in _models:
        raise ValueError(f"{args.model} is not supported!")

    print("Predicting on: {}".format(os.path.join(_datasets[args.dataset]["folder_name"],
                                                  _datasets[args.dataset]["split"])))
    print("Data subsets: {}".format(_datasets[args.dataset]["subset"]))

    if args.mode == "all" or args.mode == "infer":

        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

        global DEVICES
        if not args.use_vllm:
            DEVICES = list(range((len(DEVICES))))

        if args.use_vllm:
            if args.over_size:
                group_size = 4  # for models >= 32B
                if len(DEVICES) < group_size:
                    raise ValueError("You do not have enough GPUs!")
                split_num = len(DEVICES) // group_size
                gpus = [str(e) for e in DEVICES]
                DEVICES = [
                    ",".join(gpus[i * group_size: (i + 1) * group_size])
                    for i in range(split_num)
                ]
            else:
                DEVICES = [str(i) for i in DEVICES]
                split_num = len(DEVICES)
        else:
            DEVICES = [str(i) for i in DEVICES]
            split_num = len(DEVICES)

        model_config = model_config_handler(args.model)
        TOKENIZER_CLASS = model_config["tokenizer"]

        global_tokenizer._tokenizer = TOKENIZER_CLASS.from_pretrained(
            _models[args.model]["folder_name"], trust_remote_code=True)

        device_queue = multiprocessing.Queue()
        for dev in DEVICES:
            device_queue.put(dev)

        with ProcessPoolExecutor(
                max_workers=split_num,
                initializer=init_llm,
                initargs=(args.model, _models[args.model]["folder_name"], device_queue, args.use_vllm)) as ppexecutor:

            dummy_tasks = [ppexecutor.submit(dummy_task) for _ in range(split_num)]
            for future in dummy_tasks:
                future.result()

            for dataset in _datasets[args.dataset]["subset"]:

                save_dir = os.path.join(args.outputs, _datasets[args.dataset]["split"], dataset)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                output_file = os.path.join(save_dir, "predict.jsonl")

                episode_dir = os.path.join(
                    _datasets[args.dataset]["folder_name"], _datasets[args.dataset]["split"], dataset)
                if os.path.exists(episode_dir):
                    episode_files = os.listdir(episode_dir)
                else:
                    continue

                print("Loading dataset: {}".format(
                    os.path.join(_datasets[args.dataset]["folder_name"], _datasets[args.dataset]["split"], dataset)))

                with open(output_file, "w", encoding="utf-8") as f_out:
                    with ThreadPoolExecutor(max_workers=16) as tpexecutor:

                        for episode_batch_start in tqdm(
                            range(0, len(episode_files), args.batch_size), position=0, desc="Running episode batch"
                        ):
                            episode_files_per_batch = episode_files[
                                episode_batch_start: episode_batch_start + args.batch_size
                            ]

                            batch_future = []
                            batch_all_tasks = []

                            for episode_file in episode_files_per_batch:
                                episode_path = os.path.join(episode_dir, episode_file, f"{episode_file}.json")
                                try:
                                    with open(episode_path, "r", encoding="utf-8") as f:
                                        episode = json.load(f)
                                except Exception as e:
                                    print(f"Failed to load {episode_path}: {e}")
                                    continue

                                batch_future.append(
                                    tpexecutor.submit(
                                        prepare_inputs,
                                        args.model,
                                        episode,
                                        episode_dir,
                                        episode_file,
                                        dataset,
                                        args.dataset,
                                        args.use_vllm,
                                        args.no_think
                                    )
                                )

                            for f in tqdm(
                                as_completed(batch_future), total=len(batch_future),
                                position=1, leave=False, desc="Preparing inputs"
                            ):
                                batch_all_tasks.append(f.result())

                            batch_all_task_value = []
                            for task in batch_all_tasks:
                                for task_value in task:
                                    batch_all_task_value.append(task_value)

                            tasks = []
                            for batch_start in range(0, len(batch_all_task_value), args.batch_size):
                                batch_tasks = batch_all_task_value[batch_start: batch_start + args.batch_size]
                                tasks.append(ppexecutor.submit(run_step_batch, args.model, batch_tasks, args.use_vllm))

                            for task in tqdm(
                                as_completed(tasks), total=len(tasks), position=2,
                                leave=False, dynamic_ncols=True, desc="Running step batch"
                            ):
                                try:
                                    batch_steps = task.result()
                                except Exception as e:
                                    print(f"Error: {e}")
                                    print(f"Batch Steps: {batch_steps}")
                                    print("Traceback: ", traceback.format_exc())
                                else:
                                    for _, step in enumerate(batch_steps):
                                        try:
                                            f_out.write(json.dumps(step, ensure_ascii=False) + "\n")
                                            f_out.flush()
                                        except Exception as inner_e:
                                            print(f"Error: {inner_e}")
                                            print(f"Step: {step}")
                                            print("Traceback: ", traceback.format_exc())

                print(f"Prediction saved at: {output_file}")

            future = [ppexecutor.submit(shut_llm,) for _ in range(split_num)]
            for f in future:
                print(f.result())

        os.system(f"cat {args.outputs}/{_datasets[args.dataset]['split']}/*/predict.jsonl > {args.outputs}/all.jsonl")
        print(f"Merged prediction saved at: {args.outputs}/all.jsonl")

    if args.mode == "all" or args.mode == "eval":
        convert2aitz(f"{args.outputs}/all.jsonl", f"{args.outputs}/results", max_workers=16)

        save_dir = f"{args.outputs}/results"
        results_save_file = os.path.join(save_dir, "result.json")

        eval_data = EvalDataset(data_dir=_datasets[args.dataset]["folder_name"],
                                subset=_datasets[args.dataset]["subset"],
                                split=_datasets[args.dataset]['split'])
        print(f"Total steps: {len(eval_data)}, total episodes: {len(eval_data.episode_data)}.")

        eval_android_control = args.dataset in ['androidcontrol_high', 'androidcontrol_low']
        evaluator = ActionEvaluator(eval_android_control)
        results = list(tqdm(
            map(process_step_data, eval_data.data, [evaluator] * len(eval_data.data), [save_dir] * len(eval_data.data)),
            total=len(eval_data.data), desc="Processing steps", ncols=100))
        results = list(filter(lambda x: x is not None, results))

        try:
            os.makedirs(os.path.dirname(results_save_file), exist_ok=True)
            with open(results_save_file, "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"Evaluation results saved to {results_save_file}")
        except Exception as e:
            print(f"Error saving results to {results_save_file}: {e}")

        try:
            episode_metrics = compute_episode_metrics(results, no_open_action=False)
            print("######################################### Episode Metrics #########################################")
            for metric in episode_metrics.keys():
                print("{}: {}".format(metric, episode_metrics[metric]))
            print("###################################################################################################")

            no_open_episode_metrics = compute_episode_metrics(results, no_open_action=True)
            print("################################## Episode Metrics (without open) #################################")
            for metric in no_open_episode_metrics.keys():
                print("{}: {}".format(metric, no_open_episode_metrics[metric]))
            print("###################################################################################################")

            atomic_metrics = compute_atomic_metrics(results)
            print("######################################### Action Metrics ##########################################")
            for action in atomic_metrics.keys():
                print("{}: {}".format(action, atomic_metrics[action]))
            print("###################################################################################################")

            summary_save_file = os.path.join(args.outputs, 'summary.json')
            print(f"Evaluation summary saved to {summary_save_file}")
            with open(summary_save_file, 'w') as f:
                json.dump(episode_metrics | atomic_metrics, f, ensure_ascii=False)

        except Exception as e:
            print(f"Failed to compute evaluation metrics: {e}")

    return


if __name__ == "__main__":
    main()
