import json
import jsonschema
import os

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from guieval.utils.utils import load_json_data
from guieval.utils import UNIFIED_FIELDS, PREDICTION

EXTRACT_SCHEMA = json.load(
    open(os.path.join('guieval/utils/schema', 'schema_for_extraction.json'), encoding="utf-8"))


def parse_action(pred: PREDICTION):
    try:
        tmp_pred = deepcopy(pred)
        if 'POINT' in tmp_pred:
            tmp_pred['POINT'] = list(tmp_pred['POINT'])
        jsonschema.validate(tmp_pred, EXTRACT_SCHEMA)
        actions = {}
        parameters = {}
        status = tmp_pred.get("STATUS", "continue")

        for key in UNIFIED_FIELDS:
            if key in tmp_pred:
                actions[key] = tmp_pred[key]

        parameters["duration"] = tmp_pred.get("duration", EXTRACT_SCHEMA["properties"]["duration"]["default"])
        if "to" in tmp_pred:
            parameters["to"] = tmp_pred["to"]

        return actions, parameters, status

    except Exception:
        raise Exception(f"Error, JSON is NOT valid according to the schema: {pred}")


def process_step(task, episode_id, step_id, pred, output_path):
    try:
        actions, parameters, status = parse_action(pred)

        transformed_entry = {
            "action_predict": {
                "COA": {
                    "txt": {
                        "ACTION": actions,
                        "ARGS": parameters,
                        "STATUS": status
                    }
                }
            }
        }
        folder = f"{task}-{episode_id}"
        file_name = f"{folder}_{step_id}.json"
        output_file_path = os.path.join(output_path, folder, file_name)
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(transformed_entry, output_file, indent=4, ensure_ascii=False)
        return
    except Exception as e:
        print(f"Error processing step {step_id} in episode {episode_id}: {e}")
        return


def convert2aitz(input_path, output_path, max_workers=None):
    data = load_json_data(input_path)
    folders = set()
    tasks = []
    for item in data:
        task = item.get("category", item.get("subset", "unknown"))
        episode_id = item.get("episode_id", "unknown")
        steps = item.get("steps", [item])

        for index, each_step in enumerate(steps):
            step_id = index if "steps" in item else each_step.get("step_id", index)
            folder = f"{task}-{episode_id}"
            folders.add(folder)
            pred = each_step.get("prediction", {})
            tasks.append((task, episode_id, step_id, pred, output_path))

    for folder in folders:
        folder_path = os.path.join(output_path, folder)
        os.makedirs(folder_path, exist_ok=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_step, *task_args) for task_args in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing steps"):
            result = future.result()  # noqa: F841
