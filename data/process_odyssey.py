import json
import os

from copy import deepcopy
from pathlib import Path
from tqdm import tqdm


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
split_data_path = os.path.join(current_dir, 'gui_odyssey/test_anno/random_split.json')

out_dir = Path(os.path.join(current_dir, 'gui_odyssey/test/gui_odyssey'))
out_dir.mkdir(parents=True, exist_ok=True)


def transform_action_data(action_data: dict) -> dict:

    # directly retrieve relevant information from the data.
    episode_id: str = action_data["image"].split('/')[-1].split("_")[0]
    step_id: str = action_data["image"].split('/')[-1].split("_")[1][:-4]
    episode_length: str = action_data["step_length"]
    image_path: str = action_data["image"]
    instruction: str = action_data["question"]
    answer: str = action_data["answer"]

    # Get the picture information (w/h) of the data. They are restored under annotation/*.json.
    with open(os.path.join(
            current_dir, "gui_odyssey/annotations", f"{episode_id}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)["device_info"]
    image_width: int = data["w"]
    image_height: int = data["h"]

    # first initialize the variables.
    result_action_text = ""
    result_touch_yx = [-1.0, -1.0]
    result_lift_yx = [-1.0, -1.0]
    duration = 0
    result_action_type: int = 2

    # case: Click.
    if answer.startswith("CLICK"):
        result_action_type = 4
        x, y = list(map(float, answer[8:-1].split(",")))
        result_touch_yx = [y / 1000, x / 1000]  # get the ratio.
        result_lift_yx = deepcopy(result_touch_yx)  # same point.

    if answer.startswith("SCROLL"):
        result_action_type = 4

        result_touch_yx = [0.5, 0.5]
        result_lift_yx = deepcopy(result_touch_yx)

        # Manually create the end point as odyssey didn't give us an exact coordinate.
        if answer.endswith("UP"):
            result_lift_yx[0] -= 0.1

        elif answer.endswith("DOWN"):
            result_lift_yx[0] += 0.1

        elif answer.endswith("LEFT"):
            result_lift_yx[1] -= 0.1

        else:
            result_lift_yx[1] += 0.1

    if answer.startswith("LONG_PRESS"):
        result_action_type = 0

        x, y = list(map(float, answer[13: -1].split(",")))
        result_touch_yx = [y / 1000, x / 1000]

        result_lift_yx = deepcopy(result_touch_yx)

    if answer.startswith("TYPE"):
        result_action_type = 3
        result_action_text: str = answer[5:].strip()

    if answer.startswith("PRESS_HOME"):
        result_action_type = 6

    if answer.startswith("PRESS_BACK"):
        result_action_type = 5

    if answer.startswith("PRESS_RECENT"):
        result_action_type = 6  # mapping as HOME.

    if answer.startswith("COMPLETE"):
        result_action_type = 10

    if answer.startswith("IMPOSSIBLE"):
        result_action_type = 11

    data = {
        "episode_id": episode_id,
        "step_id": step_id,
        "episode_length": episode_length,
        "image_width": image_width,
        "image_height": image_height,
        "image_path": image_path,
        "instruction": instruction,
        "result_action_type": result_action_type,
        "result_touch_yx": str(result_touch_yx),
        "result_lift_yx": str(result_lift_yx),
        "duration": duration,  # ignore the duration.
        "result_action_text": result_action_text,
        "ui_positions": "",
        "low_instruction": ""
    }

    return data


# Construct the data.
with open(split_data_path, "r", encoding="utf-8") as f:
    eval_data_raw: dict = json.load(f)

data_eval = [transform_action_data(data) for data in tqdm(eval_data_raw)]
data_eval = [d for d in data_eval if d is not None]


# save data
def dump_traj(traj, out_root: Path, idx: int):
    if not traj:
        return

    subfolder_name = f"traj_{idx:05d}"
    subfolder_path = out_root / subfolder_name
    subfolder_path.mkdir(parents=True, exist_ok=True)

    out_filename = f"{subfolder_name}.json"
    out_file = subfolder_path / out_filename

    with out_file.open("w", encoding="utf-8") as fw:
        json.dump(traj, fw, ensure_ascii=False, indent=2)

    print(f"Save {subfolder_name}/{out_filename}  (steps={len(traj)})")


records = data_eval
traj = []
traj_idx = 1
prev_step_id, curr_instr = None, None

for rec in records:
    step_id = int(rec["step_id"])
    instr = rec["instruction"]
    rec['subset'] = 'gui_odyssey'
    rec['step_id'] = int(rec['step_id'])
    rec['ui_positions'] = "[]"

    new_traj = (
        curr_instr is None or
        instr != curr_instr or
        prev_step_id is None or
        step_id != prev_step_id + 1
    )

    if new_traj and traj:
        dump_traj(traj, out_dir, traj_idx)
        traj_idx += 1
        traj = []

    traj.append(rec)
    prev_step_id = step_id
    curr_instr = instr

dump_traj(traj, out_dir, traj_idx)
print("all done.")
