import json
import jsonschema
import os

from vllm import SamplingParams
from PIL import Image


ACTION_SCHEMA = json.load(open(os.path.join('guieval/models/utils/schema', 'schema.json'), encoding="utf-8"))
items = list(ACTION_SCHEMA.items())
insert_index = 3
items.insert(insert_index, ("required", ["thought"]))  # enable/disable thought by setting it to "required"/"optional"
ACTION_SCHEMA = dict(items)

SYSTEM_PROMPT = f'''# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束

# Schema
{json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':'))}'''

EXTRACT_SCHEMA = json.load(open(os.path.join('guieval/models/utils/schema', 'schema_for_extraction.json'),
                                encoding="utf-8"))


def resize(origin_img):
    resolution = origin_img.size
    w, h = resolution
    max_line_res = 1120

    if max_line_res is not None:
        max_line = max_line_res
        if h > max_line:
            w = int(w * max_line / h)
            h = max_line
        if w > max_line:
            h = int(h * max_line / w)
            w = max_line

    img = origin_img.resize((w, h), resample=Image.Resampling.LANCZOS)
    return img


def prepare_task_input(step, image_path, data_name, use_vllm):

    image = Image.open(image_path).convert("RGB")
    image = resize(image)
    images = [image]
    if data_name == 'androidcontrol_low':
        query = step['low_instruction']
    else:
        query = step['instruction']

    messages = []
    if use_vllm:
        messages.append(
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        )
        messages.append(
            {
                "role": "user",
                "content":
                    f"<Question>{query}</Question>\n当前屏幕截图：(<image>./</image>)",
            }
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": [
                    f"<Question>{query}</Question>\n当前屏幕截图：",
                    image
                ]
            }
        )
    return step, messages, images


def prepare_task_inputs(episode, episode_dir, episode_file, subset, dataset, use_vllm):

    res = []
    for step in episode:
        step["category"] = subset
        image_path = os.path.join(episode_dir, episode_file, f"{episode_file}_{step['step_id']}.jpeg")
        if not os.path.exists(image_path):
            image_path = image_path.replace(".jpeg", ".png")
            if not os.path.exists(image_path):
                image_path = image_path.replace(".png", ".jpg")
                if not os.path.exists(image_path):
                    image_path = step['image_path']
        res.append(prepare_task_input(step, image_path, dataset, use_vllm))

    return res


def extract_and_validate_json(input_string):
    try:
        json_obj = json.loads(input_string)
        jsonschema.validate(json_obj, EXTRACT_SCHEMA)
        return json_obj
    except json.JSONDecodeError:
        print("Error, JSON is NOT valid.")
        return input_string
    except Exception as e:
        print(f"Error, JSON is NOT valid according to the schema: {input_string}", e)
        return input_string


def run_task_batch(_llm, _tokenizer, batch_tasks, use_vllm):

    batch_steps = []
    batch_inputs = []
    for step, messages, images in batch_tasks:
        if use_vllm:
            text_prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append({"prompt": text_prompt, "multi_modal_data": {"image": images}})
        else:
            batch_inputs.append(messages)
        batch_steps.append(step)

    if use_vllm:
        sampling_params = SamplingParams(temperature=0.1, top_p=0.3, max_tokens=512)
        results = _llm.generate(batch_inputs, sampling_params, use_tqdm=False)
        predict_str = [result.outputs[0].text for result in results]
    else:
        predict_str = _llm.chat(image=None, msgs=batch_inputs, system_prompt=SYSTEM_PROMPT, tokenizer=_tokenizer,
                                temperature=0.1, top_p=0.3, n=1,)

    predict_json = [extract_and_validate_json(s) for s in predict_str]
    res_steps = []
    for step, js_res in zip(batch_steps, predict_json):
        step['pred'] = js_res
        res_steps.append(step)
    return res_steps
