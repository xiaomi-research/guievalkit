import json
import numpy as np
import os
import re

from PIL import Image
from vllm import SamplingParams

from guieval.utils.action_type import ActionType
from guieval.utils.action_utils import is_tap_action, get_direction


def get_mobile_prompt(task, history):

    prompt = (
        "You are a GUI Agent, and your primary task is to respond accurately to user requests or questions. In "
        "addition to directly answering the user's queries, you can also use tools or perform GUI operations directly"
        " until you fulfill the user's request or provide a correct answer. You should carefully read and understand "
        "the images and questions provided by the user, and engage in thinking and reflection when appropriate. The "
        "coordinates involved are all represented in thousandths (0-999).\n"
        "\n"
        "# Task:\n"
        f"{task}\n"
        "\n"
        "# Task Platform\n"
        "Mobile\n"
        "\n"
        "# Action Space\n"
        "### status\n"
        "\n"
        "Calling rule: `{\"action_type\": \"status\", \"goal_status\": \"<complete|infeasible>\"}`\n"
        "{\n"
        "    \"name\": \"status\",\n"
        "    \"description\": \"Finish the task by using the status action with complete or infeasible as "
        "goal_status.\",\n"
        "    \"parameters\": {\n"
        "        \"type\": \"object\",\n"
        "        \"properties\": {\n"
        "            \"goal_status\": {\n"
        "                \"type\": \"string\",\n"
        "                \"description\": \"The goal status of the task.\",\n"
        "                \"enum\": [\"complete\", \"infeasible\"]\n"
        "            }\n"
        "        },\n"
        "        \"required\": [\n"
        "            \"goal_status\"\n"
        "        ]\n"
        "    }\n"
        "}\n"
        "\n"
        # "### answer\n"
        # "\n"
        # "Calling rule: `{\"action_type\": \"answer\", \"text\": \"<answer_text>\"}`\n"
        # "{\n"
        # "    \"name\": \"answer\",\n"
        # "    \"description\": \"Answer user's question.\",\n"
        # "    \"parameters\": {\n"
        # "        \"type\": \"object\",\n"
        # "        \"properties\": {\n"
        # "            \"text\": {\n"
        # "                \"type\": \"string\",\n"
        # "                \"description\": \"The answer text.\"\n"
        # "            }\n"
        # "        },\n"
        # "        \"required\": [\n"
        # "            \"text\"\n"
        # "        ]\n"
        # "    }\n"
        # "}\n"
        # "\n"
        "### click\n"
        "\n"
        "Calling rule: `{\"action_type\": \"click\", \"box_2d\": [[xmin,ymin,xmax,ymax]]}`\n"
        "{\n"
        "    \"name\": \"click\",\n"
        "    \"description\": \"Click/tap on an element on the screen. Use the box_2d to indicate which element you "
        "want to click.\",\n"
        "    \"parameters\": {\n"
        "        \"type\": \"object\",\n"
        "        \"properties\": {\n"
        "            \"box_2d\": {\n"
        "                \"type\": \"array\",\n"
        "                \"description\": \"The box_2d should be [[xmin,ymin,xmax,ymax]] normalized to 0-999, "
        "indicating the position of the element.\"\n"
        "            }\n"
        "        },\n"
        "        \"required\": [\n"
        "            \"box_2d\"\n"
        "        ]\n"
        "    }\n"
        "}\n"
        "\n"
        "### long_press\n"
        "\n"
        "Calling rule: `{\"action_type\": \"long_press\", \"box_2d\": [[xmin,ymin,xmax,ymax]]}`\n"
        "{\n"
        "    \"name\": \"long_press\",\n"
        "    \"description\": \"Long press on an element on the screen, similar with the click action above, use the "
        "box_2d to indicate which element you want to long press.\",\n"
        "    \"parameters\": {\n"
        "        \"type\": \"object\",\n"
        "        \"properties\": {\n"
        "            \"box_2d\": {\n"
        "                \"type\": \"array\",\n"
        "                \"description\": \"The box_2d should be [[xmin,ymin,xmax,ymax]] normalized to 0-999, "
        "indicating the position of the element.\"\n"
        "            }\n"
        "        },\n"
        "        \"required\": [\n"
        "            \"box_2d\"\n"
        "        ]\n"
        "    }\n"
        "}\n"
        "\n"
        "### input_text\n"
        "\n"
        "Calling rule: `{\"action_type\": \"input_text\", \"text\": \"<text_input>\", \"box_2d\": "
        "[[xmin,ymin,xmax,ymax]], \"override\": true/false}`\n"
        "{\n"
        "    \"name\": \"input_text\",\n"
        "    \"description\": \"Type text into a text field (this action contains clicking the text field, typing in "
        "the text and pressing the enter). Use the box_2d to indicate the target text field.\",\n"
        "    \"parameters\": {\n"
        "        \"type\": \"object\",\n"
        "        \"properties\": {\n"
        "            \"text\": {\n"
        "                \"description\": \"The text to be input. Can be from the command, the memory, or the current"
        " screen.\"\n"
        "            },\n"
        "            \"box_2d\": {\n"
        "                \"description\": \"The box_2d should be [[xmin,ymin,xmax,ymax]] normalized to 0-999, "
        "indicating the position of the element.\"\n"
        "            },\n"
        "            \"override\": {\n"
        "                \"description\": \"If true, the text field will be cleared before typing. If false, the text"
        " will be appended.\"\n"
        "            }\n"
        "        },\n"
        "        \"required\": [\n"
        "            \"text\",\n"
        "            \"box_2d\",\n"
        "            \"override\"\n"
        "        ]\n"
        "    }\n"
        "}\n"
        "\n"
        "### keyboard_enter\n"
        "\n"
        "Calling rule: `{\"action_type\": \"keyboard_enter\"}`\n"
        "{\n"
        "    \"name\": \"keyboard_enter\",\n"
        "    \"description\": \"Press the Enter key.\",\n"
        "    \"parameters\": {\n"
        "        \"type\": \"object\",\n"
        "        \"properties\": {},\n"
        "        \"required\": []\n"
        "    }\n"
        "}\n"
        "\n"
        "### navigate_home\n"
        "\n"
        "Calling rule: `{\"action_type\": \"navigate_home\"}`\n"
        "{\n"
        "    \"name\": \"navigate_home\",\n"
        "    \"description\": \"Navigate to the home screen.\",\n"
        "    \"parameters\": {\n"
        "        \"type\": \"object\",\n"
        "        \"properties\": {},\n"
        "        \"required\": []\n"
        "    }\n"
        "}\n"
        "\n"
        "### navigate_back\n"
        "\n"
        "Calling rule: `{\"action_type\": \"navigate_back\"}`\n"
        "{\n"
        "    \"name\": \"navigate_back\",\n"
        "    \"description\": \"Navigate back.\",\n"
        "    \"parameters\": {\n"
        "        \"type\": \"object\",\n"
        "        \"properties\": {},\n"
        "        \"required\": []\n"
        "    }\n"
        "}\n"
        "\n"
        "### swipe\n"
        "\n"
        "Calling rule: `{\"action_type\": \"swipe\", \"direction\": \"<up|down|left|right>\", \"box_2d\": "
        "[[xmin,ymin,xmax,ymax]](optional)}`\n"
        "{\n"
        "    \"name\": \"swipe\",\n"
        "    \"description\": \"Swipe the screen or a scrollable UI element in one of the four directions.\",\n"
        "    \"parameters\": {\n"
        "        \"type\": \"object\",\n"
        "        \"properties\": {\n"
        "            \"direction\": {\n"
        "                \"type\": \"string\",\n"
        "                \"description\": \"The direction to swipe.\",\n"
        "                \"enum\": [\"up\", \"down\", \"left\", \"right\"]\n"
        "            },\n"
        "            \"box_2d\": {\n"
        "                \"type\": \"array\",\n"
        "                \"description\": \"The box_2d to swipe a specific UI element, leave it empty when swiping "
        "the whole screen.\"\n"
        "            }\n"
        "        },\n"
        "        \"required\": [\n"
        "            \"direction\"\n"
        "        ]\n"
        "    }\n"
        "}\n"
        "\n"
        # "### open_app\n"
        # "\n"
        # "Calling rule: `{\"action_type\": \"open_app\", \"app_name\": \"<name>\"}`\n"
        # "{\n"
        # "    \"name\": \"open_app\",\n"
        # "    \"description\": \"Open an app (nothing will happen if the app is not installed).\",\n"
        # "    \"parameters\": {\n"
        # "        \"type\": \"object\",\n"
        # "        \"properties\": {\n"
        # "            \"app_name\": {\n"
        # "                \"type\": \"string\",\n"
        # f"                \"description\": \"The name of the app to open. Supported apps: {','.join(app_names)}\"\n"
        # "            }\n"
        # "        },\n"
        # "        \"required\": [\n"
        # "            \"app_name\"\n"
        # "        ]\n"
        # "    }\n"
        # "}\n"
        # "\n"
        "### wait\n"
        "\n"
        "Calling rule: `{\"action_type\": \"wait\"}`\n"
        "{\n"
        "    \"name\": \"wait\",\n"
        "    \"description\": \"Wait for the screen to update.\",\n"
        "    \"parameters\": {\n"
        "        \"type\": \"object\",\n"
        "        \"properties\": {},\n"
        "        \"required\": []\n"
        "    }\n"
        "}\n"
        "\n"
        "# Historical Actions and Current Memory\n"
    )

    history_str = ""
    if len(history) == 0:
        history_str = "You just started, no action has been performed yet."
    else:
        for idx, h in enumerate(history):
            history_str += f"Step {idx}:\n{h}\n\n"

    prompt += history_str + "\n"

    prompt += (
        "# Output Format\n"
        "1. Memory: important information you want to remember for the future actions. The memory should be only "
        "contents on the screen that will be used in the future actions. It should satisfy that: you cannnot "
        "determine one or more future actions without this memory. \n"
        "2. Reason: the reason for the action and the memory. Your reason should include, but not limited to:- the "
        "content of the GUI, especially elements that are tightly related to the user goal- the step-by-step thinking"
        " process of how you come up with the new action. \n"
        "3. Action: the action you want to take, in the correct JSON format. The action should be one of the above "
        "list.\n"
        "\n"
        "Your answer should look like:\n"
        "Memory: ...\n"
        "Reason: ...\n"
        "Action: {\"action_type\":...}\n"
        "\n"
        "# Some Additional Notes\n"
        "General:\n"
        "- Usually there will be multiple ways to complete a task, pick the easiest one. Also when something does not"
        " work as expected (due to various reasons), sometimes a simple retry can solve the problem, but if it "
        "doesn't (you can see that from the history), SWITCH to other solutions.\n"
        "- Sometimes you may need to navigate the phone to gather information needed to complete the task, for "
        "example if user asks \"what is my schedule tomorrow\", then you may want to open the calendar app (using the"
        " `open_app` action), look up information there, answer user's question (using the `answer` action) and "
        "finish (using the `status` action with complete as goal_status).\n"
        "- For requests that are questions (or chat messages), remember to use the `answer` action to reply to user "
        "explicitly before finish! Merely displaying the answer on the screen is NOT sufficient (unless the goal is "
        "something like \"show me ...\").\n"
        "- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), you can just "
        "complete the task.\n"
        "- If we say that two items are duplicated, in most cases we require that all of their attributes are exactly"
        " the same, not just the name.\n"
        "Text Related Operations:\n"
        "- Normally to select certain text on the screen: <i> Enter text selection mode by long pressing the area "
        "where the text is, then some of the words near the long press point will be selected (highlighted with two "
        "pointers indicating the range) and usually a text selection bar will also appear with options like `copy`, "
        "`paste`, `select all`, etc. <ii> Select the exact text you need. Usually the text selected from the previous"
        " step is NOT the one you want, you need to adjust the range by dragging the two pointers. If you want to "
        "select all text in the text field, simply click the `select all` button in the bar.\n"
        "- To delete some text: first select the text you want to delete (if you want to delete all texts, just long "
        "press the text field and click the `clear all` button in the text selection bar), then click the backspace "
        "button in the keyboard.\n"
        "- To copy some text: first select the exact text you want to copy, which usually also brings up the text "
        "selection bar, then click the `copy` button in bar.\n"
        "- To paste text into a text box, first long press the text box, then usually the text selection bar will "
        "appear with a `paste` button in it.\n"
        "- When typing into a text field, sometimes an auto-complete dropdown list will appear. This usually "
        "indicating this is a enum field and you should try to select the best match by clicking the corresponding "
        "one in the list.\n"
        "Action Related:\n"
        "- Use the `input_text` action whenever you want to type something (including password) instead of clicking "
        "characters on the keyboard one by one. Sometimes there is some default text in the text field you want to "
        "type in, remember to delete them before typing.\n"
        "- Consider exploring the screen by using the `swipe` action with different directions to reveal additional "
        "content.\n"
        "- The direction parameter for the `swipe` action can be confusing sometimes as it's opposite to swipe, for "
        "example, to view content at the bottom, the `swipe` direction should be set to \"up\". It has been observed "
        "that you have difficulties in choosing the correct direction, so if one does not work, try the opposite as "
        "well.\n"
        "- To open an app if you can not find its icon, you can first press home (if necessary) and swipe up to the "
        "app drawer.\n"
        "- Swipe up means swiping from bottom to top, swipe down means swiping from top to bottom, swipe left means "
        "swiping from right to left, swipe right means swiping from left to right.\n"
        "- Use the `navigate_back` action to close/hide the soft keyboard.\n"
        "App Related:\n"
        "- In the Files app, the grid view may cause file names to be displayed incompletely. You can try switching "
        "to a different view type or use the search function directly.\n"
        "- In the Markor app, the save button is located in the top toolbar and is represented by a floppy disk "
        "icon.\n"
        "- If there are no additional requirements, when you need to add a recipe, you should include as much known "
        "information as possible, rather than only adding a small portion of the information.\n"
        "- When you open the Markor app for the first time, there may be a welcome screen. You should tap the \"right"
        " arrow\" in the bottom right corner and the \"DONE\" button to skip the related information.\n"
        "- To transfer data between different pages and different applications, you can try storing the needed "
        "information in \"Memory\" instead of using the \"Share\" function.\n"
        "- You can make full use of the search function to find your target files within a folder/directory or your "
        "target text in a long document.\n"
        "- You may scroll down or up to visit the full content of a document or a list. The important infomation in "
        "the current list should be stored in the \"Memory\" before scrolling; otherwise you will forget it.\n"
        "-- If a blank area appears at the bottom, or if the content does not change after scrolling down, it means "
        "you have reached the end.\n"
        "- When continuously scrolling through a list to find a specific item, you can briefly record the elements "
        "currently displayed on the screen in \"Memory\" to avoid endlessly scrolling even after reaching the bottom "
        "of the list.\n"
        "- To rename a note in Markor, you should first return to the note list, long press the item to be renamed, "
        "and then click the \"A\" button on the right top corner.\n"
        "- To delete a note in Markor, you should first return to the note list, long press the item to be deleted, "
        "and then click the \"trash bin\" button on the right top corner.\n"
        "- To set up a timer, you should input the digits from left to right. For example, you want to set a timer "
        "for 1 minute and 23 seconds. When you input the first \"1\", the time changes from 00h00m00s to 00h00m01s. "
        "Then, you input the second \"2\", the time changes from 00h00m01s to 00h00m12s. Finally, you input the third"
        " \"3\", the time changes from 00h00m12s to 00h01m23s. Do be confused by the intermediate results.\n"
        "- When adding a bill in Pro Expense, the bill category is a scrollable list. You can scroll through this "
        "list to discover more categories.\n"
        "- The calendar app does not automatically set the duration of an event. You need to manually adjust the "
        "interval between the start time and end time to control the event's duration.\n"
        "- In certain views (such as the month view), the calendar app may not display the full event title. To see "
        "the complete title, you need to switch to the day view or open the event details.\n"
    )

    return prompt


def parse_mobile_response(response):
    pattern = r"Memory:(.*?)Reason:(.*?)Action:(.*)"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None

    memory = match.group(1).strip()
    reason = match.group(2).strip()
    action = match.group(3).strip()

    if "<|begin_of_box|>" in action:
        action = action[
            action.index("<|begin_of_box|>") + len("<|begin_of_box|>"): action.rindex(
                "<|end_of_box|>"
            )
        ]

    parsed_action = None
    if action.startswith("{"):
        parsed_action = json.loads(action)

    return {
        "memory": memory,
        "reason": reason,
        "action": action,
        "parsed_action": parsed_action,
    }


def aitw_2_glm4p5v_action(aitw_action):

    ex_action_type = aitw_action['result_action_type']
    glm_action = {"Action": {}}

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            lift_yx = json.loads(aitw_action['result_lift_yx'])
            touch_yx = json.loads(aitw_action['result_touch_yx'])
            click_y_min = int(touch_yx[0] * 999)
            click_x_min = int(touch_yx[1] * 999)
            click_y_max = int(lift_yx[0] * 999)
            click_x_max = int(lift_yx[1] * 999)
            glm_action["Action"] = {"action_type": "click",
                                    "box_2d": [[click_x_min, click_y_min, click_x_max, click_y_max]]}
        else:
            touch_yx_new = [touch_yx[1], touch_yx[0]]
            lift_yx_new = [lift_yx[1], lift_yx[0]]
            direction = get_direction(touch_yx_new, lift_yx_new)
            glm_action["Action"] = {"action_type": "swipe",
                                           "direction": direction,
                                        "box_2d": [[int(touch_yx[1] * 999), int(touch_yx[0] * 999),
                                                      int(lift_yx[1] * 999), int(lift_yx[0] * 999)]]}

    elif ex_action_type == ActionType.PRESS_BACK:
        glm_action["Action"] = {"action_type": "navigate_back"}

    elif ex_action_type == ActionType.PRESS_HOME:
        glm_action["Action"] = {"action_type": "navigate_home"}

    elif ex_action_type == ActionType.PRESS_ENTER:
        glm_action["Action"] = {"action_type": "keyboard_enter"}

    elif ex_action_type == ActionType.TYPE:
        bbox = aitw_action.get('bbox', "")
        if bbox:
            bbox = json.loads(aitw_action['bbox'])
            bbox = bbox[0]
            click_y_min = int(bbox[0] * 999)
            click_x_min = int(bbox[1] * 999)
            click_y_max = int(bbox[2] * 999)
            click_x_max = int(bbox[3] * 999)
        else:
            lift_yx = json.loads(aitw_action['result_lift_yx'])
            touch_yx = json.loads(aitw_action['result_touch_yx'])
            click_y_min = int(touch_yx[0] * 999)
            click_x_min = int(touch_yx[1] * 999)
            click_y_max = int(lift_yx[0] * 999)
            click_x_max = int(lift_yx[1] * 999)

        glm_action["Action"] = {
                                "action_type": "input_text",
                                "text": aitw_action['result_action_text'],
                                "box_2d": [[click_x_min, click_y_min, click_x_max, click_y_max]],
                                "override": "true"}

    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        glm_action["Action"] = {"action_type": "status", "goal_status": "complete"}

    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        glm_action["Action"] = {"action_type": "status", "goal_status": "infeasible"}

    elif ex_action_type == ActionType.LONG_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        click_y_min = int(touch_yx[0] * 999)
        click_x_min = int(touch_yx[1] * 999)
        click_y_max = int(lift_yx[0] * 999)
        click_x_max = int(lift_yx[1] * 999)
        glm_action["Action"] = {"action_type": "long_press",
                                "box_2d": [[click_x_min, click_y_min, click_x_max, click_y_max]]}

    elif ex_action_type == ActionType.NO_ACTION:
        glm_action["Action"] = {"action_type": "wait"}

    elif ex_action_type == ActionType.OPEN_APP:
        glm_action["Action"] = {"action_type": "open_app", "app_name": aitw_action['result_action_app_name']}

    else:
        print('AiTW Action: ', aitw_action)
        raise NotImplementedError

    return f"""Action: {json.dumps(glm_action["Action"], ensure_ascii=False)}"""


def glm4p5v_2_minicpm(output_text):

    try:
        parsed_response = parse_mobile_response(output_text)
        glm_action = parsed_response["parsed_action"]
        action_name = glm_action['action_type']
    except Exception as e:
        print(f"Error, JSON is NOT valid: {e}")
        return {}

    if action_name == "click":
        box_2d = glm_action["box_2d"]
        x_min, y_min, x_max, y_max = box_2d[0]
        x = (x_min + x_max) // 2
        y = (y_min + y_max) // 2
        x = x / 999 * 1000
        y = y / 999 * 1000
        return {"POINT": [int(x), int(y)]}

    elif action_name == "long_press":
        box_2d = glm_action["box_2d"]
        x_min, y_min, x_max, y_max = box_2d[0]
        x = (x_min + x_max) // 2
        y = (y_min + y_max) // 2
        x = x / 999 * 1000
        y = y / 999 * 1000
        time = 1000
        return {"POINT": [int(x), int(y)], "duration": time}

    elif action_name == "swipe":
        direction = glm_action['direction']
        box_2d = glm_action.get("box_2d", "")
        if box_2d:
            x_min, y_min, x_max, y_max = box_2d[0]
            x = (x_min + x_max) // 2
            y = (y_min + y_max) // 2
            x = x / 999 * 1000
            y = y / 999 * 1000
            return {"POINT": [int(x), int(y)], "to": direction}
        return {"POINT": [500, 500], "to": direction}  # screen center point

    elif action_name == "input_text":
        box_2d = glm_action["box_2d"]
        x_min, y_min, x_max, y_max = box_2d[0]
        x = (x_min + x_max) // 2
        y = (y_min + y_max) // 2
        x = x / 999 * 1000
        y = y / 999 * 1000
        return {"INPUT": {"text": glm_action["text"], "point": [int(x), int(y)]}}

    elif action_name == "navigate_back":
        return {"PRESS": "BACK"}

    elif action_name == "navigate_home":
        return {"PRESS": "HOME"}

    elif action_name == "keyboard_enter":
        return {"PRESS": "ENTER"}

    elif action_name == "status":
        return {"STATUS": "finish"}

    elif action_name == "wait":
        return {"duration": 1000}

    elif action_name == "open_app":
        return {"OPEN_APP": glm_action["app_name"]}
    print("Error, unrecognized action.")
    return {}


def prepare_task_input(step, image_path, history_actions, data_name, _tokenizer, use_vllm):
    query = step['instruction']
    if data_name == 'androidcontrol_low':
        query = step["low_instruction"]
    image = Image.open(image_path)
    images = [image]
    glm_history = []
    for action in history_actions:
        glm_history.append(aitw_2_glm4p5v_action(action))

    prompt = get_mobile_prompt(query, glm_history)
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"<image></image>"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    return step, message, images


def prepare_task_inputs(episode, episode_dir, episode_file, subset, dataset, _tokenizer, use_vllm):

    res = []

    for index, step in enumerate(episode):
        step["category"] = subset
        history = []
        for prev_step in episode[:index]:
            image_path = os.path.join(episode_dir, episode_file, f"{episode_file}_{prev_step['step_id']}.jpeg")
            if not os.path.exists(image_path):
                image_path = image_path.replace(".jpeg", ".png")
                if not os.path.exists(image_path):
                    image_path = image_path.replace(".png", ".jpg")
                    if not os.path.exists(image_path):
                        image_path = prev_step['image_path']
            history_action = {"instruction": prev_step['instruction'],
                              "observation": prev_step.get('observation', ''),
                              "result_action_type": prev_step['result_action_type'],
                              "result_action_text": prev_step['result_action_text'],
                              "result_touch_yx": prev_step['result_touch_yx'],
                              "result_lift_yx": prev_step['result_lift_yx'],
                              "low_instruction": prev_step.get("low_instruction", ''),
                              "image_path": image_path,
                              "result_action_app_name": prev_step.get('result_action_app_name', ''),
                              "bbox": prev_step.get('bbox', ''),
                              "description": prev_step.get('description', '')
                              }
            history.append(history_action)
        image_path = os.path.join(episode_dir, episode_file, f"{episode_file}_{step['step_id']}.jpeg")
        if not os.path.exists(image_path):
            image_path = image_path.replace(".jpeg", ".png")
            if not os.path.exists(image_path):
                image_path = image_path.replace(".png", ".jpg")
                if not os.path.exists(image_path):
                    image_path = step['image_path']

        res.append(prepare_task_input(step, image_path, history, dataset, _tokenizer, use_vllm))

    return res


def run_task_batch(_llm, _tokenizer, batch_tasks, use_vllm):

    batch_steps = []
    batch_inputs = []
    batch_images = []

    for step, messages, images in batch_tasks:
        if use_vllm:
            text_prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append({"prompt": text_prompt, "multi_modal_data": {"image": images}})
        else:
            text_prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append(text_prompt)

        batch_steps.append(step)
        batch_images.append(images)

    if use_vllm:
        sampling_params = SamplingParams(temperature=0.01, max_tokens=8192)
        results = _llm.generate(batch_inputs, sampling_params, use_tqdm=False)
        predict_str = [result.outputs[0].text for result in results]
    else:
        generation_params = {'temperature': 0.01}
        inputs = _tokenizer(
            text=batch_inputs, images=batch_images, padding=True, return_tensors="pt", return_token_type_ids=False)
        inputs = inputs.to(_llm.device)
        generated_ids = _llm.generate(**inputs, max_new_tokens=8192, **generation_params)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        predict_str = _tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    res_steps = []
    for step, str_res, images in zip(batch_steps, predict_str, batch_images):
        try:
            step['pred'] = glm4p5v_2_minicpm(str_res)
        except Exception as e:
            print(f"Error, JSON is NOT valid: {e}")
            step['pred'] = {}
        res_steps.append(step)
    return res_steps
