import importlib.resources as res
from typing import Literal


MAIN_BASE = res.files('guieval.main')
MAIN_RESOURCES = res.files('guieval.main.resources')
DATA_BASE = res.files('data')


CONTENT_SOURCE = Literal['online',
                         'online_pos',
                         'online_neg',
                         'offline_rule',
                         'offline_model',  # not implemented
                         'low_instruction']


DATASET = Literal["androidcontrol_low",
                  "androidcontrol_high",
                  "cagui_agent",
                  "gui_odyssey",
                  "hypertrack",
                  "aitz"]
