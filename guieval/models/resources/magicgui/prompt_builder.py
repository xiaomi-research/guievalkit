from typing import Literal
import jinja2

from guieval.models.resources.loc import MODEL_RES_BASE

_PROMPT_TEMPLATE_PATH = (MODEL_RES_BASE / 'magicgui' / 'prompt_template.j2')
_PROMPT_TEMPLATE_CN_PATH = (MODEL_RES_BASE / 'magicgui' / 'prompt_template_cn.j2')

_PROMPT_TEMPLATE: jinja2.Template = jinja2.Template(_PROMPT_TEMPLATE_PATH.read_text())
_PROMPT_TEMPLATE_CN: jinja2.Template = jinja2.Template(_PROMPT_TEMPLATE_CN_PATH.read_text())


def build(instruction: str,
          previous_actions: list[str],
          language: Literal["English", "Chinese"] = "English") -> str:
    if language == "English":
        return _PROMPT_TEMPLATE.render(task=instruction,
                                       history='\n'.join(previous_actions) + '\n')
    elif language == "Chinese":
        return _PROMPT_TEMPLATE_CN.render(task=instruction,
                                          history='\n'.join(previous_actions) + '\n')
    else:
        raise ValueError(f"Invalid language: {language}")

    return
