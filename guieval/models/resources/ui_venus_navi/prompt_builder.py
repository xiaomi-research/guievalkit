import jinja2

from guieval.models.resources.loc import MODEL_RES_BASE

_PROMPT_TEMPLATE_PATH = (MODEL_RES_BASE / 'ui_venus_navi' / 'prompt_template.j2')

_PROMPT_TEMPLATE: jinja2.Template = jinja2.Template(_PROMPT_TEMPLATE_PATH.read_text())


def build(instruction: str, previous_actions: str, enable_think: bool = False) -> str:
    return _PROMPT_TEMPLATE.render(instruction=instruction,
                                   previous_actions=previous_actions,
                                   enable_think=enable_think)
