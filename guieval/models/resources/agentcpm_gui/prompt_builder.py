import jinja2

from guieval.models.resources.loc import MODEL_RES_BASE

_PROMPT_TEMPLATE_PATH = (MODEL_RES_BASE / 'agentcpm_gui' / 'prompt_template.j2')

_PROMPT_TEMPLATE: jinja2.Template = jinja2.Template(_PROMPT_TEMPLATE_PATH.read_text())


def build(instruction: str, *, image_count: int = 1) -> str:
    return _PROMPT_TEMPLATE.render(instruction=instruction, image_count=image_count)
