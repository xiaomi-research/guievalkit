import jinja2

from guieval.models.resources.loc import MODEL_RES_BASE
from guieval.main import StepTaskModel

_PROMPT_TEMPLATE_PATH = (MODEL_RES_BASE / 'mimo_vl' / 'prompt_template.j2')

_PROMPT_TEMPLATE: jinja2.Template = jinja2.Template(_PROMPT_TEMPLATE_PATH.read_text())


def build(step_task: StepTaskModel, history: str) -> str:
    return _PROMPT_TEMPLATE.render(step_task=step_task, history=history)
