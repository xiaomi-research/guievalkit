import jinja2

from guieval.models.resources.loc import MODEL_RES_BASE

_PROMPT_TEMPLATE_PATH = (MODEL_RES_BASE / 'glm_4_5v' / 'prompt_template.j2')

_PROMPT_TEMPLATE: jinja2.Template = jinja2.Template(_PROMPT_TEMPLATE_PATH.read_text())


def build(instruction: str, history: list[str], *,
          supported_apps: list[str] | None = None) -> str:
    return _PROMPT_TEMPLATE.render(task=instruction, history=history,
                                   supported_apps=supported_apps)
