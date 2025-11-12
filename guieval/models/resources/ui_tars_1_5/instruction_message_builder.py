import jinja2

from guieval.models.resources.loc import MODEL_RES_BASE

_PROMPT_TEMPLATE_PATH = (MODEL_RES_BASE / 'ui_tars_1_5' / 'prompt_template.j2')

_PROMPT_TEMPLATE: jinja2.Template = jinja2.Template(_PROMPT_TEMPLATE_PATH.read_text())


def build(instruction: str, *,
          language: str | None = 'Chinese',
          enable_think: bool = True) -> dict:
    prompt = _PROMPT_TEMPLATE.render(instruction=instruction, language=language, enable_think=enable_think)
    return dict(role='user',
                content=prompt)
