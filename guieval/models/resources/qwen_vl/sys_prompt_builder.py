import jinja2

from guieval.models.resources.loc import MODEL_RES_BASE

_PROMPT_TEMPLATE_PATH = (MODEL_RES_BASE / 'qwen_vl' / 'sys_prompt_template.j2')

_SYS_PROMPT_TEMPLATE: jinja2.Template = jinja2.Template(_PROMPT_TEMPLATE_PATH.read_text())


def build(enable_conclude: bool = True, enable_think: bool = True, *,
          height: int | None = None, width: int | None = None) -> str:
    return _SYS_PROMPT_TEMPLATE.render(enable_conclude=enable_conclude, enable_think=enable_think,
                                       height=height, width=width)
