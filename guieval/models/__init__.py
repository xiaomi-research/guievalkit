from guieval.models.utils import ModelRegistry

# register models
import guieval.models.gui_owl as gui_owl
import guieval.models.ui_venus_navi as ui_venus_navi
import guieval.models.qwen3_vl as qwen3_vl
import guieval.models.qwen2_5_vl as qwen2_5_vl
import guieval.models.glm_4_5v as glm_4_5v
import guieval.models.ui_tars_1_5 as ui_tars_1_5
import guieval.models.mimo_vl as mimo_vl


__all__ = ['ModelRegistry',
           'gui_owl',
           'ui_venus_navi',
           'qwen3_vl',
           'qwen2_5_vl',
           'glm_4_5v',
           'ui_tars_1_5',
           'mimo_vl']
