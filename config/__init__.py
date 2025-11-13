import json

from config.config_utils import CONFIG_BASE, model_config_handler

MODEL_PATH_FILE = CONFIG_BASE / 'model_paths.json'
MODEL_PATHS: dict = json.loads(MODEL_PATH_FILE.read_text())


__all__ = ['CONFIG_BASE',
           'model_config_handler',
           'MODEL_PATH_FILE',
           'MODEL_PATHS']
