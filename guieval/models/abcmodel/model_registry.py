import json
from typing import Dict, Any, Sequence
import threading
from typing import Type

# subsec internal
from guieval.models.abcmodel.abcmodel import ABCModel


class ModelRegistry:
    '''
    Registry for managing models, patterns, configurations and their methods.
    '''
    _models: Dict[str, ABCModel] = dict()
    _lock = threading.Lock()

    def __class_getitem__(cls, key: str) -> ABCModel:
        return cls._models[key]

    @classmethod
    def register(cls, alias: str | Sequence[str] | None = None):
        '''
        '''
        def wrapper(model_cls: Type[ABCModel]):
            with cls._lock:
                if alias is None:
                    aliases = model_cls.NAMES
                elif isinstance(alias, str):
                    aliases = [alias, ]
                elif (not isinstance(alias, Sequence) or
                      any((not isinstance(_alias, str)) for _alias in alias)):
                    raise ValueError()
                for _alias in aliases:
                    cls._models[_alias] = model_cls

            return model_cls

        return wrapper

    @classmethod
    def keys(cls):
        return cls._models.keys()

    @classmethod
    def values(cls):
        return cls._models.values()

    @classmethod
    def items(cls):
        return cls._models.items()

    @classmethod
    def dump_registry(cls) -> str:
        repr_dict = dict(
            (_model, _core.__qualname__)
            for _model, _core in cls._models.items()
        )
        return 'registry: dict[model_name, model_core] =\n' + json.dumps(repr_dict, ensure_ascii=False, indent=4)

    @classmethod
    def unregister(cls, alias: str | Sequence[str] | None = None):
        '''
        '''
        with cls._lock:
            if isinstance(alias, str):
                alias = [alias, ]
            elif (not isinstance(alias, Sequence) or
                  any((not isinstance(_alias, str)) for _alias in alias)):
                raise ValueError()

            models = [cls._models.pop(_alias, None)
                      for _alias in alias]

        return models

    @classmethod
    def get_registered_models(cls) -> list[str]:
        '''
        '''
        return list(cls._models.keys())

    @classmethod
    def get_model(cls, name: str, default: Any) -> Dict[str, Any]:
        '''
        '''
        return cls._models.get(name, default)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        '''
        '''
        return name in cls._models

    @classmethod
    def clear(cls):
        '''
        '''
        with cls._lock:
            cls._models.clear()
        return cls


__all__ = [
    'ModelRegistry',
]
