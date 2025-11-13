import threading
from typing import Callable, Generic, Hashable, TypeVar

KT = TypeVar('KeyType', bound=Hashable)
VT = TypeVar('ValueType')


class ThreadSafeMemory(Generic[KT, VT]):
    '''
    default_factory is used to create a default value for a key if the key is not found in the memory.
    if default_factory is not provided, KeyError will be raised.
    '''
    def __init__(self, *, default_factory: Callable = None):
        self._dict = {}
        self._lock = threading.Lock()
        self._default_factory = default_factory

    def __getitem__(self, key):
        with self._lock:
            if self._default_factory and self.get(key) is None:
                self._dict[key] = self._default_factory()
            return self._dict[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def get(self, key, default=None):
        return self._dict.get(key, default)
