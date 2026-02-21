#/home/yoshi/PycharmProjects/Reservoir/src/reservoir/core/presets.py
from __future__ import annotations

from enum import Enum
from typing import TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Enum)


class StrictRegistry[K: Enum, T]:
    """
    V2 Strict Registry.
    Maps typed Enums (K) to Config Objects (T).
    No strings, no aliases, no normalization.
    """

    def __init__(self, items: dict[K, T]):
        self._items = dict(items)

    def get(self, key: K) -> T | None:
        if not isinstance(key, Enum):
            raise TypeError(f"Registry lookup requires an Enum, got {type(key)}")
        return self._items.get(key)

    def __getitem__(self, key: K) -> T:
        if key not in self._items:
            raise KeyError(f"Key {key} not found in registry.")
        return self._items[key]

    def register(self, key: K, item: T) -> None:
        if not isinstance(key, Enum):
            raise TypeError(f"Registry key must be an Enum, got {type(key)}")
        self._items[key] = item

    @property
    def available_keys(self) -> list[K]:
        return list(self._items.keys())


__all__ = ["StrictRegistry"]
