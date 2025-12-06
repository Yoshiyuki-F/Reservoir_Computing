from __future__ import annotations

from typing import Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


class PresetRegistry(Generic[T]):
    """Unified registry for configuration presets with alias handling."""

    def __init__(self, items: Dict[str, T], aliases: Optional[Dict[str, str]] = None):
        self._items = dict(items)
        self._aliases = dict(aliases or {})

    def normalize_name(self, name: str) -> str:
        """Normalize preset names to a canonical key."""
        key = str(name).strip().lower()
        return self._aliases.get(key, key)

    def get(self, name: str) -> Optional[T]:
        """Retrieve a preset by name or alias."""
        key = self.normalize_name(name)
        return self._items.get(key)

    def get_or_default(self, name: str, default_key: str) -> T:
        """Retrieve a preset, falling back to a default key."""
        item = self.get(name)
        if item is not None:
            return item
        return self._items[default_key]

    def register(self, name: str, item: T, aliases: Optional[List[str]] = None) -> None:
        """Dynamically register a new preset with optional aliases."""
        key = self.normalize_name(name)
        self._items[key] = item
        if aliases:
            for alias in aliases:
                self._aliases[self.normalize_name(alias)] = key

    @property
    def available_keys(self) -> List[str]:
        return list(self._items.keys())

    def list_keys(self) -> List[str]:
        """Backward-compatible accessor for registry keys."""
        return self.available_keys


__all__ = ["PresetRegistry"]
