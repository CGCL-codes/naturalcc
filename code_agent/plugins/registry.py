from typing import Dict, List, Optional, Type

from .base import FeaturePlugin, FeatureMetadata


class FeatureRegistry:
    _instance = None
    _plugins: Dict[str, FeaturePlugin] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins = {}
        return cls._instance

    def register(self, plugin: FeaturePlugin) -> None:
        name = plugin.metadata.name
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' already registered")
        self._plugins[name] = plugin

    def get(self, name: str) -> Optional[FeaturePlugin]:
        return self._plugins.get(name)

    def list_plugins(self) -> List[FeatureMetadata]:
        return [p.metadata for p in self._plugins.values()]

    def get_schemas(self) -> Dict[str, List[Dict]]:
        result = {}
        for name, plugin in self._plugins.items():
            result[name] = [
                {
                    "name": f.name,
                    "label": f.label,
                    "type": f.type.value,
                    "required": f.required,
                    "default": f.default,
                    "placeholder": f.placeholder,
                    "help_text": f.help_text,
                    "options": f.options,
                    "accept": f.accept,
                    "multiple": f.multiple,
                }
                for f in plugin.config_schema
            ]
        return result


registry = FeatureRegistry()


def register_plugin(plugin_class: Type[FeaturePlugin]) -> Type[FeaturePlugin]:
    registry.register(plugin_class())
    return plugin_class
