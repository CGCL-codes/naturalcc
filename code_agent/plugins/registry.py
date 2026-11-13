from typing import Dict, List, Optional, Type

from .base import FeaturePlugin, FeatureMetadata


# [maint-note:71d1346c75c8]
# 数据与行为导航：FeatureRegistry。
# 字段 _plugins：Dict[str, FeaturePlugin]；初始表达式 {}。
# [/maint-note:71d1346c75c8]
class FeatureRegistry:
    _instance = None
    _plugins: Dict[str, FeaturePlugin] = {}

    # [maint-note:f6d4c020e5e0]
    # 维护导航：__new__；下列为本函数源码中的接口与路径索引。
    # 分支条件：cls._instance is None。
    # 返回路径：cls._instance。
    # 写入目标：cls._instance。
    # 写入目标：cls._instance._plugins。
    # 直接调用：super().__new__。
    # 直接调用：super。
    # [/maint-note:f6d4c020e5e0]
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins = {}
        return cls._instance

    # [maint-note:5484c291d890]
    # 维护导航：register；下列为本函数源码中的接口与路径索引。
    # 输入 plugin：FeaturePlugin。
    # 返回契约：None。
    # 分支条件：name in self._plugins。
    # 显式抛出：ValueError(f"Plugin '{name}' already registered")。
    # 写入目标：self._plugins[name]。
    # 直接调用：ValueError。
    # [/maint-note:5484c291d890]
    def register(self, plugin: FeaturePlugin) -> None:
        name = plugin.metadata.name
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' already registered")
        self._plugins[name] = plugin

    # [maint-note:d5c2fd5eda95]
    # 维护导航：get；下列为本函数源码中的接口与路径索引。
    # 输入 name：str。
    # 返回契约：Optional[FeaturePlugin]。
    # 返回路径：self._plugins.get(name)。
    # 直接调用：self._plugins.get。
    # [/maint-note:d5c2fd5eda95]
    def get(self, name: str) -> Optional[FeaturePlugin]:
        return self._plugins.get(name)

    # [maint-note:b3029bc5ce15]
    # 维护导航：list_plugins；下列为本函数源码中的接口与路径索引。
    # 返回契约：List[FeatureMetadata]。
    # 返回路径：[p.metadata for p in self._plugins.values()]。
    # 直接调用：self._plugins.values。
    # 集合派生：p 来自 self._plugins.values()。
    # [/maint-note:b3029bc5ce15]
    def list_plugins(self) -> List[FeatureMetadata]:
        return [p.metadata for p in self._plugins.values()]

    # [maint-note:4b85ed993420]
    # 维护导航：get_schemas；下列为本函数源码中的接口与路径索引。
    # 返回契约：Dict[str, List[Dict]]。
    # 遍历边界：(name, plugin) ← self._plugins.items()。
    # 返回路径：result。
    # 写入目标：result[name]。
    # 直接调用：self._plugins.items。
    # 集合派生：f 来自 plugin.config_schema。
    # [/maint-note:4b85ed993420]
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


# [maint-note:7426ac2e937c]
# 维护导航：register_plugin；下列为本函数源码中的接口与路径索引。
# 输入 plugin_class：Type[FeaturePlugin]。
# 返回契约：Type[FeaturePlugin]。
# 返回路径：plugin_class。
# 直接调用：registry.register。
# 直接调用：plugin_class。
# [/maint-note:7426ac2e937c]
def register_plugin(plugin_class: Type[FeaturePlugin]) -> Type[FeaturePlugin]:
    registry.register(plugin_class())
    return plugin_class
