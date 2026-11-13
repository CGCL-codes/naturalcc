from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional


# [maint-note:819fe2fb6e37]
# 数据与行为导航：ExecutionMode。
# 继承接口：str, Enum。
# [/maint-note:819fe2fb6e37]
class ExecutionMode(str, Enum):
    AIDER = "aider"
    DIRECT = "direct"
    HYBRID = "hybrid"


# [maint-note:1bba505f77c5]
# 数据与行为导航：ConfigFieldType。
# 继承接口：str, Enum。
# [/maint-note:1bba505f77c5]
class ConfigFieldType(str, Enum):
    TEXT = "text"
    TEXTAREA = "textarea"
    SELECT = "select"
    SWITCH = "switch"
    FILE = "file"


# [maint-note:91c83df9ef0c]
# 数据与行为导航：ConfigField。
# 字段 name：str。
# 字段 label：str。
# 字段 type：ConfigFieldType。
# 字段 required：bool；初始表达式 False。
# 字段 default：Any；初始表达式 None。
# 字段 placeholder：str；初始表达式 ''。
# 字段 help_text：str；初始表达式 ''。
# 字段 options：Optional[List[Dict[str, str]]]；初始表达式 None。
# 字段 accept：Optional[str]；初始表达式 None。
# 字段 multiple：bool；初始表达式 False。
# [/maint-note:91c83df9ef0c]
@dataclass
class ConfigField:
    name: str
    label: str
    type: ConfigFieldType
    required: bool = False
    default: Any = None
    placeholder: str = ""
    help_text: str = ""
    options: Optional[List[Dict[str, str]]] = None
    accept: Optional[str] = None
    multiple: bool = False


# [maint-note:dff17f504567]
# 数据与行为导航：FeatureMetadata。
# 字段 name：str。
# 字段 label：str。
# 字段 description：str。
# 字段 icon：Optional[str]；初始表达式 None。
# 字段 execution_mode：ExecutionMode；初始表达式 ExecutionMode.AIDER。
# [/maint-note:dff17f504567]
@dataclass
class FeatureMetadata:
    name: str
    label: str
    description: str
    icon: Optional[str] = None
    execution_mode: ExecutionMode = ExecutionMode.AIDER


class PluginResult:
    # [maint-note:fed8072924ba]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 success：bool。
    # 输入 message：str；缺省表达式 ''。
    # 输入 log：Optional[str]；缺省表达式 None。
    # 输入 files_modified：Optional[List[str]]；缺省表达式 None。
    # 输入 report：Optional[str]；缺省表达式 None。
    # 输入 artifacts：Optional[Dict[str, Any]]；缺省表达式 None。
    # 写入目标：self.success。
    # 写入目标：self.message。
    # 写入目标：self.log。
    # 写入目标：self.files_modified。
    # 写入目标：self.report。
    # 写入目标：self.artifacts。
    # [/maint-note:fed8072924ba]
    def __init__(
        self,
        success: bool,
        message: str = "",
        log: Optional[str] = None,
        files_modified: Optional[List[str]] = None,
        report: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.message = message
        self.log = log or ""
        self.files_modified = files_modified or []
        self.report = report
        self.artifacts = artifacts or {}


# [maint-note:e6e6573ec11b]
# 数据与行为导航：FeaturePlugin。
# 继承接口：ABC。
# [/maint-note:e6e6573ec11b]
class FeaturePlugin(ABC):
    # [maint-note:7c891f6f0eda]
    # 维护导航：metadata；下列为本函数源码中的接口与路径索引。
    # 返回契约：FeatureMetadata。
    # [/maint-note:7c891f6f0eda]
    @property
    @abstractmethod
    def metadata(self) -> FeatureMetadata:
        pass

    # [maint-note:3f1c0eda67bf]
    # 维护导航：config_schema；下列为本函数源码中的接口与路径索引。
    # 返回契约：List[ConfigField]。
    # [/maint-note:3f1c0eda67bf]
    @property
    @abstractmethod
    def config_schema(self) -> List[ConfigField]:
        pass

    # [maint-note:84c36fc94c4f]
    # 维护导航：validate；下列为本函数源码中的接口与路径索引。
    # 输入 config：Dict[str, Any]。
    # 输入 files：Optional[Dict[str, Any]]；缺省表达式 None。
    # 返回契约：Optional[str]。
    # 分支条件：field.required and (not config.get(field.name))。
    # 遍历边界：field ← self.config_schema。
    # 返回路径：f'{field.label} is required'。
    # 返回路径：None。
    # 直接调用：config.get。
    # [/maint-note:84c36fc94c4f]
    def validate(self, config: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> Optional[str]:
        for field in self.config_schema:
            if field.required and not config.get(field.name):
                return f"{field.label} is required"
        return None

    # [maint-note:87908aa264a7]
    # 维护导航：execute；下列为本函数源码中的接口与路径索引。
    # 输入 context：'ExecutionContext'。
    # 返回契约：Generator[str, None, None]。
    # [/maint-note:87908aa264a7]
    @abstractmethod
    def execute(self, context: "ExecutionContext") -> Generator[str, None, None]:
        pass

    # [maint-note:821e8376619b]
    # 维护导航：preview；下列为本函数源码中的接口与路径索引。
    # 输入 context：'ExecutionContext'。
    # 返回契约：str。
    # 返回路径：'No preview available.'。
    # [/maint-note:821e8376619b]
    def preview(self, context: "ExecutionContext") -> str:
        return "No preview available."


# [maint-note:03a5baf059a5]
# 数据与行为导航：ExecutionContext。
# 字段 project_dir：str。
# 字段 target_files：List[str]。
# 字段 instruction：str。
# 字段 model：str。
# 字段 api_key：Optional[str]。
# 字段 feature_config：Dict[str, Any]；初始表达式 field(default_factory=dict)。
# 字段 uploaded_files：Dict[str, Any]；初始表达式 field(default_factory=dict)。
# 字段 symbol：Optional[str]；初始表达式 None。
# 字段 completion_type：Optional[str]；初始表达式 None。
# 字段 prefix：str；初始表达式 ''。
# 字段 base_url：Optional[str]；初始表达式 None。
# [/maint-note:03a5baf059a5]
@dataclass
class ExecutionContext:
    project_dir: str
    target_files: List[str]
    instruction: str
    model: str
    api_key: Optional[str]
    feature_config: Dict[str, Any] = field(default_factory=dict)
    uploaded_files: Dict[str, Any] = field(default_factory=dict)
    # Backward compatibility fields
    symbol: Optional[str] = None
    completion_type: Optional[str] = None
    prefix: str = ""
    base_url: Optional[str] = None
