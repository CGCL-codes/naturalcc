from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional


class ExecutionMode(str, Enum):
    AIDER = "aider"
    DIRECT = "direct"
    HYBRID = "hybrid"


class ConfigFieldType(str, Enum):
    TEXT = "text"
    TEXTAREA = "textarea"
    SELECT = "select"
    SWITCH = "switch"
    FILE = "file"


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


@dataclass
class FeatureMetadata:
    name: str
    label: str
    description: str
    icon: Optional[str] = None
    execution_mode: ExecutionMode = ExecutionMode.AIDER


class PluginResult:
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


class FeaturePlugin(ABC):
    @property
    @abstractmethod
    def metadata(self) -> FeatureMetadata:
        pass

    @property
    @abstractmethod
    def config_schema(self) -> List[ConfigField]:
        pass

    def validate(self, config: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> Optional[str]:
        for field in self.config_schema:
            if field.required and not config.get(field.name):
                return f"{field.label} is required"
        return None

    @abstractmethod
    def execute(self, context: "ExecutionContext") -> Generator[str, None, None]:
        pass

    def preview(self, context: "ExecutionContext") -> str:
        return "No preview available."


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
