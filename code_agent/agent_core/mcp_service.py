from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import Unresolvable

from .contracts import RiskLevel, ToolContext, ToolResult, ToolSpec
from .mcp_stdio import McpRpcError, McpTransportError, StdioMcpClient
from .tool_registry import redact_sensitive_text, redact_sensitive_value


_SERVER_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,22}[A-Za-z0-9])?")
_REMOTE_NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SUFFIX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_RESOURCE_URI = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
_RESOURCE_NAMES = {"resources_list", "resources_read"}
_MAX_CONFIG_BYTES = 64 * 1024
_MAX_SCHEMA_BYTES = 8 * 1024
_MAX_ARGUMENT_BYTES = 16 * 1024
_MAX_RESULT_BYTES = 8 * 1024
_MAX_CATALOG_ENTRIES = 128
_MAX_CATALOG_PAGES = 16
_RESOURCE_PAGE_SIZE = 12
_OFFLINE_REGISTRY = Registry()


class McpConfigError(ValueError):
    pass


@dataclass(frozen=True)
class _ServerConfig:
    name: str
    argv: tuple[str, ...]
    env_allowlist: tuple[str, ...]
    read_only_tools: frozenset[str]
    allowed_tools: frozenset[str] | None
    resources: bool
    enabled: bool
    timeout_seconds: float


@dataclass
class _Server:
    config: _ServerConfig
    lock: Any = field(default_factory=threading.RLock)
    client: StdioMcpClient | None = None
    generation: str = ""
    resource_key: bytes = b""
    secrets: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()
    entries: dict[str, tuple[str, ToolSpec]] = field(default_factory=dict)
    resources: dict[str, tuple[str, dict[str, str]]] = field(default_factory=dict)
    error: str | None = None
    connect_failed: bool = False


@dataclass(frozen=True)
class _Dispatch:
    client: StdioMcpClient
    params: dict[str, Any]
    secrets: tuple[str, ...]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise McpConfigError("MCP configuration contains duplicate keys.")
    return value


def _string_list(value: Any, *, limit: int, pattern: re.Pattern[str] | None = None) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > limit or any(
        not isinstance(item, str) or len(item) > 4_096 or "\0" in item
        or (pattern is not None and pattern.fullmatch(item) is None)
        for item in value
    ):
        raise McpConfigError("MCP server configuration contains an invalid list.")
    return tuple(value)


def _parse_server(name: str, value: Any) -> _ServerConfig:
    if not _SERVER_NAME.fullmatch(name) or "__" in name:
        raise McpConfigError("MCP configuration contains an invalid server name.")
    known_keys = {"command", "args", "env_allowlist", "read_only_tools", "allowed_tools", "resources", "enabled", "timeout_seconds"}
    if not isinstance(value, dict) or value.keys() - known_keys:
        raise McpConfigError("MCP server configuration accepts only the supported stdio settings.")
    command = value.get("command")
    if not isinstance(command, str) or not Path(command).is_absolute() or "\0" in command:
        raise McpConfigError("MCP command must be an absolute path to an executable.")
    args = _string_list(value.get("args", []), limit=64)
    env_allowlist = _string_list(value.get("env_allowlist", []), limit=32, pattern=_ENV_NAME)
    read_only = _string_list(value.get("read_only_tools", []), limit=_MAX_CATALOG_ENTRIES, pattern=_REMOTE_NAME)
    allowed = value.get("allowed_tools")
    allowed_tools = None if "allowed_tools" not in value else frozenset(
        _string_list(allowed, limit=_MAX_CATALOG_ENTRIES, pattern=_REMOTE_NAME)
    )
    resources, enabled = value.get("resources", False), value.get("enabled", True)
    timeout = value.get("timeout_seconds", 10)
    if not isinstance(resources, bool) or not isinstance(enabled, bool):
        raise McpConfigError("MCP enabled and resources settings must be boolean.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0.1 <= timeout <= 60:
        raise McpConfigError("MCP timeout_seconds must be between 0.1 and 60.")
    return _ServerConfig(name, (command, *args), env_allowlist, frozenset(read_only), allowed_tools,
                         resources, enabled, float(timeout))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _redact_exact(value: str, exact: tuple[str, ...]) -> str:
    for secret in exact:
        value = value.replace(secret, "[REDACTED]")
    return value


def _redact_text(value: str, exact: tuple[str, ...]) -> str:
    return redact_sensitive_text(_redact_exact(value, exact))


def _safe_description(value: Any, exact: tuple[str, ...], limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return _redact_text(value, exact)[:limit]


def _qualified_name(server: str, remote: str) -> str:
    stem = f"mcp__{server}__"
    if _SUFFIX.fullmatch(remote) and remote not in _RESOURCE_NAMES and len(stem + remote) <= 64:
        return stem + remote
    # Encoded names have a reserved leading underscore; ordinary names never use it.
    return stem + "_x" + hashlib.sha256(remote.encode("utf-8")).hexdigest()[:24]


def _check_schema(value: Any, exact: tuple[str, ...], depth: int = 0, *, schema_map: bool = False) -> None:
    if depth > 16:
        raise ValueError("MCP input schema exceeds the supported nesting depth.")
    if isinstance(value, str) and _redact_text(value, exact) != value:
        raise ValueError("MCP input schema contains a credential.")
    if isinstance(value, dict):
        for key, child in value.items():
            if _redact_text(key, exact) != key:
                raise ValueError("MCP input schema contains a credential.")
            if not schema_map and key in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                not isinstance(child, str) or not child.startswith("#")
            ):
                raise ValueError("MCP input schema references must be local.")
            if not schema_map and key in {"pattern", "patternProperties"}:
                raise ValueError("MCP input schema regex validation is unsupported.")
            _check_schema(child, exact, depth + 1, schema_map=not schema_map and key in {
                "properties", "$defs", "definitions", "dependentSchemas",
            })
    if isinstance(value, list):
        for child in value:
            _check_schema(child, exact, depth + 1)


def _validator(schema: Any, exact: tuple[str, ...]) -> Draft202012Validator:
    if not isinstance(schema, dict) or schema.get("type") != "object" or len(_json_bytes(schema)) > _MAX_SCHEMA_BYTES:
        raise ValueError("MCP tool requires a bounded object input schema.")
    _check_schema(schema, exact)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=_OFFLINE_REGISTRY)


def _validate_arguments(arguments: dict[str, Any], validator: Draft202012Validator) -> None:
    try:
        if len(_json_bytes(arguments)) > _MAX_ARGUMENT_BYTES:
            raise ValueError("MCP request exceeds its argument limit.")
        validator.validate(arguments)
    except (ValueError, TypeError, ValidationError, Unresolvable, OverflowError, RecursionError):
        raise ValueError("MCP arguments do not match the configured tool schema.") from None


def _bounded_json(value: Any, exact: tuple[str, ...], depth: int = 0) -> Any:
    if depth > 12:
        return "[Nested content omitted]"
    if isinstance(value, str):
        return _redact_text(value, exact).encode("utf-8", errors="replace").decode("utf-8")
    if type(value) in (int, float) and str(value) in exact:
        return "[REDACTED]"
    if isinstance(value, list):
        return [_bounded_json(item, exact, depth + 1) for item in value]
    if isinstance(value, dict):
        binary = value.get("type") in ("image", "audio")
        return {
            _redact_text(key, exact).encode("utf-8", errors="replace").decode("utf-8"): (
                "[Binary content omitted]" if key == "blob" or (binary and key == "data")
                else _bounded_json(item, exact, depth + 1)
            )
            for key, item in value.items()
        }
    return value


def _safe_result(server: str, method: str, result: dict[str, Any], exact: tuple[str, ...]) -> ToolResult:
    safe = redact_sensitive_value(_bounded_json(result, exact))
    encoded = _json_bytes(safe)
    data = {"mcp_server": server, "mcp_method": method}
    truncated = len(_json_bytes({**data, "result": safe})) > _MAX_RESULT_BYTES
    if truncated:
        # The preview is quoted as JSON again when emitted as a tool result.
        data["preview"] = encoded[:_MAX_RESULT_BYTES // 2 - 128].decode("utf-8", errors="ignore")
    else:
        data["result"] = safe
    if result.get("isError") is True:
        summary = "The MCP server reported a tool failure."
        return ToolResult("error", summary, data=data, truncated=truncated,
                          error={"type": "McpToolError", "message": summary})
    return ToolResult("success", "MCP response received.", data=data, truncated=truncated)


def _time_left(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise McpTransportError("MCP operation reached its timeout.")
    return remaining


def _list_catalog(client: StdioMcpClient, method: str, key: str, deadline: float) -> list[dict[str, Any]]:
    entries = []
    cursor = None
    seen = set()
    for _ in range(_MAX_CATALOG_PAGES):
        page = client.request(method, {"cursor": cursor} if cursor else None, timeout=_time_left(deadline))
        page_entries = page.get(key)
        if not isinstance(page_entries, list) or any(not isinstance(entry, dict) for entry in page_entries):
            raise McpTransportError("MCP server returned an invalid catalog page.")
        entries.extend(page_entries)
        if len(entries) > _MAX_CATALOG_ENTRIES:
            raise McpTransportError("MCP catalog exceeds the supported entry count.")
        cursor = page.get("nextCursor")
        if cursor is None:
            return entries
        if not isinstance(cursor, str) or not cursor or len(cursor) > 2_048 or cursor in seen:
            raise McpTransportError("MCP server returned an invalid catalog cursor.")
        seen.add(cursor)
    raise McpTransportError("MCP catalog exceeds the supported page count.")


class McpService:
    """Own configured stdio processes and local authorization for their catalogs."""

    def __init__(self, servers: list[_ServerConfig]) -> None:
        self._servers = {config.name: _Server(config) for config in servers}
        self._closed = False

    @classmethod
    def from_file(cls, path: str | Path) -> McpService:
        try:
            with Path(path).expanduser().open("rb") as file:
                raw = file.read(_MAX_CONFIG_BYTES + 1)
            if len(raw) > _MAX_CONFIG_BYTES:
                raise McpConfigError("MCP configuration exceeds the supported size.")
            payload = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_strict_object)
        except McpConfigError:
            raise
        except (OSError, UnicodeError, ValueError, RecursionError):
            raise McpConfigError("Could not read trusted MCP configuration.") from None
        if not isinstance(payload, dict) or set(payload) != {"mcpServers"}:
            raise McpConfigError("MCP configuration must contain a mcpServers object.")
        servers = payload["mcpServers"]
        if not isinstance(servers, dict) or len(servers) > 16:
            raise McpConfigError("MCP configuration supports up to 16 servers.")
        configs = [_parse_server(name, value) for name, value in sorted(servers.items())]
        service = cls(configs)
        if any(service.redact(config.name) != config.name for config in configs):
            raise McpConfigError("MCP server names must not contain credentials.")
        return service

    def _server(self, name: str) -> _Server:
        server = self._servers.get(name)
        if server is None:
            raise KeyError("MCP server is not configured.")
        return server

    def _drop(self, server: _Server) -> None:
        client = server.client
        server.client = None
        server.entries = {}
        server.resources = {}
        server.capabilities = frozenset()
        server.secrets = ()
        server.generation = ""
        server.resource_key = b""
        if client is not None:
            client.close()

    def _live(self, server: _Server) -> StdioMcpClient | None:
        if server.client is not None and not server.client.alive:
            self._drop(server)
            server.error = "MCP server disconnected. Reconnect to continue."
        return server.client

    def _status(self, server: _Server) -> dict[str, Any]:
        live = self._live(server)
        current = live is not None and not live.catalog_dirty
        name = self.redact(server.config.name)
        return {
            "name": name,
            "enabled": server.config.enabled,
            "status": ("disabled" if not server.config.enabled else "connected" if live else
                       "error" if server.connect_failed else "disconnected"),
            "tools": sorted(server.entries) if current else [],
            "resources": current and server.config.resources and "resources" in server.capabilities,
            "tool_count": len(server.entries) if current else 0,
            "resource_count": len(server.resources) if current else 0,
            "error": server.error,
        }

    def statuses(self) -> list[dict[str, Any]]:
        result = []
        for server in self._servers.values():
            with server.lock:
                result.append(self._status(server))
        return result

    def redact(self, value: Any) -> Any:
        """Scrub known configured environment values without starting a process."""
        configured = {
            os.environ.get(key) for server in self._servers.values() for key in server.config.env_allowlist
        }
        connected = {value for server in self._servers.values() for value in server.secrets}
        exact = tuple(sorted(filter(None, configured | connected), key=lambda item: (-len(item), item)))

        def scrub(item: Any) -> Any:
            if isinstance(item, str):
                return _redact_exact(item, exact)
            if isinstance(item, dict):
                return {scrub(key): scrub(child) for key, child in item.items()}
            if isinstance(item, list):
                return [scrub(child) for child in item]
            return item

        return scrub(value) if exact else value

    def connect(self, name: str) -> dict[str, Any]:
        server = self._server(name)
        with server.lock:
            if self._closed or not server.config.enabled or self._live(server):
                return self._status(server)
            config = server.config
            env = {key: value for key in config.env_allowlist if (value := os.environ.get(key)) is not None}
            server.secrets = tuple(sorted(set(filter(None, env.values())), key=lambda item: (-len(item), item)))
            server.generation = uuid.uuid4().hex
            server.resource_key = secrets.token_bytes(32)
            deadline = time.monotonic() + config.timeout_seconds
            server.connect_failed = False
            server.error = None
            try:
                if self.redact(config.name) != config.name or _redact_text(config.name, server.secrets) != config.name:
                    raise ValueError("MCP server name contains a credential.")
                client = StdioMcpClient(list(config.argv), env)
                server.client = client
                info = client.initialize(timeout=_time_left(deadline))
                capabilities = info.get("capabilities")
                if not isinstance(capabilities, dict):
                    raise McpTransportError("MCP server did not negotiate capabilities.")
                server.capabilities = frozenset(key for key, value in capabilities.items() if isinstance(value, dict))
                self._refresh(server, client, deadline)
            except (McpTransportError, ValueError, TypeError, OverflowError, RecursionError):
                self._drop(server)
                server.connect_failed = True
                server.error = "MCP initialization or catalog discovery failed."
            return self._status(server)

    def disconnect(self, name: str) -> dict[str, Any]:
        server = self._server(name)
        with server.lock:
            self._drop(server)
            server.error = None
            server.connect_failed = False
            return self._status(server)

    def reconnect(self, name: str) -> dict[str, Any]:
        server = self._server(name)
        with server.lock:
            self.disconnect(name)
            return self.connect(name)

    def close(self) -> None:
        self._closed = True
        for name in self._servers:
            self.disconnect(name)

    def snapshot(self) -> dict[str, str]:
        versions = {}
        for server in self._servers.values():
            with server.lock:
                client = self._live(server)
                if client is None:
                    continue
                if client.catalog_dirty:
                    try:
                        self._refresh(server, client, time.monotonic() + server.config.timeout_seconds)
                    except (McpTransportError, ValueError, TypeError, OverflowError, RecursionError):
                        self._drop(server)
                        server.connect_failed = True
                        server.error = "MCP catalog refresh failed."
                if client.alive and not client.catalog_dirty:
                    versions.update({name: entry[0] for name, entry in sorted(server.entries.items())})
        return versions

    def resolve(self, name: str, version: str) -> ToolSpec | None:
        if not name.startswith("mcp__"):
            return None
        server_name, separator, _ = name[5:].partition("__")
        server = self._servers.get(server_name) if separator else None
        if server is None:
            return None
        with server.lock:
            return self._resolve(server, name, version)

    def _resolve(self, server: _Server, name: str, version: str) -> ToolSpec | None:
        client = self._live(server)
        if self._closed or client is None or client.catalog_dirty:
            return None
        entry = server.entries.get(name)
        return entry[1] if entry is not None and entry[0] == version else None

    def _refresh(self, server: _Server, client: StdioMcpClient, deadline: float) -> None:
        revision = client.catalog_revision
        config = server.config
        tools = _list_catalog(client, "tools/list", "tools", deadline) if "tools" in server.capabilities else []
        entries = {}
        discovered = set()
        for tool in tools:
            remote = tool.get("name")
            if not isinstance(remote, str) or not _REMOTE_NAME.fullmatch(remote):
                raise McpTransportError("MCP catalog contains an invalid tool name.")
            if remote in discovered:
                raise McpTransportError("MCP catalog contains duplicate tool names.")
            discovered.add(remote)
            if config.allowed_tools is not None and remote not in config.allowed_tools:
                continue
            name = _qualified_name(config.name, remote)
            if _redact_text(remote, server.secrets) != remote or _redact_text(name, server.secrets) != name:
                continue
            if name in entries:
                raise McpTransportError("MCP catalog contains duplicate tool identifiers.")
            try:
                validator = _validator(tool.get("inputSchema"), server.secrets)
            except (ValueError, TypeError, SchemaError, RecursionError):
                continue
            risk = RiskLevel.READ if remote in config.read_only_tools else RiskLevel.EXECUTE
            description = _safe_description(tool.get("description"), server.secrets, 1_000)
            description = f"MCP tool from configured server {config.name}. {description}".rstrip()
            entries[name] = self._entry(server, revision, name, description, validator, risk, "tools/call", remote)
        resources = {}
        if config.resources and "resources" in server.capabilities:
            resources = self._resource_catalog(server, _list_catalog(client, "resources/list", "resources", deadline))
            list_name = f"mcp__{config.name}__resources_list"
            read_name = f"mcp__{config.name}__resources_read"
            list_validator = _validator({
                "type": "object", "properties": {"offset": {"type": "integer", "minimum": 0}},
                "additionalProperties": False,
            }, ())
            read_validator = _validator({
                "type": "object", "properties": {"resource_id": {"type": "string", "maxLength": 64}},
                "required": ["resource_id"], "additionalProperties": False,
            }, ())
            resource_ids = tuple(sorted(resources))
            if _redact_text(list_name, server.secrets) == list_name:
                entries[list_name] = self._entry(server, revision, list_name, "List locally discovered MCP resources.",
                                                 list_validator, RiskLevel.READ, "resources/list", resource_ids)
            if _redact_text(read_name, server.secrets) == read_name:
                entries[read_name] = self._entry(server, revision, read_name,
                                                 "Read one discovered MCP resource by its resource_id.", read_validator,
                                                 RiskLevel.NETWORK, "resources/read", resource_ids)
        server.entries = entries
        server.resources = resources
        client.clear_catalog_dirty(revision)

    def _resource_catalog(self, server: _Server, raw: list[dict[str, Any]]) -> dict[str, tuple[str, dict[str, str]]]:
        resources = {}
        for item in raw:
            uri = item.get("uri")
            if not isinstance(uri, str) or len(uri) > 2_048 or not _RESOURCE_URI.match(uri):
                raise McpTransportError("MCP catalog contains an invalid resource URI.")
            resource_id = "resource_" + hmac.new(server.resource_key, uri.encode("utf-8"), "sha256").hexdigest()[:24]
            if resource_id in resources:
                raise McpTransportError("MCP catalog contains duplicate resource identifiers.")
            public = {"resource_id": resource_id}
            for key, limit in (("name", 128), ("description", 240), ("mimeType", 96)):
                if text := _safe_description(item.get(key), (uri, *server.secrets), limit):
                    public[key] = text
            resources[resource_id] = (uri, public)
        return resources

    def _entry(
        self, server: _Server, revision: int, name: str, description: str, validator: Draft202012Validator,
        risk: RiskLevel, method: str, target: str | tuple[str, ...],
    ) -> tuple[str, ToolSpec]:
        policy = server.config
        fingerprint = hashlib.sha256(_json_bytes([
            name, description, validator.schema, risk.value, method, target,
            sorted(policy.read_only_tools), sorted(policy.allowed_tools) if policy.allowed_tools is not None else None,
            policy.resources, policy.timeout_seconds,
        ])).hexdigest()
        version = f"{server.generation}:{revision}:{fingerprint}"

        def validate(arguments: dict[str, Any]) -> None:
            _validate_arguments(arguments, validator)

        def execute(context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
            return self._execute(server, name, version, revision, validator, risk, method, target, context, arguments)

        spec = ToolSpec(name, description, validator.schema, risk, execute,
                        idempotent=risk == RiskLevel.READ, parallel_safe=False,
                        default_timeout_seconds=math.ceil(policy.timeout_seconds), max_output_chars=_MAX_RESULT_BYTES,
                        validate_input=validate)
        return version, spec

    def _execute(
        self, server: _Server, name: str, version: str, revision: int, validator: Draft202012Validator, risk: RiskLevel,
        method: str, target: str | tuple[str, ...], context: ToolContext, arguments: dict[str, Any],
    ) -> ToolResult:
        if risk != RiskLevel.READ and risk not in context.approved_risks:
            return ToolResult.failure("This MCP request requires approval.", "ApprovalRequired")
        try:
            _validate_arguments(arguments, validator)
        except ValueError:
            return ToolResult.failure("MCP arguments do not match the configured tool schema.", "InvalidMcpArguments")
        timeout = server.config.timeout_seconds
        deadline_epoch = context.metadata.get("deadline_epoch")
        if isinstance(deadline_epoch, (float, int)) and not isinstance(deadline_epoch, bool) and math.isfinite(deadline_epoch):
            timeout = min(timeout, deadline_epoch - time.time())
        deadline = time.monotonic() + timeout
        try:
            while not context.cancellation.is_set():
                if server.lock.acquire(timeout=min(0.05, _time_left(deadline))):
                    try:
                        dispatch = self._prepare_dispatch(server, name, version, method, target, context, arguments)
                    finally:
                        server.lock.release()
                    if isinstance(dispatch, ToolResult):
                        return dispatch
                    return self._send(server.config.name, dispatch, revision, risk, method, context, deadline)
        except McpTransportError:
            return ToolResult.failure("The MCP request reached its timeout before dispatch.", "McpTimeout")
        return ToolResult("cancelled", "MCP request was cancelled before dispatch.")

    def _prepare_dispatch(
        self, server: _Server, name: str, version: str, method: str, target: str | tuple[str, ...],
        context: ToolContext, arguments: dict[str, Any],
    ) -> _Dispatch | ToolResult:
        if self._resolve(server, name, version) is None or server.client is None:
            return ToolResult.failure("The MCP connection or tool catalog changed. Start a new run.", "StaleMcpTool")
        if context.cancellation.is_set():
            return ToolResult("cancelled", "MCP request was cancelled before dispatch.")
        if method == "resources/list":
            offset = arguments.get("offset", 0)
            page = list(server.resources.values())[offset : offset + _RESOURCE_PAGE_SIZE]
            next_offset = offset + len(page)
            return _safe_result(server.config.name, method, {
                "resources": [item[1] for item in page],
                "next_offset": next_offset if next_offset < len(server.resources) else None,
            }, server.secrets)
        exact = server.secrets
        if method == "resources/read":
            resource = server.resources.get(arguments["resource_id"])
            if resource is None:
                return ToolResult.failure("MCP resource was not discovered for this server.", "McpResourceNotAuthorized")
            params = {"uri": resource[0]}
            exact = (resource[0], *exact)
        else:
            params = {"name": target, "arguments": arguments}
        return _Dispatch(server.client, params, exact)

    def _send(
        self, server: str, dispatch: _Dispatch, revision: int, risk: RiskLevel, method: str,
        context: ToolContext, deadline: float,
    ) -> ToolResult:
        try:
            result = dispatch.client.request(
                method, dispatch.params, timeout=_time_left(deadline), cancellation=context.cancellation,
                expected_catalog_revision=revision,
            )
        except McpRpcError:
            return ToolResult.failure("MCP server returned an RPC error.", "McpRpcError")
        except McpTransportError as exc:
            if exc.uncertain and risk != RiskLevel.READ:
                summary = "The MCP request was dispatched, but its outcome is uncertain. Verify it before trying again."
                return ToolResult("uncertain", summary, data={"mcp_server": server, "mcp_method": method},
                                  error={"type": "McpOutcomeUncertain", "message": summary})
            if context.cancellation.is_set():
                return ToolResult("cancelled", "MCP request was cancelled.")
            if dispatch.client.catalog_revision != revision:
                return ToolResult.failure("The MCP tool catalog changed before dispatch. Start a new run.", "StaleMcpTool")
            return ToolResult.failure("MCP connection failed or the request timed out.", "McpTransportError")
        return _safe_result(server, method, result, dispatch.secrets)
