import json
import sys
import threading
import time
from pathlib import Path

import pytest

from code_agent.agent_core.contracts import RiskLevel, ToolContext
from code_agent.agent_core.mcp_service import McpConfigError, McpService


_SERVER = r'''
import json
import os
import sys
import time
from pathlib import Path

secret = os.environ.get("NCC_MCP_TEST_TOKEN", "missing")
uri = "notes://one?credential=" + secret
empty = {"type": "object", "additionalProperties": False}
for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    args = request.get("params", {})
    if "id" not in request:
        continue
    if method == "initialize":
        result = {"protocolVersion": args["protocolVersion"], "capabilities": {"tools": {}, "resources": {}}}
    elif method == "tools/list" and not args:
        result = {"tools": [
            {"name": "lookup", "description": "Echo " + secret, "inputSchema": {
                "type": "object", "properties": {"pattern": {"$ref": "#/$defs/query"}},
                "$defs": {"query": {"type": ["string", "null"]}},
                "required": ["pattern"], "additionalProperties": False,
            }},
            {"name": "schema_leak", "inputSchema": {"type": "object", "description": secret}},
            {"name": "external", "inputSchema": {"type": "object", "$ref": "https://example.invalid/schema"}},
        ], "nextCursor": "page_two"}
    elif method == "tools/list":
        assert args["cursor"] == "page_two"
        result = {"tools": [
            {"name": "mutate", "annotations": {"readOnlyHint": True}, "inputSchema": empty},
            {"name": "foo.bar", "inputSchema": empty},
            {"name": "foo-bar", "inputSchema": empty},
            {"name": "resources_list", "inputSchema": empty},
            {"name": secret, "inputSchema": empty},
            {"name": "hidden", "inputSchema": empty},
        ]}
    elif method == "tools/call" and args["name"] == "mutate":
        with Path(sys.argv[1]).open("a", encoding="utf-8") as file:
            file.write("called\n")
        time.sleep(30)
        result = {}
    elif method == "tools/call":
        query = args["arguments"].get("pattern")
        result = {"echo": secret, "extra_env": os.environ.get("NCC_MCP_TEST_UNLISTED"), "query": query}
        if query == "large":
            result["large"] = (secret + '\\"') * 1_000
    elif method == "resources/list":
        result = {"resources": [{"uri": uri, "name": "Known note", "description": "Value " + secret}]}
    elif method == "resources/read":
        assert args["uri"] == uri
        result = {"contents": [{"uri": uri, "text": "Read approved note"}]}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
'''


@pytest.fixture
def mcp_service(tmp_path, monkeypatch):
    secret = "private-mcp-test-token"
    monkeypatch.setenv("NCC_MCP_TEST_TOKEN", secret)
    monkeypatch.setenv("NCC_MCP_TEST_UNLISTED", "never inherited")
    server = tmp_path / "test_server.py"
    server.write_text(_SERVER, encoding="utf-8")
    marker = tmp_path / "mutations.txt"
    config = tmp_path / "trusted.json"
    config.write_text(json.dumps({"mcpServers": {"notes": {
        "command": sys.executable, "args": [str(server), str(marker)],
        "env_allowlist": ["NCC_MCP_TEST_TOKEN"], "read_only_tools": ["lookup"],
        "allowed_tools": ["lookup", "mutate", "foo.bar", "foo-bar", "resources_list", "schema_leak", "external", secret],
        "resources": True, "timeout_seconds": 5,
    }}}), encoding="utf-8")
    service = McpService.from_file(config)
    try:
        yield service, marker, secret
    finally:
        service.close()


def _context(tmp_path: Path, *risks: RiskLevel) -> ToolContext:
    return ToolContext("run-test", tmp_path, tmp_path / "artifacts", approved_risks=set(risks))


def test_discovers_filtered_catalog_and_uses_local_policy(mcp_service, tmp_path):
    service, marker, secret = mcp_service
    assert service.statuses()[0]["status"] == "disconnected"
    assert service.redact({"echo": secret}) == {"echo": "[REDACTED]"}
    status = service.connect("notes")
    assert status["status"] == "connected"
    assert status["resources"] and status["resource_count"] == 1
    snapshot = service.snapshot()
    assert len(snapshot) == len(set(snapshot)) == 7
    assert service.snapshot() == snapshot
    assert secret not in json.dumps(status)
    assert "mcp__notes__foo-bar" in snapshot
    assert "mcp__notes__hidden" not in snapshot
    assert "mcp__notes__schema_leak" not in snapshot
    assert "mcp__notes__external" not in snapshot
    lookup = service.resolve("mcp__notes__lookup", snapshot["mcp__notes__lookup"])
    assert lookup.risk_level == RiskLevel.READ
    assert secret not in lookup.description
    assert lookup.execute(_context(tmp_path), {"pattern": 5}).error["type"] == "InvalidMcpArguments"
    result = lookup.execute(_context(tmp_path), {"pattern": None})
    assert result.status == "success"
    assert result.data["result"] == {"echo": "[REDACTED]", "extra_env": None, "query": None}
    large = lookup.execute(_context(tmp_path), {"pattern": "large"})
    assert large.truncated and secret not in json.dumps(large.data)
    assert len(json.dumps(large.data).encode()) <= 8_192

    mutate = service.resolve("mcp__notes__mutate", snapshot["mcp__notes__mutate"])
    assert mutate.risk_level == RiskLevel.EXECUTE
    assert mutate.execute(_context(tmp_path), {}).error["type"] == "ApprovalRequired"
    assert not marker.exists()

    assert service.reconnect("notes")["status"] == "connected"
    assert service.resolve(lookup.name, snapshot[lookup.name]) is None
    assert lookup.execute(_context(tmp_path), {"pattern": None}).error["type"] == "StaleMcpTool"


def test_resource_read_requires_explicit_risk_and_a_discovered_uri(mcp_service, tmp_path):
    service, _, secret = mcp_service
    service.connect("notes")
    snapshot = service.snapshot()
    list_name, read_name = "mcp__notes__resources_list", "mcp__notes__resources_read"
    listing = service.resolve(list_name, snapshot[list_name])
    reader = service.resolve(read_name, snapshot[read_name])
    result = listing.execute(_context(tmp_path), {})
    resource = result.data["result"]["resources"][0]
    assert resource["name"] == "Known note"
    assert secret not in json.dumps(result.data) and "notes://" not in json.dumps(result.data)
    args = {"resource_id": resource["resource_id"]}
    assert reader.execute(_context(tmp_path), args).error["type"] == "ApprovalRequired"
    approved = _context(tmp_path, RiskLevel.NETWORK)
    denied = reader.execute(approved, {"resource_id": "file:///arbitrary"})
    assert denied.error["type"] == "McpResourceNotAuthorized"
    contents = reader.execute(approved, args).data["result"]["contents"]
    assert contents == [{"uri": "[REDACTED]", "text": "Read approved note"}]


def test_disconnect_interrupts_a_dispatched_call_without_replaying_it(mcp_service, tmp_path):
    service, marker, _ = mcp_service
    service.connect("notes")
    name = "mcp__notes__mutate"
    mutate = service.resolve(name, service.snapshot()[name])
    results = []
    worker = threading.Thread(target=lambda: results.append(mutate.execute(_context(tmp_path, RiskLevel.EXECUTE), {})))
    worker.start()
    try:
        deadline = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        assert service.disconnect("notes")["status"] == "disconnected"
    finally:
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert results[0].status == "uncertain"
    assert results[0].error["type"] == "McpOutcomeUncertain"
    assert marker.read_text(encoding="utf-8") == "called\n"
    assert not service.snapshot()


def test_configuration_rejects_raw_env_and_implicit_path_resolution(tmp_path):
    config = tmp_path / "trusted.json"
    for item in ({"command": sys.executable, "env": {"SECRET": "raw"}}, {"command": "python"}):
        config.write_text(json.dumps({"mcpServers": {"notes": item}}), encoding="utf-8")
        with pytest.raises(McpConfigError):
            McpService.from_file(config)


def test_configuration_and_status_hide_alias_that_contains_env_value(tmp_path, monkeypatch):
    name = "private-alias-token"
    key = "NCC_MCP_ALIAS_TEST_TOKEN"
    monkeypatch.delenv(key, raising=False)
    config = tmp_path / "trusted.json"
    config.write_text(json.dumps({"mcpServers": {name: {
        "command": sys.executable, "env_allowlist": [key],
    }}}), encoding="utf-8")
    service = McpService.from_file(config)
    monkeypatch.setenv(key, name)
    try:
        assert name not in json.dumps(service.statuses())
        failed = service.connect(name)
        assert failed["status"] == "error" and name not in json.dumps(failed)
        with pytest.raises(McpConfigError) as error:
            McpService.from_file(config)
        assert name not in str(error.value)
    finally:
        service.close()
