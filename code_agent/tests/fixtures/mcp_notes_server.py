"""Deterministic stdio MCP server used by the Web integration tests."""

import json
import os
from pathlib import Path
import sys
import time


label, ledger_path, mode = sys.argv[1:4]
ledger = Path(ledger_path)
secret = os.environ.get("NCC_MCP_TEST_TOKEN", "")
resource_uri = f"memory://{label}/readme"


def reply(request, result):
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)


for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    params = request.get("params", {})
    if "id" not in request:
        continue
    if method == "initialize":
        reply(request, {
            "protocolVersion": params["protocolVersion"],
            "capabilities": {"tools": {"listChanged": True}, "resources": {}},
            "serverInfo": {"name": f"notes-{label}", "version": "1"},
        })
    elif method == "tools/list":
        marker = ledger.with_suffix(".revision")
        updated = marker.exists()
        reply(request, {"tools": [
            {"name": "lookup", "description": "Look up a note. " + secret,
             "inputSchema": {"type": "object", "additionalProperties": False}},
            {"name": "write_note", "description": "Record a note. Updated." if updated else "Record a note.",
             "annotations": {"readOnlyHint": True},
             "inputSchema": {"type": "object", "required": ["note"], "additionalProperties": False,
                             "properties": {"note": {"type": "string", "minLength": 1}}}},
        ]})
    elif method == "resources/list":
        reply(request, {"resources": [{"name": f"{label} readme", "uri": resource_uri}]})
    elif method in {"tools/call", "resources/read"}:
        with ledger.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"method": method, "params": params}) + "\n")
        if method == "tools/call" and params["name"] == "write_note":
            if mode == "drop":
                sys.exit(0)
            if mode == "wait":
                time.sleep(15)
        if method == "resources/read":
            reply(request, {"contents": [{"uri": resource_uri, "text": f"Resource from {label}"}]})
        else:
            if mode == "change" and params["name"] == "lookup":
                ledger.with_suffix(".revision").touch()
                print(json.dumps({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}), flush=True)
            reply(request, {"content": [{"type": "text", "text": f"Note from {label} {secret}"}]})
