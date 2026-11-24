import os
import sys
import threading
import time

import pytest

from code_agent.agent_core.mcp_stdio import MAX_MESSAGE_BYTES, McpRpcError, McpTransportError, StdioMcpClient


SERVER = r'''
import json
import os
import sys
import time
from pathlib import Path

mode = sys.argv[1]
signal = Path(sys.argv[2])


def send(message):
    print(json.dumps({"jsonrpc": "2.0", **message}), flush=True)


def receive():
    return json.loads(sys.stdin.readline())


request = receive()
assert request["method"] == "initialize"
assert request["params"]["protocolVersion"] == "2025-11-25"
send({"id": request["id"], "result": {
    "protocolVersion": "unsupported" if mode == "unsupported" else "2025-03-26",
    "capabilities": {"tools": {"listChanged": True}},
    "serverInfo": {"name": "fixture", "version": "1"},
}})
assert receive() == {"jsonrpc": "2.0", "method": "notifications/initialized"}
if mode == "stop-reading":
    signal.touch()
    time.sleep(30)
    raise SystemExit(0)

for line in sys.stdin:
    method_call = json.loads(line)
    method = method_call["method"]
    request_id = method_call["id"]
    if method == "test/notifications":
        send({"id": "unrelated", "method": "ping"})
        assert receive() == {"jsonrpc": "2.0", "id": "unrelated", "result": {}}
        send({"id": "denied", "method": "sampling/createMessage"})
        assert receive()["error"]["code"] == -32601
        send({"method": "notifications/tools/list_changed"})
        send({"method": "notifications/resources/list_changed"})
        send({"id": request_id, "result": {"ack": True}})
    elif method == "test/error":
        print("secret from stderr", file=sys.stderr, flush=True)
        send({"id": request_id, "error": {"code": -32602, "message": "secret from JSON error"}})
    elif method == "test/malformed":
        print("not json", flush=True)
    elif method == "test/large":
        sys.stdout.buffer.write(b"x" * (1024 * 1024 + 1) + b"\n")
        sys.stdout.buffer.flush()
    elif method == "test/hang":
        signal.touch()
        time.sleep(30)
    else:
        send({"id": request_id, "result": {"env": os.getenv("NCC_TEST_MCP_ENV")}})
'''


@pytest.fixture
def create_client(tmp_path):
    script = tmp_path / "test server.py"
    script.write_text(SERVER, encoding="utf-8")
    signal = tmp_path / "request-received"
    clients = []

    def make(mode="normal"):
        env = {"NCC_TEST_MCP_ENV": "present"}
        if "SystemRoot" in os.environ:
            env["SystemRoot"] = os.environ["SystemRoot"]
        client = StdioMcpClient([sys.executable, str(script), mode, str(signal)], env)
        clients.append(client)
        return client, signal

    yield make
    for client in clients:
        client.close()


def test_negotiate_protocol_ping_and_catalog_notifications(create_client):
    client, _ = create_client()
    assert not client.alive
    with pytest.raises(McpTransportError) as error:
        client.request("tools/list", timeout=2)
    assert not error.value.uncertain

    initialized = client.initialize(timeout=3)
    assert initialized["protocolVersion"] == "2025-03-26"
    initialized["capabilities"].clear()
    assert client.initialize(timeout=1)["capabilities"] == {"tools": {"listChanged": True}}
    assert client.alive and not client.catalog_dirty
    assert client.request("test/echo", timeout=2) == {"env": "present"}

    revision = client.catalog_revision
    assert client.request("test/notifications", {}, timeout=2) == {"ack": True}
    client.clear_catalog_dirty(revision)
    assert client.catalog_dirty and client.catalog_revision == revision + 2
    with pytest.raises(McpTransportError, match="catalog changed") as error:
        client.request("test/echo", timeout=2, expected_catalog_revision=revision)
    assert not error.value.uncertain and client.alive
    client.clear_catalog_dirty(client.catalog_revision)
    assert not client.catalog_dirty
    assert client.request("test/echo", timeout=2, expected_catalog_revision=client.catalog_revision) == {"env": "present"}
    client.close()
    client.close()
    assert not client.alive


def test_server_rpc_error_and_unsent_size_rejection_keep_connection_available(create_client):
    client, _ = create_client()
    client.initialize(timeout=3)
    with pytest.raises(McpRpcError) as error:
        client.request("test/error", timeout=2)
    assert error.value.code == -32602
    assert not error.value.uncertain and "secret" not in str(error.value)
    with pytest.raises(McpTransportError, match="size limit") as error:
        client.request("test/echo", {"arg": "x" * MAX_MESSAGE_BYTES}, timeout=2)
    assert not error.value.uncertain
    assert client.request("test/echo", timeout=2) == {"env": "present"}


def test_incompatible_server_is_closed(create_client):
    client, _ = create_client("unsupported")
    with pytest.raises(McpTransportError, match="unsupported protocol version"):
        client.initialize(timeout=3)
    assert not client.alive


@pytest.mark.parametrize("method", ["test/malformed", "test/large"])
def test_invalid_server_message_marks_pending_call_uncertain(create_client, method):
    client, _ = create_client()
    client.initialize(timeout=3)
    with pytest.raises(McpTransportError) as error:
        client.request(method, timeout=2)
    assert error.value.uncertain
    assert not client.alive


@pytest.mark.parametrize("action", ["cancel", "close"])
def test_interrupt_in_flight_request_without_replay(create_client, action):
    client, signal = create_client()
    client.initialize(timeout=3)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(McpTransportError) as error:
        client.request("test/hang", timeout=2, cancellation=cancelled)
    assert not error.value.uncertain and not signal.exists()

    cancelled.clear()
    errors = []

    def request():
        try:
            client.request("test/hang", timeout=5, cancellation=cancelled)
        except McpTransportError as exc:
            errors.append(exc)

    worker = threading.Thread(target=request, daemon=True)
    worker.start()
    for _ in range(100):
        if signal.exists():
            break
        time.sleep(0.02)
    assert signal.exists()
    client.close() if action == "close" else cancelled.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(errors) == 1 and errors[0].uncertain
    assert not client.alive


def test_timeout_unblocks_a_backpressured_pipe(create_client):
    client, _ = create_client("stop-reading")
    client.initialize(timeout=3)
    errors = []

    def request():
        try:
            client.request("test/echo", {"input": "x" * 800_000}, timeout=0.25)
        except McpTransportError as exc:
            errors.append(exc)

    worker = threading.Thread(target=request, daemon=True)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(errors) == 1 and str(errors[0]) == "MCP request timed out."
    assert errors[0].uncertain and not client.alive
