from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
MAX_MESSAGE_BYTES = 1024 * 1024
_OUTBOX_LIMIT = 16
_POLL_SECONDS = 0.025
_CATALOG_NOTIFICATIONS = {"notifications/tools/list_changed", "notifications/resources/list_changed"}


class McpTransportError(RuntimeError):
    def __init__(self, message: str, *, uncertain: bool = False) -> None:
        super().__init__(message)
        self.uncertain = uncertain


class McpRpcError(McpTransportError):
    def __init__(self, code: int) -> None:
        super().__init__("MCP server returned a JSON-RPC error.")
        self.code = code


@dataclass
class _PendingRequest:
    id: int
    expected_catalog_revision: int | None = None
    finished: threading.Event = field(default_factory=threading.Event)
    started: bool = False
    result: dict[str, Any] | None = None
    error: McpTransportError | None = None


@dataclass
class _Outbound:
    payload: bytes
    pending: _PendingRequest | None = None
    written: threading.Event | None = None


def _json_line(message: dict[str, Any]) -> bytes:
    try:
        payload = json.dumps(message, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise McpTransportError("MCP request is not valid JSON.") from None
    if len(payload) + 1 > MAX_MESSAGE_BYTES:
        raise McpTransportError("MCP request exceeds the message size limit.")
    return payload + b"\n"


def _reject_constant(value: str) -> None:
    raise ValueError("Non-JSON number")


def _valid_depth(value: Any, depth: int = 64) -> bool:
    if isinstance(value, dict):
        return depth > 0 and all(_valid_depth(item, depth - 1) for item in value.values())
    if isinstance(value, list):
        return depth > 0 and all(_valid_depth(item, depth - 1) for item in value)
    return True


def _deadline(timeout: float) -> float:
    if not math.isfinite(timeout) or timeout <= 0:
        raise McpTransportError("MCP request timeout must be positive and finite.")
    return time.monotonic() + timeout


class StdioMcpClient:
    """A single MCP subprocess with bounded, newline-delimited JSON-RPC 2.0."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        if not argv or not argv[0] or not all(isinstance(item, str) for item in argv):
            raise McpTransportError("MCP server requires an executable and a list of arguments.")
        if os.name == "nt" and os.path.splitext(argv[0].rstrip(" ."))[1].lower() in {".bat", ".cmd"}:
            raise McpTransportError("MCP server requires a native executable on Windows.")
        self._argv = list(argv)
        self._env = dict(env)
        self._process: subprocess.Popen[bytes] | None = None
        self._closed = threading.Event()
        self._state_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._outbox: queue.Queue[_Outbound] = queue.Queue(maxsize=_OUTBOX_LIMIT)
        self._write_active = False
        self._pending: _PendingRequest | None = None
        self._next_id = 0
        self._initialize_result: dict[str, Any] | None = None
        self._catalog_revision = 0
        self._clear_revision = 0

    @property
    def alive(self) -> bool:
        process = self._process
        return not self._closed.is_set() and process is not None and process.poll() is None

    @property
    def catalog_revision(self) -> int:
        with self._state_lock:
            return self._catalog_revision

    @property
    def catalog_dirty(self) -> bool:
        with self._state_lock:
            return self._catalog_revision != self._clear_revision

    def clear_catalog_dirty(self, revision: int | None = None) -> None:
        with self._state_lock:
            if revision is None or revision == self._catalog_revision:
                self._clear_revision = self._catalog_revision

    def initialize(
        self, *, timeout: float, cancellation: threading.Event | None = None
    ) -> dict[str, Any]:
        deadline = _deadline(timeout)
        self._acquire_request(deadline, cancellation)
        try:
            if self._initialize_result is not None:
                if not self.alive:
                    raise McpTransportError("MCP connection is closed.")
                return deepcopy(self._initialize_result)
            try:
                self._wait_time(deadline, cancellation)
                self._start()
                result = self._request(
                    "initialize",
                    {
                        "protocolVersion": SUPPORTED_PROTOCOL_VERSIONS[0],
                        "capabilities": {},
                        "clientInfo": {"name": "NaturalCC", "version": "1.0"},
                    },
                    deadline,
                    cancellation,
                )
                if result.get("protocolVersion") not in SUPPORTED_PROTOCOL_VERSIONS:
                    raise McpTransportError("MCP server selected an unsupported protocol version.")
                written = threading.Event()
                self._queue({"jsonrpc": "2.0", "method": "notifications/initialized"}, written=written)
                self._wait(written, deadline, cancellation)
                if not self.alive:
                    raise McpTransportError("MCP connection is closed.")
            except McpTransportError:
                self.close()
                raise
            self._initialize_result = result
            return deepcopy(result)
        finally:
            self._request_lock.release()

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float,
        cancellation: threading.Event | None = None,
        expected_catalog_revision: int | None = None,
    ) -> dict[str, Any]:
        deadline = _deadline(timeout)
        self._acquire_request(deadline, cancellation)
        try:
            if self._initialize_result is None:
                raise McpTransportError("MCP connection has not been initialized.")
            if not isinstance(method, str) or not method or (params is not None and not isinstance(params, dict)):
                raise McpTransportError("MCP request has an invalid method or parameters.")
            return self._request(method, params, deadline, cancellation, expected_catalog_revision)
        finally:
            self._request_lock.release()

    def close(self) -> None:
        with self._state_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            pending = self._pending
            if pending is not None and not pending.finished.is_set():
                pending.error = McpTransportError("MCP connection closed during the request.", uncertain=pending.started)
                pending.finished.set()
            process = self._process
            write_active = self._write_active
        if process is not None:
            self._stop_process(process, write_active=write_active)

    def _start(self) -> None:
        try:
            process = subprocess.Popen(
                self._argv,
                env=self._env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, ValueError, TypeError):
            raise McpTransportError("MCP server process could not be started.") from None
        with self._state_lock:
            started = not self._closed.is_set()
            if started:
                self._process = process
        if not started:
            self._stop_process(process)
            if process.stdout is not None:
                process.stdout.close()
            raise McpTransportError("MCP connection is closed.")
        threading.Thread(target=self._read_loop, args=(process,), daemon=True).start()
        threading.Thread(target=self._write_loop, args=(process,), daemon=True).start()

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes], *, write_active: bool = False) -> None:
        if not write_active:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except OSError:
                pass
        else:
            # In particular on Windows, close() can wait on a blocked pipe write.
            try:
                process.terminate()
            except OSError:
                pass
        try:
            process.wait(timeout=0.10)
        except subprocess.TimeoutExpired:
            try:
                process.terminate()
                process.wait(timeout=0.20)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    process.wait(timeout=0.20)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            except OSError:
                pass
        except OSError:
            pass

    def _acquire_request(self, deadline: float, cancellation: threading.Event | None) -> None:
        while True:
            delay = self._wait_time(deadline, cancellation)
            if self._request_lock.acquire(timeout=delay):
                return

    def _wait_time(self, deadline: float, cancellation: threading.Event | None) -> float:
        if cancellation is not None and cancellation.is_set():
            raise McpTransportError("MCP request was cancelled.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise McpTransportError("MCP request timed out.")
        if self._closed.is_set():
            raise McpTransportError("MCP connection is closed.")
        return min(_POLL_SECONDS, remaining)

    def _wait(self, event: threading.Event, deadline: float, cancellation: threading.Event | None) -> None:
        while not event.is_set():
            event.wait(self._wait_time(deadline, cancellation))

    def _queue(
        self,
        message: dict[str, Any],
        *,
        pending: _PendingRequest | None = None,
        written: threading.Event | None = None,
    ) -> None:
        try:
            self._outbox.put_nowait(_Outbound(_json_line(message), pending, written))
        except queue.Full:
            raise McpTransportError("MCP send queue is full.") from None

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None,
        deadline: float,
        cancellation: threading.Event | None,
        expected_catalog_revision: int | None = None,
    ) -> dict[str, Any]:
        self._wait_time(deadline, cancellation)
        self._next_id += 1
        pending = _PendingRequest(self._next_id, expected_catalog_revision)
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": pending.id, "method": method}
        if params is not None:
            message["params"] = params
        with self._state_lock:
            if self._closed.is_set():
                raise McpTransportError("MCP connection is closed.")
            self._pending = pending
        wait_error = None
        try:
            self._queue(message, pending=pending)
            self._wait(pending.finished, deadline, cancellation)
        except McpTransportError as exc:
            wait_error = exc
        finally:
            with self._state_lock:
                self._pending = None
                if pending.finished.is_set():
                    wait_error = None
                elif wait_error is not None:
                    wait_error.uncertain = pending.started
        if wait_error is not None:
            if wait_error.uncertain:
                self.close()
            raise wait_error
        if pending.error is not None:
            raise pending.error
        assert pending.result is not None
        return pending.result

    def _write_loop(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stdin is not None
        try:
            fd = process.stdin.fileno()
            while not self._closed.is_set():
                try:
                    outbound = self._outbox.get(timeout=_POLL_SECONDS)
                except queue.Empty:
                    continue
                with self._state_lock:
                    if self._closed.is_set():
                        return
                    if outbound.pending is not None:
                        if outbound.pending is not self._pending:
                            continue
                        revision = outbound.pending.expected_catalog_revision
                        if revision is not None and revision != self._catalog_revision:
                            outbound.pending.error = McpTransportError(
                                "MCP server catalog changed before the request was sent."
                            )
                            outbound.pending.finished.set()
                            continue
                        outbound.pending.started = True
                    self._write_active = True
                try:
                    data = memoryview(outbound.payload)
                    while data and not self._closed.is_set():
                        written = os.write(fd, data)
                        if not written:
                            raise OSError("Incomplete MCP write")
                        data = data[written:]
                    if not data and outbound.written is not None:
                        outbound.written.set()
                finally:
                    with self._state_lock:
                        self._write_active = False
        except (OSError, ValueError):
            self.close()
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    def _read_loop(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stdout is not None
        try:
            while not self._closed.is_set():
                line = process.stdout.readline(MAX_MESSAGE_BYTES + 1)
                if not line:
                    return
                if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
                    raise McpTransportError("MCP server sent an oversized or incomplete message.")
                message = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
                self._receive(message)
        except (McpTransportError, OSError, ValueError, RecursionError):
            pass
        finally:
            self.close()
            process.stdout.close()

    def _receive(self, message: Any) -> None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not _valid_depth(message):
            raise McpTransportError("MCP server sent an invalid JSON-RPC message.")
        if "method" in message:
            method = message["method"]
            if not isinstance(method, str) or not method:
                raise McpTransportError("MCP server sent an invalid JSON-RPC method.")
            if "id" not in message:
                if method in _CATALOG_NOTIFICATIONS:
                    with self._state_lock:
                        self._catalog_revision += 1
                return
            request_id = message["id"]
            if type(request_id) not in {int, str}:
                raise McpTransportError("MCP server sent an invalid JSON-RPC request ID.")
            response: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
            if method == "ping":
                response["result"] = {}
            else:
                response["error"] = {"code": -32601, "message": "Method not supported"}
            self._queue(response)
            return
        request_id = message.get("id")
        if type(request_id) is not int or ("result" in message) == ("error" in message):
            raise McpTransportError("MCP server sent an invalid JSON-RPC response.")
        with self._state_lock:
            pending = self._pending
            if pending is None or request_id != pending.id or not pending.started or pending.finished.is_set():
                return
            if "error" in message:
                error = message["error"]
                if not isinstance(error, dict) or type(error.get("code")) is not int:
                    raise McpTransportError("MCP server sent an invalid JSON-RPC error.")
                pending.error = McpRpcError(error["code"])
            else:
                result = message["result"]
                if not isinstance(result, dict):
                    raise McpTransportError("MCP server returned an invalid result.")
                pending.result = result
            pending.finished.set()
