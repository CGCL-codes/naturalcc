import assert from "node:assert/strict";
import test from "node:test";
import { mcpConnectionActions, mcpConnectionPath, normalizeMcpServer } from "./mcpServers.js";

test("MCP server rows keep visible capability names and omit configuration", () => {
  assert.deepEqual(normalizeMcpServer({
    name: "notes",
    enabled: true,
    status: "connected",
    tools: [
      "mcp__notes__search",
      { name: "get", qualified_name: "mcp__notes__get" },
      "mcp__notes__search",
      { command: "unused" },
      null
    ],
    resources: true,
    command: "local-config",
    env: { PRIVATE_TOKEN: "private" }
  }), {
    name: "notes",
    enabled: true,
    status: "connected",
    tools: ["mcp__notes__search", "mcp__notes__get"],
    resources: true
  });
  assert.equal(normalizeMcpServer({ status: "connected" }), null);
});

test("MCP connection controls reflect server state and encode names in the route", () => {
  assert.deepEqual(mcpConnectionActions({ enabled: false, status: "connected" }), []);
  assert.deepEqual(mcpConnectionActions({ enabled: true, status: "disabled" }), []);
  assert.deepEqual(mcpConnectionActions({ enabled: true, status: "unknown" }), []);
  assert.deepEqual(mcpConnectionActions({ enabled: true, status: "disconnected" }), [
    { action: "connect", label: "Connect" }
  ]);
  assert.deepEqual(mcpConnectionActions({ enabled: true, status: "error" }), [
    { action: "connect", label: "Reconnect" }
  ]);
  assert.deepEqual(mcpConnectionActions({ enabled: true, status: "connected" }), [
    { action: "connect", label: "Reconnect" },
    { action: "disconnect", label: "Disconnect" }
  ]);
  assert.equal(mcpConnectionPath("project notes", "connect"), "/api/agent/mcp/servers/project%20notes/connect");
});
