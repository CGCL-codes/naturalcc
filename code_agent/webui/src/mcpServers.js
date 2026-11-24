const serverStatuses = new Set(["disabled", "disconnected", "connected", "error"]);
export const MCP_SERVERS_PATH = "/api/agent/mcp/servers";

export function normalizeMcpServer(server) {
  if (typeof server?.name !== "string" || !server.name) return null;
  const tools = Array.isArray(server.tools) ? server.tools : [];
  const names = tools.map((tool) => typeof tool === "string" ? tool : tool?.qualified_name || tool?.name);
  return {
    name: server.name,
    enabled: server.enabled === true,
    status: serverStatuses.has(server.status) ? server.status : "unknown",
    tools: [...new Set(names.filter((name) => typeof name === "string" && name))],
    resources: server.resources === true
  };
}

export function mcpConnectionActions(server) {
  if (!server.enabled || server.status === "disabled" || server.status === "unknown") return [];
  if (server.status === "connected") {
    return [{ action: "connect", label: "Reconnect" }, { action: "disconnect", label: "Disconnect" }];
  }
  const label = server.status === "error" ? "Reconnect" : "Connect";
  return [{ action: "connect", label }];
}

export function mcpConnectionPath(name, action) {
  return `${MCP_SERVERS_PATH}/${encodeURIComponent(name)}/${action}`;
}
