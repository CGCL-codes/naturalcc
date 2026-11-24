import { Plug, RefreshCcw } from "lucide-react";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { MCP_SERVERS_PATH, mcpConnectionActions, mcpConnectionPath, normalizeMcpServer } from "./mcpServers.js";

const statusLabels = {
  disabled: "Disabled",
  disconnected: "Disconnected",
  connected: "Connected",
  error: "Error",
  unknown: "Unknown"
};

export function McpServersPanel({ requestJson }) {
  const [servers, setServers] = useState(null);
  const [pending, setPending] = useState("");
  const [error, setError] = useState("");
  const activeRequest = useRef(null);

  const request = useCallback(async (server, action) => {
    if (activeRequest.current) return;
    const controller = new AbortController();
    activeRequest.current = controller;
    setError("");
    const operation = action === "disconnect"
      ? `Disconnecting ${server.name}…`
      : action === "connect" ? `Connecting ${server.name}…` : "Loading MCP servers…";
    setPending(operation);
    try {
      if (action) {
        const payload = await requestJson(mcpConnectionPath(server.name, action), {
          method: "POST", signal: controller.signal
        });
        const updated = normalizeMcpServer(payload?.server);
        if (updated?.name !== server.name) throw new Error("Invalid MCP server response");
        if (!controller.signal.aborted) {
          setServers((current) => current.map((item) => item.name === updated.name ? updated : item));
        }
      } else {
        const payload = await requestJson(MCP_SERVERS_PATH, { signal: controller.signal });
        if (!Array.isArray(payload?.servers)) throw new Error("Invalid MCP server list");
        if (!controller.signal.aborted) {
          setServers(payload.servers.map(normalizeMcpServer).filter(Boolean));
        }
      }
    } catch (failure) {
      if (!controller.signal.aborted) {
        if (action) {
          setError(`Could not update ${server.name}. Refresh to check its status.`);
        } else {
          setError(failure?.status === 404
            ? "MCP server settings are unavailable from this backend."
            : "Could not load MCP servers. Try refreshing.");
        }
      }
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
        setPending("");
      }
    }
  }, [requestJson]);

  useEffect(() => {
    request();
    return () => {
      activeRequest.current?.abort();
      activeRequest.current = null;
    };
  }, [request]);

  return (
    <section className="mcp-servers-settings" aria-label="MCP servers">
      <div className="section-heading mcp-servers-heading">
        <Plug size={16} />
        <span>MCP servers</span>
        <button
          type="button"
          className="icon-button"
          title="Refresh MCP server status"
          aria-label="Refresh MCP server status"
          disabled={Boolean(pending)}
          onClick={() => request()}
        >
          <RefreshCcw size={15} />
        </button>
      </div>
      {pending && <p className="field-help" role="status">{pending}</p>}
      {error && <p className="error-box" role="alert">{error}</p>}
      {servers?.length === 0 && !error && (
        <p className="field-help">No MCP servers configured. Add them to the backend configuration, then refresh.</p>
      )}
      {servers?.length > 0 && (
        <>
          <p className="field-help">Start a new Agent run after connecting or reconnecting.</p>
          <ul className="mcp-server-list">
            {servers.map((server) => (
              <li className="mcp-server" key={server.name}>
                <div className="mcp-server-heading">
                  <strong>{server.name}</strong>
                  <span className={`codegraph-status ${server.status === "connected" ? "ready" : server.status === "error" ? "attention" : ""}`}>
                    {statusLabels[server.status]}
                  </span>
                </div>
                {server.status === "connected" && (
                  <>
                    {server.tools.length > 0 ? (
                      <details className="mcp-tools">
                        <summary>Tools ({server.tools.length})</summary>
                        <ul>{server.tools.map((name) => <li key={name}><code>{name}</code></li>)}</ul>
                      </details>
                    ) : <p className="field-help">No tools exposed.</p>}
                    {server.resources && <p className="field-help">Resources available</p>}
                  </>
                )}
                <div className="mcp-server-actions">
                  {mcpConnectionActions(server).map(({ action, label }) => (
                    <button
                      key={action}
                      type="button"
                      className="secondary-button compact-button"
                      aria-label={`${label} ${server.name}`}
                      disabled={Boolean(pending)}
                      onClick={() => request(server, action)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
