export async function followAgentEvents(runId, after, onEvent, signal, apiBase = "") {
  const response = await fetch(
    `${apiBase}/api/agent/runs/${encodeURIComponent(runId)}/events.ndjson?after=${after}`,
    { signal }
  );
  if (!response.ok || !response.body) {
    throw new Error(`Agent event stream failed (${response.status})`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (line.trim()) onEvent(JSON.parse(line));
      }
      if (done) {
        if (buffer.trim()) onEvent(JSON.parse(buffer));
        return;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
