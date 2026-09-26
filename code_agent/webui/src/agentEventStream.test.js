import assert from "node:assert/strict";
import test from "node:test";

import { followAgentEvents } from "./agentEventStream.js";

test("agent event stream parses NDJSON split across network chunks", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = "";
  globalThis.fetch = async (url) => {
    requestedUrl = url;
    return new Response(new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        controller.enqueue(encoder.encode('{"sequence":2,"type":"model.reason'));
        controller.enqueue(encoder.encode('ing.delta","payload":{"text":"Thinking"}}\n\n'));
        controller.enqueue(encoder.encode('{"sequence":3,"type":"run.completed","payload":{}}\n'));
        controller.close();
      }
    }));
  };
  try {
    const events = [];
    await followAgentEvents("run 1", 1, (event) => events.push(event), new AbortController().signal, "http://localhost");
    assert.equal(requestedUrl, "http://localhost/api/agent/runs/run%201/events.ndjson?after=1");
    assert.deepEqual(events.map((event) => event.sequence), [2, 3]);
    assert.equal(events[0].payload.text, "Thinking");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
