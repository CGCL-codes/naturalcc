// [maint-note:ecc082268db6]
// 维护导航：followAgentEvents；下列为本函数源码中的接口与路径索引。
// 输入 runId：类型由调用方与函数体约束。
// 输入 after：类型由调用方与函数体约束。
// 输入 onEvent：类型由调用方与函数体约束。
// 输入 signal：类型由调用方与函数体约束。
// 输入 apiBase：类型由调用方与函数体约束；缺省表达式 ""。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!response.ok || !response.body。
// 分支条件：line.trim()。
// 分支条件：done。
// 分支条件：buffer.trim()。
// 遍历边界：true。
// 遍历边界：const line ← lines。
// 返回路径：无返回值。
// 显式抛出：new Error(`Agent event stream failed (${response.status})`)。
// 写入目标：buffer。
// 直接调用：fetch。
// 直接调用：encodeURIComponent。
// 直接调用：response.body.getReader。
// 直接调用：reader.read。
// 直接调用：decoder.decode。
// 直接调用：buffer.split。
// 直接调用：lines.pop。
// 直接调用：line.trim。
// 直接调用：onEvent。
// 直接调用：JSON.parse。
// 直接调用：buffer.trim。
// 直接调用：reader.releaseLock。
// [/maint-note:ecc082268db6]
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
