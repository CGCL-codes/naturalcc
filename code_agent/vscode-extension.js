const http = require("http");
const net = require("net");
const path = require("path");
const { spawn } = require("child_process");
const vscode = require("vscode");

let output;
let serverProcess;
let serverPort;
let panel;

// [maint-note:08800d129afc]
// 维护导航：getPythonExecutable；下列为本函数源码中的接口与路径索引。
// 输入 context：类型由调用方与函数体约束。
// 分支条件：configured。
// 返回路径：configured。
// 返回路径：candidates.find((candidate) => candidate === "python3" || require("fs").existsSync(candidate))。
// 直接调用：vscode.workspace .getConfiguration("naturalccCodeAgent") .get("pythonPath", "") .trim。
// 直接调用：vscode.workspace .getConfiguration("naturalccCodeAgent") .get。
// 直接调用：vscode.workspace .getConfiguration。
// 直接调用：path.join。
// 直接调用：(vscode.workspace.workspaceFolders || []).map。
// 直接调用：candidates.find。
// [/maint-note:08800d129afc]
function getPythonExecutable(context) {
  const configured = vscode.workspace
    .getConfiguration("naturalccCodeAgent")
    .get("pythonPath", "")
    .trim();
  if (configured) {
    return configured;
  }

  const candidates = [
    path.join(context.extensionPath, ".venv", "bin", "python"),
    ...((vscode.workspace.workspaceFolders || []).map((folder) =>
      path.join(folder.uri.fsPath, ".venv", "bin", "python"),
    )),
    "python3",
  ];
  return candidates.find((candidate) => candidate === "python3" || require("fs").existsSync(candidate));
}

// [maint-note:d47cdee41c99]
// 维护导航：getServerEnvironment；下列为本函数源码中的接口与路径索引。
// 输入 python：类型由调用方与函数体约束。
// 分支条件：path.isAbsolute(python)。
// 返回路径：environment。
// 写入目标：environment.PATH。
// 直接调用：path.isAbsolute。
// 直接调用：path.dirname。
// [/maint-note:d47cdee41c99]
function getServerEnvironment(python) {
  const environment = { ...process.env, PYTHONUNBUFFERED: "1" };
  if (path.isAbsolute(python)) {
    const pythonBinDir = path.dirname(python);
    environment.PATH = `${pythonBinDir}${path.delimiter}${environment.PATH || ""}`;
  }
  return environment;
}

// [maint-note:c56733ebc66a]
// 维护导航：reservePort；下列为本函数源码中的接口与路径索引。
// 返回路径：new Promise((resolve, reject) => { const socket = net.createServer(); socket.unref(); socket.on("error", reject); socket.listen(0, "127.0.0.1", () => { const address = socket.address(); socket.close((error) => (error ? reject(error) : resol。
// [/maint-note:c56733ebc66a]
function reservePort() {
  return new Promise((resolve, reject) => {
    const socket = net.createServer();
    socket.unref();
    socket.on("error", reject);
    socket.listen(0, "127.0.0.1", () => {
      const address = socket.address();
      socket.close((error) => (error ? reject(error) : resolve(address.port)));
    });
  });
}

// [maint-note:8bf5c8cb3fc7]
// 维护导航：waitForHealthyService；下列为本函数源码中的接口与路径索引。
// 输入 port：类型由调用方与函数体约束。
// 输入 timeoutMs：类型由调用方与函数体约束。
// 返回路径：new Promise((resolve, reject) => { const poll = () => { const request = http.get(endpoint, { timeout: 1000 }, (response) => { response.resume(); if (response.statusCode === 200) { resolve(); } else if (Date.now() < deadline) { setTimeout(po。
// 直接调用：Date.now。
// [/maint-note:8bf5c8cb3fc7]
function waitForHealthyService(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  const endpoint = `http://127.0.0.1:${port}/api/health`;

  return new Promise((resolve, reject) => {
    // [maint-note:15ccd09c82b4]
    // 维护导航：poll；下列为本函数源码中的接口与路径索引。
    // 直接调用：http.get。
    // 直接调用：request.on。
    // [/maint-note:15ccd09c82b4]
    const poll = () => {
      const request = http.get(endpoint, { timeout: 1000 }, (response) => {
        response.resume();
        if (response.statusCode === 200) {
          resolve();
        } else if (Date.now() < deadline) {
          setTimeout(poll, 250);
        } else {
          reject(new Error(`Service returned HTTP ${response.statusCode}.`));
        }
      });
      request.on("error", () => {
        if (Date.now() < deadline) {
          setTimeout(poll, 250);
        } else {
          reject(new Error("Timed out waiting for the local NaturalCC service."));
        }
      });
      request.on("timeout", () => request.destroy());
    };
    poll();
  });
}

// [maint-note:9fec0543feb4]
// 维护导航：startServer；下列为本函数源码中的接口与路径索引。
// 输入 context：类型由调用方与函数体约束。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：serverProcess && serverPort。
// 返回路径：serverPort。
// 返回路径：port。
// 显式抛出：new Error( `${reason} Set naturalccCodeAgent.pythonPath to the Python environment created by \`uv sync\`, then try again.`, )。
// 写入目标：serverProcess。
// 写入目标：serverPort。
// 直接调用：getPythonExecutable。
// 直接调用：reservePort。
// 直接调用：vscode.workspace .getConfiguration("naturalccCodeAgent") .get。
// 直接调用：vscode.workspace .getConfiguration。
// 直接调用：path.join。
// 直接调用：output.appendLine。
// 直接调用：spawn。
// 直接调用：String。
// 直接调用：getServerEnvironment。
// 直接调用：serverProcess.stdout.on。
// 直接调用：serverProcess.stderr.on。
// 直接调用：serverProcess.on。
// 直接调用：waitForHealthyService。
// 直接调用：stopServer。
// [/maint-note:9fec0543feb4]
async function startServer(context) {
  if (serverProcess && serverPort) {
    return serverPort;
  }

  const python = getPythonExecutable(context);
  const port = await reservePort();
  const timeoutMs = vscode.workspace
    .getConfiguration("naturalccCodeAgent")
    .get("startupTimeoutMs", 30000);
  const launcher = path.join(context.extensionPath, "vscode_server.py");

  output.appendLine(`Starting NaturalCC service with ${python} on port ${port}.`);
  serverProcess = spawn(python, [launcher, "--host", "127.0.0.1", "--port", String(port)], {
    cwd: context.extensionPath,
    env: getServerEnvironment(python),
    stdio: ["ignore", "pipe", "pipe"],
  });
  serverProcess.stdout.on("data", (data) => output.append(data.toString()));
  serverProcess.stderr.on("data", (data) => output.append(data.toString()));
  serverProcess.on("error", (error) => output.appendLine(`Could not start Python: ${error.message}`));
  serverProcess.on("exit", (code, signal) => {
    output.appendLine(`NaturalCC service stopped (code=${code}, signal=${signal || "none"}).`);
    serverProcess = undefined;
    serverPort = undefined;
  });

  try {
    await waitForHealthyService(port, timeoutMs);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    await stopServer();
    throw new Error(
      `${reason} Set naturalccCodeAgent.pythonPath to the Python environment created by \`uv sync\`, then try again.`,
    );
  }

  serverPort = port;
  return port;
}

// [maint-note:58a8d545d1f8]
// 维护导航：stopServer；下列为本函数源码中的接口与路径索引。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!serverProcess。
// 返回路径：无返回值。
// 写入目标：serverProcess。
// 写入目标：serverPort。
// 直接调用：processToStop.kill。
// [/maint-note:58a8d545d1f8]
async function stopServer() {
  if (!serverProcess) {
    return;
  }
  const processToStop = serverProcess;
  serverProcess = undefined;
  serverPort = undefined;
  processToStop.kill();
}

// [maint-note:f6b447eb4b84]
// 维护导航：renderPanel；下列为本函数源码中的接口与路径索引。
// 输入 currentPanel：类型由调用方与函数体约束。
// 输入 port：类型由调用方与函数体约束。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 写入目标：currentPanel.webview.html。
// 直接调用：vscode.Uri.parse。
// 直接调用：(await vscode.env.asExternalUri(localServerUri)).toString().replace。
// 直接调用：(await vscode.env.asExternalUri(localServerUri)).toString。
// 直接调用：vscode.env.asExternalUri。
// [/maint-note:f6b447eb4b84]
async function renderPanel(currentPanel, port) {
  const localServerUri = vscode.Uri.parse(`http://127.0.0.1:${port}`);
  const serverUrl = (await vscode.env.asExternalUri(localServerUri)).toString().replace(/\/$/, "");
  currentPanel.webview.html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; frame-src ${serverUrl}; style-src 'unsafe-inline'; script-src 'nonce-naturalcc';" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    html, body, iframe { width: 100%; height: 100%; margin: 0; border: 0; overflow: hidden; }
    body { background: var(--vscode-editor-background); }
    #error { display: none; padding: 2rem; color: var(--vscode-editor-foreground); }
    button { font: inherit; padding: .5rem .9rem; cursor: pointer; }
  </style>
</head>
<body>
  <iframe title="NaturalCC Code Agent" src="${serverUrl}/"></iframe>
  <div id="error">The NaturalCC interface could not load. <button onclick="location.reload()">Reload</button></div>
</body>
</html>`;
}

// [maint-note:6fc15dcfa724]
// 维护导航：openAgent；下列为本函数源码中的接口与路径索引。
// 输入 context：类型由调用方与函数体约束。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：panel。
// 返回路径：无返回值。
// 写入目标：panel。
// 直接调用：startServer。
// 直接调用：panel.reveal。
// 直接调用：renderPanel。
// 直接调用：vscode.window.createWebviewPanel。
// 直接调用：panel.onDidDispose。
// 直接调用：String。
// 直接调用：output.appendLine。
// 直接调用：vscode.window.showErrorMessage(`NaturalCC Code Agent: ${message}`, "Show Output").then。
// 直接调用：vscode.window.showErrorMessage。
// [/maint-note:6fc15dcfa724]
async function openAgent(context) {
  try {
    const port = await startServer(context);
    if (panel) {
      panel.reveal(vscode.ViewColumn.One);
      await renderPanel(panel, port);
      return;
    }
    panel = vscode.window.createWebviewPanel(
      "naturalccCodeAgent",
      "NaturalCC Code Agent",
      vscode.ViewColumn.One,
      { enableScripts: true, retainContextWhenHidden: true },
    );
    panel.onDidDispose(() => {
      panel = undefined;
    });
    await renderPanel(panel, port);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    output.appendLine(message);
    vscode.window.showErrorMessage(`NaturalCC Code Agent: ${message}`, "Show Output").then((choice) => {
      if (choice === "Show Output") {
        output.show(true);
      }
    });
  }
}

// [maint-note:40ba5069c316]
// 维护导航：activate；下列为本函数源码中的接口与路径索引。
// 输入 context：类型由调用方与函数体约束。
// 写入目标：output。
// 直接调用：vscode.window.createOutputChannel。
// 直接调用：context.subscriptions.push。
// 直接调用：vscode.commands.registerCommand。
// [/maint-note:40ba5069c316]
function activate(context) {
  output = vscode.window.createOutputChannel("NaturalCC Code Agent");
  context.subscriptions.push(
    output,
    vscode.commands.registerCommand("naturalccCodeAgent.open", () => openAgent(context)),
    vscode.commands.registerCommand("naturalccCodeAgent.restart", async () => {
      await stopServer();
      await openAgent(context);
    }),
    vscode.commands.registerCommand("naturalccCodeAgent.stop", () => stopServer()),
  );
}

// [maint-note:fe0a39984138]
// 维护导航：deactivate；下列为本函数源码中的接口与路径索引。
// 返回路径：stopServer()。
// 直接调用：stopServer。
// [/maint-note:fe0a39984138]
function deactivate() {
  return stopServer();
}

module.exports = { activate, deactivate };
