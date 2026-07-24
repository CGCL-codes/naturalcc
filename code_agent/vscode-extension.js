const http = require("http");
const net = require("net");
const path = require("path");
const { spawn } = require("child_process");
const vscode = require("vscode");

let output;
let serverProcess;
let serverPort;
let panel;

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

function getServerEnvironment(python) {
  const environment = { ...process.env, PYTHONUNBUFFERED: "1" };
  if (path.isAbsolute(python)) {
    const pythonBinDir = path.dirname(python);
    environment.PATH = `${pythonBinDir}${path.delimiter}${environment.PATH || ""}`;
  }
  return environment;
}

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

function waitForHealthyService(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  const endpoint = `http://127.0.0.1:${port}/api/health`;

  return new Promise((resolve, reject) => {
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

async function stopServer() {
  if (!serverProcess) {
    return;
  }
  const processToStop = serverProcess;
  serverProcess = undefined;
  serverPort = undefined;
  processToStop.kill();
}

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

function deactivate() {
  return stopServer();
}

module.exports = { activate, deactivate };
