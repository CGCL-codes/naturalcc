import {
  Activity,
  Braces,
  Check,
  ChevronUp,
  Circle,
  Clipboard,
  Code2,
  FileCode,
  Folder,
  FolderOpen,
  Play,
  RefreshCcw,
  Search,
  Settings2,
  Sparkles,
  Terminal,
  Trash2,
  X
} from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const defaultCompletionTypes = ["", "member", "variable", "function", "function_body", "type"];

const emptyWorkspace = {
  normalized_root: "",
  exists: false,
  files: [],
  scanned_files: [],
  selected_files: [],
  counts: {
    visible_files: 0,
    parsable_files: 0,
    selected_files: 0,
    custom_targets: 0
  },
  primary_target: ""
};

function classNames(...items) {
  return items.filter(Boolean).join(" ");
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep the HTTP status text.
    }
    throw new Error(detail);
  }
  return response.json();
}

function metricLabel(value) {
  if (value === undefined || value === null || value === "") {
    return "not set";
  }
  return String(value);
}

const FIELD_TYPE_COMPONENTS = {
  text: ({ field, value, onChange }) => (
    <input
      id={`field-${field.name}`}
      className="text-input"
      type="text"
      value={value || ""}
      onChange={(e) => onChange(field.name, e.target.value)}
      placeholder={field.placeholder || ""}
    />
  ),

  textarea: ({ field, value, onChange }) => (
    <textarea
      id={`field-${field.name}`}
      className="text-input"
      value={value || ""}
      onChange={(e) => onChange(field.name, e.target.value)}
      placeholder={field.placeholder || ""}
      rows={4}
    />
  ),

  select: ({ field, value, onChange }) => (
    <select
      id={`field-${field.name}`}
      className="text-input"
      value={value || field.default || ""}
      onChange={(e) => onChange(field.name, e.target.value)}
    >
      {(field.options || []).map((opt) => (
        <option value={opt.value} key={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  ),

  switch: ({ field, value, onChange }) => (
    <label className="switch-row">
      <input
        type="checkbox"
        checked={!!value}
        onChange={(e) => onChange(field.name, e.target.checked)}
      />
      <span>{field.help_text || field.label}</span>
    </label>
  ),

  file: ({ field, value, onChange }) => (
    <div className="file-upload-row">
      <input
        id={`field-${field.name}`}
        type="file"
        accept={field.accept || "*"}
        multiple={field.multiple}
        onChange={(e) => onChange(field.name, e.target.files)}
      />
      {value && value[0] && (
        <span className="file-name">{value[0].name}</span>
      )}
    </div>
  ),
};

function DynamicForm({ schema, values, onChange }) {
  return (
    <div className="dynamic-form">
      {schema.map((field) => {
        const Component = FIELD_TYPE_COMPONENTS[field.type];
        if (!Component) return null;
        return (
          <div key={field.name} className="form-field">
            <label className="field-label" htmlFor={`field-${field.name}`}>
              {field.label}
              {field.required && <span className="required">*</span>}
            </label>
            <Component
              field={field}
              value={values[field.name]}
              onChange={onChange}
            />
            {field.help_text && (
              <p className="field-help">{field.help_text}</p>
            )}
          </div>
        );
      })}
    </div>
  );
}

function App() {
  const [models, setModels] = useState([]);
  const [completionTypes, setCompletionTypes] = useState(defaultCompletionTypes);
  const [projectDir, setProjectDir] = useState("");
  const [targets, setTargets] = useState([]);
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [instruction, setInstruction] = useState("");
  // Feature plugin states
  const [features, setFeatures] = useState([]);
  const [featureSchemas, setFeatureSchemas] = useState({});
  const [currentFeature, setCurrentFeature] = useState("code_completion");
  const [featureConfig, setFeatureConfig] = useState({});
  // Legacy advanced states (kept for backward compat during transition)
  const [symbol, setSymbol] = useState("");
  const [completionType, setCompletionType] = useState("");
  const [prefix, setPrefix] = useState("");
  const [workspace, setWorkspace] = useState(emptyWorkspace);
  const [browseState, setBrowseState] = useState({ path: "", parent: "", directories: [], files: [] });
  const [fileFilter, setFileFilter] = useState("");
  const fileFilterRef = useRef(null);
  const [customTarget, setCustomTarget] = useState("");
  const [terminalLog, setTerminalLog] = useState("");
  const [runCommand, setRunCommand] = useState("");
  const [previewCommand, setPreviewCommand] = useState("");
  const [provider, setProvider] = useState("openai");
  const [apiKeyLabel, setApiKeyLabel] = useState("not-set");
  const [mode, setMode] = useState("idle");
  const [busy, setBusy] = useState(false);
  const [lastError, setLastError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");

  const payload = useMemo(
    () => {
      const base = {
        project_dir: projectDir,
        target_files: targets,
        instruction,
        model,
        api_key: apiKey || null,
        feature: currentFeature,
        feature_config: { ...featureConfig },
      };
      // Backward compat: include legacy fields when using code_completion
      if (currentFeature === "code_completion") {
        base.symbol = featureConfig.symbol || symbol || null;
        base.completion_type = featureConfig.completion_type || completionType || null;
        base.prefix = featureConfig.prefix || prefix || "";
      }
      return base;
    },
    [apiKey, completionType, currentFeature, featureConfig, instruction, model, prefix, projectDir, symbol, targets]
  );

  const filteredFiles = useMemo(() => {
    const query = fileFilter.trim().toLowerCase();
    const files = workspace.files || [];
    if (!query) {
      return files.slice(0, 300);
    }
    return files.filter((file) => file.toLowerCase().includes(query)).slice(0, 300);
  }, [fileFilter, workspace.files]);

  const primaryTarget = targets[0] || "";
  const customTargetCount = useMemo(() => {
    const scanned = new Set(workspace.scanned_files || []);
    return targets.filter((file) => !scanned.has(file)).length;
  }, [targets, workspace.scanned_files]);
  const statusTitle = {
    idle: "Ready",
    preview: "Preview ready",
    running: "Running",
    success: "Complete",
    error: "Needs attention"
  }[mode] || "Ready";

  useEffect(() => {
    let active = true;
    requestJson("/api/bootstrap")
      .then((data) => {
        if (!active) {
          return;
        }
        setModels(data.models || []);
        setCompletionTypes(data.completion_types || defaultCompletionTypes);
        setModel(data.default_model || "");
        setProjectDir(data.default_project_dir || "");
        setFeatures(data.features || []);
        setFeatureSchemas(data.schemas || {});
        setCurrentFeature(data.default_feature || "code_completion");
        Promise.all([
          refreshWorkspace(data.default_project_dir || "", []),
          loadBrowse(data.default_project_dir || "")
        ]).catch((error) => setLastError(error.message));
      })
      .catch((error) => setLastError(error.message));
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!projectDir) {
      return;
    }
    const timer = window.setTimeout(() => {
      requestJson("/api/command-preview", {
        method: "POST",
        body: JSON.stringify(payload)
      })
        .then((data) => {
          setPreviewCommand(data.preview_command || "");
          setRunCommand(data.run_command || "");
          setProvider(data.provider || "openai");
          setApiKeyLabel(data.api_key_label || "not-set");
        })
        .catch((error) => setLastError(error.message));
    }, 120);
    return () => window.clearTimeout(timer);
  }, [payload, projectDir]);

  async function refreshWorkspace(dir = projectDir, nextTargets = targets) {
    setLastError("");
    const data = await requestJson("/api/workspace/scan", {
      method: "POST",
      body: JSON.stringify({ project_dir: dir, target_files: nextTargets })
    });
    setWorkspace(data);
    setTargets(data.selected_files || []);
    if (data.normalized_root) {
      setProjectDir(data.normalized_root);
    }
    setLastUpdated(new Date().toLocaleTimeString());
    return data;
  }

  async function loadBrowse(path = projectDir) {
    if (!path) {
      return;
    }
    setLastError("");
    const data = await requestJson(`/api/browse?path=${encodeURIComponent(path)}`);
    setBrowseState(data);
  }

  function toggleTarget(file) {
    setTargets((current) => {
      if (current.includes(file)) {
        return current.filter((item) => item !== file);
      }
      return [...current, file];
    });
  }

  function addCustomTarget() {
    const next = customTarget.trim().replaceAll("\\", "/");
    if (!next) {
      return;
    }
    setTargets((current) => (current.includes(next) ? current : [...current, next]));
    setCustomTarget("");
  }

  function makePrimary(file) {
    setTargets((current) => [file, ...current.filter((item) => item !== file)]);
  }

  function removeTarget(file) {
    setTargets((current) => current.filter((item) => item !== file));
  }

  async function handlePreview() {
    setBusy(true);
    setMode("running");
    setLastError("");
    setTerminalLog("Generating prompt preview...\n");
    try {
      const data = await requestJson("/api/prompt/preview", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      setTerminalLog(data.log || "");
      setPreviewCommand(data.command || previewCommand);
      setMode(data.status === "error" ? "error" : "preview");
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      setMode("error");
      setLastError(error.message);
      setTerminalLog(`Preview failed: ${error.message}\n`);
    } finally {
      setBusy(false);
    }
  }

  function buildFormData() {
    const formData = new FormData();
    formData.append("project_dir", projectDir);
    formData.append("target_files", JSON.stringify(targets));
    formData.append("instruction", instruction);
    formData.append("model", model);
    formData.append("api_key", apiKey || "");
    formData.append("feature", currentFeature);

    const cleanConfig = {};
    for (const [key, value] of Object.entries(featureConfig)) {
      if (value instanceof FileList) {
        for (let i = 0; i < value.length; i++) {
          formData.append(`files`, value[i]);
        }
      } else {
        cleanConfig[key] = value;
      }
    }
    formData.append("feature_config", JSON.stringify(cleanConfig));
    return formData;
  }

  async function handleRun() {
    setBusy(true);
    setMode("running");
    setLastError("");
    setTerminalLog("Preparing execution...\n");
    try {
      const hasFiles = Object.values(featureConfig).some(
        (v) => v instanceof FileList && v.length > 0
      );
      const options = hasFiles
        ? {
            method: "POST",
            body: buildFormData()
          }
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          };

      const response = await fetch(`${API_BASE}/api/run`, options);
      if (!response.ok || !response.body) {
        throw new Error(response.statusText || "Execution stream failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) {
            continue;
          }
          const event = JSON.parse(line);
          if (event.command) {
            setRunCommand(event.command);
          }
          if (event.log !== undefined) {
            setTerminalLog(event.log);
          }
          if (event.status) {
            setMode(event.status === "running" ? "running" : event.status);
          }
        }
      }
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      setMode("error");
      setLastError(error.message);
      setTerminalLog((current) => `${current}\nExecution failed: ${error.message}\n`);
    } finally {
      setBusy(false);
    }
  }

  function handleFeatureChange(featureName) {
    setCurrentFeature(featureName);
    setFeatureConfig({});
  }

  function handleFeatureConfigChange(name, value) {
    setFeatureConfig((prev) => ({ ...prev, [name]: value }));
  }

  function handleClear() {
    setInstruction("");
    setTerminalLog("");
    setMode("idle");
    setLastError("");
  }

  async function copyCommand() {
    const text = mode === "preview" ? previewCommand : runCommand;
    if (!text) {
      return;
    }
    await navigator.clipboard.writeText(text);
  }

  return (
    <main className="app-shell">
      <aside className="workspace-rail">
        <header className="app-title">
          <div className="mark"><Code2 size={18} /></div>
          <div>
            <h1>NaturalCC Agent</h1>
            <p>Workbench</p>
          </div>
        </header>

        <section className="panel-section">
          <div className="section-heading">
            <FolderOpen size={16} />
            <span>Workspace</span>
          </div>
          <label className="field-label" htmlFor="projectDir">Project root</label>
          <div className="path-row">
            <input
              id="projectDir"
              className="text-input"
              value={projectDir}
              onChange={(event) => setProjectDir(event.target.value)}
              onBlur={() => loadBrowse(projectDir).catch((error) => setLastError(error.message))}
              placeholder="/path/to/project"
            />
            <button
              className="icon-button"
              type="button"
              title="Refresh workspace"
              onClick={() => refreshWorkspace().then(() => loadBrowse(projectDir)).catch((error) => setLastError(error.message))}
            >
              <RefreshCcw size={16} />
            </button>
          </div>

          <div className="directory-browser">
            <div className="browser-bar">
              <button
                type="button"
                className="ghost-button"
                title="Open parent"
                disabled={!browseState.parent}
                onClick={() => loadBrowse(browseState.parent).catch((error) => setLastError(error.message))}
              >
                <ChevronUp size={15} />
                Parent
              </button>
              <button
                type="button"
                className="ghost-button"
                title="Use current browser directory"
                onClick={() => refreshWorkspace(browseState.path, targets).then(() => loadBrowse(browseState.path)).catch((error) => setLastError(error.message))}
              >
                <Check size={15} />
                Use
              </button>
            </div>
            <div className="browser-path" title={browseState.path}>{browseState.path || "No directory loaded"}</div>
            <div className="directory-list">
              {(browseState.directories || []).slice(0, 36).map((item) => (
                <button
                  type="button"
                  className="directory-item"
                  key={item.path}
                  title={item.path}
                  onClick={() => loadBrowse(item.path).catch((error) => setLastError(error.message))}
                >
                  <Folder size={14} />
                  <span>{item.name}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="panel-section file-section">
          <div className="section-heading">
            <FileCode size={16} />
            <span>Target files</span>
            <strong>{targets.length}</strong>
          </div>
          <div className="search-row">
            <Search size={15} />
            <input
              ref={fileFilterRef}
              value={fileFilter}
              onChange={(event) => setFileFilter(event.target.value)}
              placeholder="Filter files"
            />
            {fileFilter ? (
              <button
                type="button"
                className="search-clear-button"
                title="Clear filter"
                aria-label="Clear filter"
                onClick={() => {
                  setFileFilter("");
                  fileFilterRef.current?.focus();
                }}
              >
                <X size={14} />
              </button>
            ) : null}
          </div>
          <div className="file-list">
            {filteredFiles.map((file) => (
              <label className="file-row" key={file} title={file}>
                <input
                  type="checkbox"
                  checked={targets.includes(file)}
                  onChange={() => toggleTarget(file)}
                />
                <span>{file}</span>
              </label>
            ))}
          </div>
          <div className="custom-target-row">
            <input
              className="text-input"
              value={customTarget}
              onChange={(event) => setCustomTarget(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  addCustomTarget();
                }
              }}
              placeholder="Add relative target"
            />
            <button type="button" className="icon-button" title="Add target" onClick={addCustomTarget}>
              <Check size={16} />
            </button>
          </div>
        </section>
      </aside>

      <section className="center-stage">
        <div className="stage-header">
          <div>
            <span className={classNames("status-dot", mode)} />
            <span className="eyebrow">{statusTitle}</span>
            <h2>{primaryTarget || "No primary target selected"}</h2>
          </div>
          <div className="stage-actions">
            <button type="button" className="secondary-button" disabled={busy} onClick={handlePreview}>
              <Sparkles size={16} />
              Preview
            </button>
            <button type="button" className="primary-button" disabled={busy} onClick={handleRun}>
              <Play size={16} />
              Execute
            </button>
            <button type="button" className="icon-button" title="Clear task" disabled={busy} onClick={handleClear}>
              <Trash2 size={16} />
            </button>
          </div>
        </div>

        <section className="composer">
          <label className="field-label" htmlFor="instruction">Development instruction</label>
          <textarea
            id="instruction"
            value={instruction}
            onChange={(event) => setInstruction(event.target.value)}
            placeholder="Complete parse_flags and follow the existing error handling style."
          />
        </section>

        <section className="timeline">
          <div className="terminal-header">
            <div>
              <Terminal size={16} />
              <span>Execution timeline</span>
            </div>
            <button type="button" className="ghost-button" title="Copy command" onClick={copyCommand}>
              <Clipboard size={15} />
              Copy CLI
            </button>
          </div>
          <pre className="terminal-output">{terminalLog || "No output yet."}</pre>
        </section>
      </section>

      <aside className="inspector">
        <section className="panel-section">
          <div className="section-heading">
            <Settings2 size={16} />
            <span>Model</span>
          </div>
          <label className="field-label" htmlFor="model">Model</label>
          <input
            id="model"
            className="text-input"
            value={model}
            onChange={(event) => setModel(event.target.value)}
            list="model-options"
          />
          <datalist id="model-options">
            {models.map((item) => <option value={item} key={item} />)}
          </datalist>
          <label className="field-label" htmlFor="apiKey">API key</label>
          <input
            id="apiKey"
            className="text-input"
            value={apiKey}
            type="password"
            onChange={(event) => setApiKey(event.target.value)}
            placeholder="Falls back to environment"
          />
        </section>

        <section className="panel-section">
          <div className="section-heading">
            <Braces size={16} />
            <span>Feature</span>
          </div>
          <label className="field-label" htmlFor="feature">Mode</label>
          <select
            id="feature"
            className="text-input"
            value={currentFeature}
            onChange={(event) => handleFeatureChange(event.target.value)}
          >
            {features.map((f) => (
              <option value={f.name} key={f.name}>
                {f.label}
              </option>
            ))}
          </select>
          {featureSchemas[currentFeature] && (
            <DynamicForm
              schema={featureSchemas[currentFeature]}
              values={featureConfig}
              onChange={handleFeatureConfigChange}
            />
          )}
        </section>

        <section className="panel-section selected-section">
          <div className="section-heading">
            <Activity size={16} />
            <span>Status</span>
          </div>
          <div className="metrics-grid">
            <Metric label="Visible" value={workspace.counts.visible_files} />
            <Metric label="Parsable" value={workspace.counts.parsable_files} />
            <Metric label="Targets" value={targets.length} />
            <Metric label="Custom" value={customTargetCount} />
          </div>
          <dl className="info-list">
            <Info label="Provider" value={provider} />
            <Info label="API key" value={apiKeyLabel} />
            <Info label="Updated" value={lastUpdated || "not yet"} />
            <Info label="Root" value={workspace.normalized_root || projectDir} />
          </dl>
          {lastError ? <div className="error-box">{lastError}</div> : null}
        </section>

        <section className="panel-section selected-section">
          <div className="section-heading">
            <Circle size={16} />
            <span>Primary parse order</span>
          </div>
          <div className="selected-list">
            {targets.length === 0 ? (
              <p className="muted">Select at least one target file.</p>
            ) : (
              targets.map((file, index) => (
                <div className="selected-row" key={file} title={file}>
                  <span className="rank">{index + 1}</span>
                  <span className="selected-path">{file}</span>
                  {index > 0 ? (
                    <button type="button" className="mini-button" onClick={() => makePrimary(file)}>Primary</button>
                  ) : (
                    <span className="primary-pill">Primary</span>
                  )}
                  <button type="button" className="mini-icon" title="Remove target" onClick={() => removeTarget(file)}>
                    <Trash2 size={13} />
                  </button>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="panel-section command-section">
          <div className="section-heading">
            <Terminal size={16} />
            <span>Equivalent CLI</span>
          </div>
          <pre className="command-preview">{mode === "preview" ? previewCommand : runCommand}</pre>
        </section>
      </aside>
    </main>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{metricLabel(value)}</strong>
    </div>
  );
}

function Info({ label, value }) {
  return (
    <>
      <dt>{label}</dt>
      <dd title={String(value || "")}>{metricLabel(value)}</dd>
    </>
  );
}

export default App;
