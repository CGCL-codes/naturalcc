import {
  Activity,
  BookmarkPlus,
  Bot,
  Braces,
  Check,
  ChevronUp,
  Circle,
  Clipboard,
  Code2,
  Database,
  FileCode,
  Folder,
  FolderOpen,
  Menu,
  Moon,
  Network,
  PanelRightClose,
  Pencil,
  Play,
  RefreshCcw,
  Search,
  Send,
  Settings2,
  Sparkles,
  Sun,
  Terminal,
  Trash2,
  User,
  X
} from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { initialAgentState, reduceAgentEvents } from "./agentState.js";
import {
  buildAgentRunMessage,
  canControlAgentRun,
  modelInputValue,
  shouldAppendUserMessage
} from "./agentUi.js";
import {
  budgetProgress,
  defaultBudgetDraft,
  normalizeBudgetDraft,
  validateBudgetChange
} from "./budgetUi.js";
import {
  buildThreadMessagePayload,
  detectContextCandidate,
  groupThreads,
  hydrateConversationMessages,
  selectThreadAfterDeletion,
  threadNeedsCancellation
} from "./conversationState.js";
import { loadLayoutPrefs, saveLayoutPrefs, toLayoutStyle } from "./layoutPrefs.js";
import { loadThemePref, nextTheme, saveThemePref } from "./themePrefs.js";
import { MemoryReviewPanel } from "./MemoryReviewPanel.jsx";
import {
  buildMemoryProposalSelection,
  normalizeMemoryReview,
  toggleEvidenceSelection
} from "./memoryReview.js";

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

const defaultCodeGraphCapabilities = {
  codegraph: {
    enabled: false,
    auto_sync: true,
    hide_workspace_search: true
  }
};

function codeGraphStatusLabel(enabled, status) {
  if (!enabled) return "Off";
  if (!status) return "Checking";
  if (!status.installed) return "Not installed";
  if (!status.available) return "Unavailable";
  if (!status.initialized) return "Not initialized";
  if (status.stale) return "Needs sync";
  if (status.ready) return "Ready";
  return "Error";
}

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

function formatTime(date) {
  if (!date) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
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
        checked={value === undefined ? !!field.default : !!value}
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
  const chatInputRef = useRef(null);
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
  const [runtimeMode, setRuntimeMode] = useState("agent");
  const [agentState, setAgentState] = useState(initialAgentState);
  const [agentRunState, setAgentRunState] = useState(null);
  const [agentRuns, setAgentRuns] = useState([]);
  const [agentMemories, setAgentMemories] = useState([]);
  const [memoryProposals, setMemoryProposals] = useState([]);
  const [memoryError, setMemoryError] = useState("");
  const [selectedMemoryEvidence, setSelectedMemoryEvidence] = useState([]);
  const [agentThreads, setAgentThreads] = useState([]);
  const [activeThread, setActiveThread] = useState(null);
  const [threadSearch, setThreadSearch] = useState("");
  const [contextItems, setContextItems] = useState([]);
  const [contextSuggestions, setContextSuggestions] = useState([]);
  const [budgetDraft, setBudgetDraft] = useState(defaultBudgetDraft);
  const [budgetOpen, setBudgetOpen] = useState(false);
  const [codeGraphEnabled, setCodeGraphEnabled] = useState(false);
  const [codeGraphStatus, setCodeGraphStatus] = useState(null);
  const [codeGraphBusy, setCodeGraphBusy] = useState(false);
  const [runDetailsOpen, setRunDetailsOpen] = useState(false);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [threadPendingDelete, setThreadPendingDelete] = useState(null);
  const [deletingThreadId, setDeletingThreadId] = useState("");
  const [cancellingThreadId, setCancellingThreadId] = useState("");
  const [threadDeleteError, setThreadDeleteError] = useState("");
  const [renamingThreadId, setRenamingThreadId] = useState("");
  const [renamingDraft, setRenamingDraft] = useState("");
  const [renamingBusy, setRenamingBusy] = useState(false);
  const [renameError, setRenameError] = useState("");

  // Ref to prevent duplicate user messages when Send triggers Run
  const userMessageAddedRef = useRef(false);

  // New UI states
  const [messages, setMessages] = useState([]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeMessageId, setActiveMessageId] = useState(null);
  const [layoutPrefs, setLayoutPrefs] = useState(() => loadLayoutPrefs());
  const [theme, setTheme] = useState(() => loadThemePref());
  const messagesAreaRef = useRef(null);
  const messagesEndRef = useRef(null);
  const agentRunControllable = canControlAgentRun(agentState.status);
  const filteredThreads = useMemo(() => {
    const query = threadSearch.trim().toLowerCase();
    if (!query) return agentThreads;
    return agentThreads.filter((thread) => (
      thread.title?.toLowerCase().includes(query)
      || thread.preview?.toLowerCase().includes(query)
      || thread.workspace?.toLowerCase().includes(query)
    ));
  }, [agentThreads, threadSearch]);
  const threadGroups = useMemo(() => groupThreads(filteredThreads), [filteredThreads]);
  const llmBudgetProgress = budgetProgress(agentRunState?.llm_calls || 0, budgetDraft.max_llm_calls);
  const toolBudgetProgress = budgetProgress(agentRunState?.tool_calls || 0, budgetDraft.max_tool_calls);
  const inputTokenBudgetProgress = budgetProgress(
    agentRunState?.input_tokens || 0,
    budgetDraft.max_input_tokens
  );
  const promptCacheHitTokens = agentRunState?.prompt_cache_hit_tokens || 0;
  const promptCacheMissTokens = agentRunState?.prompt_cache_miss_tokens || 0;
  const promptCacheMeasuredTokens = promptCacheHitTokens + promptCacheMissTokens;
  const promptCacheLabel = promptCacheMeasuredTokens
    ? `${promptCacheHitTokens.toLocaleString()} / ${promptCacheMeasuredTokens.toLocaleString()} (${((promptCacheHitTokens / promptCacheMeasuredTokens) * 100).toFixed(1)}%)`
    : "not reported";

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
    attention: "Needs attention",
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

  useEffect(() => {
    const area = messagesAreaRef.current;
    if (area) {
      area.scrollTop = area.scrollHeight;
      return;
    }
    messagesEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages, mode]);

  useEffect(() => {
    if (drawerOpen || runDetailsOpen || threadPendingDelete) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [drawerOpen, runDetailsOpen, threadPendingDelete]);

  useEffect(() => {
    if (runtimeMode !== "agent") return;
    Promise.all([
      requestJson("/api/agent/threads"),
      requestJson("/api/agent/runs"),
      requestJson(`/api/agent/memories?project_id=${encodeURIComponent(projectDir)}`),
      requestJson(`/api/agent/memory-proposals?project_id=${encodeURIComponent(projectDir)}`)
    ])
      .then(([threadsPayload, runsPayload, memoriesPayload, proposalsPayload]) => {
        const threads = threadsPayload.threads || [];
        setAgentThreads(threads);
        setAgentRuns(runsPayload.runs || []);
        setAgentMemories(memoriesPayload.memories || []);
        setMemoryProposals((proposalsPayload.proposals || []).map(normalizeMemoryReview));
        if (!activeThread?.id && threads[0]?.id) {
          loadAgentThread(threads[0].id).catch((error) => setLastError(error.message));
        }
      })
      .catch((error) => setLastError(error.message));
  }, [runtimeMode, projectDir, activeThread?.id]);

  useEffect(() => {
    if (runtimeMode !== "agent" || !activeThread?.id) {
      setContextSuggestions([]);
      return;
    }
    const candidate = detectContextCandidate(instruction, instruction.length);
    if (candidate?.kind !== "mention" || !candidate.query.trim()) {
      setContextSuggestions([]);
      return;
    }
    const timer = window.setTimeout(() => {
      requestJson("/api/agent/context/resolve", {
        method: "POST",
        body: JSON.stringify({ thread_id: activeThread.id, value: candidate.value })
      })
        .then((payload) => setContextSuggestions(payload.matches || []))
        .catch(() => setContextSuggestions([]));
    }, 140);
    return () => window.clearTimeout(timer);
  }, [runtimeMode, activeThread?.id, instruction]);

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

  function addMessage(type, content, status = "complete", extra = {}) {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const msg = { id, type, content, timestamp: new Date(), status, ...extra };
    setMessages((prev) => [...prev, msg]);
    return id;
  }

  function updateMessage(id, updates) {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...updates } : m)));
  }

  async function handlePreview() {
    if (!instruction.trim() && !canRunWithoutInstruction()) {
      setLastError("Please enter an instruction first.");
      return;
    }
    setBusy(true);
    setMode("running");
    setLastError("");
    setTerminalLog("Generating prompt preview...\n");

    addMessage("user", instruction || defaultTaskLabel());
    const assistantId = addMessage("assistant", "Generating prompt preview...", "running");
    setActiveMessageId(assistantId);

    try {
      const data = await requestJson("/api/prompt/preview", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      setTerminalLog(data.log || "");
      setPreviewCommand(data.command || previewCommand);
      setMode(data.status === "error" ? "error" : "preview");
      setLastUpdated(new Date().toLocaleTimeString());
      updateMessage(assistantId, {
        content: data.log || "Preview generated.",
        status: data.status === "error" ? "error" : "complete",
        command: data.command || previewCommand
      });
    } catch (error) {
      setMode("error");
      setLastError(error.message);
      setTerminalLog(`Preview failed: ${error.message}\n`);
      updateMessage(assistantId, {
        content: `Preview failed: ${error.message}`,
        status: "error"
      });
    } finally {
      setBusy(false);
      setActiveMessageId(null);
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
          formData.append(key, value[i]);
        }
      } else {
        cleanConfig[key] = value;
      }
    }
    formData.append("feature_config", JSON.stringify(cleanConfig));
    return formData;
  }

  async function handlePipelineRun() {
    if (!instruction.trim() && !canRunWithoutInstruction()) {
      setLastError("Please enter an instruction first.");
      return;
    }
    setBusy(true);
    setMode("running");
    setLastError("");
    setTerminalLog("Preparing execution...\n");

    if (!userMessageAddedRef.current) {
      addMessage("user", instruction || defaultTaskLabel());
    }
    userMessageAddedRef.current = false;

    const assistantId = addMessage("assistant", "Preparing execution...", "running");
    setActiveMessageId(assistantId);

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
      let latestLog = "Preparing execution...\n";
      let latestCommand = "";
      const applyStreamEvent = (event) => {
        if (event.command) {
          latestCommand = event.command;
          setRunCommand(event.command);
        }
        if (event.log !== undefined) {
          latestLog = event.log;
          setTerminalLog(event.log);
        }
        if (event.status) {
          setMode(event.status === "running" ? "running" : event.status);
        }
        updateMessage(assistantId, {
          content: latestLog,
          command: latestCommand || runCommand,
          status: event.status === "running" ? "running" : event.status
        });
      };

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
          applyStreamEvent(event);
          await nextFrame();
        }
      }
      if (buffer.trim()) {
        applyStreamEvent(JSON.parse(buffer));
        await nextFrame();
      }
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      setMode("error");
      setLastError(error.message);
      setTerminalLog((current) => `${current}\nExecution failed: ${error.message}\n`);
      updateMessage(assistantId, {
        content: `Execution failed: ${error.message}`,
        status: "error"
      });
    } finally {
      setBusy(false);
      setActiveMessageId(null);
    }
  }

  async function refreshThreadList() {
    const payload = await requestJson("/api/agent/threads");
    setAgentThreads(payload.threads || []);
    return payload.threads || [];
  }

  async function loadAgentThread(threadId) {
    setConversationLoading(true);
    setLastError("");
    try {
      const detail = await requestJson(`/api/agent/threads/${threadId}`);
      const thread = detail.thread;
      setRuntimeMode("agent");
      setActiveThread(thread);
      setProjectDir(thread.workspace || projectDir);
      setModel(thread.model || model);
      setBudgetDraft(normalizeBudgetDraft(thread.budget, budgetDraft));
      setContextItems(thread.context_items || []);
      const graphEnabled = Boolean(thread.capabilities?.codegraph?.enabled);
      setCodeGraphEnabled(graphEnabled);
      setCodeGraphStatus(null);
      setTargets((thread.context_items || []).map((item) => item.path).filter(Boolean));
      setMessages(hydrateConversationMessages(detail.messages || []));
      setSelectedMemoryEvidence([]);
      const proposalPayload = await requestJson(
        `/api/agent/memory-proposals?project_id=${encodeURIComponent(thread.workspace || projectDir)}&thread_id=${encodeURIComponent(thread.id)}`
      );
      setMemoryProposals((proposalPayload.proposals || []).map(normalizeMemoryReview));
      if (detail.last_run?.run_id) {
        await refreshAgentState(detail.last_run.run_id, thread.workspace);
      } else {
        setAgentState(initialAgentState);
        setAgentRunState(null);
        setMode("idle");
      }
      if (graphEnabled) {
        const graphPayload = await requestJson(`/api/agent/threads/${thread.id}/codegraph/status`);
        setCodeGraphStatus(graphPayload.status || null);
      }
      setSidebarOpen(false);
      return detail;
    } finally {
      setConversationLoading(false);
    }
  }

  async function createNewThread(title = "New task") {
    if (!projectDir.trim()) {
      throw new Error("Set a workspace before creating a conversation.");
    }
    const payload = await requestJson("/api/agent/threads", {
      method: "POST",
      body: JSON.stringify({
        title,
        workspace: projectDir,
        model,
        runtime_mode: "agent",
        budget: budgetDraft,
        context_items: [],
        capabilities: {
          codegraph: {
            ...defaultCodeGraphCapabilities.codegraph,
            enabled: codeGraphEnabled
          }
        }
      })
    });
    setActiveThread(payload.thread);
    setMessages([]);
    setSelectedMemoryEvidence([]);
    setContextItems([]);
    setTargets([]);
    setAgentState(initialAgentState);
    setAgentRunState(null);
    setMode("idle");
    await refreshThreadList();
    return payload.thread;
  }

  async function ensureAgentThread(goal) {
    if (activeThread?.id) {
      return activeThread;
    }
    return createNewThread(goal.slice(0, 80) || "New task");
  }

  async function updateActiveThread(changes) {
    if (!activeThread?.id) return null;
    const payload = await requestJson(`/api/agent/threads/${activeThread.id}`, {
      method: "PATCH",
      body: JSON.stringify({
        ...changes,
        expected_version: activeThread.version
      })
    });
    setActiveThread(payload.thread);
    await refreshThreadList();
    return payload.thread;
  }

  function resetConversationView() {
    setActiveThread(null);
    setMessages([]);
    setContextItems([]);
    setContextSuggestions([]);
    setTargets([]);
    setInstruction("");
    setAgentState(initialAgentState);
    setAgentRunState(null);
    setMode("idle");
    setBusy(false);
    setActiveMessageId(null);
    setSelectedMemoryEvidence([]);
    setMemoryProposals([]);
    setRunDetailsOpen(false);
    setBudgetOpen(false);
    setCodeGraphEnabled(false);
    setCodeGraphStatus(null);
    setConversationLoading(false);
  }

  function requestThreadDeletion(event, thread) {
    event.stopPropagation();
    setThreadDeleteError("");
    setThreadPendingDelete(thread);
  }

  async function deletePendingThread() {
    const thread = threadPendingDelete;
    if (!thread || deletingThreadId || cancellingThreadId) return;
    setDeletingThreadId(thread.id);
    setThreadDeleteError("");
    setLastError("");
    try {
      await requestJson(`/api/agent/threads/${thread.id}`, { method: "DELETE" });
      const [threads, runsPayload] = await Promise.all([
        refreshThreadList(),
        requestJson("/api/agent/runs")
      ]);
      setAgentRuns(runsPayload.runs || []);
      const nextThreadId = selectThreadAfterDeletion(
        threads,
        thread.id,
        activeThread?.id
      );
      setThreadPendingDelete(null);
      if (thread.id !== activeThread?.id) {
        return;
      }
      if (nextThreadId) {
        await loadAgentThread(nextThreadId);
      } else {
        resetConversationView();
      }
    } catch (error) {
      setThreadDeleteError(error.message);
      setLastError(error.message);
    } finally {
      setDeletingThreadId("");
    }
  }

  async function cancelPendingThreadRun() {
    const thread = threadPendingDelete;
    if (!thread?.last_run_id || deletingThreadId || cancellingThreadId) return;
    setCancellingThreadId(thread.id);
    setThreadDeleteError("");
    setLastError("");
    try {
      await requestJson(`/api/agent/runs/${thread.last_run_id}/cancel`, {
        method: "POST"
      });
      const [detail, threadsPayload, runsPayload] = await Promise.all([
        requestJson(`/api/agent/threads/${thread.id}`),
        requestJson("/api/agent/threads"),
        requestJson("/api/agent/runs")
      ]);
      setThreadPendingDelete(detail.thread);
      setAgentThreads(threadsPayload.threads || []);
      setAgentRuns(runsPayload.runs || []);
      if (activeThread?.id === thread.id) {
        setActiveThread(detail.thread);
        await refreshAgentState(thread.last_run_id, thread.workspace);
      }
    } catch (error) {
      setThreadDeleteError(error.message);
      setLastError(error.message);
    } finally {
      setCancellingThreadId("");
    }
  }

  function startThreadRename(thread) {
    if (!thread?.id || renamingBusy) return;
    setRenamingThreadId(thread.id);
    setRenamingDraft(thread.title || "");
    setRenameError("");
  }

  function cancelThreadRename() {
    if (renamingBusy) return;
    setRenamingThreadId("");
    setRenamingDraft("");
    setRenameError("");
  }

  async function commitThreadRename() {
    const threadId = renamingThreadId;
    const thread = agentThreads.find((item) => item.id === threadId)
      || (activeThread?.id === threadId ? activeThread : null);
    const nextTitle = renamingDraft.trim();
    if (!thread || !nextTitle || nextTitle === (thread.title || "")) {
      cancelThreadRename();
      return;
    }
    setRenamingBusy(true);
    setRenameError("");
    setLastError("");
    try {
      const response = await requestJson(`/api/agent/threads/${thread.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          title: nextTitle,
          expected_version: thread.version
        })
      });
      if (activeThread?.id === thread.id) {
        setActiveThread((current) => (
          current && current.id === thread.id ? { ...current, ...response.thread } : current
        ));
      }
      await refreshThreadList();
      setRenamingThreadId("");
      setRenamingDraft("");
    } catch (error) {
      setRenameError(error.message);
      setLastError(error.message);
    } finally {
      setRenamingBusy(false);
    }
  }

  function addContextItem(item) {
    setContextItems((current) => {
      const key = item.absolute_path || item.path;
      if (current.some((entry) => (entry.absolute_path || entry.path) === key)) {
        return current;
      }
      const next = [...current, item];
      setTargets(next.map((entry) => entry.path).filter(Boolean));
      return next;
    });
    setContextSuggestions([]);
  }

  function removeContextItem(item) {
    setContextItems((current) => {
      const key = item.absolute_path || item.path;
      const next = current.filter((entry) => (entry.absolute_path || entry.path) !== key);
      setTargets(next.map((entry) => entry.path).filter(Boolean));
      return next;
    });
  }

  function selectContextSuggestion(item) {
    addContextItem(item);
    setInstruction((current) => current.replace(/(?:^|\s)@[^\s@]*$/, "").trimEnd());
    requestAnimationFrame(() => chatInputRef.current?.focus());
  }

  async function resolveAbsoluteContextFromInput() {
    if (!activeThread?.id) {
      setLastError("Create or select a conversation before adding context.");
      return;
    }
    const candidate = detectContextCandidate(instruction, instruction.length);
    if (candidate?.kind !== "absolute") {
      setInstruction((current) => `${current}${current && !current.endsWith(" ") ? " " : ""}@`);
      requestAnimationFrame(() => chatInputRef.current?.focus());
      return;
    }
    try {
      const payload = await requestJson("/api/agent/context/resolve", {
        method: "POST",
        body: JSON.stringify({ thread_id: activeThread.id, value: candidate.value })
      });
      if (payload.matches?.[0]) {
        addContextItem(payload.matches[0]);
        setInstruction("");
        if (payload.thread) setActiveThread(payload.thread);
      }
    } catch (error) {
      setLastError(error.message);
    }
  }

  async function saveBudget() {
    const normalized = normalizeBudgetDraft(budgetDraft, activeThread?.budget || budgetDraft);
    const errors = validateBudgetChange(normalized, agentRunState || {});
    if (Object.keys(errors).length) {
      setLastError(Object.values(errors)[0]);
      return;
    }
    try {
      if (agentState.runId && agentRunControllable) {
        await requestJson(`/api/agent/runs/${agentState.runId}/budget`, {
          method: "PATCH",
          body: JSON.stringify({ budget: normalized })
        });
        await refreshAgentState(agentState.runId);
        const threadPayload = await requestJson(`/api/agent/threads/${activeThread.id}`);
        setActiveThread(threadPayload.thread);
      } else if (activeThread?.id) {
        await updateActiveThread({ budget: normalized });
      }
      setBudgetDraft(normalized);
      setBudgetOpen(false);
      setLastError("");
    } catch (error) {
      setLastError(error.message);
    }
  }

  async function refreshAgentState(runId, memoryProjectId = projectDir) {
    const [eventPayload, runState, runsPayload, memoriesPayload, proposalsPayload] = await Promise.all([
      requestJson(`/api/agent/runs/${runId}/events`),
      requestJson(`/api/agent/runs/${runId}`),
      requestJson("/api/agent/runs"),
      requestJson(`/api/agent/memories?project_id=${encodeURIComponent(memoryProjectId)}`),
      requestJson(`/api/agent/memory-proposals?project_id=${encodeURIComponent(memoryProjectId)}`)
    ]);
    const next = reduceAgentEvents({ ...initialAgentState, runId }, eventPayload.events || []);
    setAgentState(next);
    setAgentRunState(runState);
    setAgentRuns(runsPayload.runs || []);
    setAgentMemories(memoriesPayload.memories || []);
    setMemoryProposals((proposalsPayload.proposals || []).map(normalizeMemoryReview));
    setMode(
      next.status === "completed"
        ? "success"
        : next.status === "failed" || next.status === "cancelled" || next.status === "budget_exhausted"
          ? "error"
          : next.status === "waiting_approval" || next.status === "paused"
            ? "attention"
            : "running"
    );
    return next;
  }

  async function handleAgentRun() {
    const submittedInstruction = instruction.trim();
    if (!submittedInstruction) {
      setLastError("Please enter an instruction first.");
      return;
    }
    setBusy(true);
    setMode("running");
    setLastError("");
    if (shouldAppendUserMessage(userMessageAddedRef.current)) {
      addMessage("user", submittedInstruction);
    }
    userMessageAddedRef.current = false;
    const assistantId = addMessage("assistant", "Creating durable Agent run...", "running");
    try {
      const thread = await ensureAgentThread(submittedInstruction);
      const created = await requestJson(`/api/agent/threads/${thread.id}/messages`, {
        method: "POST",
        body: JSON.stringify(
          buildThreadMessagePayload(
            submittedInstruction,
            contextItems,
            normalizeBudgetDraft(budgetDraft, thread.budget || budgetDraft),
            {
              codegraph: {
                ...defaultCodeGraphCapabilities.codegraph,
                enabled: codeGraphEnabled
              }
            },
            apiKey
          )
        )
      });
      const runId = created.run_id;
      setAgentState({ ...initialAgentState, runId, status: "queued" });
      setAgentRunState(created.state);
      const state = await requestJson(`/api/agent/runs/${runId}/run`, { method: "POST" });
      const next = await refreshAgentState(runId);
      const message = buildAgentRunMessage(state, next);
      updateMessage(assistantId, {
        content: message.content,
        status: message.status,
        approval: message.approval,
        runId
      });
      if (next.status === "completed") {
        await loadAgentThread(thread.id);
      } else {
        await refreshThreadList();
      }
    } catch (error) {
      setMode("error");
      setLastError(error.message);
      updateMessage(assistantId, { content: `Agent failed: ${error.message}`, status: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function refreshCodeGraphStatus(thread = activeThread) {
    if (!thread?.id) return null;
    setCodeGraphBusy(true);
    try {
      const payload = await requestJson(`/api/agent/threads/${thread.id}/codegraph/status`);
      setCodeGraphStatus(payload.status || null);
      return payload.status || null;
    } catch (error) {
      setLastError(error.message);
      return null;
    } finally {
      setCodeGraphBusy(false);
    }
  }

  async function handleCodeGraphToggle(enabled) {
    setCodeGraphEnabled(enabled);
    setCodeGraphStatus(null);
    if (!activeThread?.id) return;
    setCodeGraphBusy(true);
    try {
      const updated = await updateActiveThread({
        capabilities: {
          codegraph: {
            ...defaultCodeGraphCapabilities.codegraph,
            enabled
          }
        }
      });
      if (enabled) {
        const payload = await requestJson(`/api/agent/threads/${updated.id}/codegraph/status`);
        setCodeGraphStatus(payload.status || null);
      }
    } catch (error) {
      setCodeGraphEnabled(!enabled);
      setLastError(error.message);
    } finally {
      setCodeGraphBusy(false);
    }
  }

  async function runCodeGraphAction(action) {
    if (!activeThread?.id) return;
    setCodeGraphBusy(true);
    setLastError("");
    try {
      const payload = await requestJson(
        `/api/agent/threads/${activeThread.id}/codegraph/${action}`,
        {
          method: "POST",
          body: JSON.stringify(action === "visualize" ? {} : undefined)
        }
      );
      if (payload.status) setCodeGraphStatus(payload.status);
      if (payload.url) {
        window.open(`${API_BASE}${payload.url}`, "_blank", "noopener,noreferrer");
      }
    } catch (error) {
      setLastError(error.message);
    } finally {
      setCodeGraphBusy(false);
    }
  }

  async function handleAgentApproval() {
    if (!agentState.runId || !agentState.pendingApproval?.risk) return;
    setBusy(true);
    try {
      await requestJson(`/api/agent/runs/${agentState.runId}/approve`, {
        method: "POST",
        body: JSON.stringify({ risk: agentState.pendingApproval.risk })
      });
      const state = await requestJson(`/api/agent/runs/${agentState.runId}/run`, { method: "POST" });
      const next = await refreshAgentState(agentState.runId);
      const message = buildAgentRunMessage(state, next);
      addMessage(
        "assistant",
        state.final_answer
          ? message.content
          : `${message.content}\nChanged files: ${next.changedFiles.join(", ") || "none"}`,
        message.status,
        {
          approval: message.approval,
          runId: agentState.runId
        }
      );
      if (next.status === "completed") {
        await loadAgentThread(activeThread?.id);
      }
    } catch (error) {
      setLastError(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleAgentRejection() {
    if (!agentState.runId || !agentState.pendingApproval) return;
    setBusy(true);
    try {
      await requestJson(`/api/agent/runs/${agentState.runId}/reject`, { method: "POST" });
      const state = await requestJson(`/api/agent/runs/${agentState.runId}/run`, { method: "POST" });
      const next = await refreshAgentState(agentState.runId);
      const message = buildAgentRunMessage(state, next);
      addMessage(
        "assistant",
        state.final_answer ? message.content : `Rejected action.\n${message.content}`,
        message.status,
        {
          approval: message.approval,
          runId: agentState.runId
        }
      );
      if (next.status === "completed") {
        await loadAgentThread(activeThread?.id);
      }
    } catch (error) {
      setLastError(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleAgentControl(action) {
    if (!agentState.runId) return;
    setBusy(true);
    try {
      await requestJson(`/api/agent/runs/${agentState.runId}/${action}`, { method: "POST" });
      await refreshAgentState(agentState.runId);
    } catch (error) {
      setLastError(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function loadAgentRun(runId) {
    setRuntimeMode("agent");
    await refreshAgentState(runId);
  }

  async function handleMemoryAction(memoryId, action) {
    setBusy(true);
    try {
      await requestJson(
        `/api/agent/memories/${memoryId}${action === "delete" ? "" : `/${action}`}`,
        { method: action === "delete" ? "DELETE" : "POST" }
      );
      const payload = await requestJson(`/api/agent/memories?project_id=${encodeURIComponent(projectDir)}`);
      setAgentMemories(payload.memories || []);
    } catch (error) {
      setLastError(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function refreshMemoryGovernance() {
    const suffix = `project_id=${encodeURIComponent(projectDir)}${activeThread?.id ? `&thread_id=${encodeURIComponent(activeThread.id)}` : ""}`;
    const [proposalPayload, memoryPayload] = await Promise.all([
      requestJson(`/api/agent/memory-proposals?${suffix}`),
      requestJson(`/api/agent/memories?project_id=${encodeURIComponent(projectDir)}`)
    ]);
    setMemoryProposals((proposalPayload.proposals || []).map(normalizeMemoryReview));
    setAgentMemories(memoryPayload.memories || []);
  }

  function handleMemoryEvidenceToggle(messageId) {
    setSelectedMemoryEvidence((current) => toggleEvidenceSelection(current, messageId));
  }

  async function handleCreateMemoryProposals() {
    if (!activeThread?.id || !selectedMemoryEvidence.length) return;
    setBusy(true);
    setLastError("");
    setMemoryError("");
    try {
      const payload = await requestJson("/api/agent/memory-proposals/from-selection", {
        method: "POST",
        body: JSON.stringify(
          buildMemoryProposalSelection(activeThread.id, projectDir, selectedMemoryEvidence)
        )
      });
      setSelectedMemoryEvidence([]);
      await refreshMemoryGovernance();
      setRunDetailsOpen(true);
      if (!payload.count) {
        setLastUpdated("No durable memory was proposed from the selected evidence.");
      }
    } catch (error) {
      setLastError(error.message);
      setMemoryError(error.message);
      setRunDetailsOpen(true);
    } finally {
      setBusy(false);
    }
  }

  async function handleProposalAction(proposal, action, changes = null) {
    setBusy(true);
    setLastError("");
    setMemoryError("");
    try {
      if (action === "edit") {
        await requestJson(`/api/agent/memory-proposals/${proposal.proposal_id}`, {
          method: "PATCH",
          body: JSON.stringify({ expected_version: proposal.version, ...changes })
        });
      } else {
        await requestJson(`/api/agent/memory-proposals/${proposal.proposal_id}/${action}`, {
          method: "POST",
          body: JSON.stringify({ expected_version: proposal.version, reason: "" })
        });
      }
      await refreshMemoryGovernance();
    } catch (error) {
      setLastError(error.message);
      setMemoryError(error.message);
      await refreshMemoryGovernance().catch(() => {});
    } finally {
      setBusy(false);
    }
  }

  function handleRun() {
    return runtimeMode === "agent" ? handleAgentRun() : handlePipelineRun();
  }

  function canRunWithoutInstruction() {
    if (currentFeature === "knowledge_graph") {
      return true;
    }
    if (currentFeature === "code_repair") {
      return Boolean(
        (featureConfig.failure_log || "").trim()
        || (featureConfig.extra_context || "").trim()
      );
    }
    if (currentFeature === "design_to_code") {
      return Boolean(
        featureConfig.screenshot instanceof FileList && featureConfig.screenshot.length > 0
      );
    }
    return currentFeature === "code_summary"
      || (currentFeature === "vulnerability_detection" && !featureConfig.auto_fix);
  }

  function defaultTaskLabel() {
    if (currentFeature === "knowledge_graph") {
      return "Knowledge graph generation";
    }
    if (currentFeature === "code_summary") {
      return "Code summarization";
    }
    if (currentFeature === "vulnerability_detection") {
      return "Vulnerability scan";
    }
    if (currentFeature === "code_repair") {
      return "Code repair";
    }
    if (currentFeature === "design_to_code") {
      return "Design to code";
    }
    return "Code task";
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
    setMessages([]);
    setActiveMessageId(null);
  }

  async function copyCommand(command) {
    if (!command) {
      return;
    }
    await navigator.clipboard.writeText(command);
  }

  function handleSend() {
    if (busy) return;
    if (!instruction.trim() && !canRunWithoutInstruction()) {
      setLastError("Please enter an instruction first.");
      return;
    }
    userMessageAddedRef.current = true;
    addMessage("user", instruction || defaultTaskLabel());
    handleRun();
    setInstruction("");
    requestAnimationFrame(() => {
      if (chatInputRef.current) {
        chatInputRef.current.style.height = "auto";
      }
    });
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function autoResizeTextarea() {
    const el = chatInputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }

  function updateLayoutPrefs(nextValue) {
    setLayoutPrefs((current) => {
      const next = typeof nextValue === "function" ? nextValue(current) : nextValue;
      return saveLayoutPrefs(globalThis.localStorage, { ...current, ...next });
    });
  }

  function resizeDelta(kind, event, start) {
    if (kind === "sidebarWidth") {
      return event.clientX - start.x;
    }
    if (kind === "drawerWidth") {
      return start.x - event.clientX;
    }
    return event.clientY - start.y;
  }

  function startResize(kind, event) {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    const start = {
      x: event.clientX,
      y: event.clientY,
      value: layoutPrefs[kind]
    };
    document.body.classList.add("is-resizing-layout");
    const onPointerMove = (moveEvent) => {
      const delta = resizeDelta(kind, moveEvent, start);
      updateLayoutPrefs({ [kind]: start.value + delta });
    };
    const onPointerUp = () => {
      document.body.classList.remove("is-resizing-layout");
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
  }

  function handleResizeKey(kind, event) {
    const step = event.shiftKey ? 40 : 10;
    const keyDelta = {
      sidebarWidth: { ArrowLeft: -step, ArrowRight: step },
      drawerWidth: { ArrowLeft: step, ArrowRight: -step },
      agentStackHeight: { ArrowUp: -step, ArrowDown: step }
    }[kind]?.[event.key];
    if (!keyDelta) {
      return;
    }
    event.preventDefault();
    updateLayoutPrefs((current) => ({ [kind]: current[kind] + keyDelta }));
  }

  function toggleTheme() {
    setTheme((current) => saveThemePref(globalThis.localStorage, nextTheme(current)));
  }

  // ── Render ──

  return (
    <div className="app-root" data-theme={theme} style={toLayoutStyle(layoutPrefs)}>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="drawer-overlay"
          style={{ zIndex: 25 }}
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Left Sidebar */}
      <aside className={classNames("sidebar-left", sidebarOpen && "open")}>
        <header className="sidebar-header">
          <div className="mark"><Code2 size={18} /></div>
          <h1>NaturalCC Agent</h1>
        </header>
        <div className="conversation-sidebar-body">
          <button
            type="button"
            className="new-conversation-button"
            disabled={busy}
            onClick={() => createNewThread().catch((error) => setLastError(error.message))}
          >
            <Sparkles size={15} />
            New conversation
          </button>
          <div className="conversation-search">
            <Search size={14} />
            <input
              value={threadSearch}
              onChange={(event) => setThreadSearch(event.target.value)}
              placeholder="Search conversations"
            />
            {threadSearch && (
              <button type="button" onClick={() => setThreadSearch("")} aria-label="Clear search">
                <X size={13} />
              </button>
            )}
          </div>
          <nav className="conversation-history" aria-label="Conversation history">
            {threadGroups.map((group) => (
              <section className="conversation-group" key={group.label}>
                <h3>{group.label}</h3>
                {group.items.map((thread) => {
                  const isRenaming = renamingThreadId === thread.id;
                  const renameDisabled = renamingBusy && !isRenaming;
                  return (
                    <div
                      className={classNames(
                        "conversation-row-shell",
                        activeThread?.id === thread.id && "active",
                        isRenaming && "renaming"
                      )}
                      key={thread.id}
                    >
                      {isRenaming ? (
                        <>
                          <form
                            className="conversation-rename-form"
                            onSubmit={(event) => {
                              event.preventDefault();
                              commitThreadRename();
                            }}
                          >
                            <input
                              className="conversation-rename-input"
                              value={renamingDraft}
                              autoFocus
                              maxLength={200}
                              disabled={renamingBusy}
                              aria-label="Rename conversation"
                              placeholder="Conversation name"
                              onChange={(event) => setRenamingDraft(event.target.value)}
                              onKeyDown={(event) => {
                                if (event.key === "Escape") {
                                  event.preventDefault();
                                  cancelThreadRename();
                                }
                              }}
                              onBlur={(event) => {
                                // Only cancel when focus leaves the form entirely.
                                if (!event.currentTarget.parentElement?.contains(event.relatedTarget)) {
                                  cancelThreadRename();
                                }
                              }}
                            />
                            <button
                              type="submit"
                              className="conversation-rename-confirm"
                              disabled={renamingBusy || !renamingDraft.trim()}
                              aria-label="Save new name"
                              title="Save (Enter)"
                            >
                              <Check size={13} />
                            </button>
                            <button
                              type="button"
                              className="conversation-rename-cancel"
                              disabled={renamingBusy}
                              aria-label="Cancel rename"
                              title="Cancel (Esc)"
                              onMouseDown={(event) => event.preventDefault()}
                              onClick={cancelThreadRename}
                            >
                              <X size={13} />
                            </button>
                          </form>
                          {renameError && (
                            <p className="conversation-rename-error" role="alert">
                              {renameError}
                            </p>
                          )}
                        </>
                      ) : (
                        <>
                          <button
                            type="button"
                            className="conversation-row"
                            onClick={() => loadAgentThread(thread.id).catch((error) => setLastError(error.message))}
                          >
                            <span title={thread.title || "Untitled task"}>
                              {thread.title || "Untitled task"}
                            </span>
                            <small>
                              {thread.last_status || "ready"}
                              {thread.run_count ? ` · ${thread.run_count} run${thread.run_count === 1 ? "" : "s"}` : ""}
                            </small>
                          </button>
                          <button
                            type="button"
                            className="conversation-rename-button"
                            aria-label={`Rename conversation ${thread.title || "Untitled task"}`}
                            title="Rename conversation"
                            disabled={deletingThreadId === thread.id || renameDisabled}
                            onClick={(event) => {
                              event.stopPropagation();
                              startThreadRename(thread);
                            }}
                          >
                            <Pencil size={14} />
                          </button>
                          <button
                            type="button"
                            className="conversation-delete-button"
                            aria-label={`Delete conversation ${thread.title || "Untitled task"}`}
                            title="Delete conversation"
                            disabled={deletingThreadId === thread.id || renameDisabled}
                            onClick={(event) => requestThreadDeletion(event, thread)}
                          >
                            <Trash2 size={14} />
                          </button>
                        </>
                      )}
                    </div>
                  );
                })}
              </section>
            ))}
            {!threadGroups.length && (
              <p className="conversation-empty">No conversations found.</p>
            )}
          </nav>
        </div>
        <footer className="sidebar-footer">
          <button type="button" onClick={toggleTheme}>
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
            {theme === "dark" ? "Light theme" : "Dark theme"}
          </button>
          <button type="button" onClick={() => { setRunDetailsOpen(false); setDrawerOpen(true); }}>
            <Settings2 size={15} />
            Settings
          </button>
        </footer>
      </aside>
      <button
        type="button"
        className="resize-handle resize-handle-sidebar"
        aria-label="Resize left sidebar"
        title="Drag to resize sidebar"
        onPointerDown={(event) => startResize("sidebarWidth", event)}
        onKeyDown={(event) => handleResizeKey("sidebarWidth", event)}
      />

      {/* Main Chat Stage */}
      <main className="chat-stage">
        {/* Header */}
        <header className="stage-header">
          <div className="stage-title-row">
            <button
              type="button"
              className="mobile-menu-btn"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu size={18} />
            </button>
            <div>
              <span className="eyebrow">
                <span className={classNames("status-dot", mode)} />
                {conversationLoading ? "Restoring conversation" : statusTitle}
              </span>
              <h2>{activeThread?.title || "New conversation"}</h2>
            </div>
          </div>
          <div className="stage-actions">
            <input
              className="header-workspace-input"
              value={projectDir}
              title={projectDir}
              onChange={(event) => setProjectDir(event.target.value)}
              onBlur={() => {
                refreshWorkspace(projectDir, targets)
                  .then(() => activeThread?.id && updateActiveThread({ workspace: projectDir }))
                  .catch((error) => setLastError(error.message));
              }}
              placeholder="Workspace path"
            />
            <select
              className="header-select"
              value={model}
              onChange={(event) => {
                const value = event.target.value;
                setModel(value);
                if (activeThread?.id) {
                  updateActiveThread({ model: value }).catch((error) => setLastError(error.message));
                }
              }}
            >
              {models.map((item) => <option value={item} key={item}>{item}</option>)}
              {!models.includes(model) && model && <option value={model}>{model}</option>}
            </select>
            <select
              className="header-select runtime-select"
              value={runtimeMode}
              onChange={(event) => setRuntimeMode(event.target.value)}
              aria-label="Runtime mode"
            >
              <option value="agent">Agent</option>
              <option value="pipeline">Pipeline</option>
            </select>
            <button
              type="button"
              className="icon-button"
              title="Open settings"
              onClick={() => { setRunDetailsOpen(false); setDrawerOpen(true); }}
            >
              <PanelRightClose size={16} />
            </button>
          </div>
        </header>

        {runtimeMode === "agent" && (
          <section className="budget-toolbar">
            <BudgetMeter label="LLM calls" progress={llmBudgetProgress} />
            <BudgetMeter label="Tool calls" progress={toolBudgetProgress} />
            <BudgetMeter label="Input tokens" progress={inputTokenBudgetProgress} />
            <button
              type="button"
              className="run-details-button"
              onClick={() => { setDrawerOpen(false); setRunDetailsOpen(true); }}
            >
              <Activity size={14} />
              Run details
              {agentState.runId && <span className={classNames("run-status-pill", agentState.status)}>{agentState.status}</span>}
            </button>
          </section>
        )}

        {/* Messages Area */}
        <div className="messages-area" ref={messagesAreaRef}>
          {messages.length === 0 ? (
            <div className="messages-empty">
              <Sparkles size={32} />
              <p>{activeThread ? "Continue this conversation below." : "Start a new coding conversation."}</p>
              <p className="messages-empty-hint">
                Type @ to add workspace files, or paste an absolute path and choose Add context.
              </p>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={classNames("message-card", `message-${msg.type}`)}>
                <div className="message-avatar-column">
                  <div className="message-avatar">
                    {msg.type === "user" ? <User size={14} /> : <Bot size={14} />}
                  </div>
                  <span className="message-meta">{formatTime(msg.timestamp)}</span>
                </div>
                <div className="message-bubble">
                  {msg.type === "assistant" && msg.status === "running" ? (
                    <>
                      <div className="message-status-row">
                        <span className="status-dot running" />
                        <span style={{ color: "var(--amber)" }}>Running...</span>
                      </div>
                      <pre className="terminal-body">{msg.content}</pre>
                    </>
                  ) : msg.type === "assistant" ? (
                    <>
                      {msg.status === "error" && (
                        <div className="message-status-row">
                          <span className="status-dot error" />
                          <span style={{ color: "var(--danger)" }}>Error</span>
                        </div>
                      )}
                      <pre className="terminal-body">{msg.content}</pre>
                      {msg.approval && agentState.runId === msg.runId && agentState.pendingApproval && (
                        <div className="message-action-row">
                          <button
                            type="button"
                            className="message-primary-action"
                            disabled={busy}
                            onClick={handleAgentApproval}
                          >
                            <Check size={13} />
                            Approve {msg.approval.risk}
                          </button>
                          <button
                            type="button"
                            className="message-action-btn"
                            disabled={busy}
                            onClick={handleAgentRejection}
                          >
                            <X size={13} />
                            Reject
                          </button>
                        </div>
                      )}
                      {msg.command && (
                        <button
                          type="button"
                          className="message-action-btn"
                          onClick={() => copyCommand(msg.command)}
                        >
                          <Clipboard size={11} />
                          Copy CLI
                        </button>
                      )}
                      {msg.artifacts?.html && (
                        <div className="artifact-viewer">
                          <div className="artifact-header">
                            <span>
                              HTML artifact: {msg.artifacts.nodes || 0} nodes, {msg.artifacts.edges || 0} edges
                            </span>
                            <button
                              type="button"
                              className="message-action-btn"
                              onClick={() => {
                                const blob = new Blob([msg.artifacts.html], { type: "text/html" });
                                const url = URL.createObjectURL(blob);
                                const a = document.createElement("a");
                                a.href = url;
                                a.download = "knowledge_graph.html";
                                document.body.appendChild(a);
                                a.click();
                                document.body.removeChild(a);
                                URL.revokeObjectURL(url);
                              }}
                            >
                              Download
                            </button>
                          </div>
                          <iframe
                            className="artifact-iframe"
                            srcDoc={msg.artifacts.html}
                            title="Knowledge Graph"
                            sandbox="allow-scripts allow-same-origin"
                          />
                        </div>
                      )}
                    </>
                  ) : (
                    msg.content
                  )}
                  {runtimeMode === "agent" && activeThread?.id && msg.kind && (
                    <button
                      type="button"
                      className={classNames(
                        "memory-evidence-toggle",
                        selectedMemoryEvidence.includes(msg.id) && "selected"
                      )}
                      aria-pressed={selectedMemoryEvidence.includes(msg.id)}
                      onClick={() => handleMemoryEvidenceToggle(msg.id)}
                    >
                      <BookmarkPlus size={12} />
                      {selectedMemoryEvidence.includes(msg.id) ? "Selected as memory evidence" : "Use as memory evidence"}
                    </button>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input */}
        <div className="chat-input-area">
          <div className="chat-composer">
            {selectedMemoryEvidence.length > 0 && (
              <div className="memory-evidence-basket">
                <div>
                  <BookmarkPlus size={14} />
                  <span>{selectedMemoryEvidence.length} message{selectedMemoryEvidence.length === 1 ? "" : "s"} selected as evidence</span>
                </div>
                <button type="button" disabled={busy} onClick={handleCreateMemoryProposals}>
                  {busy ? "Analyzing…" : "Create memory suggestions"}
                </button>
                <button type="button" disabled={busy} onClick={() => setSelectedMemoryEvidence([])}>Clear</button>
              </div>
            )}
            {contextItems.length > 0 && (
              <div className="context-tag-row">
                {contextItems.map((item) => (
                  <span
                    className={classNames("context-tag", item.external && "external")}
                    key={item.absolute_path || item.path}
                    title={item.absolute_path || item.path}
                  >
                    <FileCode size={12} />
                    {item.external ? "External · " : "@"}{item.path}
                    <button type="button" onClick={() => removeContextItem(item)} aria-label={`Remove ${item.path}`}>
                      <X size={11} />
                    </button>
                  </span>
                ))}
              </div>
            )}
            {contextSuggestions.length > 0 && (
              <div className="context-suggestions">
                {contextSuggestions.slice(0, 8).map((item) => (
                  <button
                    type="button"
                    key={item.absolute_path || item.path}
                    onClick={() => selectContextSuggestion(item)}
                  >
                    {item.type === "directory" ? <Folder size={13} /> : <FileCode size={13} />}
                    <span>{item.path}</span>
                    {item.external && <small>external read/write</small>}
                  </button>
                ))}
              </div>
            )}
            <textarea
              ref={chatInputRef}
              className="chat-input"
              value={instruction}
              onChange={(event) => {
                setInstruction(event.target.value);
                requestAnimationFrame(autoResizeTextarea);
              }}
              onKeyDown={handleKeyDown}
              placeholder="Describe what you want to do..."
              rows={1}
            />
            <div className="composer-actions">
              <div className="composer-action-group">
                <button type="button" className="composer-tool-button" onClick={resolveAbsoluteContextFromInput}>
                  <FileCode size={13} />
                  Add context
                </button>
                <button
                  type="button"
                  className={classNames("composer-tool-button", budgetOpen && "active")}
                  onClick={() => setBudgetOpen((open) => !open)}
                >
                  <Activity size={13} />
                  Budget
                </button>
                <span className="composer-runtime-label">{runtimeMode === "agent" ? "Agent" : "Pipeline"}</span>
              </div>
              <button
                type="button"
                className="chat-send-btn"
                disabled={busy || (!instruction.trim() && !canRunWithoutInstruction())}
                onClick={handleSend}
                title="Send"
              >
                <Send size={18} />
              </button>
            </div>
            {budgetOpen && (
              <div className="budget-popover">
                <div>
                  <label htmlFor="llm-budget">LLM calls</label>
                  <input
                    id="llm-budget"
                    type="number"
                    min="0"
                    value={budgetDraft.max_llm_calls}
                    onChange={(event) => setBudgetDraft((current) => ({ ...current, max_llm_calls: event.target.value }))}
                  />
                </div>
                <div>
                  <label htmlFor="tool-budget">Tool calls</label>
                  <input
                    id="tool-budget"
                    type="number"
                    min="0"
                    value={budgetDraft.max_tool_calls}
                    onChange={(event) => setBudgetDraft((current) => ({ ...current, max_tool_calls: event.target.value }))}
                  />
                </div>
                <div>
                  <label htmlFor="input-token-budget">Input token limit</label>
                  <input
                    id="input-token-budget"
                    type="number"
                    min="0"
                    step="1000"
                    value={budgetDraft.max_input_tokens ?? ""}
                    onChange={(event) => setBudgetDraft((current) => ({ ...current, max_input_tokens: event.target.value }))}
                  />
                </div>
                <p className="budget-hint">
                  Cumulative Agent and compaction input. For an exhausted Run, the saved limit applies to the next message.
                </p>
                <button type="button" className="primary-button" onClick={saveBudget}>Save budget</button>
              </div>
            )}
          </div>
        </div>
      </main>

      {runDetailsOpen && (
        <>
          <div className="drawer-overlay" onClick={() => setRunDetailsOpen(false)} />
          <aside className="drawer run-details-drawer">
            <div className="drawer-header">
              <div>
                <h3>Run details</h3>
                <small>{agentState.runId ? agentState.runId.slice(0, 8) : "No active run"}</small>
              </div>
              <button type="button" className="drawer-close" onClick={() => setRunDetailsOpen(false)}>
                <X size={18} />
              </button>
            </div>
            <div className="drawer-body">
              <section className="run-detail-section">
                <div className="run-detail-heading">
                  <span className={classNames("run-status-pill", agentState.status)}>{agentState.status}</span>
                  {agentState.runId && <span>{agentState.events.length} events</span>}
                </div>
                {agentState.runId && agentRunControllable && (
                  <div className="run-control-actions">
                    {agentState.pendingApproval && (
                      <>
                        <button type="button" className="primary-button" disabled={busy} onClick={handleAgentApproval}>
                          Approve {agentState.pendingApproval.risk}
                        </button>
                        <button type="button" className="secondary-button" disabled={busy} onClick={handleAgentRejection}>
                          Reject
                        </button>
                      </>
                    )}
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={busy}
                      onClick={() => handleAgentControl(agentState.status === "paused" ? "resume" : "pause")}
                    >
                      {agentState.status === "paused" ? "Resume" : "Pause"}
                    </button>
                    <button type="button" className="secondary-button" disabled={busy} onClick={() => handleAgentControl("cancel")}>
                      Cancel
                    </button>
                  </div>
                )}
              </section>

              <section className="run-detail-section">
                <h4>Usage</h4>
                <BudgetMeter label="LLM calls" progress={llmBudgetProgress} />
                <BudgetMeter label="Tool calls" progress={toolBudgetProgress} />
                <BudgetMeter label="Input tokens" progress={inputTokenBudgetProgress} />
                <dl className="info-list">
                  <Info label="Tokens" value={(agentRunState?.input_tokens || 0) + (agentRunState?.output_tokens || 0)} />
                  <Info label="Prompt cache hit" value={promptCacheLabel} />
                  <Info label="Cost" value={`$${Number(agentRunState?.cost_usd || 0).toFixed(4)}`} />
                </dl>
              </section>

              <section className="run-detail-section">
                <h4>Changed files & verification</h4>
                <div className="detail-file-list">
                  {agentState.changedFiles.map((file) => <span key={file}>{file}</span>)}
                  {!agentState.changedFiles.length && <p>No changed files recorded.</p>}
                </div>
                <p className="detail-summary">
                  {agentState.verification.results.length
                    ? agentState.verification.results.map((result) => result.passed ? "Verification passed" : "Verification failed").join(" · ")
                    : agentState.verification.required ? "Verification required" : "No verification result yet"}
                </p>
              </section>

              <section className="run-detail-section">
                <h4>Event timeline</h4>
                <ol className="detail-timeline">
                  {agentState.events.slice(-30).map((event) => (
                    <li key={`${event.sequence}-${event.type}`}>
                      <strong>{event.sequence}. {event.type}</strong>
                      <span>
                        {event.payload?.result?.summary
                          || event.payload?.reason
                          || event.payload?.tool_call?.name
                          || event.payload?.decision
                          || ""}
                      </span>
                    </li>
                  ))}
                  {!agentState.events.length && <li>No run events yet.</li>}
                </ol>
              </section>

              <section className="run-detail-section memory-governance-section">
                <MemoryReviewPanel
                  proposals={memoryProposals}
                  busy={busy}
                  error={memoryError}
                  onAction={handleProposalAction}
                />
              </section>

              <section className="run-detail-section">
                <h4>Active memories</h4>
                <div className="agent-list">
                  {agentMemories.filter((memory) => memory.status === "active").slice(0, 12).map((memory) => (
                    <div className="agent-memory" key={memory.id}>
                      <span>{memory.subject}: {memory.content}</span>
                      <small>{memory.kind} · {memory.verification || "legacy_unverified"} · {memory.scope}</small>
                      <div>
                        <button type="button" onClick={() => handleMemoryAction(memory.id, "delete")}>Delete</button>
                      </div>
                    </div>
                  ))}
                  {!agentMemories.some((memory) => memory.status === "active") && <p>No active project memories.</p>}
                </div>
              </section>
            </div>
          </aside>
        </>
      )}

      {/* Right Drawer */}
      {drawerOpen && (
        <>
          <div className="drawer-overlay" onClick={() => setDrawerOpen(false)} />
          <aside className="drawer">
            <button
              type="button"
              className="resize-handle resize-handle-drawer"
              aria-label="Resize settings drawer"
              title="Drag to resize settings drawer"
              onPointerDown={(event) => startResize("drawerWidth", event)}
              onKeyDown={(event) => handleResizeKey("drawerWidth", event)}
            />
            <div className="drawer-header">
              <h3>Settings</h3>
              <button
                type="button"
                className="drawer-close"
                onClick={() => setDrawerOpen(false)}
              >
                <X size={18} />
              </button>
            </div>
            <div className="drawer-body">
              <section style={{ marginBottom: 16 }}>
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
                    placeholder="D:\path\to\project"
                  />
                  <button
                    type="button"
                    className="icon-button"
                    title="Apply workspace"
                    onClick={() => {
                      refreshWorkspace(projectDir, targets)
                        .then(() => activeThread?.id && updateActiveThread({ workspace: projectDir }))
                        .catch((error) => setLastError(error.message));
                    }}
                  >
                    <RefreshCcw size={15} />
                  </button>
                </div>
                <p className="field-help">
                  Add files from the composer with @name or an absolute path.
                </p>
              </section>

              <section className="codegraph-settings">
                <div className="section-heading">
                  <Network size={16} />
                  <span>Knowledge graph</span>
                </div>
                <label className="switch-row codegraph-switch">
                  <input
                    type="checkbox"
                    checked={codeGraphEnabled}
                    disabled={codeGraphBusy}
                    onChange={(event) => handleCodeGraphToggle(event.target.checked)}
                  />
                  <span>Use knowledge graph</span>
                  <span
                    className={classNames(
                      "codegraph-status",
                      codeGraphStatus?.ready && !codeGraphStatus?.stale && "ready",
                      codeGraphEnabled && (!codeGraphStatus?.ready || codeGraphStatus?.stale) && "attention"
                    )}
                    title={codeGraphStatus?.message || ""}
                  >
                    {codeGraphStatusLabel(codeGraphEnabled, codeGraphStatus)}
                  </span>
                </label>
                {codeGraphEnabled && (
                  <div className="codegraph-actions">
                    <button
                      type="button"
                      className="icon-button"
                      title="Refresh CodeGraph status"
                      disabled={!activeThread?.id || codeGraphBusy}
                      onClick={() => refreshCodeGraphStatus()}
                    >
                      <RefreshCcw size={15} />
                    </button>
                    {codeGraphStatus?.available && !codeGraphStatus?.initialized && (
                      <button
                        type="button"
                        className="secondary-button compact-button"
                        disabled={!activeThread?.id || codeGraphBusy}
                        onClick={() => runCodeGraphAction("init")}
                      >
                        <Database size={15} />
                        Initialize
                      </button>
                    )}
                    {codeGraphStatus?.available && codeGraphStatus?.initialized && (
                      <button
                        type="button"
                        className="secondary-button compact-button"
                        disabled={!activeThread?.id || codeGraphBusy}
                        onClick={() => runCodeGraphAction("sync")}
                      >
                        <RefreshCcw size={15} />
                        Sync
                      </button>
                    )}
                    {codeGraphStatus?.ready && (
                      <button
                        type="button"
                        className="secondary-button compact-button"
                        disabled={!activeThread?.id || codeGraphBusy}
                        onClick={() => runCodeGraphAction("visualize")}
                      >
                        <Network size={15} />
                        Open graph
                      </button>
                    )}
                  </div>
                )}
              </section>

              <section style={{ marginBottom: 16 }}>
                <div className="section-heading">
                  <Settings2 size={16} />
                  <span>Model</span>
                </div>
                <label className="field-label" htmlFor="model">Model</label>
                <input
                  id="model"
                  className="text-input"
                  value={modelInputValue(model)}
                  onChange={(event) => setModel(event.target.value)}
                  onBlur={() => {
                    if (activeThread?.id) {
                      updateActiveThread({ model }).catch((error) => setLastError(error.message));
                    }
                  }}
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

              <section style={{ marginBottom: 16 }}>
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

              <section style={{ marginBottom: 16 }}>
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

              <section>
                <div className="section-heading">
                  <Terminal size={16} />
                  <span>Equivalent CLI</span>
                </div>
                <pre className="command-preview">{mode === "preview" ? previewCommand : runCommand}</pre>
              </section>
            </div>
          </aside>
        </>
      )}

      {threadPendingDelete && (
        <div
          className="confirmation-backdrop"
          onClick={() => {
            if (!deletingThreadId && !cancellingThreadId) setThreadPendingDelete(null);
          }}
        >
          <section
            className="confirmation-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-conversation-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="confirmation-icon" aria-hidden="true">
              <Trash2 size={19} />
            </div>
            <div>
              <h3 id="delete-conversation-title">Delete conversation?</h3>
              <p className="confirmation-thread-title">
                {threadPendingDelete.title || "Untitled task"}
              </p>
              <p>
                Messages, run records, events, approvals, and execution history
                in this conversation will be permanently deleted.
              </p>
              <p className="confirmation-hint">
                {threadNeedsCancellation(threadPendingDelete.last_status)
                  ? `This conversation is currently ${threadPendingDelete.last_status}. Cancel its task before deleting it.`
                  : "This conversation has no active task and can now be deleted."}
              </p>
              {threadDeleteError && (
                <div className="confirmation-error" role="alert">
                  {threadDeleteError}
                </div>
              )}
            </div>
            <div className="confirmation-actions">
              <button
                type="button"
                className="secondary-button"
                disabled={Boolean(deletingThreadId || cancellingThreadId)}
                onClick={() => setThreadPendingDelete(null)}
              >
                Keep conversation
              </button>
              {threadNeedsCancellation(threadPendingDelete.last_status) ? (
                <button
                  type="button"
                  className="danger-button"
                  disabled={Boolean(deletingThreadId || cancellingThreadId || !threadPendingDelete.last_run_id)}
                  onClick={cancelPendingThreadRun}
                >
                  {cancellingThreadId ? "Cancelling task…" : "Cancel task first"}
                </button>
              ) : (
                <button
                  type="button"
                  className="danger-button"
                  disabled={Boolean(deletingThreadId || cancellingThreadId)}
                  onClick={deletePendingThread}
                >
                  {deletingThreadId ? "Deleting…" : "Delete permanently"}
                </button>
              )}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function BudgetMeter({ label, progress }) {
  return (
    <div className={classNames("budget-meter", progress.level)}>
      <div className="budget-meter-label">
        <span>{label}</span>
        <strong>{progress.used} / {progress.limit}</strong>
      </div>
      <div className="budget-meter-track">
        <span style={{ width: `${progress.percent}%` }} />
      </div>
    </div>
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
