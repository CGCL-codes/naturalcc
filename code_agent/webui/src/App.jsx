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
import { initialAgentState, hydrateAgentState, reduceAgentEvent } from "./agentState.js";
import { followAgentEvents } from "./agentEventStream.js";
import { approveAgentAction, approvalErrorMessage } from "./agentApproval.js";
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
import { readActiveThreadId, rememberActiveThread, threadToRestore, runtimeConfigForEdit } from "./runtimeSettings.js";
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

// [maint-note:7b154180a19a]
// 维护导航：codeGraphStatusLabel；下列为本函数源码中的接口与路径索引。
// 输入 enabled：类型由调用方与函数体约束。
// 输入 status：类型由调用方与函数体约束。
// 分支条件：!enabled。
// 分支条件：!status。
// 分支条件：!status.installed。
// 分支条件：!status.available。
// 分支条件：!status.initialized。
// 分支条件：status.stale。
// 分支条件：status.ready。
// 返回路径："Off"。
// 返回路径："Checking"。
// 返回路径："Not installed"。
// 返回路径："Unavailable"。
// 返回路径："Not initialized"。
// 返回路径："Needs sync"。
// 返回路径："Ready"。
// 返回路径："Error"。
// [/maint-note:7b154180a19a]
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

// [maint-note:b36e6bdcac7c]
// 维护导航：classNames；下列为本函数源码中的接口与路径索引。
// 输入 items：类型由调用方与函数体约束。
// 返回路径：items.filter(Boolean).join(" ")。
// 直接调用：items.filter(Boolean).join。
// 直接调用：items.filter。
// [/maint-note:b36e6bdcac7c]
function classNames(...items) {
  return items.filter(Boolean).join(" ");
}

// [maint-note:0b9ae1ab13ec]
// 维护导航：requestJson；下列为本函数源码中的接口与路径索引。
// 输入 path：类型由调用方与函数体约束。
// 输入 options：类型由调用方与函数体约束；缺省表达式 {}。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!response.ok。
// 返回路径：response.json()。
// 显式抛出：error。
// 写入目标：detail。
// 写入目标：error.status。
// 直接调用：fetch。
// 直接调用：response.json。
// 直接调用：JSON.stringify。
// [/maint-note:0b9ae1ab13ec]
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
    const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    error.status = response.status;
    throw error;
  }
  return response.json();
}

// [maint-note:0222e4ad112f]
// 维护导航：metricLabel；下列为本函数源码中的接口与路径索引。
// 输入 value：类型由调用方与函数体约束。
// 分支条件：value === undefined || value === null || value === ""。
// 返回路径："not set"。
// 返回路径：String(value)。
// 直接调用：String。
// [/maint-note:0222e4ad112f]
function metricLabel(value) {
  if (value === undefined || value === null || value === "") {
    return "not set";
  }
  return String(value);
}

// [maint-note:bbff09ef447f]
// 维护导航：formatTime；下列为本函数源码中的接口与路径索引。
// 输入 date：类型由调用方与函数体约束。
// 分支条件：!date。
// 返回路径：""。
// 返回路径：date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })。
// 直接调用：date.toLocaleTimeString。
// [/maint-note:bbff09ef447f]
function formatTime(date) {
  if (!date) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// [maint-note:133ee478d499]
// 维护导航：nextFrame；下列为本函数源码中的接口与路径索引。
// 返回路径：new Promise((resolve) => requestAnimationFrame(resolve))。
// [/maint-note:133ee478d499]
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

// [maint-note:e00eb59ee578]
// 维护导航：DynamicForm；下列为本函数源码中的接口与路径索引。
// 输入 { schema, values, onChange }：类型由调用方与函数体约束。
// 返回路径：( <div className="dynamic-form"> {schema.map((field) => { const Component = FIELD_TYPE_COMPONENTS[field.type]; if (!Component) return null; return ( <div key={field.name} className="form-field"> <label className="field-label" htmlFor={`fiel。
// 直接调用：schema.map。
// [/maint-note:e00eb59ee578]
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

// [maint-note:e1f6a2675af4]
// 维护导航：App；下列为本函数源码中的接口与路径索引。
// 返回路径：( <div className="app-root" data-theme={theme} style={toLayoutStyle(layoutPrefs)}> {/* Mobile overlay */} {sidebarOpen && ( <div className="drawer-overlay" style={{ zIndex: 25 }} onClick={() => setSidebarOpen(false)} /> )} {/* Left Sidebar 。
// 直接调用：useState。
// 直接调用：useRef。
// 直接调用：canControlAgentRun。
// 直接调用：useMemo。
// 直接调用：budgetProgress。
// 直接调用：promptCacheHitTokens.toLocaleString。
// 直接调用：promptCacheMeasuredTokens.toLocaleString。
// 直接调用：((promptCacheHitTokens / promptCacheMeasuredTokens) * 100).toFixed。
// 直接调用：useEffect。
// 直接调用：toLayoutStyle。
// 直接调用：classNames。
// 直接调用：threadGroups.map。
// 直接调用：models.map。
// 直接调用：(() => { const selectedModel = runtimeMode === "agent" ? runtimeModel : model; return !models.includes(selectedModel) && selectedModel ? <option value={selectedModel}>{selectedModel}</option> : null; })。
// 直接调用：messages.map。
// 直接调用：contextItems.map。
// 直接调用：contextSuggestions.slice(0, 8).map。
// 直接调用：contextSuggestions.slice。
// 直接调用：instruction.trim。
// 直接调用：canRunWithoutInstruction。
// 直接调用：agentState.runId.slice。
// 直接调用：Number(agentRunState?.cost_usd || 0).toFixed。
// 直接调用：Number。
// 直接调用：agentState.changedFiles.map。
// 直接调用：agentState.verification.results.map((result) => result.passed ? "Verification passed" : "Verification failed").join。
// 直接调用：agentState.verification.results.map。
// 直接调用：agentState.events.filter((event) => !event.type.endsWith(".delta")).slice(-30).map。
// 直接调用：agentState.events.filter((event) => !event.type.endsWith(".delta")).slice。
// 直接调用：agentState.events.filter。
// 直接调用：memoryExtraction.status.replaceAll。
// 直接调用：memoryExtraction.reason.replaceAll。
// 直接调用：agentMemories.filter((memory) => memory.status === "active").slice(0, 12).map。
// 直接调用：agentMemories.filter((memory) => memory.status === "active").slice。
// 直接调用：agentMemories.filter。
// 直接调用：agentMemories.some。
// 直接调用：codeGraphStatusLabel。
// 直接调用：modelInputValue。
// 直接调用：features.map。
// 直接调用：threadNeedsCancellation。
// 直接调用：Boolean。
// [/maint-note:e1f6a2675af4]
function App() {
  const [models, setModels] = useState([]);
  const [completionTypes, setCompletionTypes] = useState(defaultCompletionTypes);
  const [projectDir, setProjectDir] = useState("");
  const [targets, setTargets] = useState([]);
  const [model, setModel] = useState("");
  const [runtimeModel, setRuntimeModel] = useState("");
  const [defaultRuntimeConfig, setDefaultRuntimeConfig] = useState(null);
  const [bootstrapReady, setBootstrapReady] = useState(false);
  const [runtimeProvider, setRuntimeProvider] = useState("deepseek");
  const [runtimeProviderDefaults, setRuntimeProviderDefaults] = useState({
    deepseek: { model: "deepseek-chat" },
    openrouter: { model: "deepseek/deepseek-chat" }
  });
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
  const [workspacePickerBusy, setWorkspacePickerBusy] = useState(false);
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
  const [memoryExtraction, setMemoryExtraction] = useState(null);
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
  const approvalBusyRef = useRef(false);

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

  // [maint-note:14d7a3c2c7f6]
  // 维护导航：runtimeModelConfig；下列为本函数源码中的接口与路径索引。
  // 输入 modelOverride：类型由调用方与函数体约束；缺省表达式 runtimeModel。
  // 输入 providerOverride：类型由调用方与函数体约束；缺省表达式 runtimeProvider。
  // 返回路径：runtimeConfigForEdit( activeThread?.runtime_model_config || defaultRuntimeConfig, providerOverride, modelOverride )。
  // 直接调用：runtimeConfigForEdit。
  // [/maint-note:14d7a3c2c7f6]
  function runtimeModelConfig(modelOverride = runtimeModel, providerOverride = runtimeProvider) {
    return runtimeConfigForEdit(
      activeThread?.runtime_model_config || defaultRuntimeConfig,
      providerOverride,
      modelOverride
    );
  }

  // [maint-note:a882878b751e]
  // 维护导航：changeRuntimeProvider；下列为本函数源码中的接口与路径索引。
  // 输入 nextProvider：类型由调用方与函数体约束。
  // 分支条件：activeThread?.id。
  // 直接调用：setRuntimeProvider。
  // 直接调用：setRuntimeModel。
  // 直接调用：updateActiveThread({ model: nextModel, runtime_model_config: runtimeModelConfig(nextModel, nextProvider) }).catch。
  // 直接调用：updateActiveThread。
  // 直接调用：runtimeModelConfig。
  // [/maint-note:a882878b751e]
  function changeRuntimeProvider(nextProvider) {
    const currentDefault = runtimeProviderDefaults[runtimeProvider]?.model;
    const nextDefault = runtimeProviderDefaults[nextProvider]?.model;
    const nextModel = !runtimeModel || runtimeModel === currentDefault
      ? (nextDefault || runtimeModel)
      : runtimeModel;

    setRuntimeProvider(nextProvider);
    setRuntimeModel(nextModel);
    if (activeThread?.id) {
      updateActiveThread({
        model: nextModel,
        runtime_model_config: runtimeModelConfig(nextModel, nextProvider)
      }).catch((error) => setLastError(error.message));
    }
  }

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
      .then(async (data) => {
        if (!active) {
          return;
        }
        setModels(data.models || []);
        setCompletionTypes(data.completion_types || defaultCompletionTypes);
        setModel(data.default_model || "");
        const runtimeDefaults = data.runtime_provider_defaults || {};
        const runtimeConfig = data.runtime_default_model_config || {
          provider: "deepseek",
          model: runtimeDefaults.deepseek?.model || data.default_model || "deepseek-chat"
        };
        setDefaultRuntimeConfig(runtimeConfig);
        setRuntimeProviderDefaults((current) => ({ ...current, ...runtimeDefaults }));
        setRuntimeProvider(runtimeConfig.provider || "deepseek");
        setRuntimeModel(runtimeConfig.model || runtimeDefaults.deepseek?.model || data.default_model || "");
        setProjectDir(data.default_project_dir || "");
        setFeatures(data.features || []);
        setFeatureSchemas(data.schemas || {});
        setCurrentFeature(data.default_feature || "code_completion");
        Promise.all([
          refreshWorkspace(data.default_project_dir || "", []),
          loadBrowse(data.default_project_dir || "")
        ]).catch((error) => setLastError(error.message));
        // Restore only the conversation the user selected, after bootstrap.
        // Never let an arbitrary history entry replace the launch defaults.
        const { threads = [] } = await requestJson("/api/agent/threads");
        if (!active) return;
        setAgentThreads(threads);
        const savedId = threadToRestore(threads, readActiveThreadId());
        if (savedId) {
          await loadAgentThread(savedId);
        } else {
          rememberActiveThread(null);
        }
        if (active) setBootstrapReady(true);
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
    const runId = agentState.runId;
    setMemoryExtraction(null);
    if (!runDetailsOpen || !runId || runtimeMode !== "agent") return;
    let active = true;
    let timer;
    let pendingChecks = 0;
    // [maint-note:20f4c8ba5d94]
    // 维护导航：readExtraction；下列为本函数源码中的接口与路径索引。
    // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
    // 分支条件：!active。
    // 分支条件：["queued", "running"].includes(outcome.status) || (outcome.status === "not_scheduled" && outcome.run_status === "completed" && pendingChecks++ < 2)。
    // 分支条件：active。
    // 返回路径：无返回值。
    // 写入目标：timer。
    // 直接调用：requestJson。
    // 直接调用：setMemoryExtraction。
    // 直接调用：["queued", "running"].includes。
    // 直接调用：setTimeout。
    // [/maint-note:20f4c8ba5d94]
    async function readExtraction() {
      try {
        const outcome = await requestJson(`/api/agent/runs/${runId}/memory-extraction`);
        if (!active) return;
        setMemoryExtraction(outcome);
        if (["queued", "running"].includes(outcome.status) ||
            (outcome.status === "not_scheduled" && outcome.run_status === "completed" && pendingChecks++ < 2)) {
          timer = setTimeout(readExtraction, 1500);
        }
      } catch {
        if (active) setMemoryExtraction({ run_id: runId, status: "unavailable" });
      }
    }
    readExtraction();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [agentState.runId, agentState.status, runDetailsOpen, runtimeMode]);

  useEffect(() => {
    if (runtimeMode !== "agent" || !bootstrapReady) return;
    let active = true;
    Promise.all([
      requestJson("/api/agent/threads"),
      requestJson("/api/agent/runs"),
      requestJson(`/api/agent/memories?project_id=${encodeURIComponent(projectDir)}`),
      requestJson(`/api/agent/memory-proposals?project_id=${encodeURIComponent(projectDir)}`)
    ])
      .then(([threadsPayload, runsPayload, memoriesPayload, proposalsPayload]) => {
        if (!active) return;
        const threads = threadsPayload.threads || [];
        setAgentThreads(threads);
        setAgentRuns(runsPayload.runs || []);
        setAgentMemories(memoriesPayload.memories || []);
        setMemoryProposals((proposalsPayload.proposals || []).map(normalizeMemoryReview));
      })
      .catch((error) => { if (active) setLastError(error.message); });
    return () => { active = false; };
  }, [bootstrapReady, runtimeMode, projectDir, activeThread?.id]);

  useEffect(() => {
    const runId = agentState.runId;
    if (runtimeMode !== "agent" || !runId || !["queued", "running"].includes(agentState.status)) return;
    const controller = new AbortController();
    let after = agentState.lastSequence || 0;
    // [maint-note:4c31d166d439]
    // 维护导航：follow；下列为本函数源码中的接口与路径索引。
    // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
    // 分支条件：controller.signal.aborted。
    // 遍历边界：!controller.signal.aborted。
    // 返回路径：无返回值。
    // 直接调用：followAgentEvents。
    // [/maint-note:4c31d166d439]
    const follow = async () => {
      while (!controller.signal.aborted) {
        try {
          await followAgentEvents(runId, after, (event) => {
            if (event.sequence <= after) return;
            after = event.sequence;
            setAgentState((current) => current.runId === runId ? reduceAgentEvent(current, event) : current);
          }, controller.signal, API_BASE);
        } catch (error) {
          if (controller.signal.aborted) return;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 500));
      }
    };
    follow();
    return () => controller.abort();
  }, [agentState.runId, agentState.status, runtimeMode]);

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

  // [maint-note:92135e27defc]
  // 维护导航：refreshWorkspace；下列为本函数源码中的接口与路径索引。
  // 输入 dir：类型由调用方与函数体约束；缺省表达式 projectDir。
  // 输入 nextTargets：类型由调用方与函数体约束；缺省表达式 targets。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：data.normalized_root。
  // 返回路径：data。
  // 直接调用：setLastError。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：setWorkspace。
  // 直接调用：setTargets。
  // 直接调用：setProjectDir。
  // 直接调用：setLastUpdated。
  // 直接调用：new Date().toLocaleTimeString。
  // [/maint-note:92135e27defc]
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

  // [maint-note:84e83ed318e5]
  // 维护导航：applyWorkspace；下列为本函数源码中的接口与路径索引。
  // 输入 dir：类型由调用方与函数体约束；缺省表达式 projectDir。
  // 输入 nextTargets：类型由调用方与函数体约束；缺省表达式 targets。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：activeThread?.id。
  // 返回路径：data。
  // 直接调用：refreshWorkspace。
  // 直接调用：updateActiveThread。
  // [/maint-note:84e83ed318e5]
  async function applyWorkspace(dir = projectDir, nextTargets = targets) {
    const data = await refreshWorkspace(dir, nextTargets);
    const normalizedRoot = data.normalized_root || dir;
    if (activeThread?.id) {
      await updateActiveThread({ workspace: normalizedRoot });
    }
    return data;
  }

  // [maint-note:6470afa2cb91]
  // 维护导航：chooseWorkspaceDirectory；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!data.selected || !data.path。
  // 返回路径：无返回值。
  // 直接调用：setWorkspacePickerBusy。
  // 直接调用：setLastError。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：applyWorkspace。
  // 直接调用：loadBrowse(data.path).catch。
  // 直接调用：loadBrowse。
  // [/maint-note:6470afa2cb91]
  async function chooseWorkspaceDirectory() {
    setWorkspacePickerBusy(true);
    setLastError("");
    try {
      const data = await requestJson("/api/workspace/select-directory", {
        method: "POST",
        body: JSON.stringify({ initial_dir: projectDir })
      });
      if (!data.selected || !data.path) {
        return;
      }
      await applyWorkspace(data.path, []);
      loadBrowse(data.path).catch(() => {});
    } catch (error) {
      setLastError(error.message);
    } finally {
      setWorkspacePickerBusy(false);
    }
  }

  // [maint-note:255f37feb5a7]
  // 维护导航：loadBrowse；下列为本函数源码中的接口与路径索引。
  // 输入 path：类型由调用方与函数体约束；缺省表达式 projectDir。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!path。
  // 返回路径：无返回值。
  // 直接调用：setLastError。
  // 直接调用：requestJson。
  // 直接调用：encodeURIComponent。
  // 直接调用：setBrowseState。
  // [/maint-note:255f37feb5a7]
  async function loadBrowse(path = projectDir) {
    if (!path) {
      return;
    }
    setLastError("");
    const data = await requestJson(`/api/browse?path=${encodeURIComponent(path)}`);
    setBrowseState(data);
  }

  // [maint-note:dec7af9dfe01]
  // 维护导航：toggleTarget；下列为本函数源码中的接口与路径索引。
  // 输入 file：类型由调用方与函数体约束。
  // 直接调用：setTargets。
  // [/maint-note:dec7af9dfe01]
  function toggleTarget(file) {
    setTargets((current) => {
      if (current.includes(file)) {
        return current.filter((item) => item !== file);
      }
      return [...current, file];
    });
  }

  // [maint-note:6146c142707f]
  // 维护导航：addCustomTarget；下列为本函数源码中的接口与路径索引。
  // 分支条件：!next。
  // 返回路径：无返回值。
  // 直接调用：customTarget.trim().replaceAll。
  // 直接调用：customTarget.trim。
  // 直接调用：setTargets。
  // 直接调用：setCustomTarget。
  // [/maint-note:6146c142707f]
  function addCustomTarget() {
    const next = customTarget.trim().replaceAll("\\", "/");
    if (!next) {
      return;
    }
    setTargets((current) => (current.includes(next) ? current : [...current, next]));
    setCustomTarget("");
  }

  // [maint-note:0082750fedd2]
  // 维护导航：makePrimary；下列为本函数源码中的接口与路径索引。
  // 输入 file：类型由调用方与函数体约束。
  // 直接调用：setTargets。
  // [/maint-note:0082750fedd2]
  function makePrimary(file) {
    setTargets((current) => [file, ...current.filter((item) => item !== file)]);
  }

  // [maint-note:6298691f150c]
  // 维护导航：removeTarget；下列为本函数源码中的接口与路径索引。
  // 输入 file：类型由调用方与函数体约束。
  // 直接调用：setTargets。
  // [/maint-note:6298691f150c]
  function removeTarget(file) {
    setTargets((current) => current.filter((item) => item !== file));
  }

  // [maint-note:65907c61ed8e]
  // 维护导航：addMessage；下列为本函数源码中的接口与路径索引。
  // 输入 type：类型由调用方与函数体约束。
  // 输入 content：类型由调用方与函数体约束。
  // 输入 status：类型由调用方与函数体约束；缺省表达式 "complete"。
  // 输入 extra：类型由调用方与函数体约束；缺省表达式 {}。
  // 返回路径：id。
  // 直接调用：Date.now。
  // 直接调用：Math.random().toString(36).slice。
  // 直接调用：Math.random().toString。
  // 直接调用：Math.random。
  // 直接调用：setMessages。
  // [/maint-note:65907c61ed8e]
  function addMessage(type, content, status = "complete", extra = {}) {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const msg = { id, type, content, timestamp: new Date(), status, ...extra };
    setMessages((prev) => [...prev, msg]);
    return id;
  }

  // [maint-note:94918a402d0e]
  // 维护导航：updateMessage；下列为本函数源码中的接口与路径索引。
  // 输入 id：类型由调用方与函数体约束。
  // 输入 updates：类型由调用方与函数体约束。
  // 直接调用：setMessages。
  // [/maint-note:94918a402d0e]
  function updateMessage(id, updates) {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...updates } : m)));
  }

  // [maint-note:06ab15510d68]
  // 维护导航：handlePreview；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!instruction.trim() && !canRunWithoutInstruction()。
  // 返回路径：无返回值。
  // 直接调用：instruction.trim。
  // 直接调用：canRunWithoutInstruction。
  // 直接调用：setLastError。
  // 直接调用：setBusy。
  // 直接调用：setMode。
  // 直接调用：setTerminalLog。
  // 直接调用：addMessage。
  // 直接调用：defaultTaskLabel。
  // 直接调用：setActiveMessageId。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：setPreviewCommand。
  // 直接调用：setLastUpdated。
  // 直接调用：new Date().toLocaleTimeString。
  // 直接调用：updateMessage。
  // [/maint-note:06ab15510d68]
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

  // [maint-note:8603a82b1913]
  // 维护导航：buildFormData；下列为本函数源码中的接口与路径索引。
  // 分支条件：value instanceof FileList。
  // 遍历边界：const [key, value] ← Object.entries(featureConfig)。
  // 遍历边界：let i = 0; i < value.length; i++。
  // 返回路径：formData。
  // 写入目标：cleanConfig[key]。
  // 直接调用：formData.append。
  // 直接调用：JSON.stringify。
  // 直接调用：Object.entries。
  // [/maint-note:8603a82b1913]
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

  // [maint-note:335bd23a2cf9]
  // 维护导航：handlePipelineRun；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!instruction.trim() && !canRunWithoutInstruction()。
  // 分支条件：!userMessageAddedRef.current。
  // 分支条件：!response.ok || !response.body。
  // 分支条件：done。
  // 分支条件：!line.trim()。
  // 分支条件：buffer.trim()。
  // 遍历边界：true。
  // 遍历边界：const line ← lines。
  // 返回路径：无返回值。
  // 显式抛出：new Error(response.statusText || "Execution stream failed")。
  // 写入目标：userMessageAddedRef.current。
  // 写入目标：buffer。
  // 直接调用：instruction.trim。
  // 直接调用：canRunWithoutInstruction。
  // 直接调用：setLastError。
  // 直接调用：setBusy。
  // 直接调用：setMode。
  // 直接调用：setTerminalLog。
  // 直接调用：addMessage。
  // 直接调用：defaultTaskLabel。
  // 直接调用：setActiveMessageId。
  // 直接调用：Object.values(featureConfig).some。
  // 直接调用：Object.values。
  // 直接调用：buildFormData。
  // 直接调用：JSON.stringify。
  // 直接调用：fetch。
  // 直接调用：response.body.getReader。
  // 直接调用：reader.read。
  // 直接调用：decoder.decode。
  // 直接调用：buffer.split。
  // 直接调用：lines.pop。
  // 直接调用：line.trim。
  // 直接调用：JSON.parse。
  // 直接调用：applyStreamEvent。
  // 直接调用：nextFrame。
  // 直接调用：buffer.trim。
  // 直接调用：setLastUpdated。
  // 直接调用：new Date().toLocaleTimeString。
  // 直接调用：updateMessage。
  // [/maint-note:335bd23a2cf9]
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
      // [maint-note:b9f7369bfe27]
      // 维护导航：applyStreamEvent；下列为本函数源码中的接口与路径索引。
      // 输入 event：类型由调用方与函数体约束。
      // 分支条件：event.command。
      // 分支条件：event.log !== undefined。
      // 分支条件：event.status。
      // 写入目标：latestCommand。
      // 写入目标：latestLog。
      // 直接调用：setRunCommand。
      // 直接调用：setTerminalLog。
      // 直接调用：setMode。
      // 直接调用：updateMessage。
      // [/maint-note:b9f7369bfe27]
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

  // [maint-note:0e3a30b73860]
  // 维护导航：refreshThreadList；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 返回路径：payload.threads || []。
  // 直接调用：requestJson。
  // 直接调用：setAgentThreads。
  // [/maint-note:0e3a30b73860]
  async function refreshThreadList() {
    const payload = await requestJson("/api/agent/threads");
    setAgentThreads(payload.threads || []);
    return payload.threads || [];
  }

  // [maint-note:bad84b66a197]
  // 维护导航：loadAgentThread；下列为本函数源码中的接口与路径索引。
  // 输入 threadId：类型由调用方与函数体约束。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：detail.last_run?.run_id。
  // 分支条件：graphEnabled。
  // 返回路径：detail。
  // 直接调用：setConversationLoading。
  // 直接调用：setLastError。
  // 直接调用：setAgentState。
  // 直接调用：setAgentRunState。
  // 直接调用：requestJson。
  // 直接调用：setRuntimeMode。
  // 直接调用：setActiveThread。
  // 直接调用：rememberActiveThread。
  // 直接调用：setProjectDir。
  // 直接调用：setRuntimeModel。
  // 直接调用：setRuntimeProvider。
  // 直接调用：setBudgetDraft。
  // 直接调用：normalizeBudgetDraft。
  // 直接调用：setContextItems。
  // 直接调用：Boolean。
  // 直接调用：setCodeGraphEnabled。
  // 直接调用：setCodeGraphStatus。
  // 直接调用：setTargets。
  // 直接调用：(thread.context_items || []).map((item) => item.path).filter。
  // 直接调用：(thread.context_items || []).map。
  // 直接调用：setMessages。
  // 直接调用：hydrateConversationMessages。
  // 直接调用：setSelectedMemoryEvidence。
  // 直接调用：encodeURIComponent。
  // 直接调用：setMemoryProposals。
  // 直接调用：(proposalPayload.proposals || []).map。
  // 直接调用：refreshAgentState。
  // 直接调用：setMode。
  // 直接调用：setSidebarOpen。
  // [/maint-note:bad84b66a197]
  async function loadAgentThread(threadId) {
    setConversationLoading(true);
    setLastError("");
    setAgentState(initialAgentState);
    setAgentRunState(null);
    try {
      const detail = await requestJson(`/api/agent/threads/${threadId}`);
      const thread = detail.thread;
      setRuntimeMode("agent");
      setActiveThread(thread);
      rememberActiveThread(thread.id);
      setProjectDir(thread.workspace || projectDir);
      setRuntimeModel(thread.runtime_model_config?.model || thread.model || runtimeModel);
      setRuntimeProvider(thread.runtime_model_config?.provider || "deepseek");
      setBudgetDraft(normalizeBudgetDraft(thread.budget, budgetDraft));
      setContextItems(thread.context_items || []);
      const graphEnabled = Boolean(thread.capabilities?.codegraph?.enabled);
      setCodeGraphEnabled(graphEnabled);
      setCodeGraphStatus(null);
      setTargets((thread.context_items || []).map((item) => item.path).filter(Boolean));
      setMessages(hydrateConversationMessages(detail.messages || [], detail.last_run));
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

  // [maint-note:0086048cf1ca]
  // 维护导航：createNewThread；下列为本函数源码中的接口与路径索引。
  // 输入 title：类型由调用方与函数体约束；缺省表达式 "New task"。
  // 输入 config：类型由调用方与函数体约束；缺省表达式 defaultRuntimeConfig。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!bootstrapReady || !config。
  // 分支条件：!projectDir.trim()。
  // 返回路径：payload.thread。
  // 显式抛出：new Error("Wait for settings to finish loading before creating a conversation.")。
  // 显式抛出：new Error("Set a workspace before creating a conversation.")。
  // 直接调用：projectDir.trim。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：setActiveThread。
  // 直接调用：rememberActiveThread。
  // 直接调用：setRuntimeProvider。
  // 直接调用：setRuntimeModel。
  // 直接调用：setMessages。
  // 直接调用：setSelectedMemoryEvidence。
  // 直接调用：setContextItems。
  // 直接调用：setTargets。
  // 直接调用：setAgentState。
  // 直接调用：setAgentRunState。
  // 直接调用：setMode。
  // 直接调用：refreshThreadList。
  // [/maint-note:0086048cf1ca]
  async function createNewThread(title = "New task", config = defaultRuntimeConfig) {
    if (!bootstrapReady || !config) {
      throw new Error("Wait for settings to finish loading before creating a conversation.");
    }
    if (!projectDir.trim()) {
      throw new Error("Set a workspace before creating a conversation.");
    }
    const payload = await requestJson("/api/agent/threads", {
      method: "POST",
      body: JSON.stringify({
        title,
        workspace: projectDir,
        model: config.model,
        runtime_model_config: config,
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
    rememberActiveThread(payload.thread.id);
    setRuntimeProvider(payload.thread.runtime_model_config.provider);
    setRuntimeModel(payload.thread.runtime_model_config.model);
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

  // [maint-note:6dcb80cf52b6]
  // 维护导航：ensureAgentThread；下列为本函数源码中的接口与路径索引。
  // 输入 goal：类型由调用方与函数体约束。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：activeThread?.id。
  // 返回路径：activeThread。
  // 返回路径：createNewThread(goal.slice(0, 80) || "New task", runtimeModelConfig())。
  // 直接调用：createNewThread。
  // 直接调用：goal.slice。
  // 直接调用：runtimeModelConfig。
  // [/maint-note:6dcb80cf52b6]
  async function ensureAgentThread(goal) {
    if (activeThread?.id) {
      return activeThread;
    }
    return createNewThread(goal.slice(0, 80) || "New task", runtimeModelConfig());
  }

  // [maint-note:a747a360c3d9]
  // 维护导航：updateActiveThread；下列为本函数源码中的接口与路径索引。
  // 输入 changes：类型由调用方与函数体约束。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!activeThread?.id。
  // 返回路径：null。
  // 返回路径：payload.thread。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：setActiveThread。
  // 直接调用：refreshThreadList。
  // [/maint-note:a747a360c3d9]
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

  // [maint-note:107432de0719]
  // 维护导航：resetConversationView；下列为本函数源码中的接口与路径索引。
  // 直接调用：setActiveThread。
  // 直接调用：rememberActiveThread。
  // 直接调用：setRuntimeProvider。
  // 直接调用：setRuntimeModel。
  // 直接调用：setMessages。
  // 直接调用：setContextItems。
  // 直接调用：setContextSuggestions。
  // 直接调用：setTargets。
  // 直接调用：setInstruction。
  // 直接调用：setAgentState。
  // 直接调用：setAgentRunState。
  // 直接调用：setMode。
  // 直接调用：setBusy。
  // 直接调用：setActiveMessageId。
  // 直接调用：setSelectedMemoryEvidence。
  // 直接调用：setMemoryProposals。
  // 直接调用：setRunDetailsOpen。
  // 直接调用：setBudgetOpen。
  // 直接调用：setCodeGraphEnabled。
  // 直接调用：setCodeGraphStatus。
  // 直接调用：setConversationLoading。
  // [/maint-note:107432de0719]
  function resetConversationView() {
    setActiveThread(null);
    rememberActiveThread(null);
    setRuntimeProvider(defaultRuntimeConfig?.provider || "deepseek");
    setRuntimeModel(defaultRuntimeConfig?.model || "deepseek-chat");
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

  // [maint-note:0aada8127cc2]
  // 维护导航：requestThreadDeletion；下列为本函数源码中的接口与路径索引。
  // 输入 event：类型由调用方与函数体约束。
  // 输入 thread：类型由调用方与函数体约束。
  // 直接调用：event.stopPropagation。
  // 直接调用：setThreadDeleteError。
  // 直接调用：setThreadPendingDelete。
  // [/maint-note:0aada8127cc2]
  function requestThreadDeletion(event, thread) {
    event.stopPropagation();
    setThreadDeleteError("");
    setThreadPendingDelete(thread);
  }

  // [maint-note:3546128decee]
  // 维护导航：deletePendingThread；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!thread || deletingThreadId || cancellingThreadId。
  // 分支条件：thread.id !== activeThread?.id。
  // 分支条件：nextThreadId。
  // 返回路径：无返回值。
  // 直接调用：setDeletingThreadId。
  // 直接调用：setThreadDeleteError。
  // 直接调用：setLastError。
  // 直接调用：requestJson。
  // 直接调用：Promise.all。
  // 直接调用：refreshThreadList。
  // 直接调用：setAgentRuns。
  // 直接调用：selectThreadAfterDeletion。
  // 直接调用：setThreadPendingDelete。
  // 直接调用：loadAgentThread。
  // 直接调用：resetConversationView。
  // [/maint-note:3546128decee]
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

  // [maint-note:14a236e6208e]
  // 维护导航：cancelPendingThreadRun；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!activeRunId || deletingThreadId || cancellingThreadId。
  // 分支条件：activeThread?.id === thread.id。
  // 返回路径：无返回值。
  // 直接调用：setCancellingThreadId。
  // 直接调用：setThreadDeleteError。
  // 直接调用：setLastError。
  // 直接调用：requestJson。
  // 直接调用：Promise.all。
  // 直接调用：setThreadPendingDelete。
  // 直接调用：setAgentThreads。
  // 直接调用：setAgentRuns。
  // 直接调用：setActiveThread。
  // 直接调用：refreshAgentState。
  // [/maint-note:14a236e6208e]
  async function cancelPendingThreadRun() {
    const thread = threadPendingDelete;
    const activeRunId = thread?.active_run_id || thread?.last_run_id;
    if (!activeRunId || deletingThreadId || cancellingThreadId) return;
    setCancellingThreadId(thread.id);
    setThreadDeleteError("");
    setLastError("");
    try {
      await requestJson(`/api/agent/runs/${activeRunId}/cancel`, {
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
        await refreshAgentState(activeRunId, thread.workspace);
      }
    } catch (error) {
      setThreadDeleteError(error.message);
      setLastError(error.message);
    } finally {
      setCancellingThreadId("");
    }
  }

  // [maint-note:1ca434f1f272]
  // 维护导航：startThreadRename；下列为本函数源码中的接口与路径索引。
  // 输入 thread：类型由调用方与函数体约束。
  // 分支条件：!thread?.id || renamingBusy。
  // 返回路径：无返回值。
  // 直接调用：setRenamingThreadId。
  // 直接调用：setRenamingDraft。
  // 直接调用：setRenameError。
  // [/maint-note:1ca434f1f272]
  function startThreadRename(thread) {
    if (!thread?.id || renamingBusy) return;
    setRenamingThreadId(thread.id);
    setRenamingDraft(thread.title || "");
    setRenameError("");
  }

  // [maint-note:d36c4f8b3d9c]
  // 维护导航：cancelThreadRename；下列为本函数源码中的接口与路径索引。
  // 分支条件：renamingBusy。
  // 返回路径：无返回值。
  // 直接调用：setRenamingThreadId。
  // 直接调用：setRenamingDraft。
  // 直接调用：setRenameError。
  // [/maint-note:d36c4f8b3d9c]
  function cancelThreadRename() {
    if (renamingBusy) return;
    setRenamingThreadId("");
    setRenamingDraft("");
    setRenameError("");
  }

  // [maint-note:cd466aef4767]
  // 维护导航：commitThreadRename；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!thread || !nextTitle || nextTitle === (thread.title || "")。
  // 分支条件：activeThread?.id === thread.id。
  // 返回路径：无返回值。
  // 直接调用：agentThreads.find。
  // 直接调用：renamingDraft.trim。
  // 直接调用：cancelThreadRename。
  // 直接调用：setRenamingBusy。
  // 直接调用：setRenameError。
  // 直接调用：setLastError。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：setActiveThread。
  // 直接调用：refreshThreadList。
  // 直接调用：setRenamingThreadId。
  // 直接调用：setRenamingDraft。
  // [/maint-note:cd466aef4767]
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

  // [maint-note:569a0b567aca]
  // 维护导航：addContextItem；下列为本函数源码中的接口与路径索引。
  // 输入 item：类型由调用方与函数体约束。
  // 直接调用：setContextItems。
  // 直接调用：setContextSuggestions。
  // [/maint-note:569a0b567aca]
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

  // [maint-note:9e8ab4037162]
  // 维护导航：removeContextItem；下列为本函数源码中的接口与路径索引。
  // 输入 item：类型由调用方与函数体约束。
  // 直接调用：setContextItems。
  // [/maint-note:9e8ab4037162]
  function removeContextItem(item) {
    setContextItems((current) => {
      const key = item.absolute_path || item.path;
      const next = current.filter((entry) => (entry.absolute_path || entry.path) !== key);
      setTargets(next.map((entry) => entry.path).filter(Boolean));
      return next;
    });
  }

  // [maint-note:4796df9398f0]
  // 维护导航：selectContextSuggestion；下列为本函数源码中的接口与路径索引。
  // 输入 item：类型由调用方与函数体约束。
  // 直接调用：addContextItem。
  // 直接调用：setInstruction。
  // 直接调用：requestAnimationFrame。
  // [/maint-note:4796df9398f0]
  function selectContextSuggestion(item) {
    addContextItem(item);
    setInstruction((current) => current.replace(/(?:^|\s)@[^\s@]*$/, "").trimEnd());
    requestAnimationFrame(() => chatInputRef.current?.focus());
  }

  // [maint-note:037fd5d91681]
  // 维护导航：resolveAbsoluteContextFromInput；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!activeThread?.id。
  // 分支条件：candidate?.kind !== "absolute"。
  // 分支条件：payload.matches?.[0]。
  // 分支条件：payload.thread。
  // 返回路径：无返回值。
  // 直接调用：setLastError。
  // 直接调用：detectContextCandidate。
  // 直接调用：setInstruction。
  // 直接调用：requestAnimationFrame。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：addContextItem。
  // 直接调用：setActiveThread。
  // [/maint-note:037fd5d91681]
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

  // [maint-note:2916b2df1bc9]
  // 维护导航：saveBudget；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：Object.keys(errors).length。
  // 分支条件：agentState.runId && agentRunControllable。
  // 分支条件：activeThread?.id。
  // 返回路径：无返回值。
  // 直接调用：normalizeBudgetDraft。
  // 直接调用：validateBudgetChange。
  // 直接调用：Object.keys。
  // 直接调用：setLastError。
  // 直接调用：Object.values。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：refreshAgentState。
  // 直接调用：setActiveThread。
  // 直接调用：updateActiveThread。
  // 直接调用：setBudgetDraft。
  // 直接调用：setBudgetOpen。
  // [/maint-note:2916b2df1bc9]
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

  // [maint-note:dcdf7b0a37b8]
  // 维护导航：refreshAgentState；下列为本函数源码中的接口与路径索引。
  // 输入 runId：类型由调用方与函数体约束。
  // 输入 memoryProjectId：类型由调用方与函数体约束；缺省表达式 projectDir。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 返回路径：next。
  // 直接调用：Promise.all。
  // 直接调用：requestJson。
  // 直接调用：encodeURIComponent。
  // 直接调用：hydrateAgentState。
  // 直接调用：setAgentState。
  // 直接调用：setAgentRunState。
  // 直接调用：setAgentRuns。
  // 直接调用：setAgentMemories。
  // 直接调用：setMemoryProposals。
  // 直接调用：(proposalsPayload.proposals || []).map。
  // 直接调用：setMode。
  // [/maint-note:dcdf7b0a37b8]
  async function refreshAgentState(runId, memoryProjectId = projectDir) {
    const [eventPayload, runState, runsPayload, memoriesPayload, proposalsPayload] = await Promise.all([
      requestJson(`/api/agent/runs/${runId}/events`),
      requestJson(`/api/agent/runs/${runId}`),
      requestJson("/api/agent/runs"),
      requestJson(`/api/agent/memories?project_id=${encodeURIComponent(memoryProjectId)}`),
      requestJson(`/api/agent/memory-proposals?project_id=${encodeURIComponent(memoryProjectId)}`)
    ]);
    const next = hydrateAgentState(runId, eventPayload.events || [], runState);
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

  // [maint-note:8606e347e375]
  // 维护导航：handleAgentRun；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!submittedInstruction。
  // 分支条件：shouldAppendUserMessage(userMessageAddedRef.current)。
  // 分支条件：next.status === "completed"。
  // 返回路径：无返回值。
  // 写入目标：userMessageAddedRef.current。
  // 直接调用：instruction.trim。
  // 直接调用：setLastError。
  // 直接调用：setBusy。
  // 直接调用：setMode。
  // 直接调用：shouldAppendUserMessage。
  // 直接调用：addMessage。
  // 直接调用：ensureAgentThread。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：buildThreadMessagePayload。
  // 直接调用：normalizeBudgetDraft。
  // 直接调用：updateMessage。
  // 直接调用：setAgentState。
  // 直接调用：setAgentRunState。
  // 直接调用：refreshAgentState。
  // 直接调用：buildAgentRunMessage。
  // 直接调用：loadAgentThread。
  // 直接调用：refreshThreadList。
  // [/maint-note:8606e347e375]
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
      updateMessage(assistantId, { runId });
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

  // [maint-note:6e9c2bace58b]
  // 维护导航：refreshCodeGraphStatus；下列为本函数源码中的接口与路径索引。
  // 输入 thread：类型由调用方与函数体约束；缺省表达式 activeThread。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!thread?.id。
  // 返回路径：null。
  // 返回路径：payload.status || null。
  // 直接调用：setCodeGraphBusy。
  // 直接调用：requestJson。
  // 直接调用：setCodeGraphStatus。
  // 直接调用：setLastError。
  // [/maint-note:6e9c2bace58b]
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

  // [maint-note:97b167ee0ab5]
  // 维护导航：handleCodeGraphToggle；下列为本函数源码中的接口与路径索引。
  // 输入 enabled：类型由调用方与函数体约束。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!activeThread?.id。
  // 分支条件：enabled。
  // 返回路径：无返回值。
  // 直接调用：setCodeGraphEnabled。
  // 直接调用：setCodeGraphStatus。
  // 直接调用：setCodeGraphBusy。
  // 直接调用：updateActiveThread。
  // 直接调用：requestJson。
  // 直接调用：setLastError。
  // [/maint-note:97b167ee0ab5]
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

  // [maint-note:0f6bdae32f08]
  // 维护导航：runCodeGraphAction；下列为本函数源码中的接口与路径索引。
  // 输入 action：类型由调用方与函数体约束。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!activeThread?.id。
  // 分支条件：payload.status。
  // 分支条件：payload.url。
  // 返回路径：无返回值。
  // 直接调用：setCodeGraphBusy。
  // 直接调用：setLastError。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：setCodeGraphStatus。
  // 直接调用：window.open。
  // [/maint-note:0f6bdae32f08]
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

  // [maint-note:b8da650c76ca]
  // 维护导航：handleAgentApproval；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：busy || conversationLoading || approvalBusyRef.current || !agentState.runId || !agentState.pendingApproval?.risk。
  // 分支条件：pendingCard。
  // 分支条件：next.status === "completed"。
  // 返回路径：无返回值。
  // 写入目标：approvalBusyRef.current。
  // 直接调用：setBusy。
  // 直接调用：setLastError。
  // 直接调用：approveAgentAction。
  // 直接调用：setAgentState。
  // 直接调用：requestJson。
  // 直接调用：refreshAgentState。
  // 直接调用：buildAgentRunMessage。
  // 直接调用：next.changedFiles.join。
  // 直接调用：messages.find。
  // 直接调用：updateMessage。
  // 直接调用：addMessage。
  // 直接调用：loadAgentThread。
  // 直接调用：handleApprovalError。
  // [/maint-note:b8da650c76ca]
  async function handleAgentApproval() {
    if (busy || conversationLoading || approvalBusyRef.current || !agentState.runId || !agentState.pendingApproval?.risk) return;
    approvalBusyRef.current = true;
    setBusy(true);
    setLastError("");
    try {
      await approveAgentAction(requestJson, agentState.runId, agentState.pendingApproval);
      setAgentState((current) => ({ ...current, status: "running", pendingApproval: null }));
      const state = await requestJson(`/api/agent/runs/${agentState.runId}/run`, { method: "POST" });
      const next = await refreshAgentState(agentState.runId);
      const message = buildAgentRunMessage(state, next);
      const content = state.final_answer
        ? message.content
        : `${message.content}\nChanged files: ${next.changedFiles.join(", ") || "none"}`;
      const pendingCard = messages.find((item) => item.id === `pending-${agentState.runId}`);
      if (pendingCard) {
        updateMessage(pendingCard.id, { content, status: message.status, approval: message.approval });
      } else {
        addMessage("assistant", content, message.status, { approval: message.approval, runId: agentState.runId });
      }
      if (next.status === "completed") {
        await loadAgentThread(activeThread?.id);
      }
    } catch (error) {
      await handleApprovalError(error);
    } finally {
      approvalBusyRef.current = false;
      setBusy(false);
    }
  }

  // [maint-note:33d2abe3d96d]
  // 维护导航：handleApprovalError；下列为本函数源码中的接口与路径索引。
  // 输入 error：类型由调用方与函数体约束。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：error.status === 404 || error.status === 409。
  // 分支条件：error.status === 409。
  // 直接调用：setAgentState。
  // 直接调用：setAgentRunState。
  // 直接调用：setMode。
  // 直接调用：refreshAgentState。
  // 直接调用：refreshThreadList。
  // 直接调用：setLastError。
  // 直接调用：approvalErrorMessage。
  // [/maint-note:33d2abe3d96d]
  async function handleApprovalError(error) {
    if (error.status === 404 || error.status === 409) {
      setAgentState(initialAgentState);
      setAgentRunState(null);
      setMode("error");
      try {
        if (error.status === 409) await refreshAgentState(agentState.runId);
        await refreshThreadList();
      } catch {
        // Keep the original approval error and leave stale controls cleared.
      }
    }
    setLastError(approvalErrorMessage(error));
  }

  // [maint-note:c29af3861157]
  // 维护导航：handleAgentRejection；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：busy || conversationLoading || approvalBusyRef.current || !agentState.runId || !agentState.pendingApproval。
  // 分支条件：pendingCard。
  // 分支条件：next.status === "completed"。
  // 返回路径：无返回值。
  // 写入目标：approvalBusyRef.current。
  // 直接调用：setBusy。
  // 直接调用：requestJson。
  // 直接调用：setAgentState。
  // 直接调用：refreshAgentState。
  // 直接调用：buildAgentRunMessage。
  // 直接调用：messages.find。
  // 直接调用：updateMessage。
  // 直接调用：addMessage。
  // 直接调用：loadAgentThread。
  // 直接调用：handleApprovalError。
  // [/maint-note:c29af3861157]
  async function handleAgentRejection() {
    if (busy || conversationLoading || approvalBusyRef.current || !agentState.runId || !agentState.pendingApproval) return;
    approvalBusyRef.current = true;
    setBusy(true);
    try {
      await requestJson(`/api/agent/runs/${agentState.runId}/reject`, { method: "POST" });
      setAgentState((current) => ({ ...current, status: "running", pendingApproval: null }));
      const state = await requestJson(`/api/agent/runs/${agentState.runId}/run`, { method: "POST" });
      const next = await refreshAgentState(agentState.runId);
      const message = buildAgentRunMessage(state, next);
      const content = state.final_answer ? message.content : `Rejected action.\n${message.content}`;
      const pendingCard = messages.find((item) => item.id === `pending-${agentState.runId}`);
      if (pendingCard) {
        updateMessage(pendingCard.id, { content, status: message.status, approval: message.approval });
      } else {
        addMessage("assistant", content, message.status, { approval: message.approval, runId: agentState.runId });
      }
      if (next.status === "completed") {
        await loadAgentThread(activeThread?.id);
      }
    } catch (error) {
      await handleApprovalError(error);
    } finally {
      approvalBusyRef.current = false;
      setBusy(false);
    }
  }

  // [maint-note:dccff7cef2c2]
  // 维护导航：handleAgentControl；下列为本函数源码中的接口与路径索引。
  // 输入 action：类型由调用方与函数体约束。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!agentState.runId。
  // 返回路径：无返回值。
  // 直接调用：setBusy。
  // 直接调用：requestJson。
  // 直接调用：refreshAgentState。
  // 直接调用：setLastError。
  // [/maint-note:dccff7cef2c2]
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

  // [maint-note:d181af9cd1cb]
  // 维护导航：loadAgentRun；下列为本函数源码中的接口与路径索引。
  // 输入 runId：类型由调用方与函数体约束。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 直接调用：setRuntimeMode。
  // 直接调用：refreshAgentState。
  // [/maint-note:d181af9cd1cb]
  async function loadAgentRun(runId) {
    setRuntimeMode("agent");
    await refreshAgentState(runId);
  }

  // [maint-note:1373932c3b45]
  // 维护导航：handleMemoryAction；下列为本函数源码中的接口与路径索引。
  // 输入 memoryId：类型由调用方与函数体约束。
  // 输入 action：类型由调用方与函数体约束。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 直接调用：setBusy。
  // 直接调用：requestJson。
  // 直接调用：encodeURIComponent。
  // 直接调用：setAgentMemories。
  // 直接调用：setLastError。
  // [/maint-note:1373932c3b45]
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

  // [maint-note:a37ae32782d2]
  // 维护导航：refreshMemoryGovernance；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 直接调用：encodeURIComponent。
  // 直接调用：Promise.all。
  // 直接调用：requestJson。
  // 直接调用：setMemoryProposals。
  // 直接调用：(proposalPayload.proposals || []).map。
  // 直接调用：setAgentMemories。
  // [/maint-note:a37ae32782d2]
  async function refreshMemoryGovernance() {
    const suffix = `project_id=${encodeURIComponent(projectDir)}${activeThread?.id ? `&thread_id=${encodeURIComponent(activeThread.id)}` : ""}`;
    const [proposalPayload, memoryPayload] = await Promise.all([
      requestJson(`/api/agent/memory-proposals?${suffix}`),
      requestJson(`/api/agent/memories?project_id=${encodeURIComponent(projectDir)}`)
    ]);
    setMemoryProposals((proposalPayload.proposals || []).map(normalizeMemoryReview));
    setAgentMemories(memoryPayload.memories || []);
  }

  // [maint-note:ff6dc37da242]
  // 维护导航：handleMemoryEvidenceToggle；下列为本函数源码中的接口与路径索引。
  // 输入 messageId：类型由调用方与函数体约束。
  // 直接调用：setSelectedMemoryEvidence。
  // [/maint-note:ff6dc37da242]
  function handleMemoryEvidenceToggle(messageId) {
    setSelectedMemoryEvidence((current) => toggleEvidenceSelection(current, messageId));
  }

  // [maint-note:ccacdabdda5f]
  // 维护导航：handleCreateMemoryProposals；下列为本函数源码中的接口与路径索引。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!activeThread?.id || !selectedMemoryEvidence.length。
  // 分支条件：!payload.count。
  // 返回路径：无返回值。
  // 直接调用：setBusy。
  // 直接调用：setLastError。
  // 直接调用：setMemoryError。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：buildMemoryProposalSelection。
  // 直接调用：setSelectedMemoryEvidence。
  // 直接调用：refreshMemoryGovernance。
  // 直接调用：setRunDetailsOpen。
  // 直接调用：setLastUpdated。
  // [/maint-note:ccacdabdda5f]
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

  // [maint-note:dd3e950c92de]
  // 维护导航：handleProposalAction；下列为本函数源码中的接口与路径索引。
  // 输入 proposal：类型由调用方与函数体约束。
  // 输入 action：类型由调用方与函数体约束。
  // 输入 changes：类型由调用方与函数体约束；缺省表达式 null。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：action === "edit"。
  // 直接调用：setBusy。
  // 直接调用：setLastError。
  // 直接调用：setMemoryError。
  // 直接调用：requestJson。
  // 直接调用：JSON.stringify。
  // 直接调用：refreshMemoryGovernance。
  // 直接调用：refreshMemoryGovernance().catch。
  // [/maint-note:dd3e950c92de]
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

  // [maint-note:135955ca8fb5]
  // 维护导航：handleRun；下列为本函数源码中的接口与路径索引。
  // 返回路径：runtimeMode === "agent" ? handleAgentRun() : handlePipelineRun()。
  // 直接调用：handleAgentRun。
  // 直接调用：handlePipelineRun。
  // [/maint-note:135955ca8fb5]
  function handleRun() {
    return runtimeMode === "agent" ? handleAgentRun() : handlePipelineRun();
  }

  // [maint-note:b7301c3d9de1]
  // 维护导航：canRunWithoutInstruction；下列为本函数源码中的接口与路径索引。
  // 分支条件：currentFeature === "knowledge_graph"。
  // 分支条件：currentFeature === "code_repair"。
  // 分支条件：currentFeature === "design_to_code"。
  // 返回路径：true。
  // 返回路径：Boolean( (featureConfig.failure_log || "").trim() || (featureConfig.extra_context || "").trim() )。
  // 返回路径：Boolean( featureConfig.screenshot instanceof FileList && featureConfig.screenshot.length > 0 )。
  // 返回路径：currentFeature === "code_summary" || (currentFeature === "vulnerability_detection" && !featureConfig.auto_fix)。
  // 直接调用：Boolean。
  // 直接调用：(featureConfig.failure_log || "").trim。
  // 直接调用：(featureConfig.extra_context || "").trim。
  // [/maint-note:b7301c3d9de1]
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

  // [maint-note:adb8aaaa835c]
  // 维护导航：defaultTaskLabel；下列为本函数源码中的接口与路径索引。
  // 分支条件：currentFeature === "knowledge_graph"。
  // 分支条件：currentFeature === "code_summary"。
  // 分支条件：currentFeature === "vulnerability_detection"。
  // 分支条件：currentFeature === "code_repair"。
  // 分支条件：currentFeature === "design_to_code"。
  // 返回路径："Knowledge graph generation"。
  // 返回路径："Code summarization"。
  // 返回路径："Vulnerability scan"。
  // 返回路径："Code repair"。
  // 返回路径："Design to code"。
  // 返回路径："Code task"。
  // [/maint-note:adb8aaaa835c]
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

  // [maint-note:bb0093517016]
  // 维护导航：handleFeatureChange；下列为本函数源码中的接口与路径索引。
  // 输入 featureName：类型由调用方与函数体约束。
  // 直接调用：setCurrentFeature。
  // 直接调用：setFeatureConfig。
  // [/maint-note:bb0093517016]
  function handleFeatureChange(featureName) {
    setCurrentFeature(featureName);
    setFeatureConfig({});
  }

  // [maint-note:79f2f85a7134]
  // 维护导航：handleFeatureConfigChange；下列为本函数源码中的接口与路径索引。
  // 输入 name：类型由调用方与函数体约束。
  // 输入 value：类型由调用方与函数体约束。
  // 直接调用：setFeatureConfig。
  // [/maint-note:79f2f85a7134]
  function handleFeatureConfigChange(name, value) {
    setFeatureConfig((prev) => ({ ...prev, [name]: value }));
  }

  // [maint-note:f77b2b4e1723]
  // 维护导航：handleClear；下列为本函数源码中的接口与路径索引。
  // 直接调用：setInstruction。
  // 直接调用：setTerminalLog。
  // 直接调用：setMode。
  // 直接调用：setLastError。
  // 直接调用：setMessages。
  // 直接调用：setActiveMessageId。
  // [/maint-note:f77b2b4e1723]
  function handleClear() {
    setInstruction("");
    setTerminalLog("");
    setMode("idle");
    setLastError("");
    setMessages([]);
    setActiveMessageId(null);
  }

  // [maint-note:0254f6108f15]
  // 维护导航：copyCommand；下列为本函数源码中的接口与路径索引。
  // 输入 command：类型由调用方与函数体约束。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!command。
  // 返回路径：无返回值。
  // 直接调用：navigator.clipboard.writeText。
  // [/maint-note:0254f6108f15]
  async function copyCommand(command) {
    if (!command) {
      return;
    }
    await navigator.clipboard.writeText(command);
  }

  // [maint-note:c6d8815d7d6f]
  // 维护导航：handleSend；下列为本函数源码中的接口与路径索引。
  // 分支条件：busy。
  // 分支条件：!instruction.trim() && !canRunWithoutInstruction()。
  // 返回路径：无返回值。
  // 写入目标：userMessageAddedRef.current。
  // 直接调用：instruction.trim。
  // 直接调用：canRunWithoutInstruction。
  // 直接调用：setLastError。
  // 直接调用：addMessage。
  // 直接调用：defaultTaskLabel。
  // 直接调用：handleRun。
  // 直接调用：setInstruction。
  // 直接调用：requestAnimationFrame。
  // [/maint-note:c6d8815d7d6f]
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

  // [maint-note:f072b3dc7994]
  // 维护导航：handleKeyDown；下列为本函数源码中的接口与路径索引。
  // 输入 e：类型由调用方与函数体约束。
  // 分支条件：e.key === "Enter" && !e.shiftKey。
  // 直接调用：e.preventDefault。
  // 直接调用：handleSend。
  // [/maint-note:f072b3dc7994]
  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // [maint-note:df039f864c24]
  // 维护导航：autoResizeTextarea；下列为本函数源码中的接口与路径索引。
  // 分支条件：!el。
  // 返回路径：无返回值。
  // 写入目标：el.style.height。
  // 直接调用：Math.min。
  // [/maint-note:df039f864c24]
  function autoResizeTextarea() {
    const el = chatInputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }

  // [maint-note:eee49c70d677]
  // 维护导航：updateLayoutPrefs；下列为本函数源码中的接口与路径索引。
  // 输入 nextValue：类型由调用方与函数体约束。
  // 直接调用：setLayoutPrefs。
  // [/maint-note:eee49c70d677]
  function updateLayoutPrefs(nextValue) {
    setLayoutPrefs((current) => {
      const next = typeof nextValue === "function" ? nextValue(current) : nextValue;
      return saveLayoutPrefs(globalThis.localStorage, { ...current, ...next });
    });
  }

  // [maint-note:fb4807877ab8]
  // 维护导航：resizeDelta；下列为本函数源码中的接口与路径索引。
  // 输入 kind：类型由调用方与函数体约束。
  // 输入 event：类型由调用方与函数体约束。
  // 输入 start：类型由调用方与函数体约束。
  // 分支条件：kind === "sidebarWidth"。
  // 分支条件：kind === "drawerWidth"。
  // 返回路径：event.clientX - start.x。
  // 返回路径：start.x - event.clientX。
  // 返回路径：event.clientY - start.y。
  // [/maint-note:fb4807877ab8]
  function resizeDelta(kind, event, start) {
    if (kind === "sidebarWidth") {
      return event.clientX - start.x;
    }
    if (kind === "drawerWidth") {
      return start.x - event.clientX;
    }
    return event.clientY - start.y;
  }

  // [maint-note:032ce8325a4e]
  // 维护导航：startResize；下列为本函数源码中的接口与路径索引。
  // 输入 kind：类型由调用方与函数体约束。
  // 输入 event：类型由调用方与函数体约束。
  // 分支条件：event.button !== 0。
  // 返回路径：无返回值。
  // 直接调用：event.preventDefault。
  // 直接调用：document.body.classList.add。
  // 直接调用：window.addEventListener。
  // [/maint-note:032ce8325a4e]
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
    // [maint-note:dff1dcf80f6b]
    // 维护导航：onPointerMove；下列为本函数源码中的接口与路径索引。
    // 输入 moveEvent：类型由调用方与函数体约束。
    // 直接调用：resizeDelta。
    // 直接调用：updateLayoutPrefs。
    // [/maint-note:dff1dcf80f6b]
    const onPointerMove = (moveEvent) => {
      const delta = resizeDelta(kind, moveEvent, start);
      updateLayoutPrefs({ [kind]: start.value + delta });
    };
    // [maint-note:add5e80f87c6]
    // 维护导航：onPointerUp；下列为本函数源码中的接口与路径索引。
    // 直接调用：document.body.classList.remove。
    // 直接调用：window.removeEventListener。
    // [/maint-note:add5e80f87c6]
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

  // [maint-note:77a473bb6042]
  // 维护导航：handleResizeKey；下列为本函数源码中的接口与路径索引。
  // 输入 kind：类型由调用方与函数体约束。
  // 输入 event：类型由调用方与函数体约束。
  // 分支条件：!keyDelta。
  // 返回路径：无返回值。
  // 直接调用：event.preventDefault。
  // 直接调用：updateLayoutPrefs。
  // [/maint-note:77a473bb6042]
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

  // [maint-note:32dd64183263]
  // 维护导航：toggleTheme；下列为本函数源码中的接口与路径索引。
  // 直接调用：setTheme。
  // [/maint-note:32dd64183263]
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
                applyWorkspace(projectDir, targets)
                  .catch((error) => setLastError(error.message));
              }}
              placeholder="Workspace path"
            />
            <button
              type="button"
              className="icon-button"
              title="Choose workspace folder"
              disabled={workspacePickerBusy}
              onClick={chooseWorkspaceDirectory}
            >
              <FolderOpen size={16} />
            </button>
            <select
              className="header-select"
              value={runtimeMode === "agent" ? runtimeModel : model}
              onChange={(event) => {
                const value = event.target.value;
                if (runtimeMode === "agent") {
                  setRuntimeModel(value);
                } else {
                  setModel(value);
                }
                if (runtimeMode === "agent" && activeThread?.id) {
                  updateActiveThread({ model: value, runtime_model_config: runtimeModelConfig(value) }).catch((error) => setLastError(error.message));
                }
              }}
            >
              {models.map((item) => <option value={item} key={item}>{item}</option>)}
              {(() => {
                const selectedModel = runtimeMode === "agent" ? runtimeModel : model;
                return !models.includes(selectedModel) && selectedModel
                  ? <option value={selectedModel}>{selectedModel}</option>
                  : null;
              })()}
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
                  {msg.type === "assistant" && msg.runId === agentState.runId
                    && agentState.thinking.some((item) => item.text) && (
                    <details className="thinking-panel" defaultOpen>
                      <summary>
                        Model thinking
                        {agentState.status === "running" && <span className="thinking-live">Live</span>}
                      </summary>
                      <div className="thinking-panel-body">
                        {agentState.thinking.filter((item) => item.text).map((item) => (
                          <div key={item.callIndex} className="thinking-step">
                            <span>Model call {item.callIndex}</span>
                            <pre>{item.text}</pre>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                  {msg.type === "assistant" && msg.status === "running" ? (
                    <>
                      <div className="message-status-row">
                        <span className="status-dot running" />
                        <span style={{ color: "var(--amber)" }}>Running...</span>
                      </div>
                      <pre className="terminal-body">{msg.runId === agentState.runId && agentState.streamedContent
                        ? agentState.streamedContent : msg.content}</pre>
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
                      {msg.approval && agentState.runId === msg.runId && agentState.pendingApproval
                        && msg.approval.tool_call?.id === agentState.pendingApproval.tool_call?.id && (
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
                  {agentState.events.filter((event) => !event.type.endsWith(".delta")).slice(-30).map((event) => (
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
                {memoryExtraction?.run_id === agentState.runId && (
                  <p role="status">
                    Memory extraction: {memoryExtraction.status.replaceAll("_", " ")}
                    {memoryExtraction.reason && ` · ${memoryExtraction.reason.replaceAll("_", " ")}`}
                    {memoryExtraction.status === "completed" &&
                      ` · ${memoryExtraction.proposal_count} proposed, ${memoryExtraction.applied_count} applied`}
                  </p>
                )}
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => {
                    setMemoryError("");
                    refreshMemoryGovernance().catch((error) => setMemoryError(error.message));
                  }}
                >
                  Refresh memories
                </button>
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
                    title="Choose workspace folder"
                    disabled={workspacePickerBusy}
                    onClick={chooseWorkspaceDirectory}
                  >
                    <FolderOpen size={15} />
                  </button>
                  <button
                    type="button"
                    className="icon-button"
                    title="Apply workspace"
                    onClick={() => {
                      applyWorkspace(projectDir, targets)
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
                <label className="field-label" htmlFor="runtimeProvider">Provider</label>
                <select
                  id="runtimeProvider"
                  className="text-input"
                  value={runtimeProvider}
                  onChange={(event) => changeRuntimeProvider(event.target.value)}
                >
                  <option value="deepseek">DeepSeek official</option>
                  <option value="openrouter">OpenRouter</option>
                </select>
                <label className="field-label" htmlFor="model">Model</label>
                <input
                  id="model"
                  className="text-input"
                  value={modelInputValue(runtimeModel)}
                  onChange={(event) => setRuntimeModel(event.target.value)}
                  onBlur={() => {
                    if (activeThread?.id) {
                      updateActiveThread({ model: runtimeModel, runtime_model_config: runtimeModelConfig() }).catch((error) => setLastError(error.message));
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
                  <Info
                    label={runtimeMode === "agent" ? "Runtime provider" : "Pipeline provider"}
                    value={runtimeMode === "agent" ? runtimeProvider : provider}
                  />
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
                {threadNeedsCancellation(threadPendingDelete.active_status)
                  ? `This conversation has an active ${threadPendingDelete.active_status} task. Cancel it before deleting.`
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
              {threadNeedsCancellation(threadPendingDelete.active_status) ? (
                <button
                  type="button"
                  className="danger-button"
                  disabled={Boolean(deletingThreadId || cancellingThreadId || !threadPendingDelete.active_run_id)}
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

// [maint-note:75b69967f52b]
// 维护导航：BudgetMeter；下列为本函数源码中的接口与路径索引。
// 输入 { label, progress }：类型由调用方与函数体约束。
// 返回路径：( <div className={classNames("budget-meter", progress.level)}> <div className="budget-meter-label"> <span>{label}</span> <strong>{progress.used} / {progress.limit}</strong> </div> <div className="budget-meter-track"> <span style={{ width: `。
// 直接调用：classNames。
// [/maint-note:75b69967f52b]
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

// [maint-note:1854a8752f89]
// 维护导航：Metric；下列为本函数源码中的接口与路径索引。
// 输入 { label, value }：类型由调用方与函数体约束。
// 返回路径：( <div className="metric"> <span>{label}</span> <strong>{metricLabel(value)}</strong> </div> )。
// 直接调用：metricLabel。
// [/maint-note:1854a8752f89]
function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{metricLabel(value)}</strong>
    </div>
  );
}

// [maint-note:765054e83139]
// 维护导航：Info；下列为本函数源码中的接口与路径索引。
// 输入 { label, value }：类型由调用方与函数体约束。
// 返回路径：( <> <dt>{label}</dt> <dd title={String(value || "")}>{metricLabel(value)}</dd> </> )。
// 直接调用：String。
// 直接调用：metricLabel。
// [/maint-note:765054e83139]
function Info({ label, value }) {
  return (
    <>
      <dt>{label}</dt>
      <dd title={String(value || "")}>{metricLabel(value)}</dd>
    </>
  );
}

export default App;
