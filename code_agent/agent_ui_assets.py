APP_HEADER_HTML = """
<div class="app-header">
    <h1>NaturalCC Agent Console</h1>
    <p>左侧配置工程与模型，右侧处理任务、输出和状态。</p>
    <div class="header-chips">
        <span class="header-chip">NaturalCC</span>
        <span class="header-chip">Aider</span>
        <span class="header-chip">Console</span>
    </div>
</div>
"""


USAGE_GUIDE = """
### 操作说明
1. 左侧先确认项目根目录，并选择一个或多个 Aider 目标文件。
2. 第一个目标文件会作为主解析文件。
3. 推荐先预览 Prompt，再执行。
4. 左右栏超出内容时，直接在各自面板内滚动。
"""


CUSTOM_CSS = """
:root {
    --app-bg: linear-gradient(180deg, #f5f1e8 0%, #f7f9fc 52%, #eef5f5 100%);
    --card-bg: rgba(255, 255, 255, 0.92);
    --border: rgba(20, 57, 79, 0.12);
    --text-main: #153449;
    --text-subtle: #5d7282;
    --accent: #0f766e;
    --accent-soft: rgba(15, 118, 110, 0.12);
    --amber: #d97706;
    --danger: #b9382f;
    --danger-soft: rgba(185, 56, 47, 0.12);
    --shadow: 0 22px 56px rgba(24, 52, 73, 0.10);
    --shell-height: clamp(646px, calc(100dvh - 182px), 724px);
}

*,
*::before,
*::after {
    box-sizing: border-box;
}

.gradio-container {
    background: var(--app-bg);
    width: min(1760px, calc(100vw - 12px)) !important;
    max-width: none !important;
    margin: 0 auto !important;
    min-height: auto;
    padding: 4px !important;
}

.gradio-container footer {
    display: none !important;
}

.header-shell {
    padding: 0 !important;
    margin: 0 0 4px !important;
}

.header-shell .html-container,
.header-shell .prose {
    padding: 0 !important;
    margin: 0 !important;
}

.app-header {
    background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 36%),
        linear-gradient(135deg, #0f172a 0%, #153449 58%, #115e59 100%);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 24px;
    box-shadow: var(--shadow);
    color: #f8fafc;
    padding: 6px 12px;
    margin-bottom: 0;
}

.app-header h1 {
    margin: 0;
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #ffffff;
    text-shadow: 0 1px 16px rgba(0, 0, 0, 0.32);
}

.app-header p {
    margin: 2px 0 0;
    color: rgba(248, 250, 252, 0.88);
    font-size: 0.78rem;
    line-height: 1.3;
}

.header-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 5px;
}

.header-chip {
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 999px;
    padding: 4px 9px;
    font-size: 0.72rem;
    color: rgba(248, 250, 252, 0.86);
    background: rgba(255, 255, 255, 0.08);
}

.workspace-row {
    align-items: stretch;
    gap: 6px;
    width: 100%;
    min-height: 0;
    margin: 0 !important;
}

.panel-shell {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 24px;
    box-shadow: var(--shadow);
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    padding: 6px !important;
    min-width: 0 !important;
    height: var(--shell-height);
    max-height: var(--shell-height);
    overflow: hidden;
}

.panel-shell > * {
    width: 100%;
    min-width: 0;
}

.panel-body {
    display: flex;
    flex: 1 1 auto;
    flex-direction: column;
    min-height: 0;
    overflow: auto;
    overscroll-behavior: contain;
    padding-right: 1px;
    scrollbar-width: none;
}

.panel-body:hover,
.panel-body:focus-within,
.panel-body:active {
    scrollbar-width: thin;
}

.panel-body::-webkit-scrollbar {
    width: 0;
    height: 0;
}

.panel-body:hover::-webkit-scrollbar,
.panel-body:focus-within::-webkit-scrollbar,
.panel-body:active::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

.panel-body:hover::-webkit-scrollbar-thumb,
.panel-body:focus-within::-webkit-scrollbar-thumb,
.panel-body:active::-webkit-scrollbar-thumb {
    background: rgba(21, 52, 73, 0.28);
    border-radius: 999px;
}

.panel-banner {
    margin-bottom: 3px;
}

.panel-title {
    margin-bottom: 2px;
}

.panel-title h2 {
    margin: 0;
    color: var(--text-main);
    font-size: 1.04rem;
    letter-spacing: -0.03em;
}

.panel-title p {
    margin: 2px 0 0;
    color: var(--text-subtle);
    line-height: 1.3;
    font-size: 0.76rem;
}

.panel-view {
    min-height: 0;
    overflow: visible;
    border: 1px solid rgba(21, 52, 73, 0.08);
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.86);
    padding: 4px !important;
    margin-bottom: 4px;
}

.panel-view:last-child {
    margin-bottom: 0;
}

.panel-view.hide,
.hide.panel-view,
.panel-view[style*="display: none"],
.panel-view[hidden] {
    display: none !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    min-height: 0 !important;
}

.panel-view > * {
    min-width: 0;
}

.panel-view .gr-column,
.panel-view .gr-group {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

.section-copy h3 {
    margin: 0 0 4px;
    color: var(--text-main);
    font-size: 0.96rem;
}

.section-copy p {
    margin: 0 0 8px;
    color: var(--text-subtle);
    line-height: 1.35;
    font-size: 0.8rem;
}

.signal-card {
    border-radius: 18px;
    border: 1px solid rgba(21, 52, 73, 0.10);
    padding: 6px 8px;
    margin-bottom: 4px;
}

.signal-card.signal-success {
    background: rgba(15, 118, 110, 0.08);
    border-color: rgba(15, 118, 110, 0.18);
}

.signal-card.signal-info {
    background: rgba(21, 52, 73, 0.05);
}

.signal-card.signal-warning {
    background: rgba(217, 119, 6, 0.09);
    border-color: rgba(217, 119, 6, 0.16);
}

.signal-card.signal-error {
    background: var(--danger-soft);
    border-color: rgba(185, 56, 47, 0.18);
}

.signal-eyebrow {
    color: var(--accent);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 3px;
}

.signal-error .signal-eyebrow {
    color: var(--danger);
}

.signal-warning .signal-eyebrow {
    color: var(--amber);
}

.signal-title {
    color: var(--text-main);
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 2px;
}

.signal-copy {
    color: var(--text-subtle);
    line-height: 1.35;
    font-size: 0.78rem;
}

.signal-list {
    margin: 0;
    padding-left: 18px;
    color: var(--text-subtle);
    line-height: 1.4;
    font-size: 0.78rem;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 4px;
    margin: 0 0 4px;
}

.metric-card {
    border-radius: 16px;
    border: 1px solid rgba(15, 118, 110, 0.10);
    background: linear-gradient(180deg, rgba(15, 118, 110, 0.08), rgba(255, 255, 255, 0.88));
    padding: 6px 7px;
}

.metric-label {
    display: block;
    color: var(--text-subtle);
    font-size: 0.74rem;
    margin-bottom: 3px;
}

.metric-value {
    display: block;
    color: var(--text-main);
    font-size: 0.9rem;
    font-weight: 700;
}

.info-rows {
    display: grid;
    gap: 4px;
}

.info-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: flex-start;
    padding: 5px 7px;
    border-radius: 12px;
    background: rgba(21, 52, 73, 0.04);
    color: var(--text-subtle);
    font-size: 0.8rem;
}

.info-row strong {
    color: var(--text-main);
    font-weight: 600;
    text-align: right;
}

.action-row {
    flex-wrap: wrap !important;
    gap: 4px;
}

.action-row > * {
    flex: 1 1 160px;
    min-width: 0 !important;
}

.panel-footer {
    display: flex;
    flex-wrap: wrap !important;
    gap: 4px;
    border-top: 1px solid rgba(21, 52, 73, 0.10);
    margin-top: 2px;
    padding-top: 4px;
    position: static;
    background: transparent;
    padding-bottom: 1px;
}

.panel-footer > * {
    flex: 1 1 180px;
    min-width: 0 !important;
}

.footer-btn {
    min-height: 33px !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
}

#config-panel textarea,
#execution-panel textarea {
    font-family: "IBM Plex Mono", ui-monospace, monospace !important;
}

#execution-panel [data-testid="textbox"],
#execution-panel [data-testid="textbox"] textarea,
#execution-panel .status-markdown,
#execution-panel .status-markdown .prose,
#execution-panel pre {
    scrollbar-width: none;
}

#execution-panel [data-testid="textbox"]::-webkit-scrollbar,
#execution-panel [data-testid="textbox"] textarea::-webkit-scrollbar,
#execution-panel .status-markdown::-webkit-scrollbar,
#execution-panel .status-markdown .prose::-webkit-scrollbar,
#execution-panel pre::-webkit-scrollbar {
    width: 0;
    height: 0;
}

#execution-panel [data-testid="textbox"]:hover,
#execution-panel [data-testid="textbox"]:focus-within,
#execution-panel [data-testid="textbox"] textarea:hover,
#execution-panel [data-testid="textbox"] textarea:focus,
#execution-panel .status-markdown:hover,
#execution-panel .status-markdown:focus-within,
#execution-panel .status-markdown .prose:hover,
#execution-panel .status-markdown .prose:focus-within,
#execution-panel pre:hover {
    scrollbar-width: thin;
}

#execution-panel [data-testid="textbox"]:hover::-webkit-scrollbar,
#execution-panel [data-testid="textbox"]:focus-within::-webkit-scrollbar,
#execution-panel [data-testid="textbox"] textarea:hover::-webkit-scrollbar,
#execution-panel [data-testid="textbox"] textarea:focus::-webkit-scrollbar,
#execution-panel .status-markdown:hover::-webkit-scrollbar,
#execution-panel .status-markdown:focus-within::-webkit-scrollbar,
#execution-panel .status-markdown .prose:hover::-webkit-scrollbar,
#execution-panel .status-markdown .prose:focus-within::-webkit-scrollbar,
#execution-panel pre:hover::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

#execution-panel [data-testid="textbox"]:hover::-webkit-scrollbar-thumb,
#execution-panel [data-testid="textbox"]:focus-within::-webkit-scrollbar-thumb,
#execution-panel [data-testid="textbox"] textarea:hover::-webkit-scrollbar-thumb,
#execution-panel [data-testid="textbox"] textarea:focus::-webkit-scrollbar-thumb,
#execution-panel .status-markdown:hover::-webkit-scrollbar-thumb,
#execution-panel .status-markdown:focus-within::-webkit-scrollbar-thumb,
#execution-panel .status-markdown .prose:hover::-webkit-scrollbar-thumb,
#execution-panel .status-markdown .prose:focus-within::-webkit-scrollbar-thumb,
#execution-panel pre:hover::-webkit-scrollbar-thumb {
    background: rgba(21, 52, 73, 0.28);
    border-radius: 999px;
}

.console-content-row {
    gap: 6px;
    align-items: stretch;
}

.console-content-row > * {
    min-width: 0 !important;
}

@media (max-width: 1440px), (max-height: 860px) {
    .gradio-container {
        width: calc(100vw - 10px) !important;
        padding: 3px !important;
    }

    .panel-shell {
        height: clamp(612px, calc(100dvh - 172px), 688px);
        max-height: clamp(612px, calc(100dvh - 172px), 688px);
    }

    .app-header h1 {
        font-size: 1.04rem;
    }

    .app-header p,
    .panel-title p,
    .section-copy p,
    .signal-copy,
    .signal-list,
    .info-row {
        font-size: 0.74rem;
    }

    .panel-title h2 {
        font-size: 0.98rem;
    }
}

@media (max-width: 1220px) {
    .workspace-row {
        gap: 10px;
    }

    .panel-shell {
        height: auto;
        max-height: none;
    }

    .console-content-row {
        gap: 6px;
    }

    .metric-grid {
        grid-template-columns: 1fr;
    }

    .info-row {
        flex-direction: column;
        align-items: flex-start;
    }
}

@media (max-width: 1380px) {
    .console-content-row {
        flex-direction: column;
    }
}
"""
