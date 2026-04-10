from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class UIComponents:
    """References to Gradio components that participate in event wiring."""

    general_config_view: Any
    advanced_config_view: Any
    general_config_btn: Any
    advanced_config_btn: Any
    instruction_view: Any
    console_view: Any
    status_view: Any
    instruction_panel_btn: Any
    console_panel_btn: Any
    status_panel_btn: Any
    choose_dir_btn: Any
    refresh_btn: Any
    project_dir_input: Any
    target_files_input: Any
    model_dropdown: Any
    api_key_input: Any
    symbol_input: Any
    completion_type_input: Any
    prefix_input: Any
    instruction_input: Any
    config_banner: Any
    project_overview: Any
    advanced_control_summary: Any
    instruction_feedback: Any
    command_summary: Any
    command_preview_output: Any
    generated_content_output: Any
    status_detail: Any
    preview_btn: Any
    run_btn: Any
    clear_btn: Any


@dataclass
class CallbackHandlers:
    """UI callback functions supplied by agent_ui.py."""

    switch_left_view: Callable[[str], tuple]
    switch_right_view: Callable[[str], tuple]
    choose_project_dir: Callable[..., tuple]
    refresh_workspace_views: Callable[..., tuple]
    sync_workspace_panels: Callable[..., tuple]
    sync_advanced_controls: Callable[..., str]
    build_instruction_feedback: Callable[..., str]
    preview_prompt_from_ui: Callable[..., tuple]
    run_aider_stream_from_ui: Callable[..., Any]
    clear_task_view: Callable[..., tuple]


def bind_ui_events(ui: UIComponents, handlers: CallbackHandlers) -> None:
    """Wire button and input listeners to the Gradio callbacks."""

    left_nav_outputs = [
        ui.general_config_view,
        ui.advanced_config_view,
        ui.general_config_btn,
        ui.advanced_config_btn,
    ]
    right_nav_outputs = [
        ui.instruction_view,
        ui.console_view,
        ui.status_view,
        ui.instruction_panel_btn,
        ui.console_panel_btn,
        ui.status_panel_btn,
    ]
    workspace_inputs = [
        ui.project_dir_input,
        ui.target_files_input,
        ui.model_dropdown,
        ui.api_key_input,
        ui.symbol_input,
        ui.completion_type_input,
        ui.prefix_input,
        ui.instruction_input,
    ]
    workspace_outputs = [
        ui.project_dir_input,
        ui.target_files_input,
        ui.config_banner,
        ui.project_overview,
        ui.advanced_control_summary,
        ui.instruction_feedback,
    ]
    workspace_summary_outputs = [
        ui.config_banner,
        ui.project_overview,
        ui.advanced_control_summary,
        ui.instruction_feedback,
    ]
    advanced_inputs = [
        ui.model_dropdown,
        ui.target_files_input,
        ui.symbol_input,
        ui.completion_type_input,
        ui.prefix_input,
    ]
    execution_inputs = [
        ui.target_files_input,
        ui.instruction_input,
        ui.model_dropdown,
        ui.api_key_input,
        ui.project_dir_input,
        ui.symbol_input,
        ui.completion_type_input,
        ui.prefix_input,
    ]
    execution_outputs = [
        ui.instruction_feedback,
        ui.command_summary,
        ui.command_preview_output,
        ui.generated_content_output,
        ui.status_detail,
        *right_nav_outputs,
    ]

    # Left card footer, left button: switch to the "常规配置" pane.
    ui.general_config_btn.click(
        fn=lambda: handlers.switch_left_view("general"),
        outputs=left_nav_outputs,
    )
    # Left card footer, right button: switch to the "工程概览 / 进阶" pane.
    ui.advanced_config_btn.click(
        fn=lambda: handlers.switch_left_view("advanced"),
        outputs=left_nav_outputs,
    )

    # Right card footer, left button: switch to the "开发指令" pane.
    ui.instruction_panel_btn.click(
        fn=lambda: handlers.switch_right_view("instruction"),
        outputs=right_nav_outputs,
    )
    # Right card footer, center button: switch to the "命令行生成内容" pane.
    ui.console_panel_btn.click(
        fn=lambda: handlers.switch_right_view("console"),
        outputs=right_nav_outputs,
    )
    # Right card footer, right button: switch to the "操作说明 / 状态" pane.
    ui.status_panel_btn.click(
        fn=lambda: handlers.switch_right_view("status"),
        outputs=right_nav_outputs,
    )

    # Left card body, first action row, left button: open the native project directory picker.
    ui.choose_dir_btn.click(
        fn=handlers.choose_project_dir,
        inputs=workspace_inputs,
        outputs=workspace_outputs,
    )
    # Left card body, first action row, right button: rescan the current workspace snapshot.
    ui.refresh_btn.click(
        fn=handlers.refresh_workspace_views,
        inputs=workspace_inputs,
        outputs=workspace_outputs,
    )

    # Basic config inputs: keep the left summaries and instruction hint in sync.
    ui.project_dir_input.change(
        fn=handlers.refresh_workspace_views,
        inputs=workspace_inputs,
        outputs=workspace_outputs,
    )
    ui.target_files_input.change(
        fn=handlers.sync_workspace_panels,
        inputs=workspace_inputs,
        outputs=workspace_summary_outputs,
    )
    ui.model_dropdown.change(
        fn=handlers.sync_workspace_panels,
        inputs=workspace_inputs,
        outputs=workspace_summary_outputs,
    )
    ui.api_key_input.change(
        fn=handlers.sync_workspace_panels,
        inputs=workspace_inputs,
        outputs=workspace_summary_outputs,
    )

    # Advanced controls: refresh the compact summary after symbol-related changes.
    ui.symbol_input.change(
        fn=handlers.sync_advanced_controls,
        inputs=advanced_inputs,
        outputs=[ui.advanced_control_summary],
    )
    ui.completion_type_input.change(
        fn=handlers.sync_advanced_controls,
        inputs=advanced_inputs,
        outputs=[ui.advanced_control_summary],
    )
    ui.prefix_input.change(
        fn=handlers.sync_advanced_controls,
        inputs=advanced_inputs,
        outputs=[ui.advanced_control_summary],
    )

    # Right card, instruction editor: update the live summary card while typing.
    ui.instruction_input.change(
        fn=handlers.build_instruction_feedback,
        inputs=[ui.instruction_input, ui.target_files_input],
        outputs=[ui.instruction_feedback],
    )

    # Right card, instruction pane action row, left button: preview the generated prompt only.
    ui.preview_btn.click(
        fn=handlers.preview_prompt_from_ui,
        inputs=execution_inputs,
        outputs=execution_outputs,
    )
    # Right card, instruction pane action row, center button: run the full Aider execution flow.
    ui.run_btn.click(
        fn=handlers.run_aider_stream_from_ui,
        inputs=execution_inputs,
        outputs=execution_outputs,
    )
    # Right card, instruction pane action row, right button: clear the task input and outputs.
    ui.clear_btn.click(
        fn=handlers.clear_task_view,
        inputs=[ui.target_files_input, ui.model_dropdown, ui.project_dir_input],
        outputs=[ui.instruction_input, *execution_outputs],
    )
