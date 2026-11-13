import base64
import os
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from code_agent.plugins.base import (
    ConfigField,
    ConfigFieldType,
    ExecutionContext,
    ExecutionMode,
    FeatureMetadata,
    FeaturePlugin,
    PluginResult,
)
from code_agent.plugins.registry import register_plugin


DIRECT_PROMPT = (
    "You are an expert web developer who specializes in HTML and CSS.\n"
    "A user will provide you with a screenshot of a webpage.\n"
    "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
    "Include all CSS code in the HTML file itself.\n"
    "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
    "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
    "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
    "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
    "Respond with the content of the HTML+CSS file (directly start with the code, do not add any additional explanation):\n"
)

TEXT_AUGMENTED_PROMPT_TEMPLATE = (
    "You are an expert web developer who specializes in HTML and CSS.\n"
    "A user will provide you with a screenshot of a webpage, along with all texts that they want to put on the webpage.\n"
    "The text elements are:\n{texts}\n"
    "You should generate the correct layout structure for the webpage, and put the texts in the correct places "
    "so that the resultant webpage will look the same as the given one.\n"
    "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
    "Include all CSS code in the HTML file itself.\n"
    "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
    "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
    "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
    "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
    "Respond with the content of the HTML+CSS file (directly start with the code, do not add any additional explanation):\n"
)


# [maint-note:d96870c40f55]
# 维护导航：_encode_image；下列为本函数源码中的接口与路径索引。
# 输入 image_path：str。
# 返回契约：str。
# 返回路径：base64.b64encode(f.read()).decode('utf-8')。
# 直接调用：open。
# 直接调用：base64.b64encode(f.read()).decode。
# 直接调用：base64.b64encode。
# 直接调用：f.read。
# 资源作用域：open(image_path, 'rb')；退出该 with 时调用对应上下文清理。
# [/maint-note:d96870c40f55]
def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# [maint-note:cab75a1a8ac4]
# 维护导航：_cleanup_response；下列为本函数源码中的接口与路径索引。
# 输入 response：str。
# 返回契约：str。
# 分支条件：response.startswith('```')。
# 分支条件：response.endswith('```')。
# 分支条件：response.lower().startswith('html')。
# 分支条件：'<!DOCTYPE' in response。
# 分支条件：'</html>' in response。
# 返回路径：response.strip()。
# 直接调用：response.strip。
# 直接调用：response.startswith。
# 直接调用：response[3:].strip。
# 直接调用：response.endswith。
# 直接调用：response[:-3].strip。
# 直接调用：response.lower().startswith。
# 直接调用：response.lower。
# 直接调用：response[4:].strip。
# 直接调用：response.split。
# [/maint-note:cab75a1a8ac4]
def _cleanup_response(response: str) -> str:
    response = response.strip()
    if response.startswith("```"):
        response = response[3:].strip()
    if response.endswith("```"):
        response = response[:-3].strip()
    if response.lower().startswith("html"):
        response = response[4:].strip()
    if "<!DOCTYPE" in response:
        response = "<!DOCTYPE" + response.split("<!DOCTYPE", 1)[1]
    if "</html>" in response:
        response = response.split("</html>")[0] + "</html>"
    return response.strip()


# [maint-note:6a6bc93c514e]
# 维护导航：_extract_texts_from_html；下列为本函数源码中的接口与路径索引。
# 输入 html_content：str。
# 返回契约：List[str]。
# 分支条件：text and text != 'html'。
# 分支条件：text。
# 遍历边界：script ← soup.find_all('script')。
# 遍历边界：style ← soup.find_all('style')。
# 遍历边界：element ← soup.find_all(string=True)。
# 返回路径：[]。
# 返回路径：texts。
# 捕获异常：ImportError。
# 直接调用：re.compile。
# 直接调用：comment_pattern.sub。
# 直接调用：BeautifulSoup。
# 直接调用：soup.find_all。
# 直接调用：script.decompose。
# 直接调用：style.decompose。
# 直接调用：str(element).strip。
# 直接调用：str。
# 直接调用：' '.join。
# 直接调用：text.split。
# 直接调用：texts.append。
# [/maint-note:6a6bc93c514e]
def _extract_texts_from_html(html_content: str) -> List[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    comment_pattern = re.compile(r"<!.*?>", re.DOTALL)
    html_content = comment_pattern.sub("", html_content)

    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup.find_all("script"):
        script.decompose()
    for style in soup.find_all("style"):
        style.decompose()

    texts = []
    for element in soup.find_all(string=True):
        text = str(element).strip()
        if text and text != "html":
            text = " ".join(text.split())
            if text:
                texts.append(text)
    return texts


# [maint-note:a8ee0cc003d8]
# 维护导航：_resolve_api_key；下列为本函数源码中的接口与路径索引。
# 输入 context：ExecutionContext。
# 返回契约：Optional[str]。
# 分支条件：api_key。
# 分支条件：value。
# 遍历边界：env_var ← ('OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY')。
# 返回路径：api_key。
# 返回路径：value。
# 返回路径：None。
# 直接调用：os.environ.get。
# [/maint-note:a8ee0cc003d8]
def _resolve_api_key(context: ExecutionContext) -> Optional[str]:
    api_key = context.api_key
    if api_key:
        return api_key
    for env_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        value = os.environ.get(env_var)
        if value:
            return value
    return None


MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


# [maint-note:60ea1e7da8c6]
# 维护导航：_guess_mime_type；下列为本函数源码中的接口与路径索引。
# 输入 image_path：str。
# 返回契约：str。
# 返回路径：MIME_TYPES.get(suffix, 'image/jpeg')。
# 直接调用：Path(image_path).suffix.lower。
# 直接调用：Path。
# 直接调用：MIME_TYPES.get。
# [/maint-note:60ea1e7da8c6]
def _guess_mime_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    return MIME_TYPES.get(suffix, "image/jpeg")


NON_VISION_MODEL_HINTS = (
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-v3",
    "gpt-3.5",
    "text-davinci",
    "text-embedding",
    "qwen2.5-coder",
)


# [maint-note:ffa82042905b]
# 维护导航：_normalize_model_id；下列为本函数源码中的接口与路径索引。
# 输入 model：str。
# 返回契约：str。
# 分支条件：model.startswith('openrouter/')。
# 返回路径：model[len('openrouter/'):]。
# 返回路径：model。
# 直接调用：model.startswith。
# 直接调用：len。
# [/maint-note:ffa82042905b]
def _normalize_model_id(model: str) -> str:
    """Strip openrouter/ prefix for direct OpenRouter API calls. Aider uses litellm which
    handles the prefix internally, but raw OpenAI-compatible API calls do not need it."""
    if model.startswith("openrouter/"):
        return model[len("openrouter/"):]
    return model


# [maint-note:1e37f4466209]
# 维护导航：_check_vision_capability；下列为本函数源码中的接口与路径索引。
# 输入 model：str。
# 返回契约：None。
# 分支条件：hint in normalized。
# 遍历边界：hint ← NON_VISION_MODEL_HINTS。
# 显式抛出：ValueError(f"Model '{model}' does not support vision/image input. Please select a vision-capable model such as gpt-4o, claude-3.5-sonnet, or gemini-pro.")。
# 直接调用：model.lower。
# 直接调用：ValueError。
# [/maint-note:1e37f4466209]
def _check_vision_capability(model: str) -> None:
    normalized = model.lower()
    for hint in NON_VISION_MODEL_HINTS:
        if hint in normalized:
            raise ValueError(
                f"Model '{model}' does not support vision/image input. "
                f"Please select a vision-capable model such as gpt-4o, claude-3.5-sonnet, or gemini-pro."
            )


# [maint-note:388072cfaee9]
# 维护导航：_call_vision_api；下列为本函数源码中的接口与路径索引。
# 输入 api_key：str。
# 输入 model：str。
# 输入 base64_image：str。
# 输入 image_path：str。
# 输入 prompt：str。
# 返回契约：str。
# 返回路径：response.choices[0].message.content or ''。
# 直接调用：_normalize_model_id。
# 直接调用：_check_vision_capability。
# 直接调用：os.environ.get。
# 直接调用：OpenAI。
# 直接调用：_guess_mime_type。
# 直接调用：client.chat.completions.create。
# [/maint-note:388072cfaee9]
def _call_vision_api(api_key: str, model: str, base64_image: str, image_path: str, prompt: str) -> str:
    from openai import OpenAI

    api_model = _normalize_model_id(model)
    _check_vision_capability(api_model)

    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    client = OpenAI(base_url=base_url, api_key=api_key)

    mime_type = _guess_mime_type(image_path)
    response = client.chat.completions.create(
        model=api_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.0,
        timeout=120,
    )

    return response.choices[0].message.content or ""


# [maint-note:9c6c0fcaf328]
# 数据与行为导航：DesignToCodePlugin。
# 继承接口：FeaturePlugin。
# [/maint-note:9c6c0fcaf328]
@register_plugin
class DesignToCodePlugin(FeaturePlugin):
    # [maint-note:1f2fe30d1249]
    # 维护导航：metadata；下列为本函数源码中的接口与路径索引。
    # 返回契约：FeatureMetadata。
    # 返回路径：FeatureMetadata(name='design_to_code', label='Design to Code', description='Convert a webpage screenshot into HTML/CSS code using vision-language models.', execution_mode=ExecutionMode.DIRECT)。
    # 直接调用：FeatureMetadata。
    # [/maint-note:1f2fe30d1249]
    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="design_to_code",
            label="Design to Code",
            description="Convert a webpage screenshot into HTML/CSS code using vision-language models.",
            execution_mode=ExecutionMode.DIRECT,
        )

    # [maint-note:5a82aa884620]
    # 维护导航：config_schema；下列为本函数源码中的接口与路径索引。
    # 返回契约：List[ConfigField]。
    # 返回路径：[ConfigField(name='prompt_strategy', label='Prompt Strategy', type=ConfigFieldType.SELECT, required=False, default='direct', options=[{'value': 'direct', 'label': 'Direct (screenshot only)'}, {'value': 'text_augmented', 'label': 'Text Augme。
    # 直接调用：ConfigField。
    # [/maint-note:5a82aa884620]
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="prompt_strategy",
                label="Prompt Strategy",
                type=ConfigFieldType.SELECT,
                required=False,
                default="direct",
                options=[
                    {"value": "direct", "label": "Direct (screenshot only)"},
                    {"value": "text_augmented", "label": "Text Augmented (screenshot + reference HTML texts)"},
                ],
                help_text="Direct: use screenshot only. Text Augmented: extract texts from a reference HTML file.",
            ),
            ConfigField(
                name="screenshot",
                label="Screenshot / 截图",
                type=ConfigFieldType.FILE,
                required=True,
                accept=".png,.jpg,.jpeg,.webp,.gif",
                help_text="Upload a screenshot of the webpage to convert.",
            ),
            ConfigField(
                name="text_source",
                label="Reference HTML (for Text Augmented)",
                type=ConfigFieldType.FILE,
                required=False,
                accept=".html,.htm",
                help_text="Optional: provide a reference HTML file to extract text elements from.",
            ),
            ConfigField(
                name="output_name",
                label="Output Filename",
                type=ConfigFieldType.TEXT,
                required=False,
                default="design_output.html",
                placeholder="design_output.html",
                help_text="Name of the generated HTML file (relative to project directory).",
            ),
            ConfigField(
                name="extra_instruction",
                label="Extra Instruction / 额外指令",
                type=ConfigFieldType.TEXTAREA,
                required=False,
                default="",
                placeholder='e.g. "Use Tailwind CSS", "Make it responsive"',
                help_text="Additional constraints appended to the prompt.",
            ),
        ]

    # [maint-note:fded20aa817d]
    # 维护导航：validate；下列为本函数源码中的接口与路径索引。
    # 输入 config：Dict[str, Any]。
    # 输入 files：Optional[Dict[str, Any]]；缺省表达式 None。
    # 返回契约：Optional[str]。
    # 分支条件：not has_screenshot。
    # 分支条件：strategy not in {'direct', 'text_augmented'}。
    # 分支条件：not output_name。
    # 分支条件：not output_name.lower().endswith(('.html', '.htm'))。
    # 分支条件：strategy == 'text_augmented'。
    # 分支条件：not has_text_source。
    # 返回路径：'Screenshot / 截图 is required'。
    # 返回路径：'prompt_strategy must be one of: direct, text_augmented'。
    # 返回路径：'output_name cannot be empty'。
    # 返回路径：'output_name must end with .html or .htm'。
    # 返回路径：'text_augmented strategy requires a reference HTML file (text_source)'。
    # 返回路径：None。
    # 直接调用：bool。
    # 直接调用：config.get。
    # 直接调用：files.get。
    # 直接调用：str(config.get('prompt_strategy', 'direct')).strip。
    # 直接调用：str。
    # 直接调用：str(config.get('output_name', 'design_output.html')).strip。
    # 直接调用：output_name.lower().endswith。
    # 直接调用：output_name.lower。
    # [/maint-note:fded20aa817d]
    def validate(self, config: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> Optional[str]:
        # Check required FILE fields against uploaded_files
        has_screenshot = bool(config.get("screenshot")) or bool(files and (files.get("screenshot") or files.get("files")))
        if not has_screenshot:
            return "Screenshot / 截图 is required"

        strategy = str(config.get("prompt_strategy", "direct")).strip()
        if strategy not in {"direct", "text_augmented"}:
            return "prompt_strategy must be one of: direct, text_augmented"

        output_name = str(config.get("output_name", "design_output.html")).strip()
        if not output_name:
            return "output_name cannot be empty"
        if not output_name.lower().endswith((".html", ".htm")):
            return "output_name must end with .html or .htm"

        if strategy == "text_augmented":
            has_text_source = bool(config.get("text_source")) or bool(files and (files.get("text_source") or files.get("files")))
            if not has_text_source:
                return "text_augmented strategy requires a reference HTML file (text_source)"

        return None

    # [maint-note:5e81f32b9c43]
    # 维护导航：preview；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：str。
    # 分支条件：extra。
    # 返回路径：'\n'.join(lines)。
    # 直接调用：str(config.get('prompt_strategy', 'direct')).strip。
    # 直接调用：str。
    # 直接调用：config.get。
    # 直接调用：str(config.get('output_name', 'design_output.html')).strip。
    # 直接调用：str(config.get('extra_instruction', '')).strip。
    # 直接调用：lines.append。
    # 直接调用：'\n'.join。
    # [/maint-note:5e81f32b9c43]
    def preview(self, context: ExecutionContext) -> str:
        config = context.feature_config or {}
        strategy = str(config.get("prompt_strategy", "direct")).strip()
        output_name = str(config.get("output_name", "design_output.html")).strip()
        extra = str(config.get("extra_instruction", "")).strip()

        lines = [
            "Design to Code Preview",
            f"- Strategy: {strategy}",
            f"- Output: {output_name}",
            f"- Model: {context.model}",
        ]
        if extra:
            lines.append(f"- Extra: {extra}")

        lines.append("")
        lines.append("--- Prompt ---")
        lines.append(DIRECT_PROMPT[:200] + "...")
        return "\n".join(lines)

    # [maint-note:5c1a2c5b751f]
    # 维护导航：execute；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：Generator[Any, None, None]。
    # 分支条件：screenshot_file is None。
    # 分支条件：screenshot_path is None。
    # 分支条件：prompt is None。
    # 分支条件：not api_key。
    # 分支条件：extra_instruction。
    # 返回路径：无返回值。
    # 捕获异常：Exception。
    # 直接调用：str(config.get('prompt_strategy', 'direct')).strip。
    # 直接调用：str。
    # 直接调用：config.get。
    # 直接调用：str(config.get('output_name', 'design_output.html')).strip。
    # 直接调用：str(config.get('extra_instruction', '')).strip。
    # 直接调用：self._get_uploaded_file。
    # 直接调用：PluginResult。
    # 直接调用：self._save_uploaded_file。
    # 直接调用：os.path.basename。
    # 直接调用：self._build_prompt。
    # 直接调用：_resolve_api_key。
    # 直接调用：_encode_image。
    # 直接调用：_call_vision_api。
    # 直接调用：_cleanup_response。
    # 直接调用：self._apply_extra_instruction。
    # 直接调用：Path。
    # 直接调用：output_path.parent.mkdir。
    # 直接调用：output_path.write_text。
    # 直接调用：len。
    # 直接调用：log_lines.append。
    # 直接调用：'\n'.join。
    # 生成器交付：'[Design2Code] Starting design-to-code conversion...\n'。
    # 生成器交付：PluginResult(success=False, message='Screenshot file is required but was not uploaded.', log='[Design2Code] Error: No screenshot file found in upload.\n')。
    # 生成器交付：PluginResult(success=False, message='Failed to save uploaded screenshot.', log='[Design2Code] Error: Could not save screenshot file.\n')。
    # 生成器交付：f'[Design2Code] Screenshot saved: {os.path.basename(screenshot_path)}\n'。
    # 生成器交付：PluginResult(success=False, message='Failed to build prompt.', log='[Design2Code] Error: Prompt building failed.\n')。
    # 生成器交付：f'[Design2Code] Calling vision model ({context.model})...\n'。
    # 生成器交付：PluginResult(success=False, message='API key not configured.', log='[Design2Code] Error: No API key found. Set it in settings or via OPENROUTER_API_KEY / OPENAI_API_KEY env var.\n')。
    # 生成器交付：PluginResult(success=False, message=f'Vision API call failed: {exc}', log=f'[Design2Code] Error during API call: {exc}\n')。
    # 生成器交付：'[Design2Code] Cleaning up response...\n'。
    # 生成器交付：PluginResult(success=False, message=f'Failed to write output file: {exc}', log=f'[Design2Code] Error writing file: {exc}\n')。
    # 生成器交付：f'[Design2Code] Output written to: {output_path}\n'。
    # 生成器交付：PluginResult(success=True, message=f'HTML generated and saved to {output_name}', log='\n'.join(log_lines) + '\n', files_modified=[str(output_path)], report=f'Generated HTML file: {output_name}\nSize: {len(html)} characters', artifacts={'htm。
    # [/maint-note:5c1a2c5b751f]
    def execute(self, context: ExecutionContext) -> Generator[Any, None, None]:
        config = context.feature_config or {}
        strategy = str(config.get("prompt_strategy", "direct")).strip()
        output_name = str(config.get("output_name", "design_output.html")).strip()
        extra_instruction = str(config.get("extra_instruction", "")).strip()
        uploaded_files = context.uploaded_files or {}

        yield "[Design2Code] Starting design-to-code conversion...\n"

        screenshot_file = self._get_uploaded_file(uploaded_files, "screenshot")
        if screenshot_file is None:
            yield PluginResult(
                success=False,
                message="Screenshot file is required but was not uploaded.",
                log="[Design2Code] Error: No screenshot file found in upload.\n",
            )
            return

        screenshot_path = self._save_uploaded_file(screenshot_file, context.project_dir)
        if screenshot_path is None:
            yield PluginResult(
                success=False,
                message="Failed to save uploaded screenshot.",
                log="[Design2Code] Error: Could not save screenshot file.\n",
            )
            return

        yield f"[Design2Code] Screenshot saved: {os.path.basename(screenshot_path)}\n"

        prompt = self._build_prompt(strategy, extra_instruction, uploaded_files)
        if prompt is None:
            yield PluginResult(
                success=False,
                message="Failed to build prompt.",
                log="[Design2Code] Error: Prompt building failed.\n",
            )
            return

        yield f"[Design2Code] Calling vision model ({context.model})...\n"

        api_key = _resolve_api_key(context)
        if not api_key:
            yield PluginResult(
                success=False,
                message="API key not configured.",
                log="[Design2Code] Error: No API key found. Set it in settings or via OPENROUTER_API_KEY / OPENAI_API_KEY env var.\n",
            )
            return

        try:
            base64_image = _encode_image(screenshot_path)
            raw_response = _call_vision_api(api_key, context.model, base64_image, screenshot_path, prompt)
        except Exception as exc:
            yield PluginResult(
                success=False,
                message=f"Vision API call failed: {exc}",
                log=f"[Design2Code] Error during API call: {exc}\n",
            )
            return

        yield "[Design2Code] Cleaning up response...\n"
        html = _cleanup_response(raw_response)

        if extra_instruction:
            html = self._apply_extra_instruction(html, extra_instruction)

        output_path = Path(context.project_dir) / output_name
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding="utf-8")
        except Exception as exc:
            yield PluginResult(
                success=False,
                message=f"Failed to write output file: {exc}",
                log=f"[Design2Code] Error writing file: {exc}\n",
            )
            return

        yield f"[Design2Code] Output written to: {output_path}\n"

        log_lines = [
            "[Design2Code] Conversion completed successfully.",
            f"- Strategy: {strategy}",
            f"- Model: {context.model}",
            f"- Output: {output_path}",
            f"- HTML size: {len(html)} chars",
        ]
        if extra_instruction:
            log_lines.append(f"- Extra instruction applied: {extra_instruction}")

        yield PluginResult(
            success=True,
            message=f"HTML generated and saved to {output_name}",
            log="\n".join(log_lines) + "\n",
            files_modified=[str(output_path)],
            report=f"Generated HTML file: {output_name}\nSize: {len(html)} characters",
            artifacts={"html": html, "output_path": str(output_path), "strategy": strategy},
        )

    # [maint-note:9ba245a0cd74]
    # 维护导航：_get_uploaded_file；下列为本函数源码中的接口与路径索引。
    # 输入 uploaded_files：Dict[str, Any]。
    # 输入 field_name：str。
    # 分支条件：value is None。
    # 分支条件：hasattr(value, 'filename') and hasattr(value, 'file')。
    # 分支条件：isinstance(value, list)。
    # 分支条件：hasattr(item, 'filename') and hasattr(item, 'file')。
    # 遍历边界：item ← value。
    # 返回路径：None。
    # 返回路径：value。
    # 返回路径：item。
    # 直接调用：uploaded_files.get。
    # 直接调用：hasattr。
    # 直接调用：isinstance。
    # [/maint-note:9ba245a0cd74]
    def _get_uploaded_file(self, uploaded_files: Dict[str, Any], field_name: str):
        value = uploaded_files.get(field_name)
        if value is None:
            value = uploaded_files.get("files")
        if value is None:
            return None

        if hasattr(value, "filename") and hasattr(value, "file"):
            return value
        if isinstance(value, list):
            for item in value:
                if hasattr(item, "filename") and hasattr(item, "file"):
                    return item
        return None

    # [maint-note:6d4dca0e04f1]
    # 维护导航：_save_uploaded_file；下列为本函数源码中的接口与路径索引。
    # 输入 upload_file：类型由调用方与函数体约束。
    # 输入 project_dir：str。
    # 返回契约：Optional[str]。
    # 分支条件：hasattr(upload_file, 'filename') and hasattr(upload_file, 'file')。
    # 分支条件：isinstance(upload_file, (str, Path)) and Path(upload_file).exists()。
    # 返回路径：str(dest)。
    # 返回路径：str(upload_file)。
    # 返回路径：None。
    # 直接调用：hasattr。
    # 直接调用：Path。
    # 直接调用：temp_dir.mkdir。
    # 直接调用：os.path.basename。
    # 直接调用：upload_file.file.read。
    # 直接调用：dest.write_bytes。
    # 直接调用：str。
    # 直接调用：isinstance。
    # 直接调用：Path(upload_file).exists。
    # [/maint-note:6d4dca0e04f1]
    def _save_uploaded_file(self, upload_file, project_dir: str) -> Optional[str]:
        if hasattr(upload_file, "filename") and hasattr(upload_file, "file"):
            temp_dir = Path(project_dir) / ".design2code_tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            safe_name = os.path.basename(upload_file.filename or "screenshot.png")
            dest = temp_dir / safe_name
            content = upload_file.file.read()
            dest.write_bytes(content)
            return str(dest)

        if isinstance(upload_file, (str, Path)) and Path(upload_file).exists():
            return str(upload_file)

        return None

    # [maint-note:3d0e1bf2c312]
    # 维护导航：_build_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 strategy：str。
    # 输入 extra_instruction：str。
    # 输入 uploaded_files：Dict[str, Any]。
    # 返回契约：Optional[str]。
    # 分支条件：strategy == 'direct'。
    # 分支条件：strategy == 'text_augmented'。
    # 分支条件：texts is None。
    # 分支条件：extra_instruction。
    # 返回路径：None。
    # 返回路径：prompt。
    # 直接调用：self._extract_texts。
    # 直接调用：'\n'.join。
    # 直接调用：TEXT_AUGMENTED_PROMPT_TEMPLATE.format。
    # 表达式分流：texts；真值与假值分支分别计算，不同时执行。
    # 集合派生：t 来自 texts。
    # [/maint-note:3d0e1bf2c312]
    def _build_prompt(
        self,
        strategy: str,
        extra_instruction: str,
        uploaded_files: Dict[str, Any],
    ) -> Optional[str]:
        if strategy == "direct":
            prompt = DIRECT_PROMPT
        elif strategy == "text_augmented":
            texts = self._extract_texts(uploaded_files)
            if texts is None:
                return None
            text_block = "\n".join(f"- {t}" for t in texts) if texts else "(no text elements found)"
            prompt = TEXT_AUGMENTED_PROMPT_TEMPLATE.format(texts=text_block)
        else:
            return None

        if extra_instruction:
            prompt += f"\n\nAdditional requirements:\n{extra_instruction}\n"

        return prompt

    # [maint-note:df5ca1564289]
    # 维护导航：_extract_texts；下列为本函数源码中的接口与路径索引。
    # 输入 uploaded_files：Dict[str, Any]。
    # 返回契约：Optional[List[str]]。
    # 分支条件：text_file is None。
    # 分支条件：path is None。
    # 返回路径：[]。
    # 返回路径：None。
    # 返回路径：_extract_texts_from_html(content)。
    # 捕获异常：Exception。
    # 直接调用：self._get_uploaded_file。
    # 直接调用：self._save_uploaded_file。
    # 直接调用：Path(path).read_text。
    # 直接调用：Path。
    # 直接调用：_extract_texts_from_html。
    # [/maint-note:df5ca1564289]
    def _extract_texts(self, uploaded_files: Dict[str, Any]) -> Optional[List[str]]:
        text_file = self._get_uploaded_file(uploaded_files, "text_source")
        if text_file is None:
            return []

        path = self._save_uploaded_file(text_file, "/tmp")
        if path is None:
            return None

        try:
            content = Path(path).read_text(encoding="utf-8")
            return _extract_texts_from_html(content)
        except Exception:
            return None

    # [maint-note:149e4d91eff8]
    # 维护导航：_apply_extra_instruction；下列为本函数源码中的接口与路径索引。
    # 输入 html：str。
    # 输入 instruction：str。
    # 返回契约：str。
    # 分支条件：'tailwind' in instruction.lower() or 'Tailwind' in instruction。
    # 分支条件：'tailwindcss.com' not in html and 'cdn.tailwindcss.com' not in html。
    # 分支条件：'<head>' in html。
    # 分支条件：'<html>' in html。
    # 返回路径：html。
    # 直接调用：instruction.lower。
    # 直接调用：html.replace。
    # [/maint-note:149e4d91eff8]
    def _apply_extra_instruction(self, html: str, instruction: str) -> str:
        if "tailwind" in instruction.lower() or "Tailwind" in instruction:
            if "tailwindcss.com" not in html and "cdn.tailwindcss.com" not in html:
                script = '<script src="https://cdn.tailwindcss.com"></script>'
                if "<head>" in html:
                    html = html.replace("<head>", f"<head>\n    {script}")
                elif "<html>" in html:
                    html = html.replace("<html>", f"<html>\n<head>\n    {script}\n</head>")
        return html
