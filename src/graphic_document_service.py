"""
Graphic Document Generator: AI-generated markdown document on a topic with AI-chosen illustrations.
Content length and style are controlled by a fixed system prompt; user configures provider, model, and max images.
Also produces an HTML version with base64-embedded images, saved to data/graphic_documents.
"""
import base64
import re
import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple

try:
    import markdown
except ImportError:
    markdown = None  # pip install markdown for HTML export

from src.llm_factory import LLMFactory, LLMProvider
from src.config import settings
from src.image_generation import generate_image_and_save, get_images_dir

logger = logging.getLogger(__name__)

# Fixed system prompt (not user-editable): controls length and tone. AI must be creative and innovative.
GRAPHIC_DOCUMENT_SYSTEM_PROMPT = """You are a creative, innovative writer. Your task is to write a detailed, engaging markdown document on the given topic.

Requirements:
- Write substantial but focused content: aim for 4–8 sections with 2–4 paragraphs each. Do not write an essay that is excessively long.
- Be creative, original, and innovative in your angles and examples.
- Use clear markdown: headers (##, ###), lists, bold/italic where appropriate.
- Where an illustration would strengthen the document, insert exactly N placeholders, each on its own line, in this exact format: [IMAGE: one-line description of the image to generate]
- Use exactly N placeholders, with N given in the prompt. Spread them across the document where they fit best.
- Output only valid markdown. No preamble or meta-commentary."""


def _get_llm_caller(provider_str: str, model_name: str):
    provider_str = (provider_str or "gemini").lower().strip()
    if provider_str == "gemini":
        provider = LLMProvider.GEMINI
        api_key = settings.gemini_api_key
        model = model_name or settings.gemini_default_model
    elif provider_str == "qwen":
        provider = LLMProvider.QWEN
        api_key = settings.qwen_api_key
        model = model_name or settings.qwen_default_model
    elif provider_str == "mistral":
        provider = LLMProvider.MISTRAL
        api_key = settings.mistral_api_key
        model = model_name or settings.mistral_default_model
    else:
        provider = LLMProvider.GEMINI
        api_key = settings.gemini_api_key
        model = model_name or settings.gemini_default_model
    return LLMFactory.create_caller(provider=provider, api_key=api_key, model=model)


def _polish_image_prompt(llm_caller, idea: str) -> str:
    """Turn a short image idea into a vivid prompt for image generation."""
    prompt = f"""You are an expert at creating detailed image generation prompts. 
Transform this idea into a single, detailed, vivid prompt optimized for AI image generation (2-4 sentences). 
Include subject, setting, lighting, style, and mood. Respond with ONLY the enhanced prompt, nothing else.

Idea: {idea}"""
    out = llm_caller.generate(prompt).strip()
    if out.startswith('"') and out.endswith('"'):
        out = out[1:-1]
    return out or idea


def get_graphic_documents_dir() -> str:
    """Return the directory where HTML exports are saved (for API file serving)."""
    return os.path.join(os.path.dirname(__file__), "..", "data", "graphic_documents")


def _slug(s: str) -> str:
    """Safe filename slug from topic."""
    s = re.sub(r"[^\w\s-]", "", s)[:50].strip()
    return re.sub(r"[-\s]+", "_", s) or "document"


def _normalize_markdown_images(md: str) -> str:
    """
    Ensure image syntax is on a single line so parsers (ReactMarkdown, Python markdown) render correctly.
    - Collapse ]\\n( to ](
    - Collapse newlines inside alt text (already single-line from our replacements; fix any from raw LLM).
    """
    # Fix broken image links: ] optional whitespace newline optional whitespace (
    md = re.sub(r'\]\s*\n\s*\(', '](', md)
    return md


def _build_and_save_html(markdown_content: str, topic: str) -> Optional[str]:
    """
    Convert markdown to HTML, embed images as base64, wrap in full document, save to data/graphic_documents.
    Returns the saved filename (e.g. document_20250119_123456.html), or None if the markdown package is not installed.
    """
    if markdown is None:
        logger.warning("markdown package not installed; run pip install markdown for HTML export")
        return None
    html_body = markdown.markdown(markdown_content, extensions=["extra", "nl2br"])
    images_dir = get_images_dir()
    # Replace src="/images/file/FILENAME" with base64 data URI
    def repl(match):
        filename = match.group(1).strip()
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            return match.group(0)
        path = os.path.join(images_dir, filename)
        if not os.path.isfile(path):
            return match.group(0)
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            return f'src="data:image/png;base64,{b64}"'
        except Exception:
            return match.group(0)

    html_body = re.sub(r'src="/images/file/([^"]+)"', repl, html_body)
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_slug(topic)}</title>
<style>
:root {{ --bg: #0f0f12; --surface: #18181c; --text: #e4e4e7; --muted: #a1a1aa; --accent: #818cf8; --border: #27272a; --code-bg: #1e1e24; }}
* {{ box-sizing: border-box; }}
body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; max-width: 720px; margin: 0 auto; padding: 2rem 1.5rem; line-height: 1.65; color: var(--text); background: var(--bg); }}
h1 {{ font-size: 1.85rem; font-weight: 700; margin-top: 0; margin-bottom: 0.75em; color: #fff; letter-spacing: -0.02em; border-bottom: 1px solid var(--border); padding-bottom: 0.5em; }}
h2 {{ font-size: 1.35rem; font-weight: 600; margin-top: 2em; margin-bottom: 0.5em; color: var(--text); }}
h3 {{ font-size: 1.1rem; font-weight: 600; margin-top: 1.5em; margin-bottom: 0.4em; color: var(--muted); }}
p {{ margin: 0 0 1em; color: var(--text); }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
img {{ max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); border: 1px solid var(--border); }}
ul, ol {{ margin: 0 0 1em; padding-left: 1.5em; color: var(--text); }}
li {{ margin-bottom: 0.35em; }}
blockquote {{ margin: 1em 0; padding: 0.5em 0 0.5em 1em; border-left: 4px solid var(--accent); background: var(--surface); border-radius: 0 6px 6px 0; color: var(--muted); }}
pre {{ background: var(--code-bg); padding: 1rem 1.25rem; border-radius: 8px; overflow-x: auto; font-size: 0.875rem; line-height: 1.5; border: 1px solid var(--border); margin: 1em 0; }}
code {{ background: var(--code-bg); padding: 0.2em 0.45em; border-radius: 4px; font-size: 0.9em; font-family: 'JetBrains Mono', 'Fira Code', monospace; border: 1px solid var(--border); }}
pre code {{ padding: 0; background: none; border: none; }}
hr {{ border: none; height: 1px; background: var(--border); margin: 2em 0; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
    out_dir = get_graphic_documents_dir()
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(topic)
    filename = f"{slug}_{timestamp}.html"
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    return filename


def _extract_image_placeholders(markdown: str, max_n: int) -> List[Tuple[str, str]]:
    """
    Find up to max_n placeholders [IMAGE: description]. Returns list of (full_match, description).
    """
    pattern = re.compile(r'\[IMAGE:\s*([^\]]+)\]', re.IGNORECASE)
    matches = list(pattern.finditer(markdown))
    result = []
    for m in matches[:max_n]:
        result.append((m.group(0), m.group(1).strip()))
    return result


def generate_graphic_document(
    topic: str,
    llm_provider: str = "gemini",
    model_name: str = "",
    max_images: int = 3,
    polish_image_prompts: bool = True,
) -> dict:
    """
    Generate a markdown document on the given topic with AI-chosen illustrations.

    - topic: user-provided topic.
    - llm_provider: gemini, qwen, or mistral.
    - model_name: optional model override.
    - max_images: 1–5; number of image placeholders and generated images.
    - polish_image_prompts: if True, use LLM to polish each image description before generation.

    Returns dict: success (bool), markdown (str), error (str if failed), images_generated (int).
    """
    max_images = max(1, min(5, int(max_images)))
    try:
        llm = _get_llm_caller(llm_provider, model_name)
    except Exception as e:
        logger.exception("Failed to create LLM caller for graphic document")
        return {"success": False, "markdown": "", "error": str(e), "images_generated": 0}

    user_message = f"""Topic: {topic}

Write the document. Use exactly {max_images} image placeholders in the format [IMAGE: description] where illustrations would help."""
    full_prompt = f"{GRAPHIC_DOCUMENT_SYSTEM_PROMPT}\n\n---\n\n{user_message}"

    try:
        raw_markdown = llm.generate(full_prompt)
    except Exception as e:
        logger.exception("LLM failed during graphic document content generation")
        return {"success": False, "markdown": "", "error": str(e), "images_generated": 0}

    if not raw_markdown or not raw_markdown.strip():
        return {"success": False, "markdown": "", "error": "AI returned empty content.", "images_generated": 0}

    placeholders = _extract_image_placeholders(raw_markdown, max_images)
    if not placeholders:
        normalized = _normalize_markdown_images(raw_markdown)
        html_filename = None
        try:
            html_filename = _build_and_save_html(normalized, topic)
        except Exception as e:
            logger.warning("Failed to build/save HTML for graphic document: %s", e)
        return {
            "success": True,
            "markdown": normalized,
            "error": None,
            "images_generated": 0,
            "html_filename": html_filename,
        }

    result_markdown = raw_markdown
    images_done = 0
    for full_match, description in placeholders:
        prompt_for_gen = _polish_image_prompt(llm, description) if polish_image_prompts else description
        gen_result = generate_image_and_save(prompt_for_gen, save=True)
        if gen_result.get("saved") and gen_result.get("filename"):
            # Single-line alt: no newlines so markdown image syntax parses everywhere
            alt = re.sub(r'\s+', ' ', (description or '').strip())
            repl = f"![{alt}](/images/file/{gen_result['filename']})"
            result_markdown = result_markdown.replace(full_match, repl, 1)
            images_done += 1
        else:
            err = gen_result.get("save_error", "Unknown error")
            logger.warning("Image generation failed for placeholder %s: %s", description, err)
            result_markdown = result_markdown.replace(full_match, f"*[Image: {description} — generation failed]*", 1)

    result_markdown = _normalize_markdown_images(result_markdown)

    html_filename = None
    try:
        html_filename = _build_and_save_html(result_markdown, topic)
    except Exception as e:
        logger.warning("Failed to build/save HTML for graphic document: %s", e)

    return {
        "success": True,
        "markdown": result_markdown,
        "error": None,
        "images_generated": images_done,
        "html_filename": html_filename,
    }
