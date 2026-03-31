"""
Customization tools: induce params/SQL from system prompt + user query, then execute request or db tool.
"""
import json
import logging
import re
from typing import Optional, Dict, Any, Union

from .models import CustomizationProfile, LLMProviderType
from .config import settings


logger = logging.getLogger(__name__)


def _extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Try to find ```json ... ``` block first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    # Or find {...} in the text
    if "{" in text and "}" in text:
        start = text.find("{")
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def induce_request_params(
    req_profile: Any,
    system_prompt: str,
    user_query: str,
    llm,
) -> Dict[str, Any]:
    """
    Use LLM to induce request params and/or body from system prompt + user query.
    Returns dict with optional 'params' and 'body' keys.
    """
    schema_hint = {
        "params": req_profile.params if req_profile.params else {},
        "body": req_profile.body if isinstance(req_profile.body, (dict, list)) else None,
    }
    from .request_tools import should_wrap_request_json_body

    wrap_note = ""
    if should_wrap_request_json_body(req_profile):
        wrap_note = (
            "\nNote: The HTTP API expects a JSON array of items at the root, but you should still put the single payload "
            'in the "body" key as one JSON object (not an array); the client will wrap it as [{...}] when sending.\n'
        )
    prompt = f"""You are an API parameter extractor. Follow the system instructions to extract parameters from the user's natural language input.{wrap_note}

## System instructions
{system_prompt}

## API configuration
- Method: {req_profile.method.value if req_profile.method else 'GET'}
- URL: {req_profile.url}
- Description: {req_profile.description or 'No description'}
- Expected params (query string): {json.dumps(schema_hint.get('params') or {{}}, indent=2)}
- Expected body structure (for POST/PUT): {json.dumps(schema_hint.get('body') or {{}}, indent=2) if schema_hint.get('body') else 'No body or use params'}

## User query
{user_query}

Output a valid JSON object with keys "params" and/or "body" containing the extracted values. Use only the keys that the API expects. If the user did not provide a value for a required field, use null or omit it. Return ONLY valid JSON, no other text."""

    response = await llm.ainvoke(prompt)
    result = _extract_json_from_response(response)
    if not result or not isinstance(result, dict):
        logger.warning("LLM did not return valid JSON for request params, using empty")
        return {}
    return result


async def execute_customization_with_tools(
    profile: CustomizationProfile,
    query: str,
    *,
    request_tools_manager=None,
    db_tools_manager=None,
    rag_system=None,
    text_to_sql_service=None,
    context: Optional[Dict[str, Any]] = None,
    provider_str: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> str:
    """
    Execute a customization query, optionally using a request tool or db tool.
    When request_tool_id or db_tool_id is set, uses LLM to induce params/SQL from system prompt + user query.
    """
    from .llm_factory import LLMFactory, LLMProvider
    from .llm_langchain_wrapper import LangChainLLMWrapper

    provider_str = provider_str or (
        profile.llm_provider.value if profile.llm_provider else settings.default_llm_provider
    )
    if provider_str == "gemini":
        provider = LLMProviderType.GEMINI
        api_key = settings.gemini_api_key
        model_name = model_name or profile.model_name or settings.gemini_default_model
    elif provider_str == "qwen":
        provider = LLMProviderType.QWEN
        api_key = settings.qwen_api_key
        model_name = model_name or profile.model_name or settings.qwen_default_model
    elif provider_str == "mistral":
        provider = LLMProviderType.MISTRAL
        api_key = settings.mistral_api_key
        model_name = model_name or profile.model_name or settings.mistral_default_model
    elif provider_str == "groq":
        provider = LLMProviderType.GROQ
        api_key = getattr(settings, "groq_api_key", "")
        model_name = model_name or profile.model_name or getattr(settings, "groq_default_model", "llama-3.3-70b-versatile")
    else:
        provider = LLMProviderType.GEMINI
        api_key = settings.gemini_api_key
        model_name = model_name or profile.model_name or settings.gemini_default_model

    llm_caller = LLMFactory.create_caller(
        provider=LLMProvider(provider_str),
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.api_timeout,
    )
    llm = LangChainLLMWrapper(llm_caller=llm_caller)

    # Build base prompt parts
    system_prompt = profile.system_prompt
    flow_context = (context or {}).get("flow_context_formatted", "")
    if flow_context:
        system_prompt = f"{flow_context}\n\n{system_prompt}"

    rag_context = ""
    if profile.rag_collection and rag_system:
        results = rag_system.query_collection(profile.rag_collection, query, n_results=3)
        if results:
            rag_context = "\n\n".join(r["content"] for r in results[:3])

    full_system = system_prompt
    if rag_context:
        full_system += f"\n\nContext (from knowledge base):\n{rag_context}"
    full_system += f"\n\nUser query: {query}"

    # ---- Request tool path ----
    if profile.request_tool_id and request_tools_manager:
        req_profile = request_tools_manager.get_profile(profile.request_tool_id)
        if not req_profile:
            return f"Error: Request tool '{profile.request_tool_id}' not found."

        induced = await induce_request_params(req_profile, system_prompt, query, llm)
        params_override = induced.get("params")
        body_override = induced.get("body")

        exec_kw: Dict[str, Any] = {}
        if params_override is not None:
            merged = {**(req_profile.params or {}), **params_override}
            exec_kw["params"] = {k: v for k, v in merged.items() if v is not None}
        if body_override is not None:
            exec_kw["body"] = body_override

        import asyncio

        result = await asyncio.to_thread(
            lambda: request_tools_manager.execute_request(req_profile.id, **exec_kw),
        )

        if not result.get("success"):
            return f"API call failed: {result.get('error', 'Unknown error')}. Response: {result}"

        response_data = result.get("response_data")
        if profile.tool_response_mode == "summarize" and response_data is not None:
            summarize_prompt = f"""{full_system}

## API response
{json.dumps(response_data, indent=2) if isinstance(response_data, (dict, list)) else str(response_data)}

Summarize the API response for the user in a clear, natural way. Address their original query."""
            return await llm.ainvoke(summarize_prompt)

        return json.dumps(response_data, indent=2) if isinstance(response_data, (dict, list)) else str(response_data)

    # ---- DB tool path ----
    if profile.db_tool_id and db_tools_manager:
        if not text_to_sql_service:
            from .text_to_sql import TextToSQLService
            text_to_sql_service = TextToSQLService(db_tools_manager=db_tools_manager)

        question = f"{system_prompt}\n\nUser question: {query}"
        tts_result = text_to_sql_service.run(
            question=question,
            db_tool_id=profile.db_tool_id,
            provider=provider_str,
            model=model_name,
        )
        if tts_result.get("error"):
            return f"Database query failed: {tts_result['error']}"

        if profile.tool_response_mode == "summarize":
            return tts_result.get("summary", json.dumps(tts_result.get("rows", []), indent=2))

        summary = tts_result.get("summary", "")
        rows = tts_result.get("rows", [])
        columns = tts_result.get("columns", [])
        if rows:
            return f"{summary}\n\n```\n{json.dumps(rows[:100], indent=2)}\n```"
        return summary or "No results."

    # ---- No tool: standard LLM response ----
    return await llm.ainvoke(full_system)
