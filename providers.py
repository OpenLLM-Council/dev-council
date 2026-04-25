"""Ollama-only model access for dev-council."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Generator


PROVIDERS: dict[str, dict] = {
    "local": {
        "type": "ollama",
        "label": "Ollama Local",
        "base_url_key": "ollama_local_base_url",
        "default_base_url": "http://localhost:11434",
        "api_key_key": "",
        "context_limit": 128000,
    },
    "cloud": {
        "type": "ollama",
        "label": "Ollama Cloud",
        "base_url_key": "ollama_cloud_base_url",
        "default_base_url": "",
        "api_key_key": "ollama_cloud_api_key",
        "context_limit": 128000,
    },
}


class TextChunk:
    def __init__(self, text: str):
        self.text = text


class ThinkingChunk:
    def __init__(self, text: str):
        self.text = text


class AssistantTurn:
    """Completed assistant turn with text + tool calls."""

    def __init__(self, text: str, tool_calls: list[dict], in_tokens: int, out_tokens: int):
        self.text = text
        self.tool_calls = tool_calls
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens


def detect_provider(model: str) -> str:
    if "/" in model:
        provider = model.split("/", 1)[0].strip().lower()
        if provider in PROVIDERS:
            return provider
        if provider == "ollama":
            return "local"
    return "local"


def bare_model(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def get_base_url(provider_name: str, config: dict) -> str:
    provider = PROVIDERS.get(provider_name, PROVIDERS["local"])
    key = provider["base_url_key"]
    return str(config.get(key) or provider["default_base_url"]).rstrip("/")


def get_api_key(provider_name: str, config: dict) -> str:
    provider = PROVIDERS.get(provider_name, PROVIDERS["local"])
    key = provider.get("api_key_key") or ""
    if not key:
        return ""
    return str(config.get(key, ""))


def calc_cost(model: str, in_tok: int, out_tok: int) -> float:
    return 0.0


def tools_to_ollama(tool_schemas: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        }
        for schema in tool_schemas
    ]


def messages_to_ollama(messages: list) -> list[dict]:
    result = []
    for message in messages:
        role = message["role"]
        if role == "user":
            item = {"role": "user", "content": message.get("content", "")}
            if message.get("images"):
                item["images"] = message["images"]
            result.append(item)
            continue

        if role == "assistant":
            item = {"role": "assistant", "content": message.get("content", "") or ""}
            tool_calls = []
            for tool_call in message.get("tool_calls", []):
                tool_calls.append(
                    {
                        "function": {
                            "name": tool_call["name"],
                            "arguments": tool_call.get("input", {}),
                        }
                    }
                )
            if tool_calls:
                item["tool_calls"] = tool_calls
            result.append(item)
            continue

        if role == "tool":
            result.append(
                {
                    "role": "tool",
                    "content": message.get("content", ""),
                    "tool_name": message.get("name", ""),
                }
            )
    return result


def messages_to_ollama_plain(messages: list) -> list[dict]:
    result = []
    for message in messages:
        role = message["role"]
        if role == "user":
            item = {"role": "user", "content": message.get("content", "")}
            if message.get("images"):
                item["images"] = message["images"]
            result.append(item)
            continue
        if role == "assistant":
            content = message.get("content", "") or ""
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                names = ", ".join(tc.get("name", "") for tc in tool_calls)
                content = f"{content}\n\n[Tool calls requested: {names}]".strip()
            result.append({"role": "assistant", "content": content})
            continue
        if role == "tool":
            name = message.get("name", "tool")
            content = message.get("content", "")
            result.append({"role": "user", "content": f"[Tool result from {name}]\n{content}"})
    return result


def stream_ollama(
    provider_name: str,
    model: str,
    system: str,
    messages: list,
    tool_schemas: list,
    config: dict,
) -> Generator:
    base_url = get_base_url(provider_name, config)
    if not base_url:
        raise ValueError(
            f"{PROVIDERS[provider_name]['label']} base URL is not configured. "
            f"Set {PROVIDERS[provider_name]['base_url_key']} in /config."
        )

    headers = {"Content-Type": "application/json"}
    api_key = get_api_key(provider_name, config)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages_to_ollama(messages),
        "stream": True,
        "options": {"num_ctx": config.get("context_limit", PROVIDERS[provider_name]["context_limit"])},
    }
    if tool_schemas and not config.get("no_tools"):
        payload["tools"] = tools_to_ollama(tool_schemas)

    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )

    text = ""
    tool_calls: list[dict] = []
    in_tokens = 0
    out_tokens = 0

    try:
        response_cm = urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as exc:
        has_tool_protocol = "tools" in payload or any(
            message.get("role") == "tool" or message.get("tool_calls")
            for message in messages
        )
        if exc.code in {400, 500} and has_tool_protocol:
            payload.pop("tools", None)
            payload["messages"] = [{"role": "system", "content": system}] + messages_to_ollama_plain(messages)
            request = urllib.request.Request(
                f"{base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
            )
            response_cm = urllib.request.urlopen(request, timeout=300)
        else:
            raise

    with response_cm as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            message = data.get("message", {})
            thinking = message.get("thinking")
            if thinking:
                yield ThinkingChunk(thinking)

            content = message.get("content", "")
            if content:
                text += content
                yield TextChunk(content)

            for tool_call in message.get("tool_calls", []):
                function = tool_call.get("function", {})
                tool_calls.append(
                    {
                        "id": f"call_{len(tool_calls)}",
                        "name": function.get("name", ""),
                        "input": function.get("arguments", {}) or {},
                    }
                )

            if data.get("done"):
                in_tokens = int(data.get("prompt_eval_count") or 0)
                out_tokens = int(data.get("eval_count") or 0)

    yield AssistantTurn(text, tool_calls, in_tokens, out_tokens)


def stream(
    model: str,
    system: str,
    messages: list,
    tool_schemas: list,
    config: dict,
) -> Generator:
    provider_name = detect_provider(model)
    model_name = bare_model(model)
    yield from stream_ollama(provider_name, model_name, system, messages, tool_schemas, config)


def list_ollama_models(base_url: str, api_key: str = "") -> list[str]:
    if not base_url:
        return []
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    return [item["name"] for item in data.get("models", []) if item.get("name")]
