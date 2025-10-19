from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple

import requests
from flask import Response, jsonify, make_response

from .config import CHATGPT_RESPONSES_URL
from .http import build_cors_headers
from .session import ensure_session_id
from flask import request as flask_request
from .utils import get_effective_chatgpt_auth


def normalize_model_name(name: str | None, debug_model: str | None = None) -> str:
    if isinstance(debug_model, str) and debug_model.strip():
        return debug_model.strip()
    if not isinstance(name, str) or not name.strip():
        return "gpt-5"
    base = name.split(":", 1)[0].strip()
    for sep in ("-", "_"):
        lowered = base.lower()
        for effort in ("minimal", "low", "medium", "high"):
            suffix = f"{sep}{effort}"
            if lowered.endswith(suffix):
                base = base[: -len(suffix)]
                break
    mapping = {
        "gpt5": "gpt-5",
        "gpt-5-latest": "gpt-5",
        "gpt-5": "gpt-5",
        "gpt5-codex": "gpt-5-codex",
        "gpt-5-codex": "gpt-5-codex",
        "gpt-5-codex-latest": "gpt-5-codex",
        "codex": "codex-mini-latest",
        "codex-mini": "codex-mini-latest",
        "codex-mini-latest": "codex-mini-latest",
        # fake gpt-5-mini
        "gpt-5-mini": "gpt-5",
    }
    return mapping.get(base, "gpt-5")


def start_upstream_request(
    model: str,
    input_items: List[Dict[str, Any]],
    *,
    instructions: str | None = None,
    tools: List[Dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    parallel_tool_calls: bool = False,
    reasoning_param: Dict[str, Any] | None = None,
    include: List[str] | None = None,
    store: bool | None = False,
    extra_payload: Dict[str, Any] | None = None,
):
    access_token, account_id = get_effective_chatgpt_auth()
    if not access_token or not account_id:
        resp = make_response(
            jsonify(
                {
                    "error": {
                        "message": "Missing ChatGPT credentials. Run 'python3 chatmock.py login' first.",
                    }
                }
            ),
            401,
        )
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return None, resp

    include_values: List[str] = []
    if isinstance(include, list):
        for item in include:
            if isinstance(item, str) and item:
                include_values.append(item)
    if isinstance(reasoning_param, dict):
        include_values.append("reasoning.encrypted_content")

    client_session_id = None
    try:
        client_session_id = (
            flask_request.headers.get("X-Session-Id")
            or flask_request.headers.get("session_id")
            or None
        )
    except Exception:
        client_session_id = None
    session_id = ensure_session_id(instructions, input_items, client_session_id)

    responses_payload = {
        "model": model,
        "instructions": instructions if isinstance(instructions, str) and instructions.strip() else instructions,
        "input": input_items,
        "tools": tools or [],
        "tool_choice": tool_choice if tool_choice in ("auto", "none") or isinstance(tool_choice, dict) else "auto",
        "parallel_tool_calls": bool(parallel_tool_calls),
        "store": bool(store) if store is not None else False,
        "stream": True,
        "prompt_cache_key": session_id,
    }
    if include_values:
        # preserve order while removing duplicates
        seen: set[str] = set()
        deduped: List[str] = []
        for value in include_values:
            if value not in seen:
                deduped.append(value)
                seen.add(value)
        responses_payload["include"] = deduped

    if reasoning_param is not None:
        responses_payload["reasoning"] = reasoning_param

    if isinstance(extra_payload, dict):
        for key, value in extra_payload.items():
            if value is None:
                continue
            responses_payload[key] = value

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "session_id": session_id,
        "Accept-Encoding": "identity",
        "Connection": "keep-alive"
    }

    # print('responses_payload\n\n---\n\n', responses_payload, '\n\n---\n\n')

    try:
        upstream = requests.post(
            CHATGPT_RESPONSES_URL,
            headers=headers,
            json=responses_payload,
            stream=True,
            timeout=600,
        )
        return upstream, None
    except Exception as e:
        # Covers requests errors and everything else
        msg = f"Error while contacting ChatGPT: {type(e).__name__}: {e}"
        status = 502 if isinstance(e, requests.RequestException) else 500
        resp = make_response(jsonify({"error": {"message": msg}}), status)
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return None, resp
