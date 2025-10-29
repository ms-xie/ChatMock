from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import requests
from flask import Blueprint, Response, current_app, jsonify, make_response, request, stream_with_context

from .config import BASE_INSTRUCTIONS, GPT5_CODEX_INSTRUCTIONS
from .limits import record_rate_limits_from_response
from .http import build_cors_headers
from .reasoning import apply_reasoning_to_message, build_reasoning_param, extract_reasoning_from_model_name, extract_reasoning_from_last_input, clean_reasoning_tag_in_query
from .upstream import normalize_model_name, start_upstream_request
from .utils import (
    convert_chat_messages_to_responses_input,
    convert_tools_chat_to_responses,
    sse_translate_chat,
    sse_translate_text,
)

API_KEY_CUSTOM_SUFFIX = os.environ.get("API_KEY_CUSTOM_SUFFIX", "-chatmock")
EMBEDDINGS_ENDPOINT = os.environ.get("OPENAI_EMBEDDINGS_ENDPOINT", "").strip() or "https://api.openai.com/v1/embeddings"
openai_bp = Blueprint("openai", __name__)


def _instructions_for_model(model: str) -> str:
    base = current_app.config.get("BASE_INSTRUCTIONS", BASE_INSTRUCTIONS)
    if model == "gpt-5-codex":
        codex = current_app.config.get("GPT5_CODEX_INSTRUCTIONS") or GPT5_CODEX_INSTRUCTIONS
        if isinstance(codex, str) and codex.strip():
            return codex
    return base


@openai_bp.route("/v1/chat/completions", methods=["POST"])
def chat_completions() -> Response:
    verbose = bool(current_app.config.get("VERBOSE"))
    reasoning_effort = current_app.config.get("REASONING_EFFORT", "medium")
    reasoning_summary = current_app.config.get("REASONING_SUMMARY", "auto")
    reasoning_compat = current_app.config.get("REASONING_COMPAT", "think-tags")
    debug_model = current_app.config.get("DEBUG_MODEL")

    if verbose:
        try:
            body_preview = (request.get_data(cache=True, as_text=True) or "")[:2000]
            print("IN POST /v1/chat/completions\n" + body_preview)
        except Exception:
            pass

    raw = request.get_data(cache=True, as_text=True) or ""
    try:
        payload = json.loads(raw) if raw else {}
    except Exception:
        try:
            payload = json.loads(raw.replace("\r", "").replace("\n", ""))
        except Exception:
            return jsonify({"error": {"message": "Invalid JSON body"}}), 400

    requested_model = payload.get("model")
    model = normalize_model_name(requested_model, debug_model)
    messages = payload.get("messages")
    if messages is None and isinstance(payload.get("prompt"), str):
        messages = [{"role": "user", "content": payload.get("prompt") or ""}]
    if messages is None and isinstance(payload.get("input"), str):
        messages = [{"role": "user", "content": payload.get("input") or ""}]
    if messages is None:
        messages = []
    if not isinstance(messages, list):
        return jsonify({"error": {"message": "Request must include messages: []"}}), 400

    if isinstance(messages, list):
        sys_idx = next((i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "system"), None)
        if isinstance(sys_idx, int):
            sys_msg = messages.pop(sys_idx)
            content = sys_msg.get("content") if isinstance(sys_msg, dict) else ""
            messages.insert(0, {"role": "user", "content": content})
    is_stream = bool(payload.get("stream"))
    stream_options = payload.get("stream_options") if isinstance(payload.get("stream_options"), dict) else {}
    include_usage = bool(stream_options.get("include_usage", False))

    tools_responses = convert_tools_chat_to_responses(payload.get("tools"))
    tool_choice = payload.get("tool_choice", "auto")
    parallel_tool_calls = bool(payload.get("parallel_tool_calls", False))
    responses_tools_payload = payload.get("responses_tools") if isinstance(payload.get("responses_tools"), list) else []
    extra_tools: List[Dict[str, Any]] = []
    had_responses_tools = False
    if isinstance(responses_tools_payload, list):
        for _t in responses_tools_payload:
            if not isinstance(_t, dict):
                continue
            tool_type = _t.get("type")
            if not isinstance(tool_type, str):
                continue
            if tool_type == "web_search_preview":
                return (
                    jsonify(
                        {
                            "error": {
                                "message": "web_search_preview is no longer supported in responses_tools",
                                "code": "RESPONSES_TOOL_UNSUPPORTED",
                            }
                        }
                    ),
                    400,
                )
            if tool_type != "web_search":
                return (
                    jsonify(
                        {
                            "error": {
                                "message": "Only web_search is supported in responses_tools",
                                "code": "RESPONSES_TOOL_UNSUPPORTED",
                            }
                        }
                    ),
                    400,
                )
            extra_tools.append(_t)

        if not extra_tools and bool(current_app.config.get("DEFAULT_WEB_SEARCH")):
            responses_tool_choice = payload.get("responses_tool_choice")
            if not (isinstance(responses_tool_choice, str) and responses_tool_choice == "none"):
                extra_tools = [{"type": "web_search"}]

        if extra_tools:
            import json as _json
            MAX_TOOLS_BYTES = 32768
            try:
                size = len(_json.dumps(extra_tools))
            except Exception:
                size = 0
            if size > MAX_TOOLS_BYTES:
                return jsonify({"error": {"message": "responses_tools too large", "code": "RESPONSES_TOOLS_TOO_LARGE"}}), 400
            had_responses_tools = True
            tools_responses = (tools_responses or []) + extra_tools

    responses_tool_choice = payload.get("responses_tool_choice")
    if isinstance(responses_tool_choice, str) and responses_tool_choice in ("auto", "none"):
        tool_choice = responses_tool_choice

    input_items = convert_chat_messages_to_responses_input(messages)
    if not input_items and isinstance(payload.get("prompt"), str) and payload.get("prompt").strip():
        input_items = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": payload.get("prompt")}]}
        ]

    extra_payload: Dict[str, Any] = {}
    for key in (
        # "max_output_tokens",
        "max_tool_calls",
        # "temperature",
        "top_p",
        "top_logprobs",
        # "service_tier",
        "safety_identifier",
        "prompt_cache_key",
        "previous_response_id",
    ):
        if key in payload:
            extra_payload[key] = payload.get(key)
    
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None
    if metadata is not None:
        extra_payload["metadata"] = metadata
    if isinstance(stream_options, dict) and stream_options:
        extra_payload["stream_options"] = stream_options

    extra_payload_ignore: Dict[str, Any] = {}
    for key in (
        "max_output_tokens",
        "temperature",
        "service_tier"
    ):
        if key in payload:
            extra_payload_ignore[key] = payload.get(key)

    if extra_payload_ignore:
        print(f'This is not official response endpoint, these parameter will ignore:\n{extra_payload_ignore}\n')

    # TODO: add parse reasoning effort in query as /v1/responses
    model_reasoning = extract_reasoning_from_model_name(requested_model)
    reasoning_overrides = payload.get("reasoning") if isinstance(payload.get("reasoning"), dict) else model_reasoning
    reasoning_param = build_reasoning_param(reasoning_effort, reasoning_summary, reasoning_overrides)

    if reasoning_param.get('effort') == "minimal":
        tools_responses = [
            t for t in tools_responses
            if t.get("type") != "web_search"
        ]
    if isinstance(reasoning_overrides, dict):
        for k, v in reasoning_overrides.items():
            if k not in reasoning_param and v is not None:
                reasoning_param[k] = v

    upstream, error_resp = start_upstream_request(
        model,
        input_items,
        instructions=_instructions_for_model(model),
        tools=tools_responses,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        reasoning_param=reasoning_param,
        extra_payload=extra_payload,
    )
    if error_resp is not None:
        return error_resp

    record_rate_limits_from_response(upstream)

    # Check status
    if upstream.status_code != 200:
        # You can get the plain text or JSON error message
        try:
            error_text = upstream.text  # Raw text body
            # or try JSON decoding if the server returns JSON
            error_json = upstream.json()  
        except ValueError:
            error_json = None

        print("Status code:", upstream.status_code)
        print("Response JSON:", error_json)
        print("Response JSON:", error_json)
        return (
            jsonify(
                {
                    "error": {
                        "message": f"Upstream error: {error_json or error_text}",
                        "code": "RESPONSES_TOOLS_REJECTED",
                    }
                }
            ),
            (upstream.status_code if upstream is not None else 500),
        )
    
    created = int(time.time())

    if is_stream:
        resp = Response(
            sse_translate_chat(
                upstream,
                requested_model or model,
                created,
                verbose=verbose,
                vlog=print if verbose else None,
                reasoning_compat=reasoning_compat,
                include_usage=include_usage,
            ),
            status=upstream.status_code,
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
        resp.headers.setdefault("X-Accel-Buffering", "no")
        # Preserve your CORS handling
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp

    full_text = ""
    reasoning_summary_text = ""
    reasoning_full_text = ""
    response_id = "chatcmpl"
    tool_calls: List[Dict[str, Any]] = []
    error_message: str | None = None
    usage_obj: Dict[str, int] | None = None

    def _extract_usage(evt: Dict[str, Any]) -> Dict[str, int] | None:
        try:
            usage = (evt.get("response") or {}).get("usage")
            if not isinstance(usage, dict):
                return None
            pt = int(usage.get("input_tokens") or 0)
            ct = int(usage.get("output_tokens") or 0)
            tt = int(usage.get("total_tokens") or (pt + ct))
            return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
        except Exception:
            return None
    try:
        for raw in upstream.iter_lines(chunk_size=1, decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else raw
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):].strip()
            if not data:
                continue
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
            except Exception:
                continue
            kind = evt.get("type")
            mu = _extract_usage(evt)
            if mu:
                usage_obj = mu
            if isinstance(evt.get("response"), dict) and isinstance(evt["response"].get("id"), str):
                response_id = evt["response"].get("id") or response_id
            if kind == "response.output_text.delta":
                full_text += evt.get("delta") or ""
            elif kind == "response.reasoning_summary_text.delta":
                reasoning_summary_text += evt.get("delta") or ""
            elif kind == "response.reasoning_text.delta":
                reasoning_full_text += evt.get("delta") or ""
            elif kind == "response.output_item.done":
                item = evt.get("item") or {}
                if isinstance(item, dict) and item.get("type") == "function_call":
                    call_id = item.get("call_id") or item.get("id") or ""
                    name = item.get("name") or ""
                    args = item.get("arguments") or ""
                    if isinstance(call_id, str) and isinstance(name, str) and isinstance(args, str):
                        tool_calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": args},
                            }
                        )
            elif kind == "response.failed":
                error_message = evt.get("response", {}).get("error", {}).get("message", "response.failed")
            elif kind == "response.completed":
                break
    finally:
        upstream.close()

    if error_message:
        resp = make_response(jsonify({"error": {"message": error_message}}), 502)
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp

    message: Dict[str, Any] = {"role": "assistant", "content": full_text if full_text else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    message = apply_reasoning_to_message(message, reasoning_summary_text, reasoning_full_text, reasoning_compat)
    completion = {
        "id": response_id or "chatcmpl",
        "object": "chat.completion",
        "created": created,
        "model": requested_model or model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        **({"usage": usage_obj} if usage_obj else {}),
    }
    resp = make_response(jsonify(completion), upstream.status_code)
    for k, v in build_cors_headers().items():
        resp.headers.setdefault(k, v)
    return resp


# /v1/completions route deprecated and removed on 2025-10-29.


@openai_bp.route("/v1/responses", methods=["POST"])
def responses() -> Response:
    verbose = bool(current_app.config.get("VERBOSE"))
    debug_model = current_app.config.get("DEBUG_MODEL")
    reasoning_effort = current_app.config.get("REASONING_EFFORT", "medium")
    reasoning_summary = current_app.config.get("REASONING_SUMMARY", "auto")

    def _normalize_input(source: Any) -> List[Dict[str, Any]]:
        def _as_message(text: str) -> Dict[str, Any] | None:
            if not isinstance(text, str) or not text.strip():
                return None
            return {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }

        if isinstance(source, str):
            item = _as_message(source)
            return [item] if item else []
        if isinstance(source, dict):
            return [source]
        if isinstance(source, list):
            items: List[Dict[str, Any]] = []
            for entry in source:
                if isinstance(entry, dict):
                    items.append(entry)
                elif isinstance(entry, str):
                    msg = _as_message(entry)
                    if msg:
                        items.append(msg)
            return items
        return []

    def _normalize_tools(raw: Any) -> List[Dict[str, Any]]:
        tools_out: List[Dict[str, Any]] = []
        if not isinstance(raw, list):
            return tools_out
        for tool in raw:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == 'web_search_preview':
                continue
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                conv = convert_tools_chat_to_responses([tool])
                if conv:
                    converted = conv[0]
                    fn = tool.get("function", {})
                    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                        converted["name"] = fn.get("name")
                    if "description" in tool and isinstance(tool.get("description"), str):
                        converted["description"] = tool.get("description")
                    if "strict" in tool:
                        converted["strict"] = bool(tool.get("strict"))
                    elif isinstance(fn, dict) and "strict" in fn:
                        converted["strict"] = bool(fn.get("strict"))
                    tools_out.append(converted)
                continue
            if tool.get("type") == "function" and isinstance(tool.get("name"), str):
                params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {"type": "object", "properties": {}}
                normalized = {
                    "type": "function",
                    "name": tool.get("name"),
                    "description": tool.get("description") or "",
                    "parameters": params,
                    "strict": bool(tool.get("strict")),
                }
                tools_out.append(normalized)
                continue
            tools_out.append(tool)
        return tools_out

    raw_body = request.get_data(cache=True, as_text=True) or ""
    if verbose:
        try:
            preview = raw_body
            print("IN POST /v1/responses\n" + preview)
        except Exception:
            pass

    try:
        payload = json.loads(raw_body) if raw_body else {}
    except Exception:
        try:
            payload = json.loads(raw_body.replace("\r", "").replace("\n", ""))
        except Exception:
            return jsonify({"error": {"message": "Invalid JSON body"}}), 400

    if bool(payload.get("background")):
        return jsonify({"error": {"message": "background responses are not supported"}}), 400

    stream_req = bool(payload.get("stream"))
    stream_options = payload.get("stream_options") if isinstance(payload.get("stream_options"), dict) else {}

    requested_model = payload.get("model")
    model = normalize_model_name(requested_model, debug_model)

    instructions = payload.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        instructions = _instructions_for_model(model)

    input_items = _normalize_input(payload.get("input"))
    if not input_items and isinstance(payload.get("messages"), list):
        input_items = convert_chat_messages_to_responses_input(payload.get("messages"))
    if not input_items and isinstance(payload.get("prompt"), str):
        input_items = _normalize_input(payload.get("prompt"))
    if not input_items and isinstance(payload.get("input"), dict):
        input_items = _normalize_input([payload.get("input")])

    tool_choice = payload.get("tool_choice", "auto")
    parallel_tool_calls_value = payload.get("parallel_tool_calls")
    if parallel_tool_calls_value is None:
        parallel_tool_calls = True
    else:
        parallel_tool_calls = bool(parallel_tool_calls_value)

    base_tools = _normalize_tools(payload.get("tools"))
    tools_responses = list(base_tools)
    responses_tools_payload = payload.get("responses_tools") if isinstance(payload.get("responses_tools"), list) else []
    had_responses_tools = False
    extra_tools: List[Dict[str, Any]] = []
    if isinstance(responses_tools_payload, list):
        for _tool in responses_tools_payload:
            if not isinstance(_tool, dict):
                continue
            tool_type = _tool.get("type")
            if not isinstance(tool_type, str):
                continue
            if tool_type == "web_search_preview":
                return (
                    jsonify(
                        {
                            "error": {
                                "message": "web_search_preview is no longer supported in responses_tools",
                                "code": "RESPONSES_TOOL_UNSUPPORTED",
                            }
                        }
                    ),
                    400,
                )
            if tool_type != "web_search":
                return (
                    jsonify(
                        {
                            "error": {
                                "message": "Only web_search is supported in responses_tools",
                                "code": "RESPONSES_TOOL_UNSUPPORTED",
                            }
                        }
                    ),
                    400,
                )
            extra_tools.append(_tool)
        if not extra_tools and bool(current_app.config.get("DEFAULT_WEB_SEARCH")):
            responses_tool_choice = payload.get("responses_tool_choice")
            if not (isinstance(responses_tool_choice, str) and responses_tool_choice == "none"):
                extra_tools = [{"type": "web_search"}]
        if extra_tools:
            try:
                size = len(json.dumps(extra_tools))
            except Exception:
                size = 0
            if size > 32768:
                return (
                    jsonify({"error": {"message": "responses_tools too large", "code": "RESPONSES_TOOLS_TOO_LARGE"}}),
                    400,
                )
            had_responses_tools = True
            tools_responses = (tools_responses or []) + extra_tools

    responses_tool_choice = payload.get("responses_tool_choice")
    if isinstance(responses_tool_choice, str) and responses_tool_choice in ("auto", "none"):
        tool_choice = responses_tool_choice

    include_values = payload.get("include") if isinstance(payload.get("include"), list) else []
    store_requested = payload.get("store")
    store_flag = bool(store_requested) if isinstance(store_requested, bool) else False

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None
    extra_payload: Dict[str, Any] = {}
    for key in (
        # "max_output_tokens",
        "max_tool_calls",
        # "temperature",
        "top_p",
        "top_logprobs",
        # "service_tier",
        "safety_identifier",
        "prompt_cache_key",
        "previous_response_id",
    ):
        if key in payload:
            extra_payload[key] = payload.get(key)
    if metadata is not None:
        extra_payload["metadata"] = metadata
    if isinstance(stream_options, dict) and stream_options:
        extra_payload["stream_options"] = stream_options

    extra_payload_ignore: Dict[str, Any] = {}
    for key in (
        "max_output_tokens",
        "service_tier",
        "temperature"
    ):
        if key in payload:
            extra_payload_ignore[key] = payload.get(key)

    model_reasoning = extract_reasoning_from_model_name(requested_model)
    reasoning_overrides = payload.get("reasoning") if isinstance(payload.get("reasoning"), dict) else model_reasoning
    
    input_items, reasoning_overrides_query = extract_reasoning_from_last_input(input_items)
    input_items = clean_reasoning_tag_in_query(input_items)

    if reasoning_overrides_query:
        reasoning_overrides = reasoning_overrides_query
        
    reasoning_param = build_reasoning_param(reasoning_effort, reasoning_summary, reasoning_overrides)
    
    if reasoning_param.get('effort') == "minimal":
        tools_responses = [
            t for t in tools_responses
            if t.get("type") != "web_search"
        ]
    if isinstance(reasoning_overrides, dict):
        for k, v in reasoning_overrides.items():
            if k not in reasoning_param and v is not None:
                reasoning_param[k] = v

    if extra_payload_ignore:
        print(f'This is not official response endpoint, these parameter will ignore:\n{extra_payload_ignore}\n')

    upstream, error_resp = start_upstream_request(
        model,
        input_items,
        instructions=instructions,
        tools=tools_responses,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        reasoning_param=reasoning_param,
        include=include_values,
        store=store_flag,
        extra_payload=extra_payload,
    )
    if error_resp is not None:
        return error_resp

    record_rate_limits_from_response(upstream)
    # Check status
    if upstream.status_code != 200:
        # You can get the plain text or JSON error message
        try:
            error_text = upstream.text  # Raw text body
            # or try JSON decoding if the server returns JSON
            error_json = upstream.json()  
        except ValueError:
            error_json = None

        print("Status code:", upstream.status_code)
        print("Response JSON:", error_json)
        print("Response JSON:", error_json)
        return (
            jsonify(
                {
                    "error": {
                        "message": f"Upstream error: {error_json or error_text}",
                        "code": "RESPONSES_TOOLS_REJECTED",
                    }
                }
            ),
            (upstream.status_code if upstream is not None else 500),
        )

    created = int(time.time())

    if stream_req:
        line_iter = upstream.iter_lines(chunk_size=2048, decode_unicode=False)
        model_for_fix = (requested_model or model or "") or ""
        needs_reasoning_spacing_fix = "codex" not in model_for_fix.lower()
        saw_reasoning_summary_part = False
        pending_reasoning_newline = False

        def _maybe_fix_reasoning_summary(raw_line: Any) -> Any:
            nonlocal saw_reasoning_summary_part, pending_reasoning_newline
            if raw_line is None or not needs_reasoning_spacing_fix:
                return raw_line
            if isinstance(raw_line, (bytes, bytearray)):
                decoded = raw_line.decode("utf-8", errors="ignore")
                as_bytes = True
            else:
                decoded = str(raw_line)
                as_bytes = False
            if not decoded.startswith("data: "):
                return raw_line
            payload = decoded[len("data: ") :].strip()
            if not payload or payload == "[DONE]":
                return raw_line
            try:
                evt = json.loads(payload)
            except Exception:
                return raw_line
            kind = evt.get("type")
            if kind == "response.reasoning_summary_part.added":
                if saw_reasoning_summary_part:
                    pending_reasoning_newline = True
                else:
                    saw_reasoning_summary_part = True
                return raw_line
            if kind == "response.reasoning_summary_text.delta" and pending_reasoning_newline:
                delta_text = evt.get("delta")
                pending_reasoning_newline = False
                if isinstance(delta_text, str) and delta_text and not delta_text.startswith("\n"):
                    evt["delta"] = "\n\n" + delta_text
                    new_payload = json.dumps(evt, separators=(",", ":"))
                    new_line = f"data: {new_payload}"
                    if as_bytes:
                        return new_line.encode("utf-8")
                    return new_line
                if isinstance(delta_text, str) and delta_text.startswith("\n") and not delta_text.startswith("\n\n"):
                    evt["delta"] = "\n" + delta_text
                    new_payload = json.dumps(evt, separators=(",", ":"))
                    new_line = f"data: {new_payload}"
                    if as_bytes:
                        return new_line.encode("utf-8")
                    return new_line
            return raw_line

        try:
            first_line = next(line_iter)
        except StopIteration:
            first_line = None
        else:
            first_line = _maybe_fix_reasoning_summary(first_line)

        def _relay():
            try:
                if first_line is not None:
                    # iter_lines 移除換行；SSE 需還原
                    line_bytes = first_line if isinstance(first_line, (bytes, bytearray)) else str(first_line).encode("utf-8")
                    yield line_bytes + b"\n"
                for line in line_iter:
                    # 注意：空行需保留，代表事件分隔
                    fixed_line = _maybe_fix_reasoning_summary(line)
                    line_bytes = fixed_line if isinstance(fixed_line, (bytes, bytearray)) else str(fixed_line).encode("utf-8")
                    yield line_bytes + b"\n"
            except (GeneratorExit, BrokenPipeError):
                # 客戶端斷線
                pass
            except requests.exceptions.ChunkedEncodingError as exc:
                if verbose:
                    print(f"Streaming upstream ended early: {exc}")
            except requests.exceptions.RequestException as exc:
                if verbose:
                    print(f"Streaming upstream error: {exc}")
            finally:
                upstream.close()

        headers = {
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Content-Type": upstream.headers.get(
                "Content-Type",
                "text/event-stream; charset=utf-8"
            ),
            "X-Accel-Buffering": "no",
        }
        resp = Response(
            stream_with_context(_relay()),
            status=upstream.status_code,
            headers=headers,
            direct_passthrough=True,
        )

        # Preserve your CORS handling
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp

    response_id = "resp"
    latest_response: Dict[str, Any] | None = None
    usage_obj: Dict[str, Any] | None = None
    status_value: str | None = None
    collected_text: Dict[int, str] = {}
    error_message: str | None = None
    error_payload: Dict[str, Any] | None = None

    def _extract_usage(evt: Dict[str, Any]) -> Dict[str, Any] | None:
        try:
            usage = (evt.get("response") or {}).get("usage")
            if not isinstance(usage, dict):
                return None
            return usage
        except Exception:
            return None

    try:
        for raw_line in upstream.iter_lines(chunk_size=1, decode_unicode=False):
            if not raw_line:
                continue
            decoded = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, (bytes, bytearray)) else raw_line
            if not decoded.startswith("data: "):
                continue
            data = decoded[len("data: "):].strip()
            if not data:
                continue
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
            except Exception:
                continue
            if isinstance(evt.get("response"), dict):
                latest_response = evt.get("response")
                if isinstance(latest_response.get("usage"), dict):
                    usage_obj = latest_response.get("usage")
                if isinstance(latest_response.get("id"), str):
                    response_id = latest_response.get("id") or response_id
                if isinstance(latest_response.get("status"), str):
                    status_value = latest_response.get("status")
            if isinstance(evt.get("response"), dict) and isinstance(evt["response"].get("created"), int):
                created = evt["response"].get("created") or created
            kind = evt.get("type")
            if kind == "response.output_text.delta":
                idx = evt.get("output_index")
                delta_txt = evt.get("delta") or ""
                if isinstance(idx, int) and delta_txt:
                    collected_text[idx] = collected_text.get(idx, "") + delta_txt
            elif kind == "response.error":
                error_payload = evt.get("error") if isinstance(evt.get("error"), dict) else None
                error_message = ((error_payload or {}).get("message") or "response.error") if (error_payload or {}) else "response.error"
                break
            elif kind == "response.failed":
                failure = evt.get("response", {}).get("error", {}) if isinstance(evt.get("response"), dict) else {}
                if isinstance(failure, dict):
                    error_payload = failure
                    error_message = failure.get("message", "response.failed")
                else:
                    error_message = "response.failed"
                break
            usage_candidate = _extract_usage(evt)
            if usage_candidate:
                usage_obj = usage_candidate
            if kind == "response.completed":
                break
    except requests.exceptions.ChunkedEncodingError as exc:
        if verbose:
            print(f"Non-stream upstream ended early: {exc}")
        if error_message is None:
            error_message = "Upstream response ended prematurely"
        if error_payload is None:
            error_payload = {"code": "UPSTREAM_STREAM_ERROR"}
    except requests.exceptions.RequestException as exc:
        if verbose:
            print(f"Non-stream upstream error: {exc}")
        if error_message is None:
            error_message = "Upstream request error"
        if error_payload is None:
            error_payload = {"code": "UPSTREAM_REQUEST_ERROR"}
    finally:
        upstream.close()

    if error_message:
        err_body = {"message": error_message}
        if isinstance(error_payload, dict):
            err_body.update(error_payload)
        resp = make_response(jsonify({"error": err_body}), 502)
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp

    if latest_response is not None:
        result = dict(latest_response)
        result.setdefault("id", response_id)
        result.setdefault("object", "response")
        result.setdefault("created", created)
        result["model"] = requested_model or result.get("model") or model
        if metadata is not None:
            result.setdefault("metadata", metadata)
        if usage_obj and not isinstance(result.get("usage"), dict):
            result["usage"] = usage_obj
        if not result.get("status") and status_value:
            result["status"] = status_value
        resp = make_response(jsonify(result), upstream.status_code)
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp

    fallback_output: List[Dict[str, Any]] = []
    if collected_text:
        for idx in sorted(collected_text.keys()):
            text_value = collected_text.get(idx) or ""
            if not text_value:
                continue
            fallback_output.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "id": f"{response_id}-output-{idx}",
                    "content": [{"type": "output_text", "text": text_value}],
                }
            )

    fallback_response = {
        "id": response_id,
        "object": "response",
        "created": created,
        "model": requested_model or model,
        "output": fallback_output,
        "status": status_value or "completed",
    }
    if metadata is not None:
        fallback_response["metadata"] = metadata
    if usage_obj:
        fallback_response["usage"] = usage_obj
    resp = make_response(jsonify(fallback_response), upstream.status_code)
    for k, v in build_cors_headers().items():
        resp.headers.setdefault(k, v)
    return resp


@openai_bp.route("/v1/embeddings", methods=["POST"])
def embeddings() -> Response:
    verbose = bool(current_app.config.get("VERBOSE"))
    if verbose:
        try:
            preview = (request.get_data(cache=True, as_text=True) or "")[:2000]
            print("IN POST /v1/embeddings\n" + preview)
        except Exception:
            pass

    api_key = current_app.config.get("EXPECTED_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        env_key = os.getenv("OPENAI_API_KEY")
        api_key = env_key.strip() if isinstance(env_key, str) else ""
    
    api_key = api_key.strip(API_KEY_CUSTOM_SUFFIX)
    
    if not api_key:
        resp = make_response(jsonify({"error": {"message": "Server missing OpenAI API key"}}), 500)
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp

    raw_body = request.get_data(cache=True) or b""

    upstream_headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": request.headers.get("Content-Type", "application/json"),
    }

    user_agent = request.headers.get("User-Agent")
    if isinstance(user_agent, str) and user_agent.strip():
        upstream_headers["User-Agent"] = user_agent.strip()
    else:
        upstream_headers["User-Agent"] = "ChatMockProxy/1.0"

    for header_name, header_value in request.headers.items():
        if not isinstance(header_value, str):
            continue
        lower = header_name.lower()
        if lower.startswith("openai-") or lower.startswith("x-openai-"):
            upstream_headers[header_name] = header_value

    try:
        upstream = requests.post(
            EMBEDDINGS_ENDPOINT,
            headers=upstream_headers,
            data=raw_body,
            timeout=120,
        )
    except requests.RequestException as exc:
        if verbose:
            print(f"Embeddings upstream error: {exc}")
        resp = make_response(
            jsonify({"error": {"message": f"Error contacting OpenAI embeddings: {exc}"}}),
            502,
        )
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp
    except Exception as exc:  # pragma: no cover - safety net
        if verbose:
            print(f"Unexpected embeddings error: {exc}")
        resp = make_response(
            jsonify({"error": {"message": "Unexpected error contacting OpenAI embeddings"}}),
            500,
        )
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp

    resp = make_response(upstream.content, upstream.status_code)
    content_type = upstream.headers.get("Content-Type")
    resp.headers["Content-Type"] = content_type or "application/json"

    for header_name, header_value in upstream.headers.items():
        lower = header_name.lower()
        if lower in ("content-type", "content-length"):
            continue
        if lower.startswith("openai-") or lower.startswith("x-request-id") or lower.startswith("x-ratelimit-") or lower.startswith("cf-"):
            resp.headers[header_name] = header_value

    for k, v in build_cors_headers().items():
        resp.headers.setdefault(k, v)

    upstream.close()
    return resp


@openai_bp.route("/v1/models", methods=["GET"])
def list_models() -> Response:
    expose_variants = bool(current_app.config.get("EXPOSE_REASONING_MODELS"))
    model_groups = [
        ("gpt-5", ["high", "medium", "low", "minimal"]),
        ("gpt-5-codex", ["high", "medium", "low"]),
        ("codex-mini", []),
        ("gpt-5-mini", [])
    ]
    model_ids: List[str] = []
    for base, efforts in model_groups:
        model_ids.append(base)
        if expose_variants:
            model_ids.extend([f"{base}-{effort}" for effort in efforts])
    data = [{"id": mid, "object": "model", "owned_by": "owner"} for mid in model_ids]
    models = {"object": "list", "data": data}
    resp = make_response(jsonify(models), 200)
    for k, v in build_cors_headers().items():
        resp.headers.setdefault(k, v)
    return resp
