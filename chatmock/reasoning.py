from __future__ import annotations

from typing import Any, Dict, List
import re

def build_reasoning_param(
    base_effort: str = "medium", base_summary: str = "auto", overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    effort = (base_effort or "").strip().lower()
    summary = (base_summary or "").strip().lower()

    valid_efforts = {"minimal", "low", "medium", "high"}
    valid_summaries = {"auto", "concise", "detailed", "none"}

    if isinstance(overrides, dict):
        o_eff = str(overrides.get("effort", "")).strip().lower()
        o_sum = str(overrides.get("summary", "")).strip().lower()
        if o_eff in valid_efforts and o_eff:
            effort = o_eff
        if o_sum in valid_summaries and o_sum:
            summary = o_sum
    if effort not in valid_efforts:
        effort = "medium"
    if summary not in valid_summaries:
        summary = "auto"

    reasoning: Dict[str, Any] = {"effort": effort}
    
    # if minimal effort, do not summary
    if summary != "none" and effort != "minimal":
        reasoning["summary"] = summary
    return reasoning


def apply_reasoning_to_message(
    message: Dict[str, Any],
    reasoning_summary_text: str,
    reasoning_full_text: str,
    compat: str,
) -> Dict[str, Any]:
    try:
        compat = (compat or "think-tags").strip().lower()
    except Exception:
        compat = "think-tags"

    if compat == "o3":
        rtxt_parts: list[str] = []
        if isinstance(reasoning_summary_text, str) and reasoning_summary_text.strip():
            rtxt_parts.append(reasoning_summary_text)
        if isinstance(reasoning_full_text, str) and reasoning_full_text.strip():
            rtxt_parts.append(reasoning_full_text)
        rtxt = "\n\n".join([p for p in rtxt_parts if p])
        if rtxt:
            message["reasoning"] = {"content": [{"type": "text", "text": rtxt}]}
        return message

    if compat in ("legacy", "current"):
        if reasoning_summary_text:
            message["reasoning_summary"] = reasoning_summary_text
        if reasoning_full_text:
            message["reasoning"] = reasoning_full_text
        return message

    rtxt_parts: list[str] = []
    if isinstance(reasoning_summary_text, str) and reasoning_summary_text.strip():
        rtxt_parts.append(reasoning_summary_text)
    if isinstance(reasoning_full_text, str) and reasoning_full_text.strip():
        rtxt_parts.append(reasoning_full_text)
    rtxt = "\n\n".join([p for p in rtxt_parts if p])
    if rtxt:
        think_block = f"<think>{rtxt}</think>"
        content_text = message.get("content") or ""
        if isinstance(content_text, str):
            message["content"] = think_block + (content_text or "")
    return message


def extract_reasoning_from_model_name(model: str | None) -> Dict[str, Any] | None:
    """Infer reasoning overrides from a model."""
    if not isinstance(model, str) or not model:
        return None
    s = model.strip().lower()
    if not s:
        return None
    efforts = {"minimal", "low", "medium", "high", "mini"}

    if ":" in s:
        maybe = s.rsplit(":", 1)[-1].strip()
        if maybe in efforts:
            return {"effort": maybe}

    for sep in ("-", "_"):
        if s.endswith(sep + "minimal"):
            return {"effort": "minimal"}
        if s.endswith(sep + "mini"):
            return {"effort": "minimal"}
        if s.endswith(sep + "low"):
            return {"effort": "low"}
        if s.endswith(sep + "medium"):
            return {"effort": "medium"}
        if s.endswith(sep + "high"):
            return {"effort": "high"}

    return None

def extract_last_query(input_items):
    user_input = None
    for i in range(len(input_items)-1, -1, -1):
        item = input_items[i]
        if item['role'] == 'user':
            for j in range(len(item["content"])-1, -1, -1):
                if item["content"][j]["type"] == "input_text":
                    user_input = item["content"][j]['text']
                    return user_input
    
    return None

def clean_reasoning_tag_in_query(input_items):
    for item in input_items:
        if item['role'] == 'user':
            for content in item["content"]:
                if content["type"] == "input_text":
                    content['text'] = re.sub(r"#([LMH])\b", "", content['text'], flags=re.IGNORECASE).strip()

    return input_items

def extract_reasoning_from_last_input(input_items):
    if not isinstance(input_items, List) or not input_items:
        return input_items, None
    
    user_input = extract_last_query(input_items)
    
    if not user_input:
        return input_items, None
    
    match = re.search(r"#([LMH])\b", user_input, re.IGNORECASE)
    
    if match:
        reasoning_effort = {
            "L": "low",
            "M": "medium",
            "H": "high",
        }[match.group(1).upper()]

        return input_items, {"effort": reasoning_effort}
    
    return input_items, None