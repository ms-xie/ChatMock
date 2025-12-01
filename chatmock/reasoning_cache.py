from __future__ import annotations

import threading
import copy
from collections import OrderedDict
from typing import Any, Dict, Optional


# Simple in-memory LRU cache for reasoning items.
# Stored per-process; enough for dev/proxy use.
_MAX_ITEMS = 256
_lock = threading.Lock()
_store: "OrderedDict[str, Dict[str, Optional[Any]]]" = OrderedDict()


def _trim() -> None:
    while len(_store) > _MAX_ITEMS:
        _store.popitem(last=False)


def set_encrypted(item_id: str, encrypted_content: Optional[str], priority: int) -> None:
    """Upsert encrypted_content with priority (higher wins)."""
    if not item_id or encrypted_content is None:
        return
    with _lock:
        entry = _store.get(item_id, {"encrypted_content": None, "summary_text": None, "item": None, "priority": -1})
        if priority >= int(entry.get("priority", -1)):
            entry["encrypted_content"] = encrypted_content
            entry["priority"] = priority
        _store[item_id] = entry
        _store.move_to_end(item_id)
        _trim()


def append_summary_delta(item_id: str, delta: str) -> None:
    if not item_id or not isinstance(delta, str):
        return
    with _lock:
        entry = _store.get(item_id, {"encrypted_content": None, "summary_text": "", "item": None, "priority": -1})
        entry["summary_text"] = (entry.get("summary_text") or "") + delta
        _store[item_id] = entry
        _store.move_to_end(item_id)
        _trim()


def set_summary(item_id: str, text: Optional[str]) -> None:
    if not item_id or text is None:
        return
    with _lock:
        entry = _store.get(item_id, {"encrypted_content": None, "summary_text": "", "item": None, "priority": -1})
        entry["summary_text"] = text
        _store[item_id] = entry
        _store.move_to_end(item_id)
        _trim()


def set_item(item_id: str, item: Optional[Dict[str, Any]], priority: int) -> None:
    """Cache a full output item (message/function_call/reasoning) with priority."""
    if not item_id or not isinstance(item, dict):
        return
    with _lock:
        entry = _store.get(item_id, {"encrypted_content": None, "summary_text": None, "item": None, "priority": -1})
        if priority >= int(entry.get("priority", -1)):
            entry["item"] = copy.deepcopy(item)
            entry["priority"] = priority
        _store[item_id] = entry
        _store.move_to_end(item_id)
        _trim()


def get(item_id: str) -> Optional[Dict[str, Any]]:
    if not item_id:
        return None
    with _lock:
        entry = _store.get(item_id)
        if entry is None:
            return None
        # touch for LRU
        _store.move_to_end(item_id)
        return copy.deepcopy(dict(entry))
