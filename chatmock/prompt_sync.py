from __future__ import annotations

from collections.abc import MutableMapping
from datetime import date, datetime, timezone
from importlib import reload as _reload_module
import sys
from pathlib import Path

import requests
import urllib.request

from . import config as _config_module
from .utils import eprint, get_accounts_base_dir

GPT5_GITHUB_PROMPT_URL_BASE = "https://raw.githubusercontent.com/openai/codex/refs/tags/{tag}/codex-rs/core/"


ROOT = Path(__file__).parent.parent.resolve()

prompt_list = [
    "gpt_5_1_prompt.md",
    "gpt_5_codex_prompt.md",
    "prompt.md",
    "gpt-5.1-codex-max_prompt.md"
]

def get_latest_tag() -> str:
    # 這個 URL 會 302 轉跳到最新的 release 頁面
    releases_latest = f"https://github.com/openai/codex/releases/latest"
    # urllib 會自動跟隨 redirect，最後的 URL 就是實際 release 頁面
    with urllib.request.urlopen(releases_latest) as resp:
        final_url = resp.geturl()
    # 例如：https://github.com/openai/codex/releases/tag/rust-v0.58.0
    # 取最後一段當成 tag 名稱
    tag = final_url.rsplit("/", 1)[-1]
    return tag

_PROMPT_SYNC_STATE_FILENAME = "prompt_sync_last_sync"


def _state_file_path() -> Path:
    base = Path(get_accounts_base_dir(ensure_exists=True))
    return base / _PROMPT_SYNC_STATE_FILENAME


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _load_last_sync_date() -> date | None:
    path = _state_file_path()
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        eprint(f"Unable to read prompt sync state: {exc}")
        return None
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _write_last_sync_date(sync_date: date) -> None:
    path = _state_file_path()
    try:
        path.write_text(sync_date.isoformat(), encoding="utf-8")
    except OSError as exc:
        eprint(f"Unable to write prompt sync state: {exc}")


def _already_synced_today() -> bool:
    last_sync = _load_last_sync_date()
    if last_sync is None:
        return False
    return last_sync >= _today_utc()


def reload_prompt_instructions(
    target: MutableMapping[str, object] | None = None,
    verbose: bool = False,
) -> bool:
    """
    Reload the prompt constants from chatmock.config and apply them to `target`
    (typically `app.config`) so live servers can pick up the new text.
    """
    try:
        _reload_module(_config_module)
    except Exception as exc:
        eprint(f"Unable to reload prompt configuration: {exc}")
        return False

    if target is not None:
        target.update(
            BASE_INSTRUCTIONS=_config_module.BASE_INSTRUCTIONS,
            GPT5_CODEX_INSTRUCTIONS=_config_module.GPT5_CODEX_INSTRUCTIONS,
            GPT5_1_INSTRUCTIONS=_config_module.GPT5_1_INSTRUCTIONS,
            GPT5_1_CODEX_MAX_INSTRUCTIONS=_config_module.GPT5_1_CODEX_MAX_INSTRUCTIONS
        )

    if verbose:
        eprint("Prompt instructions reloaded from disk.")

    return True


def sync_prompt_from_official_github(
    timeout: float = 15.0,
    verbose: bool = False,
    target_config: MutableMapping[str, object] | None = None,
) -> bool:
    """
    Download the canonical GPT-5 prompt, refresh chatmock.config, and write it to ROOT.
    Returns True on success, False otherwise.
    """
    latest_tag = get_latest_tag()
    print('latest_tag: ', latest_tag)
    for prompt_file_name in prompt_list:
        resolved_url = GPT5_GITHUB_PROMPT_URL_BASE.format(tag=latest_tag) + prompt_file_name
        path = ROOT / prompt_file_name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"Unable to create destination directory {path.parent}: {exc}", file=sys.stderr)
            return False

        try:
            response = requests.get(resolved_url, timeout=timeout)
            response.raise_for_status()
            content = response.text or ""
        except requests.RequestException as exc:
            print(f"Unable to download GPT-5 prompt from {resolved_url}: {exc}", file=sys.stderr)
            return False

        if not content.strip():
            print(f"Downloaded content from {resolved_url} was empty; skipping write.", file=sys.stderr)
            return False

        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            print(f"Unable to write GPT-5 prompt to {path}: {exc}", file=sys.stderr)
            return False

        if verbose:
            print(f"Synced GPT-5 prompt from {resolved_url} to {path}", file=sys.stderr)
    
    reload_prompt_instructions(target=target_config, verbose=verbose)
    return True


def sync_prompt_from_official_github_if_due(
    timeout: float = 15.0,
    verbose: bool = False,
    force: bool = False,
    target_config: MutableMapping[str, object] | None = None,
) -> bool:
    """
    Sync the official GPT-5 prompts once per UTC day unless `force` is True.
    """
    if not force and _already_synced_today():
        if verbose:
            eprint("Prompt already synced for today; skipping download.")
        return True

    success = sync_prompt_from_official_github(
        timeout=timeout,
        verbose=True,
        target_config=target_config,
    )
    if success:
        _write_last_sync_date(_today_utc())
    return success
