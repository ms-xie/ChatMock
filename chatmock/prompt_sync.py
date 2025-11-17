from __future__ import annotations

from datetime import date, datetime, timezone
import sys
from pathlib import Path

import requests

from .utils import eprint, get_accounts_base_dir

GPT5_GITHUB_PROMPT_URL_BASE = "https://raw.githubusercontent.com/openai/codex/refs/heads/main/codex-rs/core/"

ROOT = Path(__file__).parent.parent.resolve()

prompt_list = [
    "gpt_5_1_prompt.md",
    "gpt_5_codex_prompt.md",
    "prompt.md"
]

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

def sync_prompt_from_official_github(
    timeout: float = 15.0,
    verbose: bool = False,
) -> bool:
    """
    Download the canonical GPT-5 prompt and write it to ROOT.
    Returns True on success, False otherwise.
    """
    for prompt_file_name in prompt_list:
        resolved_url = GPT5_GITHUB_PROMPT_URL_BASE + prompt_file_name
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
    
    return True


def sync_prompt_from_official_github_if_due(
    timeout: float = 15.0,
    verbose: bool = False,
    force: bool = False,
) -> bool:
    """
    Sync the official GPT-5 prompts once per UTC day unless `force` is True.
    """
    if not force and _already_synced_today():
        if verbose:
            eprint("Prompt already synced for today; skipping download.")
        return True

    success = sync_prompt_from_official_github(timeout=timeout, verbose=True)
    if success:
        _write_last_sync_date(_today_utc())
    return success
