from __future__ import annotations

import errno
import argparse
import json
import os
import sys
import time
import webbrowser
from datetime import datetime

from .app import create_app
from .config import CLIENT_ID_DEFAULT
from .limits import RateLimitWindow, StoredRateLimitSnapshot, compute_reset_at, load_rate_limit_snapshot
from .oauth import OAuthHTTPServer, OAuthHandler, REQUIRED_PORT, URL_BASE
from .utils import (
    eprint,
    get_accounts_base_dir,
    get_active_account_slug,
    load_chatgpt_tokens,
    parse_jwt_claims,
    read_auth_file,
    list_known_accounts,
    set_active_account_slug,
    ensure_unique_account_slug,
    remove_account,
    update_account_metadata,
    write_auth_file,
)

from typing import Any, Dict, List, Optional

_STATUS_LIMIT_BAR_SEGMENTS = 30
_STATUS_LIMIT_BAR_FILLED = "█"
_STATUS_LIMIT_BAR_EMPTY = "░"
_STATUS_LIMIT_BAR_PARTIAL = "▓"

_PLAN_NAME_MAP = {
    "plus": "Plus",
    "pro": "Pro",
    "free": "Free",
    "team": "Team",
    "enterprise": "Enterprise",
}


def _clamp_percent(value: float) -> float:
    try:
        percent = float(value)
    except Exception:
        return 0.0
    if percent != percent:
        return 0.0
    if percent < 0.0:
        return 0.0
    if percent > 100.0:
        return 100.0
    return percent


def _render_progress_bar(percent_used: float) -> str:
    ratio = max(0.0, min(1.0, percent_used / 100.0))
    filled_exact = ratio * _STATUS_LIMIT_BAR_SEGMENTS
    filled = int(filled_exact)
    partial = filled_exact - filled
    
    has_partial = partial > 0.5
    if has_partial:
        filled += 1
    
    filled = max(0, min(_STATUS_LIMIT_BAR_SEGMENTS, filled))
    empty = _STATUS_LIMIT_BAR_SEGMENTS - filled
    
    if has_partial and filled > 0:
        bar = _STATUS_LIMIT_BAR_FILLED * (filled - 1) + _STATUS_LIMIT_BAR_PARTIAL + _STATUS_LIMIT_BAR_EMPTY * empty
    else:
        bar = _STATUS_LIMIT_BAR_FILLED * filled + _STATUS_LIMIT_BAR_EMPTY * empty
    
    return f"[{bar}]"


def _get_usage_color(percent_used: float) -> str:
    if percent_used >= 90:
        return "\033[91m" 
    elif percent_used >= 75:
        return "\033[93m"  
    elif percent_used >= 50:
        return "\033[94m"  
    else:
        return "\033[92m" 


def _reset_color() -> str:
    """ANSI reset color code"""
    return "\033[0m"


def _format_window_duration(minutes: int | None) -> str | None:
    if minutes is None:
        return None
    try:
        total = int(minutes)
    except Exception:
        return None
    if total <= 0:
        return None
    minutes = total
    weeks, remainder = divmod(minutes, 7 * 24 * 60)
    days, remainder = divmod(remainder, 24 * 60)
    hours, remainder = divmod(remainder, 60)
    parts = []
    if weeks:
        parts.append(f"{weeks} week" + ("s" if weeks != 1 else ""))
    if days:
        parts.append(f"{days} day" + ("s" if days != 1 else ""))
    if hours:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if remainder:
        parts.append(f"{remainder} minute" + ("s" if remainder != 1 else ""))
    if not parts:
        parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    return " ".join(parts)


def _format_reset_duration(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    try:
        value = int(seconds)
    except Exception:
        return None
    if value < 0:
        value = 0
    days, remainder = divmod(value, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, remainder = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts and remainder:
        parts.append("under 1m")
    if not parts:
        parts.append("0m")
    return " ".join(parts)


def _format_local_datetime(dt: datetime) -> str:
    local = dt.astimezone()
    tz_name = local.tzname() or "local"
    return f"{local.strftime('%b %d, %Y %H:%M')} {tz_name}"


def _plan_label(raw: Optional[str]) -> str:
    if not isinstance(raw, str):
        return "Unknown"
    value = raw.strip()
    if not value:
        return "Unknown"
    return _PLAN_NAME_MAP.get(value.lower(), value.title())


def _profile_from_tokens(access_token: Optional[str], id_token: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    email: Optional[str] = None
    plan_raw: Optional[str] = None

    id_claims = parse_jwt_claims(id_token) or {}
    if isinstance(id_claims, dict):
        for key in ("email", "preferred_username", "name", "sub"):
            candidate = id_claims.get(key)
            if isinstance(candidate, str) and candidate:
                email = candidate
                break

    access_claims = parse_jwt_claims(access_token) or {}
    if isinstance(access_claims, dict):
        auth_claims = access_claims.get("https://api.openai.com/auth")
        if isinstance(auth_claims, dict):
            raw_plan = auth_claims.get("chatgpt_plan_type")
            if isinstance(raw_plan, str):
                plan_raw = raw_plan.strip().lower() or None
            elif raw_plan is not None:
                plan_raw = str(raw_plan).strip().lower() or None

    return email, plan_raw


def _format_account_usage(snapshot: Optional[StoredRateLimitSnapshot]) -> tuple[List[str], bool]:
    if snapshot is None:
        return ["    Usage: no data recorded yet."], False

    lines: List[str] = []
    limit_hit = False
    lines.append(f"    Last updated: {_format_local_datetime(snapshot.captured_at)}")

    windows: List[tuple[str, str, RateLimitWindow]] = []
    if snapshot.snapshot.primary is not None:
        windows.append(("⚡", "5 hour limit", snapshot.snapshot.primary))
    if snapshot.snapshot.secondary is not None:
        windows.append(("📅", "Weekly limit", snapshot.snapshot.secondary))

    if not windows:
        lines.append("    Usage data available but no limit windows provided.")
        return lines, False

    for icon_label, desc, window in windows:
        percent_used = _clamp_percent(window.used_percent)
        limit_hit = limit_hit or percent_used >= 100.0
        remaining = max(0.0, 100.0 - percent_used)
        color = _get_usage_color(percent_used)
        reset_color = _reset_color()
        progress = _render_progress_bar(percent_used)
        lines.append(
            f"    {icon_label}  {desc}: {color}{progress}{reset_color} {color}{percent_used:5.1f}% used{reset_color} | {remaining:5.1f}% left"
        )
        reset_in = _format_reset_duration(window.resets_in_seconds)
        reset_at = compute_reset_at(snapshot.captured_at, window)
        if reset_in and reset_at:
            lines.append(f"       ⏳ Resets in {reset_in} at {_format_local_datetime(reset_at)}")
        elif reset_in:
            lines.append(f"       ⏳ Resets in {reset_in}")
        elif reset_at:
            lines.append(f"       ⏳ Resets at {_format_local_datetime(reset_at)}")

    return lines, limit_hit


def _window_to_dict(window: Optional[RateLimitWindow]) -> Optional[Dict[str, Any]]:
    if window is None:
        return None
    return {
        "used_percent": window.used_percent,
        "window_minutes": window.window_minutes,
        "resets_in_seconds": window.resets_in_seconds,
    }


def _collect_accounts_state() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in list_known_accounts():
        slug = entry.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        snapshot = load_rate_limit_snapshot(account_slug=slug)
        usage_lines, limit_hit = _format_account_usage(snapshot)
        usage_data: Dict[str, Any] = {}
        if snapshot is not None:
            usage_data = {
                "captured_at": snapshot.captured_at.isoformat(),
                "primary": _window_to_dict(snapshot.snapshot.primary),
                "secondary": _window_to_dict(snapshot.snapshot.secondary),
            }
        rows.append(
            {
                "slug": slug,
                "label": entry.get("label") or entry.get("email") or slug,
                "email": entry.get("email"),
                "plan": entry.get("plan"),
                "account_id": entry.get("account_id"),
                "last_used": entry.get("last_used"),
                "usage_lines": usage_lines,
                "limit_hit": limit_hit,
                "usage": usage_data,
            }
        )
    return rows


def _usage_signature(rows: List[Dict[str, Any]]) -> tuple:
    signature: List[tuple[Any, ...]] = []
    for row in rows:
        usage = row.get("usage") or {}
        primary = usage.get("primary") or {}
        secondary = usage.get("secondary") or {}
        signature.append(
            (
                row.get("slug"),
                usage.get("captured_at"),
                primary.get("used_percent"),
                primary.get("resets_in_seconds"),
                secondary.get("used_percent"),
                secondary.get("resets_in_seconds"),
            )
        )
    return tuple(signature)


def _print_account_dashboard(rows: List[Dict[str, Any]], active_slug: Optional[str]) -> None:
    print("")
    print("📂 ChatMock Account Manager")
    if rows:
        print("Accounts marked with !LIMIT will rotate automatically when limits reset.")
    else:
        print("Link a ChatGPT account to begin.")
    print("")

    if not rows:
        print("Actions: n=new  q=quit")
        print("")
        return

    for idx, row in enumerate(rows, start=1):
        marker = ">>" if row["slug"] == active_slug else "  "
        limit_flag = " !LIMIT" if row["limit_hit"] else ""
        plan_display = _plan_label(row.get("plan"))
        label = row.get("label") or row["slug"]
        print(f"{marker} [{idx}] {label} ({plan_display}){limit_flag}")
        if row.get("email"):
            print(f"    Email: {row['email']}")
        if row.get("account_id"):
            print(f"    Account ID: {row['account_id']}")
        if row.get("last_used"):
            print(f"    Last used: {row['last_used']}")
        for line in row["usage_lines"]:
            print(line)
        print("")

    print("'>>' marks the active account. !LIMIT means the account is exhausted until reset.")
    print("Actions: [number]=activate  n=new  d=delete  r=rename  q=quit")
    


def _prompt_account_index(rows: List[Dict[str, Any]], prompt_text: str) -> Optional[int]:
    if not rows:
        eprint("No accounts available.")
        return None
    try:
        choice = input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        print("")
        return None
    if not choice:
        return None
    if not choice.isdigit():
        eprint("Please enter a numeric selection.")
        return None
    idx = int(choice)
    if idx < 1 or idx > len(rows):
        eprint("Selection out of range.")
        return None
    return idx - 1


def _handle_remove_account(rows: List[Dict[str, Any]]) -> bool:
    idx = _prompt_account_index(rows, "Enter account number to delete: ")
    if idx is None:
        return False
    row = rows[idx]
    try:
        confirm = input(f"Delete account '{row['label']}'? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("")
        return False
    if confirm not in ("y", "yes"):
        eprint("Deletion cancelled.")
        return False
    if not remove_account(row["slug"]):
        eprint("Failed to remove account.")
        return False
    eprint(f"Removed account '{row['label']}'.")
    return True


def _handle_rename_account(rows: List[Dict[str, Any]]) -> bool:
    idx = _prompt_account_index(rows, "Enter account number to rename: ")
    if idx is None:
        return False
    row = rows[idx]
    try:
        new_label = input("Enter new label: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("")
        return False
    if not new_label:
        eprint("Rename cancelled.")
        return False
    update_account_metadata(row["slug"], label=new_label)
    eprint(f"Updated label for '{row['label']}' → '{new_label}'.")
    return True


def _readline_with_timeout(timeout_seconds: float) -> Optional[str]:
    if timeout_seconds <= 0:
        line = sys.stdin.readline()
        if line == "":
            raise EOFError
        return line

    deadline = time.monotonic() + timeout_seconds
    try:
        import select  # type: ignore
    except Exception:  # pragma: no cover - select should be present on POSIX.
        select = None  # type: ignore

    if select is not None:
        remaining = timeout_seconds
        while remaining > 0:
            ready, _, _ = select.select([sys.stdin], [], [], remaining)
            if ready:
                line = sys.stdin.readline()
                if line == "":
                    raise EOFError
                return line
            remaining = deadline - time.monotonic()
        return None

    if os.name == "nt":
        import msvcrt

        buffer: List[str] = []
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    print("")
                    return "".join(buffer)
                if ch in ("\b", "\x7f"):
                    if buffer:
                        buffer.pop()
                        msvcrt.putwch("\b")
                        msvcrt.putwch(" ")
                        msvcrt.putwch("\b")
                    continue
                buffer.append(ch)
                msvcrt.putwch(ch)
            else:
                time.sleep(0.05)
        return None

    time.sleep(max(0.0, deadline - time.monotonic()))
    return None


def _handle_login_new_account(client_id: str, no_browser: bool, verbose: bool) -> bool:
    slug_hint = ensure_unique_account_slug("account")
    base_dir = get_accounts_base_dir(ensure_exists=True)

    try:
        bind_host = os.getenv("CHATGPT_LOCAL_LOGIN_BIND", "127.0.0.1")
        httpd = OAuthHTTPServer(
            (bind_host, REQUIRED_PORT),
            OAuthHandler,
            home_dir=base_dir,
            client_id=client_id,
            verbose=verbose,
            account_slug=slug_hint,
            persist_immediately=False,
        )
    except OSError as exc:
        eprint(f"ERROR: {exc}")
        if exc.errno == errno.EADDRINUSE:
            eprint("Another login session may already be running.")
        return False

    auth_url = httpd.auth_url()
    interrupted = False

    with httpd:
        eprint(f"Starting local login server on {URL_BASE}")
        if not no_browser:
            try:
                webbrowser.open(auth_url, new=1, autoraise=True)
            except Exception as browser_exc:
                eprint(f"Failed to open browser: {browser_exc}")
        eprint(f"If your browser did not open, navigate to:\n{auth_url}")

        def _stdin_paste_worker() -> None:
            try:
                eprint(
                    "If the browser can't reach this machine, paste the full redirect URL here and press Enter (or leave blank to keep waiting):"
                )
                line = sys.stdin.readline().strip()
                if not line:
                    return
                from urllib.parse import parse_qs, urlparse

                parsed = urlparse(line)
                params = parse_qs(parsed.query)
                code = (params.get("code") or [None])[0]
                state = (params.get("state") or [None])[0]
                if not code:
                    eprint("Input did not contain an auth code. Ignoring.")
                    return
                if state and state != httpd.state:
                    eprint("State mismatch. Ignoring pasted URL for safety.")
                    return
                eprint("Received redirect URL. Completing login without callback…")
                bundle, _ = httpd.exchange_code(code)
                if httpd.persist_auth(bundle):
                    httpd.shutdown()
            except Exception as exc:
                eprint(f"Failed to process pasted redirect URL: {exc}")

        try:
            import threading

            threading.Thread(target=_stdin_paste_worker, daemon=True).start()
        except Exception:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            interrupted = True
            eprint("\nKeyboard interrupt received, aborting login.")

    if interrupted:
        return False

    if httpd.exit_code != 0 or httpd.last_auth_bundle is None:
        eprint("Login was not completed.")
        return False

    bundle = httpd.last_auth_bundle
    email, plan_raw = _profile_from_tokens(bundle.token_data.access_token, bundle.token_data.id_token)
    slug_seed = email or bundle.token_data.account_id or "account"
    final_slug = ensure_unique_account_slug(slug_seed)
    auth_payload = {
        "OPENAI_API_KEY": bundle.api_key,
        "tokens": {
            "id_token": bundle.token_data.id_token,
            "access_token": bundle.token_data.access_token,
            "refresh_token": bundle.token_data.refresh_token,
            "account_id": bundle.token_data.account_id,
        },
        "last_refresh": bundle.last_refresh,
    }

    if not write_auth_file(auth_payload, account_slug=final_slug):
        eprint("ERROR: Unable to persist auth file for new account.")
        return False

    update_account_metadata(
        final_slug,
        label=email or final_slug,
        email=email,
        plan=plan_raw,
        account_id=bundle.token_data.account_id,
        last_used=bundle.last_refresh,
    )
    set_active_account_slug(final_slug)

    plan_display = _plan_label(plan_raw)
    eprint(f"Linked new account '{email or final_slug}' ({plan_display}). Now active.")
    return True


def cmd_login(no_browser: bool, verbose: bool) -> int:
    client_id = CLIENT_ID_DEFAULT
    if not client_id:
        eprint("ERROR: No OAuth client id configured. Set CHATGPT_LOCAL_CLIENT_ID.")
        return 1

    auto_refresh_seconds = 10.0
    last_signature: Optional[tuple] = None
    prompt_visible = False
    force_render = True

    while True:
        rows = _collect_accounts_state()
        active_slug = get_active_account_slug()
        signature = _usage_signature(rows)

        if force_render or signature != last_signature:
            if prompt_visible:
                print("")
                prompt_visible = False
            _print_account_dashboard(rows, active_slug)
            last_signature = signature
            force_render = False

        if not prompt_visible:
            print("Selection: ", end="", flush=True)
            prompt_visible = True

        try:
            choice_raw = _readline_with_timeout(auto_refresh_seconds)
        except EOFError:
            print("")
            return 0
        except KeyboardInterrupt:
            print("")
            return 1

        if choice_raw is None:
            continue

        prompt_visible = False
        choice = choice_raw.strip()

        if not choice:
            continue

        lowered = choice.lower()
        if lowered in {"q", "quit", "exit"}:
            return 0
        if lowered in {"n", "new"}:
            _handle_login_new_account(client_id, no_browser, verbose)
            force_render = True
            continue
        if lowered in {"d", "del", "delete"}:
            _handle_remove_account(rows)
            force_render = True
            continue
        if lowered in {"r", "rename"}:
            _handle_rename_account(rows)
            force_render = True
            continue
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(rows):
                slug = rows[idx - 1]["slug"]
                set_active_account_slug(slug)
                eprint(f"Active account set to '{rows[idx - 1]['label']}'.")
                force_render = True
            else:
                eprint("Selection out of range.")
            continue

        eprint("Unrecognized option. Enter a number, 'n', 'd', 'r', or 'q'.")
        continue


def cmd_serve(
    host: str,
    port: int,
    verbose: bool,
    reasoning_effort: str,
    reasoning_summary: str,
    reasoning_compat: str,
    debug_model: str | None,
    expose_reasoning_models: bool,
    default_web_search: bool,
) -> int:
    app = create_app(
        verbose=verbose,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        reasoning_compat=reasoning_compat,
        debug_model=debug_model,
        expose_reasoning_models=expose_reasoning_models,
        default_web_search=default_web_search,
    )

    app.run(host=host, debug=False, use_reloader=False, port=port, threaded=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatGPT Local: login & OpenAI-compatible proxy")
    sub = parser.add_subparsers(dest="command", required=True)

    p_login = sub.add_parser("login", help="Authorize with ChatGPT and store tokens")
    p_login.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    p_login.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    p_serve = sub.add_parser("serve", help="Run local OpenAI-compatible server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    p_serve.add_argument(
        "--debug-model",
        dest="debug_model",
        default=os.getenv("CHATGPT_LOCAL_DEBUG_MODEL"),
        help="Forcibly override requested 'model' with this value",
    )
    p_serve.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=os.getenv("CHATGPT_LOCAL_REASONING_EFFORT", "medium").lower(),
        help="Reasoning effort level for Responses API (default: medium)",
    )
    p_serve.add_argument(
        "--reasoning-summary",
        choices=["auto", "concise", "detailed", "none"],
        default=os.getenv("CHATGPT_LOCAL_REASONING_SUMMARY", "auto").lower(),
        help="Reasoning summary verbosity (default: auto)",
    )
    p_serve.add_argument(
        "--reasoning-compat",
        choices=["legacy", "o3", "think-tags", "current"],
        default=os.getenv("CHATGPT_LOCAL_REASONING_COMPAT", "think-tags").lower(),
        help=(
            "Compatibility mode for exposing reasoning to clients (legacy|o3|think-tags). "
            "'current' is accepted as an alias for 'legacy'"
        ),
    )
    p_serve.add_argument(
        "--expose-reasoning-models",
        action="store_true",
        default=os.getenv("CHATGPT_LOCAL_EXPOSE_REASONING_MODELS", "").strip().lower() in ("1", "true", "yes", "on"),
        help=(
            "Expose gpt-5 reasoning effort variants (minimal|low|medium|high) as separate models from /v1/models. "
            "This allows choosing effort via model selection in compatible UIs."
        ),
    )
    p_serve.add_argument(
        "--enable-web-search",
        action="store_true",
        default=os.getenv("CHATGPT_ENABLE_WEB_SEARCH", "").strip().lower() in ("1", "true", "yes", "on"),
        help="Enable default web_search tool when a request omits responses_tools (off by default)",
    )

    p_info = sub.add_parser("info", help="Print current stored tokens and derived account id")
    p_info.add_argument("--json", action="store_true", help="Output raw auth.json contents")

    args = parser.parse_args()

    if args.command == "login":
        sys.exit(cmd_login(no_browser=args.no_browser, verbose=args.verbose))
    elif args.command == "serve":
        sys.exit(
            cmd_serve(
                host=args.host,
                port=args.port,
                verbose=args.verbose,
                reasoning_effort=args.reasoning_effort,
                reasoning_summary=args.reasoning_summary,
                reasoning_compat=args.reasoning_compat,
                debug_model=args.debug_model,
                expose_reasoning_models=args.expose_reasoning_models,
                default_web_search=args.enable_web_search,
            )
        )
    elif args.command == "info":
        auth = read_auth_file()
        if getattr(args, "json", False):
            print(json.dumps(auth or {}, indent=2))
            sys.exit(0)

        # Refresh active account tokens to keep metadata up to date.
        load_chatgpt_tokens()

        rows = _collect_accounts_state()
        active_slug = get_active_account_slug()

        if not rows:
            print("📂 Accounts")
            print("  • No accounts stored. Run: python3 chatmock.py login")
            print("")
            sys.exit(0)

        print("📂 Accounts")
        for row in rows:
            slug = row["slug"]
            marker = "[ACTIVE]" if slug == active_slug else "         "

            email = row.get("email")
            plan_raw = row.get("plan")
            if not email or not plan_raw:
                auth_data = read_auth_file(account_slug=slug) or {}
                tokens = auth_data.get("tokens") if isinstance(auth_data.get("tokens"), dict) else {}
                token_email, token_plan = _profile_from_tokens(
                    tokens.get("access_token"),
                    tokens.get("id_token"),
                )
                email = email or token_email
                plan_raw = plan_raw or token_plan

            plan_display = _plan_label(plan_raw)
            label = row.get("label") or email or slug

            print(f"  {marker} {label} ({plan_display})")
            if email:
                print(f"    Email: {email}")
            if row.get("account_id"):
                print(f"    Account ID: {row['account_id']}")
            if row.get("last_used"):
                print(f"    Last used: {row['last_used']}")

            usage_lines = row.get("usage_lines") or ["    Usage: no data recorded yet."]
            for line in usage_lines:
                print(line)
            print("")

        print("  [ACTIVE] indicates the active account.")
        print("Accounts rotate automatically when limits are reached.")
        sys.exit(0)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
