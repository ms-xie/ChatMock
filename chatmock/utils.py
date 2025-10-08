from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import CLIENT_ID_DEFAULT, OAUTH_TOKEN_URL


_ACCOUNTS_DIR_NAME = "accounts"
_ACCOUNTS_STATE_FILE = "accounts.json"


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _base_storage_dir() -> str:
    home = os.getenv("CHATGPT_LOCAL_HOME") or os.getenv("CODEX_HOME")
    if not home:
        home = os.path.expanduser("~/.chatgpt-local")
    return home


def get_accounts_base_dir(ensure_exists: bool = False) -> str:
    base = _base_storage_dir()
    if ensure_exists:
        try:
            os.makedirs(base, exist_ok=True)
        except Exception:
            pass
    return base


def _accounts_root(ensure_exists: bool = False) -> str:
    root = os.path.join(get_accounts_base_dir(ensure_exists=ensure_exists), _ACCOUNTS_DIR_NAME)
    if ensure_exists:
        try:
            os.makedirs(root, exist_ok=True)
        except Exception:
            pass
    return root


def _accounts_state_path() -> str:
    return os.path.join(get_accounts_base_dir(ensure_exists=False), _ACCOUNTS_STATE_FILE)


def _load_accounts_state() -> Dict[str, Any]:
    path = _accounts_state_path()
    try:
        with open(path, "r", encoding="utf-8") as fp:
            raw = json.load(fp)
    except FileNotFoundError:
        return {"active": None, "accounts": {}}
    except Exception:
        return {"active": None, "accounts": {}}

    accounts: Dict[str, Dict[str, Any]] = {}
    raw_accounts = raw.get("accounts") if isinstance(raw, dict) else None
    if isinstance(raw_accounts, dict):
        for slug, meta in raw_accounts.items():
            if not isinstance(slug, str) or not slug:
                continue
            if isinstance(meta, dict):
                accounts[slug] = meta
    active = raw.get("active") if isinstance(raw, dict) else None
    if not isinstance(active, str) or active not in accounts:
        active = None
    return {"active": active, "accounts": accounts}


def _write_accounts_state(state: Dict[str, Any]) -> None:
    base = get_accounts_base_dir(ensure_exists=True)
    path = os.path.join(base, _ACCOUNTS_STATE_FILE)
    payload = {
        "active": state.get("active"),
        "accounts": state.get("accounts", {}),
    }
    try:
        with open(path, "w", encoding="utf-8") as fp:
            if hasattr(os, "fchmod"):
                try:
                    os.fchmod(fp.fileno(), 0o600)
                except OSError:
                    pass
            json.dump(payload, fp, indent=2)
    except Exception as exc:
        eprint(f"ERROR: unable to persist accounts state ({exc})")


def _normalize_account_slug(identifier: Optional[str]) -> str:
    if not isinstance(identifier, str):
        return ""
    slug = re.sub(r"[^a-z0-9]+", "-", identifier.lower()).strip("-")
    return slug


def ensure_unique_account_slug(preferred: Optional[str] = None) -> str:
    base_slug = _normalize_account_slug(preferred)
    if not base_slug:
        base_slug = "account"
    state = _load_accounts_state()
    existing = set(state.get("accounts", {}).keys())
    slug = base_slug
    counter = 2
    while slug in existing:
        slug = f"{base_slug}-{counter}"
        counter += 1
    return slug


def list_known_accounts() -> List[Dict[str, Any]]:
    state = _load_accounts_state()
    accounts: List[Dict[str, Any]] = []
    for slug, meta in state.get("accounts", {}).items():
        if not isinstance(slug, str) or not slug:
            continue
        entry = {"slug": slug}
        if isinstance(meta, dict):
            entry.update(meta)
        accounts.append(entry)
    accounts.sort(key=lambda item: item.get("label") or item["slug"])
    return accounts


def get_active_account_slug() -> Optional[str]:
    state = _load_accounts_state()
    active = state.get("active")
    if isinstance(active, str) and active in state.get("accounts", {}):
        return active
    return None


def set_active_account_slug(slug: Optional[str]) -> None:
    state = _load_accounts_state()
    accounts = dict(state.get("accounts", {}))
    if slug is None:
        state["active"] = None
    else:
        normalized = slug.strip()
        if normalized:
            now = _now_iso8601()
            meta = dict(accounts.get(normalized) or {})
            if not meta.get("created_at"):
                meta["created_at"] = now
            meta["last_used"] = now
            meta["updated_at"] = now
            if not meta.get("label"):
                meta["label"] = normalized
            accounts[normalized] = meta
            state["active"] = normalized
        else:
            state["active"] = None
    state["accounts"] = accounts
    _write_accounts_state(state)


def update_account_metadata(slug: str, **fields: Any) -> Dict[str, Any]:
    normalized = (slug or "").strip()
    if not normalized:
        raise ValueError("account slug is required")
    state = _load_accounts_state()
    accounts = dict(state.get("accounts", {}))
    meta = dict(accounts.get(normalized) or {})
    now = _now_iso8601()
    changed = False
    if not meta.get("created_at"):
        meta["created_at"] = now
        changed = True
    if not meta.get("label"):
        meta["label"] = normalized
        changed = True
    for key, value in fields.items():
        if value is None:
            continue
        if meta.get(key) != value:
            meta[key] = value
            changed = True
    if changed:
        meta["updated_at"] = now
    accounts[normalized] = meta
    state["accounts"] = accounts
    _write_accounts_state(state)
    return meta


def remove_account(slug: str) -> bool:
    normalized = (slug or "").strip()
    if not normalized:
        return False
    state = _load_accounts_state()
    accounts = dict(state.get("accounts", {}))
    removed = accounts.pop(normalized, None) is not None
    if state.get("active") == normalized:
        state["active"] = None
    state["accounts"] = accounts
    _write_accounts_state(state)
    directory = os.path.join(_accounts_root(ensure_exists=False), normalized)
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        pass
    except Exception as exc:
        eprint(f"WARNING: unable to fully remove account '{normalized}': {exc}")
    return removed


def get_account_directory(slug: str, ensure_exists: bool = False) -> str:
    normalized = (slug or "").strip()
    if not normalized:
        return get_accounts_base_dir(ensure_exists=ensure_exists)
    path = os.path.join(_accounts_root(ensure_exists=ensure_exists), normalized)
    if ensure_exists:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
    return path


def get_home_dir(account_slug: Optional[str] = None, ensure_exists: bool = False) -> str:
    slug = account_slug or get_active_account_slug()
    if slug:
        return get_account_directory(slug, ensure_exists=ensure_exists)
    return get_accounts_base_dir(ensure_exists=ensure_exists)


def read_auth_file(account_slug: Optional[str] = None) -> Dict[str, Any] | None:
    candidate_paths: List[str] = []
    slug = account_slug or get_active_account_slug()
    if slug:
        account_path = os.path.join(get_account_directory(slug, ensure_exists=False), "auth.json")
        candidate_paths.append(account_path)
    for base in [
        get_accounts_base_dir(ensure_exists=False),
        os.getenv("CHATGPT_LOCAL_HOME"),
        os.getenv("CODEX_HOME"),
        os.path.expanduser("~/.chatgpt-local"),
        os.path.expanduser("~/.codex"),
    ]:
        if not base:
            continue
        path = os.path.join(base, "auth.json")
        if path not in candidate_paths:
            candidate_paths.append(path)
    for path in candidate_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return None


def write_auth_file(auth: Dict[str, Any], account_slug: Optional[str] = None) -> bool:
    slug = account_slug or get_active_account_slug()
    if slug:
        home = get_account_directory(slug, ensure_exists=True)
    else:
        home = get_accounts_base_dir(ensure_exists=True)
    path = os.path.join(home, "auth.json")
    try:
        with open(path, "w", encoding="utf-8") as fp:
            if hasattr(os, "fchmod"):
                os.fchmod(fp.fileno(), 0o600)
            json.dump(auth, fp, indent=2)
        if slug:
            update_account_metadata(slug, last_used=_now_iso8601())
        return True
    except Exception as exc:
        eprint(f"ERROR: unable to write auth file: {exc}")
        return False


def parse_jwt_claims(token: str) -> Dict[str, Any] | None:
    if not token or token.count(".") != 2:
        return None
    try:
        _, payload, _ = token.split(".")
        padded = payload + "=" * (-len(payload) % 4)
        data = base64.urlsafe_b64decode(padded.encode())
        return json.loads(data.decode())
    except Exception:
        return None


def generate_pkce() -> "PkceCodes":
    from .models import PkceCodes

    code_verifier = secrets.token_hex(64)
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return PkceCodes(code_verifier=code_verifier, code_challenge=code_challenge)


def convert_chat_messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _normalize_image_data_url(url: str) -> str:
        try:
            if not isinstance(url, str):
                return url
            if not url.startswith("data:image/"):
                return url
            if ";base64," not in url:
                return url
            header, data = url.split(",", 1)
            try:
                from urllib.parse import unquote

                data = unquote(data)
            except Exception:
                pass
            data = data.strip().replace("\n", "").replace("\r", "")
            data = data.replace("-", "+").replace("_", "/")
            pad = (-len(data)) % 4
            if pad:
                data = data + ("=" * pad)
            try:
                base64.b64decode(data, validate=True)
            except Exception:
                return url
            return f"{header},{data}"
        except Exception:
            return url

    input_items: List[Dict[str, Any]] = []
    system_content = None
    for message in messages:
        role = message.get("role")
        if role in ["system", "developer"]:
            system_content = message.get("content")
            continue

        if role == "tool":
            call_id = message.get("tool_call_id") or message.get("id")
            if isinstance(call_id, str) and call_id:
                content = message.get("content", "")
                if isinstance(content, list):
                    texts = []
                    for part in content:
                        if isinstance(part, dict):
                            t = part.get("text") or part.get("content")
                            if isinstance(t, str) and t:
                                texts.append(t)
                    content = "\n".join(texts)
                if isinstance(content, str):
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": content,
                        }
                    )
            continue
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            for tc in message.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tc_type = tc.get("type", "function")
                if tc_type != "function":
                    continue
                call_id = tc.get("id") or tc.get("call_id")
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = fn.get("name") if isinstance(fn, dict) else None
                args = fn.get("arguments") if isinstance(fn, dict) else None
                if isinstance(call_id, str) and isinstance(name, str) and isinstance(args, str):
                    input_items.append(
                        {
                            "type": "function_call",
                            "name": name,
                            "arguments": args,
                            "call_id": call_id,
                        }
                    )

        content = message.get("content", "")
        content_items: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text = part.get("text") or part.get("content") or ""
                    if isinstance(text, str) and text:
                        kind = "output_text" if role == "assistant" else "input_text"
                        content_items.append({"type": kind, "text": text})
                elif ptype == "image_url":
                    image = part.get("image_url")
                    url = image.get("url") if isinstance(image, dict) else image
                    if isinstance(url, str) and url:
                        content_items.append({"type": "input_image", "image_url": _normalize_image_data_url(url)})
        elif isinstance(content, str) and content:
            kind = "output_text" if role == "assistant" else "input_text"
            content_items.append({"type": kind, "text": content})

        if not content_items:
            continue
        role_out = "assistant" if role == "assistant" else "user"
        input_items.append({"type": "message", "role": role_out, "content": content_items})
    
    if system_content:
        for item in input_items:
            if item['role'] == 'user':
                item['content'] = f"{system_content}\n\n---\n\n{item['content']}"
    return input_items


def convert_tools_chat_to_responses(tools: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            continue
        desc = fn.get("description") if isinstance(fn, dict) else None
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "name": name,
                "description": desc or "",
                "strict": False,
                "parameters": params,
            }
        )
    return out


def _account_limit_reached(slug: str) -> bool:
    if not isinstance(slug, str) or not slug:
        return False
    try:
        from .limits import compute_reset_at, load_rate_limit_snapshot  # noqa: WPS433
    except Exception:
        return False
    snapshot = load_rate_limit_snapshot(account_slug=slug)
    if snapshot is None:
        return False
    windows = []
    if snapshot.snapshot.primary is not None:
        windows.append(snapshot.snapshot.primary)
    if snapshot.snapshot.secondary is not None:
        windows.append(snapshot.snapshot.secondary)
    for window in windows:
        try:
            used = float(window.used_percent)
        except (TypeError, ValueError):
            continue
        if used >= 100.0:
            reset_at = compute_reset_at(snapshot.captured_at, window)
            if reset_at is not None:
                if reset_at.tzinfo is None:
                    reset_at = reset_at.replace(tzinfo=datetime.timezone.utc)
                now = datetime.datetime.now(datetime.timezone.utc)
                if reset_at <= now:
                    continue
            return True
    return False


def _secondary_reset_eta_seconds(slug: str) -> Optional[float]:
    if not isinstance(slug, str) or not slug:
        return None
    try:
        from .limits import compute_reset_at, load_rate_limit_snapshot  # noqa: WPS433
    except Exception:
        return None

    snapshot = load_rate_limit_snapshot(account_slug=slug)
    if snapshot is None or snapshot.snapshot.secondary is None:
        return None

    window = snapshot.snapshot.secondary
    reset_at = compute_reset_at(snapshot.captured_at, window)
    if reset_at is None:
        if window.resets_in_seconds is None:
            return None
        return float(window.resets_in_seconds)

    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    eta = (reset_at - now).total_seconds()
    if eta < 0:
        return 0.0
    return eta


def _select_account_with_capacity(preferred: Optional[str]) -> Optional[str]:
    known = list_known_accounts()
    order: List[str] = []
    if isinstance(preferred, str) and preferred:
        order.append(preferred)
    for entry in known:
        slug = entry.get("slug") if isinstance(entry, dict) else None
        if isinstance(slug, str) and slug and slug not in order:
            order.append(slug)
    index_by_slug = {slug: idx for idx, slug in enumerate(order)}
    candidates: List[Tuple[str, Optional[float]]] = []
    for slug in order:
        if slug and not _account_limit_reached(slug):
            eta = _secondary_reset_eta_seconds(slug)
            candidates.append((slug, eta))
    if candidates:
        def _candidate_key(item: Tuple[str, Optional[float]]) -> Tuple[int, float, float]:
            slug, eta = item
            # Prefer accounts with no recent usage snapshot (eta is None) to gather data early.
            has_snapshot = 1 if eta is not None else 0
            eta_value = float(eta) if eta is not None else 0.0
            fallback_index = float(index_by_slug.get(slug, float("inf")))
            return (has_snapshot, eta_value, fallback_index)

        candidates.sort(key=_candidate_key)
        return candidates[0][0]
    return order[0] if order else preferred


def _extract_account_profile(
    id_token: Optional[str],
    access_token: Optional[str],
) -> Dict[str, Optional[str]]:
    profile: Dict[str, Optional[str]] = {"email": None, "plan": None}
    id_claims = parse_jwt_claims(id_token) or {}
    if isinstance(id_claims, dict):
        for key in ("email", "preferred_username", "name", "sub"):
            value = id_claims.get(key)
            if isinstance(value, str) and value:
                profile["email"] = value
                break
    access_claims = parse_jwt_claims(access_token) or {}
    plan_raw = None
    if isinstance(access_claims, dict):
        auth_claims = access_claims.get("https://api.openai.com/auth")
        if isinstance(auth_claims, dict):
            plan_raw = auth_claims.get("chatgpt_plan_type")
    if isinstance(plan_raw, str) and plan_raw:
        profile["plan"] = plan_raw
    elif plan_raw is not None:
        profile["plan"] = str(plan_raw)
    return profile


def _bootstrap_account_from_auth(auth: Dict[str, Any]) -> Optional[str]:
    tokens = auth.get("tokens") if isinstance(auth.get("tokens"), dict) else {}
    id_token = tokens.get("id_token") if isinstance(tokens, dict) else None
    access_token = tokens.get("access_token") if isinstance(tokens, dict) else None
    account_id = tokens.get("account_id") if isinstance(tokens, dict) else None
    if not isinstance(account_id, str) or not account_id:
        account_id = _derive_account_id(id_token)

    profile = _extract_account_profile(id_token, access_token)
    slug_seed = profile.get("email") or account_id or "account"
    slug = ensure_unique_account_slug(slug_seed)
    if not write_auth_file(auth, account_slug=slug):
        return None

    label = profile.get("email") or slug
    update_account_metadata(
        slug,
        label=label,
        email=profile.get("email"),
        plan=profile.get("plan"),
        account_id=account_id,
    )

    legacy_limits = os.path.join(get_accounts_base_dir(ensure_exists=False), "usage_limits.json")
    if os.path.isfile(legacy_limits):
        destination = os.path.join(get_account_directory(slug, ensure_exists=True), "usage_limits.json")
        if not os.path.exists(destination):
            try:
                shutil.copy2(legacy_limits, destination)
            except Exception:
                pass
    return slug


def load_chatgpt_tokens(
    ensure_fresh: bool = True,
    account_slug: Optional[str] = None,
) -> tuple[str | None, str | None, str | None]:
    requested_slug = (account_slug or "").strip() or None
    active = get_active_account_slug()
    slug = requested_slug or active

    if not slug:
        known = list_known_accounts()
        if known:
            slug = known[0]["slug"]

    if requested_slug is None and slug:
        candidate = _select_account_with_capacity(slug)
        if candidate:
            slug = candidate

    if slug and slug != active and requested_slug is None:
        set_active_account_slug(slug)
        active = slug

    auth = read_auth_file(account_slug=slug)
    if not isinstance(auth, dict):
        return None, None, None

    if slug is None:
        slug = _bootstrap_account_from_auth(auth)
        if slug:
            set_active_account_slug(slug)
            auth = read_auth_file(account_slug=slug) or auth

    tokens = auth.get("tokens") if isinstance(auth.get("tokens"), dict) else {}
    access_token: Optional[str] = tokens.get("access_token")
    account_id: Optional[str] = tokens.get("account_id")
    id_token: Optional[str] = tokens.get("id_token")
    refresh_token: Optional[str] = tokens.get("refresh_token")
    last_refresh = auth.get("last_refresh")

    if ensure_fresh and isinstance(refresh_token, str) and refresh_token and CLIENT_ID_DEFAULT:
        needs_refresh = _should_refresh_access_token(access_token, last_refresh)
        if needs_refresh or not (isinstance(access_token, str) and access_token):
            refreshed = _refresh_chatgpt_tokens(refresh_token, CLIENT_ID_DEFAULT)
            if refreshed:
                access_token = refreshed.get("access_token") or access_token
                id_token = refreshed.get("id_token") or id_token
                refresh_token = refreshed.get("refresh_token") or refresh_token
                account_id = refreshed.get("account_id") or account_id

                updated_tokens = dict(tokens)
                if isinstance(access_token, str) and access_token:
                    updated_tokens["access_token"] = access_token
                if isinstance(id_token, str) and id_token:
                    updated_tokens["id_token"] = id_token
                if isinstance(refresh_token, str) and refresh_token:
                    updated_tokens["refresh_token"] = refresh_token
                if isinstance(account_id, str) and account_id:
                    updated_tokens["account_id"] = account_id

                persisted = _persist_refreshed_auth(auth, updated_tokens, account_slug=slug)
                if persisted is not None:
                    auth, tokens = persisted
                else:
                    tokens = updated_tokens

    if not isinstance(account_id, str) or not account_id:
        account_id = _derive_account_id(id_token)

    if slug:
        profile = _extract_account_profile(id_token, access_token)
        now_iso = _now_iso8601()
        update_account_metadata(
            slug,
            label=profile.get("email") or slug,
            email=profile.get("email"),
            plan=profile.get("plan"),
            account_id=account_id,
            last_used=now_iso,
        )

    access_token = access_token if isinstance(access_token, str) and access_token else None
    id_token = id_token if isinstance(id_token, str) and id_token else None
    account_id = account_id if isinstance(account_id, str) and account_id else None
    return access_token, account_id, id_token


def _should_refresh_access_token(access_token: Optional[str], last_refresh: Any) -> bool:
    if not isinstance(access_token, str) or not access_token:
        return True

    claims = parse_jwt_claims(access_token) or {}
    exp = claims.get("exp") if isinstance(claims, dict) else None
    now = datetime.datetime.now(datetime.timezone.utc)
    if isinstance(exp, (int, float)):
        try:
            expiry = datetime.datetime.fromtimestamp(float(exp), datetime.timezone.utc)
        except (OverflowError, OSError, ValueError):
            expiry = None
        if expiry is not None:
            return expiry <= now + datetime.timedelta(minutes=5)

    if isinstance(last_refresh, str):
        refreshed_at = _parse_iso8601(last_refresh)
        if refreshed_at is not None:
            return refreshed_at <= now - datetime.timedelta(minutes=55)
    return False


def _refresh_chatgpt_tokens(refresh_token: str, client_id: str) -> Optional[Dict[str, Optional[str]]]:
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "scope": "openid profile email",
    }

    try:
        resp = requests.post(OAUTH_TOKEN_URL, json=payload, timeout=30)
    except requests.RequestException as exc:
        eprint(f"ERROR: failed to refresh ChatGPT token: {exc}")
        return None

    if resp.status_code >= 400:
        eprint(f"ERROR: refresh token request returned status {resp.status_code}")
        return None

    try:
        data = resp.json()
    except ValueError as exc:
        eprint(f"ERROR: unable to parse refresh token response: {exc}")
        return None

    id_token = data.get("id_token")
    access_token = data.get("access_token")
    new_refresh_token = data.get("refresh_token") or refresh_token
    if not isinstance(id_token, str) or not isinstance(access_token, str):
        eprint("ERROR: refresh token response missing expected tokens")
        return None

    account_id = _derive_account_id(id_token)
    new_refresh_token = new_refresh_token if isinstance(new_refresh_token, str) and new_refresh_token else refresh_token
    return {
        "id_token": id_token,
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "account_id": account_id,
    }


def _persist_refreshed_auth(
    auth: Dict[str, Any],
    updated_tokens: Dict[str, Any],
    *,
    account_slug: Optional[str] = None,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    updated_auth = dict(auth)
    updated_auth["tokens"] = updated_tokens
    updated_auth["last_refresh"] = _now_iso8601()
    if write_auth_file(updated_auth, account_slug=account_slug):
        return updated_auth, updated_tokens
    eprint("ERROR: unable to persist refreshed auth tokens")
    return None


def _derive_account_id(id_token: Optional[str]) -> Optional[str]:
    if not isinstance(id_token, str) or not id_token:
        return None
    claims = parse_jwt_claims(id_token) or {}
    auth_claims = claims.get("https://api.openai.com/auth") if isinstance(claims, dict) else None
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


def _parse_iso8601(value: str) -> Optional[datetime.datetime]:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    except Exception:
        return None


def _now_iso8601() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def get_effective_chatgpt_auth() -> tuple[str | None, str | None]:
    access_token, account_id, id_token = load_chatgpt_tokens()
    if not account_id:
        account_id = _derive_account_id(id_token)
    return access_token, account_id


def sse_translate_chat(
    upstream,
    model: str,
    created: int,
    verbose: bool = False,
    vlog=None,
    reasoning_compat: str = "think-tags",
    *,
    include_usage: bool = False,
):
    response_id = "chatcmpl-stream"
    compat = (reasoning_compat or "think-tags").strip().lower()
    think_open = False
    think_closed = False
    saw_output = False
    saw_any_summary = False
    pending_summary_paragraph = False
    upstream_usage = None
    ws_state: dict[str, Any] = {}
    ws_index: dict[str, int] = {}
    ws_next_index: int = 0
    
    def _serialize_tool_args(eff_args: Any) -> str:
        """
        Serialize tool call arguments with proper JSON handling.
        
        Args:
            eff_args: Arguments to serialize (dict, list, str, or other)
            
        Returns:
            JSON string representation of the arguments
        """
        if isinstance(eff_args, (dict, list)):
            return json.dumps(eff_args)
        elif isinstance(eff_args, str):
            try:
                parsed = json.loads(eff_args)
                if isinstance(parsed, (dict, list)):
                    return json.dumps(parsed) 
                else:
                    return json.dumps({"query": eff_args})  
            except (json.JSONDecodeError, ValueError):
                return json.dumps({"query": eff_args})
        else:
            return "{}"
    
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
        for raw in upstream.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else raw
            if verbose and vlog:
                vlog(line)
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
            if isinstance(evt.get("response"), dict) and isinstance(evt["response"].get("id"), str):
                response_id = evt["response"].get("id") or response_id

            if isinstance(kind, str) and ("web_search_call" in kind):
                try:
                    call_id = evt.get("item_id") or "ws_call"
                    if verbose and vlog:
                        try:
                            vlog(f"CM_TOOLS {kind} id={call_id} -> tool_calls(web_search)")
                        except Exception:
                            pass
                    item = evt.get('item') if isinstance(evt.get('item'), dict) else {}
                    params_dict = ws_state.setdefault(call_id, {}) if isinstance(ws_state.get(call_id), dict) else {}
                    def _merge_from(src):
                        if not isinstance(src, dict):
                            return
                        for whole in ('parameters','args','arguments','input'):
                            if isinstance(src.get(whole), dict):
                                params_dict.update(src.get(whole))
                        if isinstance(src.get('query'), str): params_dict.setdefault('query', src.get('query'))
                        if isinstance(src.get('q'), str): params_dict.setdefault('query', src.get('q'))
                        for rk in ('recency','time_range','days'):
                            if src.get(rk) is not None and rk not in params_dict: params_dict[rk] = src.get(rk)
                        for dk in ('domains','include_domains','include'):
                            if isinstance(src.get(dk), list) and 'domains' not in params_dict: params_dict['domains'] = src.get(dk)
                        for mk in ('max_results','topn','limit'):
                            if src.get(mk) is not None and 'max_results' not in params_dict: params_dict['max_results'] = src.get(mk)
                    _merge_from(item)
                    _merge_from(evt if isinstance(evt, dict) else None)
                    params = params_dict if params_dict else None
                    if isinstance(params, dict):
                        try:
                            ws_state.setdefault(call_id, {}).update(params)
                        except Exception:
                            pass
                    eff_params = ws_state.get(call_id, params if isinstance(params, (dict, list, str)) else {})
                    args_str = _serialize_tool_args(eff_params)
                    if call_id not in ws_index:
                        ws_index[call_id] = ws_next_index
                        ws_next_index += 1
                    _idx = ws_index.get(call_id, 0)
                    delta_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": _idx,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {"name": "web_search", "arguments": args_str},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(delta_chunk)}\n\n".encode("utf-8")
                    if kind.endswith(".completed") or kind.endswith(".done"):
                        finish_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                            ],
                        }
                        yield f"data: {json.dumps(finish_chunk)}\n\n".encode("utf-8")
                except Exception:
                    pass

            if kind == "response.output_text.delta":
                delta = evt.get("delta") or ""
                if compat == "think-tags" and think_open and not think_closed:
                    close_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": "</think>"}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(close_chunk)}\n\n".encode("utf-8")
                    think_open = False
                    think_closed = True
                saw_output = True
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            elif kind == "response.output_item.done":
                item = evt.get("item") or {}
                if isinstance(item, dict) and (item.get("type") == "function_call" or item.get("type") == "web_search_call"):
                    call_id = item.get("call_id") or item.get("id") or ""
                    name = item.get("name") or ("web_search" if item.get("type") == "web_search_call" else "")
                    raw_args = item.get("arguments") or item.get("parameters")
                    if isinstance(raw_args, dict):
                        try:
                            ws_state.setdefault(call_id, {}).update(raw_args)
                        except Exception:
                            pass
                    eff_args = ws_state.get(call_id, raw_args if isinstance(raw_args, (dict, list, str)) else {})
                    try:
                        args = _serialize_tool_args(eff_args)
                    except Exception:
                        args = "{}"
                    if item.get("type") == "web_search_call" and verbose and vlog:
                        try:
                            vlog(f"CM_TOOLS response.output_item.done web_search_call id={call_id} has_args={bool(args)}")
                        except Exception:
                            pass
                    if call_id not in ws_index:
                        ws_index[call_id] = ws_next_index
                        ws_next_index += 1
                    _idx = ws_index.get(call_id, 0)
                    if isinstance(call_id, str) and isinstance(name, str) and isinstance(args, str):
                        delta_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": _idx,
                                                "id": call_id,
                                                "type": "function",
                                                "function": {"name": name, "arguments": args},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(delta_chunk)}\n\n".encode("utf-8")

                        finish_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                        }
                        yield f"data: {json.dumps(finish_chunk)}\n\n".encode("utf-8")
            elif kind == "response.reasoning_summary_part.added":
                if compat in ("think-tags", "o3"):
                    if saw_any_summary:
                        pending_summary_paragraph = True
                    else:
                        saw_any_summary = True
            elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                delta_txt = evt.get("delta") or ""
                if compat == "o3":
                    if kind == "response.reasoning_summary_text.delta" and pending_summary_paragraph:
                        nl_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning": {"content": [{"type": "text", "text": "\n"}]}},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(nl_chunk)}\n\n".encode("utf-8")
                        pending_summary_paragraph = False
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning": {"content": [{"type": "text", "text": delta_txt}]}},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                elif compat == "think-tags":
                    if not think_open and not think_closed:
                        open_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": "<think>"}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(open_chunk)}\n\n".encode("utf-8")
                        think_open = True
                    if think_open and not think_closed:
                        if kind == "response.reasoning_summary_text.delta" and pending_summary_paragraph:
                            nl_chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": "\n"}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(nl_chunk)}\n\n".encode("utf-8")
                            pending_summary_paragraph = False
                        content_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": delta_txt}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(content_chunk)}\n\n".encode("utf-8")
                else:
                    if kind == "response.reasoning_summary_text.delta":
                        chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_summary": delta_txt, "reasoning": delta_txt},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    else:
                        chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {"reasoning": delta_txt}, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            elif isinstance(kind, str) and kind.endswith(".done"):
                pass
            elif kind == "response.output_text.done":
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            elif kind == "response.failed":
                err = evt.get("response", {}).get("error", {}).get("message", "response.failed")
                chunk = {"error": {"message": err}}
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            elif kind == "response.completed":
                m = _extract_usage(evt)
                if m:
                    upstream_usage = m
                if compat == "think-tags" and think_open and not think_closed:
                    close_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": "</think>"}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(close_chunk)}\n\n".encode("utf-8")
                    think_open = False
                    think_closed = True
                if include_usage and upstream_usage:
                    try:
                        usage_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                            "usage": upstream_usage,
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n".encode("utf-8")
                    except Exception:
                        pass
                yield b"data: [DONE]\n\n"
                break
    finally:
        upstream.close()


def sse_translate_text(upstream, model: str, created: int, verbose: bool = False, vlog=None, *, include_usage: bool = False):
    response_id = "cmpl-stream"
    upstream_usage = None
    
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
        for raw_line in upstream.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, (bytes, bytearray)) else raw_line
            if verbose and vlog:
                vlog(line)
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):].strip()
            if not data or data == "[DONE]":
                if data == "[DONE]":
                    chunk = {
                        "id": response_id,
                        "object": "text_completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                continue
            try:
                evt = json.loads(data)
            except Exception:
                continue
            kind = evt.get("type")
            if isinstance(evt.get("response"), dict) and isinstance(evt["response"].get("id"), str):
                response_id = evt["response"].get("id") or response_id
            if kind == "response.output_text.delta":
                delta_text = evt.get("delta") or ""
                chunk = {
                    "id": response_id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": delta_text, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            elif kind == "response.output_text.done":
                chunk = {
                    "id": response_id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            elif kind == "response.completed":
                m = _extract_usage(evt)
                if m:
                    upstream_usage = m
                if include_usage and upstream_usage:
                    try:
                        usage_chunk = {
                            "id": response_id,
                            "object": "text_completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "text": "", "finish_reason": None}],
                            "usage": upstream_usage,
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n".encode("utf-8")
                    except Exception:
                        pass
                yield b"data: [DONE]\n\n"
                break
    finally:
        upstream.close()
