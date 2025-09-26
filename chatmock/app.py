from __future__ import annotations

import os

from flask import Flask, jsonify, make_response, request

from .config import BASE_INSTRUCTIONS, GPT5_CODEX_INSTRUCTIONS
from .http import build_cors_headers
from .routes_openai import openai_bp
from .routes_ollama import ollama_bp
from .utils import read_auth_file


def _load_expected_api_key() -> str | None:
    env_key = os.getenv("OPENAI_API_KEY")
    if isinstance(env_key, str) and env_key.strip():
        return env_key.strip()

    auth = read_auth_file()
    if isinstance(auth, dict):
        stored_key = auth.get("OPENAI_API_KEY")
        if isinstance(stored_key, str) and stored_key.strip():
            return stored_key.strip()
    return None


def _extract_presented_api_key() -> str | None:
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if isinstance(auth_header, str) and auth_header.strip():
        token = auth_header.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if token:
            return token

    for header_name in ("X-OpenAI-Api-Key", "x-openai-api-key", "X-Api-Key", "x-api-key", "OpenAI-Api-Key", "openai-api-key"):
        candidate = request.headers.get(header_name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    query_key = request.args.get("api_key")
    if isinstance(query_key, str) and query_key.strip():
        return query_key.strip()

    return None


def create_app(
    verbose: bool = False,
    reasoning_effort: str = "medium",
    reasoning_summary: str = "auto",
    reasoning_compat: str = "think-tags",
    debug_model: str | None = None,
    expose_reasoning_models: bool = False,
    default_web_search: bool = False,
) -> Flask:
    app = Flask(__name__)

    app.config.update(
        VERBOSE=bool(verbose),
        REASONING_EFFORT=reasoning_effort,
        REASONING_SUMMARY=reasoning_summary,
        REASONING_COMPAT=reasoning_compat,
        DEBUG_MODEL=debug_model,
        BASE_INSTRUCTIONS=BASE_INSTRUCTIONS,
        GPT5_CODEX_INSTRUCTIONS=GPT5_CODEX_INSTRUCTIONS,
        EXPOSE_REASONING_MODELS=bool(expose_reasoning_models),
        DEFAULT_WEB_SEARCH=bool(default_web_search),
    )

    app.config["EXPECTED_API_KEY"] = _load_expected_api_key()

    @app.before_request
    def _require_api_key():
        if request.method == "OPTIONS":
            return None

        expected_key = app.config.get("EXPECTED_API_KEY")
        if not isinstance(expected_key, str) or not expected_key:
            expected_key = _load_expected_api_key()
            if isinstance(expected_key, str) and expected_key:
                app.config["EXPECTED_API_KEY"] = expected_key

        if not isinstance(expected_key, str) or not expected_key:
            resp = make_response(jsonify({"error": {"message": "Server missing API key configuration"}}), 401)
            resp.headers.setdefault("WWW-Authenticate", "Bearer")
            for k, v in build_cors_headers().items():
                resp.headers.setdefault(k, v)
            return resp

        presented_key = _extract_presented_api_key()
        if not isinstance(presented_key, str) or not presented_key:
            resp = make_response(jsonify({"error": {"message": "Missing API key"}}), 401)
            resp.headers.setdefault("WWW-Authenticate", "Bearer")
            for k, v in build_cors_headers().items():
                resp.headers.setdefault(k, v)
            return resp

        if presented_key != expected_key:
            resp = make_response(jsonify({"error": {"message": "Invalid API key"}}), 403)
            resp.headers.setdefault("WWW-Authenticate", "Bearer")
            for k, v in build_cors_headers().items():
                resp.headers.setdefault(k, v)
            return resp

    @app.get("/")
    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.after_request
    def _cors(resp):
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp

    app.register_blueprint(openai_bp)
    app.register_blueprint(ollama_bp)

    return app
