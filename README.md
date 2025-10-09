# ChatMock

- OpenAI & Ollama compatible API proxy that reuses your Codex ChatGPT Plus/Pro plan (**Paid Account Required**).
- Fork of `RayBytes/ChatMock`, maintained with a Docker-first workflow while still supporting a lightweight Python CLI.

## Major Updates

- `/v1/responses` endpoint fully compatible with the Codex VS Code extension.
- Multi-account vault with automatic selection and rotation to spread usage across sessions.
- Terminal UI for browsing, selecting, and managing stored accounts without stopping the main proxy server.
- Built-in Tailscale sidecar to expose ChatMock securely across your Tailnet.
- `/v1/embeddings` routing configurable between OpenAI and custom embedding backends.
- Per-request reasoning control via inline effort tags or CLI flags.

## Requirements

- ChatGPT account with Codex CLI OAuth access.
- Python 3.11+ for the CLI or Docker + docker compose for containers.
- Optional: Tailscale account when exposing ChatMock over your Tailnet.

## Docker Quickstart

- Copy env template and adjust secrets: `cp .env.example .env`.
- Build images: `docker compose build`.
- Complete the OAuth flow once:
  ```
  docker compose run --rm --service-ports chatmock-login login
  ```
- Log in to Tailscale once, after login you can Ctrl+C to exit:
  ```
  docker compose up tailscaled
  ```
- Start the stack (ChatMock + `tailscaled` sidecar): `docker compose up -d chatmock`.
- Reach the API on `http://localhost:8000/v1`; connect remotely after `tailscale up` in the sidecar.
- Need more detail? See [DOCKER.md](https://github.com/ms-xie/ChatMock/blob/main/DOCKER.md) for the full container playbook.

## CLI Quickstart

- Install dependencies (e.g. `pip install -r requirements.txt`) and stay in the project root.
- Authenticate once: `python chatmock.py login`; confirm details with `python chatmock.py info`.
- Run the local API gateway: `python chatmock.py serve` (defaults to `http://127.0.0.1:8000`).
- Point SDKs at the `/v1` suffix, with `low` reasoning effort:
  ```python
  from openai import OpenAI

  client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="<key>")
  reply = client.chat.completions.create(
      model="gpt-5",
      messages=[{"role": "user", "content": "hello world #L"}],
  )
  print(reply.choices[0].message.content)
  ```
  Use the `OPENAI_API_KEY` value you saved in `.env` wherever the snippets below show `<key>`.

## Feature Highlights

- **OpenAI-compatible endpoints** — `/v1/chat/completions`, `/v1/completions`, `/v1/responses` reuse existing OpenAI clients.
  ```bash
  curl http://127.0.0.1:8000/v1/responses \
       -H "Authorization: Bearer <key>" \
       -H "Content-Type: application/json" \
       -d '{"model":"gpt-5","input":[{"role":"user","content":[{"type":"input_text","text":"hello world #L"}]}]}'
  ```
- **Ollama bridge** — `/api/chat`, `/api/show`, `/api/tags` let Ollama-oriented tools talk to ChatGPT-backed models.
  ```bash
  curl http://127.0.0.1:8000/api/chat \
       -H "Content-Type: application/json" \
       -d '{"model":"gpt-5","messages":[{"role":"user","content":"draft a haiku about proxies"}]}'
  ```
- **Account rotation & usage insight** — CLI surfaces stored sessions and limits.
  ```bash
  docker compose run --rm --service-ports chatmock-login login
  ```
- **Web search passthrough** — enable at launch with `--enable-web-search`, then request search tools per call.<br>
Minimal effort (including the `gpt-5-mini` alias) skips web search, so raise the effort when you need live results.
  ```json
  {
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Find current METAR rules #M"}],
    "responses_tools": [{"type": "web_search"}],
    "responses_tool_choice": "auto"
  }
  ```
- **Per-request reasoning control** — inline tags dictate effort without editing config.
  ```bash
  curl http://127.0.0.1:8000/v1/responses \
       -H "Authorization: Bearer <key>" \
       -H "Content-Type: application/json" \
       -d '{"model":"gpt-5","input":[{"role":"user","content":[{"type":"input_text","text":"Explain FFT basics #M"}]}]}'
  ```

## Configuration Checklist

- `.env` keys:
  - `PORT`, `VERBOSE`, `OPENAI_API_KEY`, `API_KEY_CUSTOM_SUFFIX`.
  - `CHATGPT_LOCAL_REASONING_EFFORT`, `CHATGPT_LOCAL_REASONING_SUMMARY`, `CHATGPT_LOCAL_REASONING_COMPAT`.
  - `CHATGPT_ENABLE_WEB_SEARCH`, `CHATGPT_LOCAL_DEBUG_MODEL`, `CHATGPT_LOCAL_CLIENT_ID`.
- CLI flags mirror envs:
  - `python chatmock.py serve --reasoning-effort high --reasoning-summary concise`.
  - `python chatmock.py serve --enable-web-search`.
  - `python chatmock.py serve --expose-reasoning-models`.

## Reasoning & Compatibility Notes

- Inline tags `#L`, `#M`, `#H` or the `--reasoning-effort` flag switch between minimal/low/medium/high effort.
- `gpt-5-mini` aliases to `gpt-5` with minimal effort and no reasoning summary for quick replies.
- Set `--reasoning-compat legacy` to push summaries into reasoning tags instead of the response body.

## Supported Models

- `gpt-5`
- `gpt-5-codex`
- `codex-mini`
- `gpt-5-mini` (minimal effort alias)

## Operational Tips

- Toggle verbose logging by setting `VERBOSE=true` (check logs with `docker compose logs -f chatmock`).
- Check available CLI options anytime: `python chatmock.py serve --help`.
- Use responsibly; this community project is not affiliated with OpenAI.
