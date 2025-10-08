# Docker Deployment

## Prepare

- Git clone this repo and `cd` into it.
  ```
  git clone https://github.com/ms-xie/ChatMock.git
  cd ChatMock
  ```
- Copy the env template once: `cp .env.example .env`.
- Fill in `OPENAI_API_KEY`, `API_KEY_CUSTOM_SUFFIX`, and any reasoning defaults you need.
- Build or rebuild images whenever dependencies change: `docker compose build`.

## Authenticate

- `chatmock-login` launches a TUI that can add, remove, rename accounts and monitor usage.
- Start the OAuth helper with published ports:
  ```
  docker compose run --rm --service-ports chatmock-login login
  ```
- Paste the printed URL into a browser and, if callbacks fail, paste the redirect URL back into the terminal.
- Re-run the same command anytime you want to rotate stored ChatGPT accounts.

## Run the Stack

- Bring up ChatMock (and the `tailscaled` sidecar) in the background: `docker compose up -d chatmock`.
- Verify the service: `docker compose ps` should list `chatmock` and `tailscaled` as `running`.
- Hit the API locally: `http://localhost:8000/v1`.

## Remote Access via Tailscale

- Ensure the sidecar is authenticated: `docker compose exec tailscaled tailscale up`.
- Use Tailnet routing or Funnel (disabled by default) to expose ChatMock without opening public ports.
- Example status check:
  ```
  docker compose exec tailscaled tailscale status
  ```
- Read the [entrypoint-tailscale.sh](https://github.com/ms-xie/ChatMock/blob/main/docker/entrypoint-tailscale.sh) for detailed.

## Configuration Cheatsheet

- `.env` keys:
  - `PORT`, `VERBOSE`, `OPENAI_API_KEY`, `API_KEY_CUSTOM_SUFFIX`.
  - `CHATGPT_LOCAL_REASONING_EFFORT`, `CHATGPT_LOCAL_REASONING_SUMMARY`, `CHATGPT_LOCAL_REASONING_COMPAT`.
  - `CHATGPT_LOCAL_DEBUG_MODEL`, `CHATGPT_LOCAL_CLIENT_ID`, `CHATGPT_ENABLE_WEB_SEARCH`.
- Override at runtime with `docker compose run -e KEY=value ...` for ad-hoc tests.

## Usage Tips

- Inline `#L`, `#M`, `#H` tags inside the final user message to force reasoning effort tiers.
- Select `gpt-5-mini` when you want minimal effort, no reasoning summary, and faster turnarounds.
- Enable verbose logging for diagnostics:
  ```
  VERBOSE=true docker compose up -d chatmock
  docker compose logs -f chatmock
  ```

## Smoke Test

- Validate the proxy after startup:
  ```
  curl -s http://localhost:8000/v1/responses \
       -H "Content-Type: application/json" \
       -d '{"model":"gpt-5","input":[{"role":"user","content":[{"type":"input_text","text":"Hello world #L"}]}]}' \
       | jq .
  ```

## Shut Down

- Stop everything: `docker compose down`.
