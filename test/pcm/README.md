# Testing PrivateChatManager (PCM)

Live end-to-end test for the PCM streaming path (`stream: true`).

## Prerequisites

1. A running PCM (e.g. the `examples/` Docker Compose stack exposing port `8080`).
2. A reachable Triton privacy-filter and downstream LLM for PCM to use.
3. `httpx` available in the interpreter you run the script with (already a
   PCM dependency; e.g. `private_chat_manager/.venv`).

## Test Script

- Script: `test.py`
- Default URL: `http://localhost:8080`

It sends a user message containing several kinds of PII, consumes the SSE
deltas, then inspects the stored session and asserts:

- the response is `text/event-stream` terminated by `[DONE]`;
- at least one placeholder was created;
- the hidden history (sent to the LLM) does **not** contain the PII;
- the raw history preserves the original message;
- every placeholder maps back to its original value in the raw history;
- no placeholder leaks to the client;
- the client-visible content equals the de-anonymised session history.

## Usage

Run from the repository root (`pnyx-privacy-filter/`):

```bash
./private_chat_manager/.venv/bin/python ./test/pcm/test.py --url http://localhost:8080
```

Useful flags:

```bash
python ./test/pcm/test.py --help
python ./test/pcm/test.py --message "My name is Jane Doe" --session-id <uuid>
```

The script exits non-zero if any check fails.

## Observing PCM

With the `examples/` stack, follow the manager logs while the test runs:

```bash
cd examples/
docker compose logs -f pnyx-pcm
```

Useful events (set `PCM_LOG_LEVEL=DEBUG` and `PCM_VERBOSE_LOG_EVENTS=...` in
`examples/.env`): `message redacted`, `LLM payload`, `stream raw response`,
`stream session saved`, `stream finished`. The session itself can be fetched
with `GET /v1/sessions/{id}` using the `X-Session-ID` returned by the stream.
