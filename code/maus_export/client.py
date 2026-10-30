"""MCP client for the Maus dashboard (https://dash.mausrobotics.com/mcp).

Auth: OAuth 2.1 with dynamic client registration + PKCE. The first run opens a
browser to sign in; tokens are cached in ~/.maus-export/auth.json
(outside the repo).
Set MAUS_TOKEN to skip the browser flow and use a bearer token directly.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import anyio
import httpx2
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

SERVER_URL = os.environ.get("MAUS_MCP_URL", "https://dash.mausrobotics.com/mcp")
CALLBACK_PORT = int(os.environ.get("MAUS_CALLBACK_PORT", "8765"))
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/callback"
AUTH_FILE = Path.home() / ".maus-export" / "auth.json"


class FileTokenStorage:
    """Stores OAuth tokens and the registered client in a local JSON file."""

    def __init__(self, path: Path = AUTH_FILE):
        self.path = path

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, key: str, value: dict[str, Any]) -> None:
        data = self._load()
        data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._load().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._save("tokens", tokens.model_dump(mode="json", exclude_none=True))

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._load().get("client")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(
        self, client_info: OAuthClientInformationFull
    ) -> None:
        self._save(
            "client", client_info.model_dump(mode="json", exclude_none=True)
        )

    def access_token(self) -> str | None:
        return (self._load().get("tokens") or {}).get("access_token")


def _wait_for_callback() -> AuthorizationCodeResult:
    """One-shot local HTTP server; returns the ?code=&state= it gets."""
    result: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            qs = parse_qs(urlparse(self.path).query)
            if "code" in qs:
                result.update({k: v[0] for k, v in qs.items()})
                body = b"<h2>Signed in. You can close this tab.</h2>"
                self.send_response(200)
            else:
                body = b"Missing code"
                self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("localhost", CALLBACK_PORT), Handler)
    try:
        while "code" not in result:
            server.handle_request()
    finally:
        server.server_close()
    return AuthorizationCodeResult(
        code=result["code"], state=result.get("state"), iss=result.get("iss")
    )


async def _redirect_handler(url: str) -> None:
    print(f"\nOpen this URL to sign in to the Maus dashboard:\n  {url}\n")
    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()


async def _callback_handler() -> AuthorizationCodeResult:
    return await anyio.to_thread.run_sync(_wait_for_callback)


def _http_client(storage: FileTokenStorage) -> httpx2.AsyncClient:
    timeout = httpx2.Timeout(60.0, read=300.0)
    token = os.environ.get("MAUS_TOKEN")
    if token:
        return httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {token}"}, timeout=timeout
        )
    auth = OAuthClientProvider(
        server_url=SERVER_URL,
        client_metadata=OAuthClientMetadata(
            client_name="maus-export (BaselHack 2026)",
            redirect_uris=[REDIRECT_URI],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            scope="openid profile email offline_access",
        ),
        storage=storage,
        redirect_handler=_redirect_handler,
        callback_handler=_callback_handler,
    )
    return httpx2.AsyncClient(auth=auth, timeout=timeout)


PROTOCOL_VERSION = "2026-07-28"


class MausClient:
    """Calls dashboard tools over MCP streamable HTTP (protocol 2026-07-28).

    We speak JSON-RPC directly instead of using the SDK session: the SDK's
    strict result validation rejects this server's tools/list (missing
    cacheScope/ttlMs). The SDK's
    OAuth provider still handles sign-in and token refresh on the HTTP client.
    """

    def __init__(self, http: httpx2.AsyncClient, storage: FileTokenStorage):
        self.http = http
        self.storage = storage
        self._id = 0

    async def call(self, tool: str, **args: Any) -> Any:
        self._id += 1
        body = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": "tools/call",
            "params": {
                "name": tool,
                "arguments": {k: v for k, v in args.items() if v is not None},
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION
                },
            },
        }
        headers = {
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
            "Mcp-Method": "tools/call",
            "Mcp-Name": tool,
        }
        r = await self.http.post(SERVER_URL, json=body, headers=headers)
        msg = _parse_rpc(r.text, r.headers.get("content-type", ""))
        if "error" in msg:
            raise RuntimeError(f"{tool} failed: {msg['error']}")
        res = msg["result"]
        if res.get("isError"):
            raise RuntimeError(f"{tool} failed: {res.get('content')}")
        if res.get("structuredContent") is not None:
            return res["structuredContent"]
        return json.loads(
            "".join(c.get("text", "") for c in res.get("content", []))
        )

    def bearer_token(self) -> str | None:
        return os.environ.get("MAUS_TOKEN") or self.storage.access_token()


def _parse_rpc(text: str, content_type: str) -> dict[str, Any]:
    if "text/event-stream" in content_type:
        for line in text.splitlines():
            if line.startswith("data:"):
                msg = json.loads(line[5:].strip())
                if "result" in msg or "error" in msg:
                    return msg
        raise RuntimeError("no JSON-RPC response in event stream")
    return json.loads(text)


@asynccontextmanager
async def connect() -> AsyncIterator[MausClient]:
    storage = FileTokenStorage()
    async with _http_client(storage) as http:
        yield MausClient(http, storage)
