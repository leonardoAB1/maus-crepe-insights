"""MCP client for the Maus dashboard (https://dash.mausrobotics.com/mcp).

Auth: OAuth 2.1 with dynamic client registration + PKCE. The first run opens a
browser to sign in; tokens are cached in ~/.maus-export/auth.json (outside the repo).
Set MAUS_TOKEN to skip the browser flow and use a bearer token directly.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import parse_qs, urlparse

import anyio
import httpx2
from mcp.client.auth import OAuthClientProvider
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
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
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, key: str, value: dict[str, Any]) -> None:
        data = self._load()
        data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._load().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._save("tokens", tokens.model_dump(mode="json", exclude_none=True))

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._load().get("client")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._save("client", client_info.model_dump(mode="json", exclude_none=True))

    def access_token(self) -> str | None:
        return (self._load().get("tokens") or {}).get("access_token")


def _wait_for_callback() -> AuthorizationCodeResult:
    """Run a one-shot local HTTP server and return the ?code=&state= it receives."""
    result: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            qs = parse_qs(urlparse(self.path).query)
            if "code" in qs:
                result.update({k: v[0] for k, v in qs.items()})
                body = b"<h2>Signed in to the Maus dashboard. You can close this tab.</h2>"
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
    return AuthorizationCodeResult(code=result["code"], state=result.get("state"), iss=result.get("iss"))


async def _redirect_handler(url: str) -> None:
    print(f"\nOpen this URL to sign in to the Maus dashboard:\n  {url}\n")
    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()


async def _callback_handler() -> AuthorizationCodeResult:
    return await anyio.to_thread.run_sync(_wait_for_callback)


def _http_client(storage: FileTokenStorage) -> httpx2.AsyncClient:
    timeout = httpx2.Timeout(60.0, read=300.0)
    token = os.environ.get("MAUS_TOKEN")
    if token:
        return httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=timeout)
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


class MausClient:
    """Thin wrapper: call a dashboard tool and get its JSON result back."""

    def __init__(self, session: ClientSession, storage: FileTokenStorage):
        self.session = session
        self.storage = storage

    async def call(self, tool: str, **args: Any) -> Any:
        res = await self.session.call_tool(tool, {k: v for k, v in args.items() if v is not None})
        if getattr(res, "is_error", False):
            text = " ".join(getattr(c, "text", "") for c in res.content)
            raise RuntimeError(f"{tool} failed: {text}")
        if res.structured_content is not None:
            return res.structured_content
        text = "".join(getattr(c, "text", "") for c in res.content)
        return json.loads(text)

    def bearer_token(self) -> str | None:
        return os.environ.get("MAUS_TOKEN") or self.storage.access_token()


@asynccontextmanager
async def connect() -> AsyncIterator[MausClient]:
    storage = FileTokenStorage()
    async with _http_client(storage) as http:
        async with streamable_http_client(SERVER_URL, http_client=http) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield MausClient(session, storage)
