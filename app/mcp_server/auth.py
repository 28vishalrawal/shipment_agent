"""Bearer authentication for the MCP mount.

NemoClaw attaches exactly one bearer credential per registered server and keeps
the raw value on the host, outside the sandbox, resolving it at egress. This
middleware is the other end of that arrangement: it authenticates the sandbox,
not a user, so it is deliberately separate from `app.api.security`, which issues
and validates per-principal JWTs with scopes.

Keeping them separate matters. If the MCP mount accepted app JWTs, a leaked
sandbox credential would carry whatever scopes that token had; as written it
carries exactly one capability — call the published read-only tools.
"""
from __future__ import annotations

import hmac
import logging

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("mcp.auth")

_UNAUTHORIZED = {
    "type": "http.response.start",
    "status": 401,
    "headers": [
        (b"content-type", b"application/json"),
        (b"www-authenticate", b"Bearer"),
    ],
}
_BODY = b'{"error":"unauthorized"}'


class BearerAuthMiddleware:
    """Reject any request to the wrapped app without the configured token."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        if not token:
            raise ValueError("MCP bearer token must not be empty")
        self._app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if not self._authorized(scope):
            # No detail about which part failed: a caller probing the endpoint
            # learns only that it is protected.
            logger.warning("mcp_auth_rejected path=%s", scope.get("path"))
            await send(_UNAUTHORIZED)
            await send({"type": "http.response.body", "body": _BODY})
            return

        await self._app(scope, receive, send)

    def _authorized(self, scope: Scope) -> bool:
        for key, value in scope.get("headers", []):
            if key.lower() != b"authorization":
                continue
            try:
                decoded = value.decode("latin-1")
            except UnicodeDecodeError:
                return False
            prefix, _, presented = decoded.partition(" ")
            if prefix.lower() != "bearer" or not presented:
                return False
            # Constant-time so response latency does not leak a prefix match.
            return hmac.compare_digest(presented.strip(), self._token)
        return False