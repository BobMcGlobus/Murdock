"""Web UI sign-in and API token management."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from .auth import SESSION_COOKIE, SESSION_MAX_AGE, VIA_SESSION, ApiAuth

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _auth(request: Request) -> ApiAuth:
    return request.app.state.auth


def _set_session(request: Request, response: Response, session_id: str) -> None:
    # Behind a TLS-terminating proxy the request itself is plain http.
    https = (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto") == "https"
    )
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="strict",
        secure=https,
        path="/",
    )


class AuthStatus(BaseModel):
    authenticated: bool
    via: Optional[str] = None


class LoginIn(BaseModel):
    token: str


class TokenOut(BaseModel):
    token: str
    pinned: bool


@router.get("/status", response_model=AuthStatus)
async def auth_status(request: Request) -> AuthStatus:
    via = _auth(request).authorize(request.scope).via
    return AuthStatus(authenticated=via is not None, via=via)


@router.post("/login", response_model=AuthStatus)
async def login(body: LoginIn, request: Request, response: Response) -> AuthStatus:
    auth = _auth(request)
    if not auth.token_matches(body.token):
        raise HTTPException(status_code=401, detail="invalid API token")
    _set_session(request, response, auth.create_session())
    return AuthStatus(authenticated=True, via=VIA_SESSION)


@router.post("/logout", response_model=AuthStatus)
async def logout(request: Request, response: Response) -> AuthStatus:
    _auth(request).revoke_session(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return AuthStatus(authenticated=False)


@router.get("/token", response_model=TokenOut)
async def get_token(request: Request) -> TokenOut:
    auth = _auth(request)
    return TokenOut(token=auth.token, pinned=auth.pinned)


@router.post("/token/rotate", response_model=TokenOut)
async def rotate_token(request: Request, response: Response) -> TokenOut:
    """Replace the API token. The calling browser stays signed in."""
    auth = _auth(request)
    try:
        token = auth.rotate()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if request.state.auth == VIA_SESSION:
        _set_session(request, response, auth.create_session())
    return TokenOut(token=token, pinned=False)
