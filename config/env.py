"""Environment variable access layer — single point of env-var interaction.

Every module in Radiance should read env vars through this module rather than
calling os.environ directly. This makes env vars discoverable, testable, and
prevents scattered magic-string lookups.
"""
from __future__ import annotations

import os
from typing import MutableMapping, Optional


class ENV:
    """Well-known environment variable names used across Radiance.

    Usage:  value = os.environ.get(ENV.RADIANCE_TURBO_DECODER, "")
    """

    RADIANCE_TURBO_DECODER = "RADIANCE_TURBO_DECODER"
    RADIANCE_CACHE_SIZE = "RADIANCE_CACHE_SIZE"
    RADIANCE_LICENSE_KEY = "RADIANCE_LICENSE_KEY"
    OCIO = "OCIO"

    # DCC Integration Settings
    RADIANCE_NUKE_HOST = "RADIANCE_NUKE_HOST"
    RADIANCE_NUKE_PORT = "RADIANCE_NUKE_PORT"
    RADIANCE_MCP_HOST = "RADIANCE_MCP_HOST"
    RADIANCE_MCP_PORT = "RADIANCE_MCP_PORT"
    RADIANCE_HTTP_HOST = "RADIANCE_HTTP_HOST"
    RADIANCE_HTTP_PORT = "RADIANCE_HTTP_PORT"
    RADIANCE_COMFY_URL = "RADIANCE_COMFY_URL"
    RADIANCE_DCC_AUTH_TOKEN = "RADIANCE_DCC_AUTH_TOKEN"
    RADIANCE_DEV = "RADIANCE_DEV"

    # Internal flags that must be set before OpenCV/OpenMP-backed imports.
    KMP_DUPLICATE_LIB_OK = "KMP_DUPLICATE_LIB_OK"
    OPENCV_IO_ENABLE_OPENEXR = "OPENCV_IO_ENABLE_OPENEXR"


RUNTIME_ENV_DEFAULTS: dict[str, str] = {
    ENV.KMP_DUPLICATE_LIB_OK: "TRUE",
    ENV.OPENCV_IO_ENABLE_OPENEXR: "1",
}


def configure_runtime_environment(
    environ: Optional[MutableMapping[str, str]] = None,
) -> MutableMapping[str, str]:
    target = os.environ if environ is None else environ
    for key, value in RUNTIME_ENV_DEFAULTS.items():
        target.setdefault(key, value)
    return target


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def get_env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def get_nuke_host() -> str:
    return get_env(ENV.RADIANCE_NUKE_HOST, "127.0.0.1")


def get_nuke_port() -> int:
    return get_env_int(ENV.RADIANCE_NUKE_PORT, 1986)


def get_mcp_host() -> str:
    return get_env(ENV.RADIANCE_MCP_HOST, "127.0.0.1")


def get_mcp_port() -> int:
    return get_env_int(ENV.RADIANCE_MCP_PORT, 1987)


def get_http_host() -> str:
    return get_env(ENV.RADIANCE_HTTP_HOST, "127.0.0.1")


def get_http_port() -> int:
    return get_env_int(ENV.RADIANCE_HTTP_PORT, 7863)


def get_comfy_url() -> str:
    return get_env(ENV.RADIANCE_COMFY_URL, "http://127.0.0.1:8188")


def get_dcc_auth_token() -> str:
    return get_env(ENV.RADIANCE_DCC_AUTH_TOKEN, "")
