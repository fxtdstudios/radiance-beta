"""One place that decides whether Radiance may fetch model weights.

Several nodes download their weights on first use: Real-ESRGAN, HAT-L, SwinIR,
Depth Anything V2, DSINE. Between them that is 67 MB to 2.4 GB, and it used to
start the moment someone queued a graph — no prompt, no confirmation, and on a
metered or air-gapped machine no way to see it coming until the bandwidth was
already spent.

The gates were also inconsistent: `nodes/upscale` honoured an opt-*out*
(`RADIANCE_UPSCALE_OFFLINE=1`), while `nodes/vfx/multipass` had no gate at all.
Two mechanisms, one of them missing, both invisible from the node UI.

This module is the single decision point. The default is **ask first**: a
download is refused with an actionable message naming the file, the size and
where to put it, unless the operator has said yes.

Saying yes, in order of precedence:

    RADIANCE_ALLOW_DOWNLOADS=0   never download, even if something else says to
    RADIANCE_ALLOW_DOWNLOADS=1   allow every model download
    RADIANCE_UPSCALE_OFFLINE=1   legacy opt-out, still honoured for upscale

Studios that want the old always-download behaviour set
`RADIANCE_ALLOW_DOWNLOADS=1` once in the ComfyUI launch environment.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("radiance.consent")

ALLOW_ENV = "RADIANCE_ALLOW_DOWNLOADS"
LEGACY_UPSCALE_OFFLINE_ENV = "RADIANCE_UPSCALE_OFFLINE"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _flag(name: str) -> Optional[bool]:
    """Tri-state read of an environment flag: True, False, or unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return None


def downloads_allowed(*, legacy_offline_env: Optional[str] = None) -> bool:
    """True when the operator has consented to fetching model weights.

    *legacy_offline_env* names an older opt-out variable to keep honouring, so
    existing studio configs do not silently start downloading again.
    """
    explicit = _flag(ALLOW_ENV)
    if explicit is not None:
        return explicit

    if legacy_offline_env and _flag(legacy_offline_env) is True:
        return False       # the old opt-out said no; that still means no

    return False           # default: ask first


def refusal_message(
    what: str,
    size_mb: float | int | None = None,
    dest: str | None = None,
    url: str | None = None,
) -> str:
    """The message shown when a download is refused.

    Actionable rather than apologetic: it names what was wanted, how big it is,
    where to put it, and the one setting that changes the answer.
    """
    size = f" (~{size_mb} MB)" if size_mb else ""
    lines = [
        f"[Radiance] '{what}'{size} is not installed and automatic downloads are off.",
        f"           Set {ALLOW_ENV}=1 to allow Radiance to fetch model weights,",
        "           or install the file manually:",
    ]
    if url:
        lines.append(f"             from: {url}")
    if dest:
        lines.append(f"             to:   {dest}")
    return "\n".join(lines)


def require_consent(
    what: str,
    size_mb: float | int | None = None,
    dest: str | None = None,
    url: str | None = None,
    *,
    legacy_offline_env: Optional[str] = None,
) -> bool:
    """Check consent and log the refusal if there is none.

    Returns True when the caller may proceed with the download.
    """
    if downloads_allowed(legacy_offline_env=legacy_offline_env):
        return True
    logger.error(refusal_message(what, size_mb=size_mb, dest=dest, url=url))
    return False
