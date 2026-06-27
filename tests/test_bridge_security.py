"""Security tests for the DCC bridge AST validator and loopback gate."""
import pytest

dcc = pytest.importorskip("radiance.nodes.pipeline.dcc")


SAFE = [
    "1 + 2",
    "[x*2 for x in range(3)]",
    "json.dumps({'a': 1})",
    "max(1, 2, 3)",
    "y = 5",
]
DANGEROUS = [
    "__import__('os').system('id')",
    "open('/etc/passwd').read()",
    "().__class__.__bases__",
    "eval('1+1')",
    "exec('x=1')",
    "import os",
    "lambda: 1",
    "globals()",
    "(1).__class__",
    "json.__class__",
]


@pytest.mark.parametrize("code", SAFE)
def test_validator_allows_safe(code):
    ok, reason = dcc._validate(code)
    assert ok, f"should allow {code!r}: {reason}"


@pytest.mark.parametrize("code", DANGEROUS)
def test_validator_blocks_dangerous(code):
    ok, reason = dcc._validate(code)
    assert not ok, f"should block {code!r}"


def test_remote_bridge_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RADIANCE_ALLOW_REMOTE_BRIDGE", raising=False)
    assert dcc._remote_bridge_allowed() is False
    monkeypatch.setenv("RADIANCE_ALLOW_REMOTE_BRIDGE", "1")
    assert dcc._remote_bridge_allowed() is True
