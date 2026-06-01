"""Regression coverage for #3256 / #3263: the global model.context_length cap
must apply ONLY when the session model equals model.default.

Bug: a global `model.context_length` (set for the default model, e.g. 232000)
was applied to EVERY model, silently shrinking a non-default model's real
context window (e.g. a 1M-context variant). The fix scopes the override to
`model.default` only, in `_resolve_context_length_for_session_model`.

This test drives the real resolver with a monkeypatched
`agent.model_metadata.get_model_context_length` that records the
`config_context_length` it receives, so we assert the gating without needing
the real agent metadata catalog.
"""
import sys
import types
from pathlib import Path as _Path


def _install_fake_get_model_context_length(monkeypatch, recorder):
    """Install a fake get_model_context_length into a stand-in agent.model_metadata."""
    mod = types.ModuleType("agent.model_metadata")

    def _fake(model, base_url="", config_context_length=None, provider="", custom_providers=None):
        recorder["model"] = model
        recorder["config_context_length"] = config_context_length
        # Pretend the real per-model metadata window is 1,000,000 unless the
        # caller forced a config cap, in which case honor the cap (mirrors the
        # real helper's contract).
        if config_context_length:
            return int(config_context_length)
        return 1_000_000

    mod.get_model_context_length = _fake
    # Ensure a parent `agent` package exists so `from agent.model_metadata import ...` resolves.
    if "agent" not in sys.modules:
        agent_pkg = types.ModuleType("agent")
        agent_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "agent", agent_pkg)
    monkeypatch.setitem(sys.modules, "agent.model_metadata", mod)


def _resolver():
    import api.routes as routes
    return routes._resolve_context_length_for_session_model


def test_global_cap_applies_to_default_model(monkeypatch):
    """When the session model IS model.default, the global cap is passed through."""
    import api.config as config
    rec = {}
    _install_fake_get_model_context_length(monkeypatch, rec)
    monkeypatch.setattr(
        config, "get_config",
        lambda *a, **k: {"model": {"default": "claude-opus-4.8", "context_length": 232000}},
    )
    result = _resolver()("claude-opus-4.8")
    assert rec["config_context_length"] == 232000, "default model must receive the global cap"
    assert result == 232000


def test_global_cap_NOT_applied_to_non_default_model(monkeypatch):
    """When the session model is NOT model.default, the global cap is dropped so
    the model's real (larger) window is used — the core #3256/#3263 fix."""
    import api.config as config
    rec = {}
    _install_fake_get_model_context_length(monkeypatch, rec)
    monkeypatch.setattr(
        config, "get_config",
        lambda *a, **k: {"model": {"default": "claude-opus-4.8", "context_length": 232000}},
    )
    result = _resolver()("claude-opus-4.7-1m-internal")
    assert rec["config_context_length"] is None, (
        "non-default model must NOT receive the global cap (it would clobber real metadata)"
    )
    assert result == 1_000_000, "non-default model should resolve to its real 1M window, not the 232K cap"


def test_no_default_configured_still_applies_cap(monkeypatch):
    """If model.default is unset, the cap applies (backward-compatible)."""
    import api.config as config
    rec = {}
    _install_fake_get_model_context_length(monkeypatch, rec)
    monkeypatch.setattr(
        config, "get_config",
        lambda *a, **k: {"model": {"context_length": 200000}},
    )
    result = _resolver()("some-model")
    assert rec["config_context_length"] == 200000
    assert result == 200000


def test_empty_model_returns_zero(monkeypatch):
    rec = {}
    _install_fake_get_model_context_length(monkeypatch, rec)
    assert _resolver()("") == 0
    assert _resolver()(None) == 0


# --- #3263 dual-gate MUST-FIX invariants (Codex regression gate, v0.51.192) ---
# These pin the two consistency fixes applied after the gate found that the
# default-only guard dropped the stale cap but didn't (a) recompute a persisted
# stale context_length, or (b) rescale the terminal SSE threshold. Both live
# deep inside _run_agent_streaming, so we pin them at the source-structure level
# (the live-snapshot path already had behavioral coverage; these guard the two
# sibling paths from silently regressing back to the stale value).
_STREAMING_SRC = (_Path(__file__).resolve().parent.parent / "api" / "streaming.py").read_text(encoding="utf-8")


def test_persistence_fallback_also_runs_when_skip_cc_cl():
    """The per-turn persistence fallback must recompute the real cap when the
    stale compressor cap was skipped — not only when context_length is falsy.
    Otherwise a previously-persisted stale 232K survives forever."""
    assert "(not getattr(s, 'context_length', 0)) or _skip_cc_cl:" in _STREAMING_SRC, (
        "persistence fallback gate must also fire on _skip_cc_cl (#3263 MUST-FIX 1)"
    )


def test_persistence_rescales_threshold_when_cap_skipped():
    """When the stale cap is skipped and the real cap recomputed, the persisted
    threshold_tokens must be rescaled to the real cap (or cleared), so a reload
    matches the live snapshot."""
    assert "if _skip_cc_cl:" in _STREAMING_SRC
    assert "s.threshold_tokens = int(_orig_thresh * _real_cap / _orig_cap)" in _STREAMING_SRC, (
        "persistence path must rescale threshold_tokens to the real cap (#3263 MUST-FIX 2)"
    )


def test_sse_done_payload_rescales_threshold_when_cap_dropped():
    """The terminal SSE usage payload must rescale threshold_tokens when it
    dropped the stale compressor cap, so the indicator doesn't revert on stream
    end (messages.js overwrites S.lastUsage with this payload)."""
    assert "_dropped_stale_cap_sse" in _STREAMING_SRC
    assert "usage['threshold_tokens'] = int(_orig_cc_thresh_sse * _fb_cl / _orig_cc_cl_sse)" in _STREAMING_SRC, (
        "SSE done payload must rescale threshold_tokens to the resolved window (#3263 MUST-FIX 3)"
    )
