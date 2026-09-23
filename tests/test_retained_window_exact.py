from __future__ import annotations

import pytest
import torch

# Modified for the Fast namespace and nonlocal batching tests; see NOTICE.
from vdn_h3_fast import retained, window
from vdn_h3_fast.softmax_provider import KEY_V4


@pytest.mark.parametrize('video_start,frames,tail', [(3, 17, 2), (0, 17, 0), (3, 1, 0)])
def test_batched_nonlocal_matches_separate(video_start, frames, tail):
    torch.manual_seed(31)
    tokens, heads, dim = 7, 2, 8
    video_end = video_start + frames * tokens
    seq = video_end + tail
    q, k, v = [torch.randn(seq, heads, dim) for _ in range(3)]
    calls = []
    def provider(native, q, k, v, **kwargs):
        calls.append(kwargs['kind'])
        return native()
    args = (q, k, v, video_start, video_end, frames, tokens,
            window.window_bounds(frames, 1, 3), dim ** -.5)
    options = {KEY_V4: provider}
    ref = retained.window_softmax_grouped_runtime(*args, anchor_frames='both', transformer_options=options)
    local_calls = calls.count('local')
    calls.clear()
    out = retained.window_softmax_grouped_runtime(*args, anchor_frames='both', transformer_options=options, batch_nonlocal=True)
    assert torch.allclose(ref, out, atol=1e-6, rtol=1e-6)
    assert calls.count('local') == local_calls
    assert calls.count('global') == 1
    assert calls.count('anchor') == 0


def test_retained_windows_do_not_forward_transformer_options(monkeypatch):
    """Retained grouped windows must preserve released exact-SDPA semantics."""
    original = window._sdpa
    seen_options = []

    def recording_sdpa(q, k, v, scale, transformer_options=None):
        seen_options.append(transformer_options)
        return original(q, k, v, scale, transformer_options)

    monkeypatch.setattr(window, "_sdpa", recording_sdpa)

    torch.manual_seed(520)
    video_start, tokens, frames, heads, dim = 3, 4, 6, 2, 8
    video_end = video_start + frames * tokens
    seq = video_end + 2
    q = torch.randn(seq, heads, dim)
    k = torch.randn(seq, heads, dim)
    v = torch.randn(seq, heads, dim)
    bounds = window.window_bounds(frames, 1, 3)
    override = {"optimized_attention_override": object(), "sentinel": True}

    got = retained.window_softmax_grouped_runtime(
        q,
        k,
        v,
        video_start,
        video_end,
        frames,
        tokens,
        bounds,
        dim ** -0.5,
        anchor_frames="both",
        transformer_options=override,
    )

    assert seen_options
    assert all(option is None for option in seen_options)

    want = window.window_softmax_grouped(
        q,
        k,
        v,
        video_start,
        video_end,
        frames,
        tokens,
        bounds,
        dim ** -0.5,
        anchor_frames="both",
        transformer_options=None,
    )
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)
