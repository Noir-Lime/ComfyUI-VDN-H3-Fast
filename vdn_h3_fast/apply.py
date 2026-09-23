"""Modified for ComfyUI-VDN-H3-Fast; upstream provenance in NOTICE.

Applying the converted VDN adapters onto a cloned ModelPatcher.

The application machinery is a port of the ComfyUI-MiniMax-H3-Turbo node's
battle-tested paths (bypass injection with a memory-frugal additive LoRA, merge for
quantized-fused weights, and the e-grid re-injection for curve/pruned bases' adaln
updates), adapted to VDN's already-converted ComfyUI key space.
"""
import logging
import math
import os

import torch
import torch.nn.functional as F

import comfy.ldm.minimax.model
import folder_paths
import comfy.lora
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter

from .spec import vdn_folders

_log = logging.getLogger("comfy.vdn")

def _turbo_grid_path():
    # A standalone copy lives with the VDN stage; an installed Turbo node is
    # accepted for existing workflows. Never silently omit pruned AdaLN deltas.
    for root in vdn_folders():
        path = os.path.join(root, "h3_silu_temb_grid.safetensors")
        if os.path.isfile(path):
            return path
    for root in folder_paths.get_folder_paths("custom_nodes"):
        path = os.path.join(root, "ComfyUI-MiniMax-H3-Turbo", "h3_silu_temb_grid.safetensors")
        if os.path.isfile(path):
            return path
    return os.path.join(vdn_folders()[0], "h3_silu_temb_grid.safetensors")



class _FrugalLoRA(comfy.weight_adapter.LoRAAdapter):
    """LoRA bypass adapter with an in-place additive path (ported from the
    MiniMax-H3-Turbo node): accumulates up(down(x)) * scale straight into the base
    output instead of allocating the full-size projection three times."""

    def _cast_pair(self, down, up, x):
        """The per-call .to(dtype) casts of the LoRA pair, cached per (dtype,
        device): bypass_forward runs once per module per step, and the cast result
        is identical every time. Bypass mode only; merge never sees this path."""
        key = (x.dtype, x.device)
        cache = getattr(self, "_cast_cache", None)
        if cache is None:
            cache = self._cast_cache = {}
        hit = cache.get(key)
        if hit is None:
            hit = (down.to(dtype=x.dtype), up.to(dtype=x.dtype))
            cache[key] = hit
        return hit

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        base_out = org_forward(x, *args, **kwargs)
        if getattr(self, "is_conv", False):
            return super().bypass_forward(org_forward, x, *args, **kwargs)
        up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
        rank = down.shape[0]
        scale = (alpha / rank if alpha is not None else 1.0) \
            * getattr(self, "multiplier", 1.0)
        down, up = self._cast_pair(down, up, x)
        return base_out.add_(torch.nn.functional.linear(
            torch.nn.functional.linear(x, down), up), alpha=scale)


def _int8_fused_fc2(dm, modules):
    """MLP fc2 modules riding ComfyUI's fused int8 matmul: their fused forward reads
    linear.weight directly and never calls the module forward, so a bypass hook would
    silently drop the LoRA. Those must go through the merge/weight-function path.
    (Ported from the MiniMax-H3-Turbo node.)"""
    fused = []
    for m in modules:
        if not m.endswith(".mlp.fc2"):
            continue
        try:
            w = comfy.utils.get_attr(dm, m + ".weight")
        except Exception:
            continue
        if (getattr(w, "_layout_cls", None) == "TensorWiseINT8Layout"
                and not getattr(getattr(w, "_params", None), "transposed", False)):
            fused.append(m)
    return fused


def apply_adapters(new_model, converted_by_name, strength, mode, verbose=False):
    """converted_by_name: {adapter_name: {comfy_module: (A, B, scale)}}. Bypass is the
    sharp default; merge is the low-VRAM/quantized-friendly path. `strength` is a
    float applied to every adapter, or {adapter_name: float} for per-adapter
    control (missing names default to 1.0). Returns a report dict."""
    per_name = strength if isinstance(strength, dict) else None
    dm = new_model.get_model_object("diffusion_model")
    pruned = _is_pruned_base(dm)
    report = {}
    all_hooks = []
    for name, converted in converted_by_name.items():
        s = per_name.get(name, 1.0) if per_name is not None else strength
        modules = sorted(converted.keys())
        lora = {}
        for path, (a, b, scale) in converted.items():
            lora[path + ".lora_A.weight"] = a.contiguous()
            lora[path + ".lora_B.weight"] = b.contiguous()
            lora[path + ".alpha"] = torch.tensor(scale * a.shape[0])
        key_map = {m: f"diffusion_model.{m}.weight" for m in modules}
        loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
        sd_keys = set(new_model.model.state_dict().keys())

        if mode == "merge":
            if set(loaded) != set(key_map.values()):
                raise RuntimeError("VDN adapter conversion omitted a requested weight")
            adaln = [m for m in modules if pruned and _is_adaln(m)]
            adaln_keys = {key_map[m] for m in adaln}
            mergeable = {k: v for k, v in loaded.items() if k not in adaln_keys}
            if adaln:
                # Keep the full-width Turbo delta without bypass-rounding all
                # backbone projections or modifying the collapsed weights.
                _inject_adaln_egrid(new_model, dm, lora, adaln, s)
            n = len(new_model.add_patches(mergeable, s))
            if n != len(mergeable):
                raise RuntimeError("VDN merge omitted a requested backbone weight")
            report[name] = f"{n} weights merged, {len(adaln)} pruned AdaLN reinjected; none skipped"
            continue

        backbone = [m for m in modules if not _is_adaln(m)]
        adaln = [m for m in modules if _is_adaln(m)]
        fc2_fused = set(_int8_fused_fc2(dm, backbone))
        bypass_mods = [m for m in backbone if m not in fc2_fused]

        n = 0
        if bypass_mods:
            n += _bypass(new_model, loaded, key_map, bypass_mods, sd_keys, s,
                         all_hooks)
        if fc2_fused:
            n += len(new_model.add_patches(
                {k: v for k, v in loaded.items()
                 if k in {key_map[m] for m in fc2_fused}}, s))
        if adaln:
            if pruned:
                grid_path = _turbo_grid_path()
                if not os.path.exists(grid_path):
                    # Curve bases cannot take the adaln deltas through the normal
                    # paths (that is what the per-block '[96768, 8]' reshape errors
                    # were); skipping them is near-visual-neutral per community
                    # testing. One warning instead of 50 ERROR lines.
                    _log.warning("[vdn] pruned base: %d adaln adapters skipped "
                                 "(needs the silu-temb grid from the "
                                 "ComfyUI-MiniMax-H3-Turbo node at %s)",
                                 len(adaln), grid_path)
                    report[name] = (f"{n} adapters ({len(bypass_mods)} bypass, "
                                    f"{len(fc2_fused)} int8-fc2 merged, "
                                    f"{len(adaln)} adaln SKIPPED)")
                    continue
                _inject_adaln_egrid(new_model, dm, lora, adaln, s)
                n += len(adaln)
            else:
                n += _bypass(new_model, loaded, key_map, adaln, sd_keys, s,
                             all_hooks)
        report[name] = (f"{n} adapters ({len(bypass_mods)} bypass, "
                        f"{len(fc2_fused)} int8-fc2 merged, {len(adaln)} adaln)")
    _install_injection(new_model, all_hooks)
    return report


def _is_adaln(module):
    return module.endswith(".adaln_proj.linear")


def _is_pruned_base(dm):
    """Curve/pruned bases collapse adaln_proj.linear to a tiny t-feature input
    (the [96768, 8] weights); the trained weight takes the full silu(t_emb)
    width. The model flag alone missed some pruned checkpoints (issues #3/#5),
    so the weight shape is the reliable tell."""
    if getattr(dm, "use_adaln_curves", False):
        return True
    try:
        w = comfy.utils.get_attr(dm, "blocks.0.adaln_proj.linear.weight")
        return w.dim() == 2 and w.shape[-1] < 64
    except Exception:
        return False


def _bypass(new_model, loaded, key_map, modules, sd_keys, strength, hooks):
    manager = comfy.weight_adapter.BypassInjectionManager()
    n = 0
    for module in modules:
        key = key_map[module]
        adapter = loaded.get(key)
        if adapter is None or key not in sd_keys:
            continue
        if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            adapter = _FrugalLoRA(adapter.loaded_keys, adapter.weights)
        elif not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            continue
        manager.add_adapter(key, adapter, strength=strength)
        n += 1
    manager.create_injections(new_model.model)
    hooks.extend(manager.hooks)
    return n


def _install_injection(new_model, hooks):
    """All bypass hooks go through ONE PatcherInjection whose eject unwinds in
    reverse. ComfyUI applies injection sets in list order on load and on unload;
    with two stacked adapter sets (default + turbo), forward-order eject restores
    a stale hook as module.forward, and the next load captures that hook as its
    own "original" -- infinite self-recursion on the second run (observed as
    RecursionError after a model reload). LIFO eject always restores the true
    forward, so load/unload cycles are stable.

    STACK-PROOFING: every Apply-VDN run clones the patcher but every clone shares
    ONE inner model, and ComfyUI ejects a clone's injections only when that clone
    is UNLOADED -- on a big-VRAM card nothing unloads between runs, so re-running
    with any widget changed (e.g. flipping lora_mode) would otherwise stack
    ANOTHER full set of bypass hooks on the same modules: 2x, 3x the LoRA delta
    per rerun, progressively grainy/fried output (merge is immune -- weight
    patches go through backup/restore). The live hook set is tracked on the
    shared inner model and ejected before a new set goes in."""
    if not hooks:
        return
    owner = new_model.model      # shared by every clone of this model

    def inject_all(model_patcher):
        old = getattr(owner, "_vdn_live_hooks", None)
        if old:
            for hook in reversed(old):
                hook.eject()
        for hook in hooks:
            hook.inject()
        owner._vdn_live_hooks = hooks

    def eject_all(model_patcher):
        for hook in reversed(hooks):
            hook.eject()
        if getattr(owner, "_vdn_live_hooks", None) is hooks:
            owner._vdn_live_hooks = None

    injection = comfy.patcher_extension.PatcherInjection(
        inject=inject_all, eject=eject_all)
    new_model.set_injections("vdn_lora", [injection])


# ------------------------------------------------------- pruned-base adaln path --

_EGRID = None


def _egrid():
    global _EGRID
    if _EGRID is None:
        path = _turbo_grid_path()
        if not os.path.exists(path):
            raise RuntimeError(
                "This VDN adapter updates the adaln projections, but the loaded base "
                "is a pruned (curve) MiniMax-H3 whose adaln weights were collapsed. "
                "Re-injection needs the silu(t_emb) grid bundled with the "
                "ComfyUI-MiniMax-H3-Turbo node; expected it at: " + path)
        _EGRID = comfy.utils.load_torch_file(path)["silu_t_emb_grid"]
    return _EGRID


def _interp_egrid(unique_t, e, device, dtype):
    # Same FP32 positions/interpolation as native MiniMaxH3Model._forward.
    e = e.to(device=device, dtype=torch.float32)
    t = torch.tensor(unique_t, dtype=torch.float32, device=device)
    pos = t.clamp(0.0, 1.0) * (e.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=e.shape[0] - 2)
    return torch.lerp(e[i0], e[i0 + 1], (pos - i0).unsqueeze(1)).to(dtype)


def _unique_t(timestep, shift_v, shift_a, payload):
    """Native unmasked layout only. In particular, AddGuide audio is a row.

    Do not guess from reference types: keyframe audio and ref-video audio also
    contribute. Masked per-token timesteps require a separate verified contract.
    """
    layout = payload.get("layout")
    if layout is None:
        raise RuntimeError("VDN AdaLN requires the native prebuilt PackedLayout")
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - comfy.ldm.minimax.model.time_shift_sigma(sigma_v, shift_v, shift_a))
    tv_cond = max(t_v, float(payload.get("visual_cond_noise_aug", 0.999)))
    ta_cond = max(t_a, float(payload.get("audio_cond_noise_aug", 1.0)))
    seg_t = dict(text=t_v, video=t_v, audio=t_a, cond=tv_cond,
                 ref_img=tv_cond, cond_audio=ta_cond, ref_audio=ta_cond)
    return sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments})


def _make_adaln_forward(base, a, b, shared):
    """Full-width delta at exact current timestep labels, not nearest-grid lookup.

    The original module tree and its offload ownership stay untouched. Missing or
    mismatched rows fail, rather than silently omitting/padding the adapter.
    """
    def forward(t_emb):
        st, expected = shared.get("silu_temb"), shared.get("curve_coords")
        if (base.apply_silu or st is None or expected is None
                or expected.shape != t_emb.shape or not torch.equal(expected, t_emb)
                or st.shape[0] != t_emb.shape[0]):
            raise RuntimeError("VDN AdaLN timestep rows differ from the native curve; refusing skipped delta")
        x = base.linear(t_emb)
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(st.float(), a.to(x.device, torch.float32)),
                             b.to(x.device, torch.float32))
            x = (x.float() + delta).to(x.dtype)
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)
    return forward


def _inject_adaln_egrid(new_model, dm, lora, adaln_modules, strength):
    e = _egrid()
    shared = {}
    tt = getattr(dm, "adaln_t_table", None)
    if tt is None or tt.ndim != 2 or tt.shape[0] < 2 or e.ndim != 2 or e.shape[0] < 2:
        raise RuntimeError("VDN AdaLN needs valid native curve and full-width embedding grids")

    def wrap(executor, *args, **kwargs):
        shared.clear()
        if kwargs.get("denoise_mask") is not None or kwargs.get("audio_denoise_mask") is not None:
            raise RuntimeError("VDN AdaLN masked timestep reinjection is not validated")
        ts = args[1] if len(args) > 1 else kwargs.get("timestep")
        ctx = args[2] if len(args) > 2 else kwargs.get("context")
        opts = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        shift_v = float(opts.get("minimax_h3_sigma_shift_video", dm.sigma_shift_video))
        shift_a = float(opts.get("minimax_h3_sigma_shift_audio", dm.sigma_shift_audio))
        labels = _unique_t(ts, shift_v, shift_a, kwargs.get("minimax_payload") or {})
        shared["silu_temb"] = _interp_egrid(labels, e, ctx.device, torch.float32)
        shared["curve_coords"] = _interp_egrid(labels, tt, ctx.device, torch.float32)
        try:
            return executor(*args, **kwargs)
        finally:
            shared.clear()

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "vdn_adaln", wrap)
    for name in adaln_modules:
        a = lora[name + ".lora_A.weight"]
        alpha = float(lora[name + ".alpha"])
        b = lora[name + ".lora_B.weight"] * (strength * alpha / a.shape[0])
        if a.shape[1] != e.shape[1]:
            raise RuntimeError("VDN AdaLN adapter input width does not match embedding grid")
        key = "diffusion_model." + name.rsplit(".linear", 1)[0]
        new_model.add_object_patch(
            key + ".forward",
            _make_adaln_forward(new_model.get_model_object(key), a, b, shared))
