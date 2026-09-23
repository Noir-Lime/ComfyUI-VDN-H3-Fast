"""Drag-and-drop MODEL-in/MODEL-out optimized VDN-H3 node.

Modified from xmarre/ComfyUI-VDN-H3-Plus; see NOTICE.
"""
import logging

import comfy.model_management

from . import policy, spec
from .adapters import convert_adapter
from .apply import apply_adapters
from .ck_dense import install as install_ck_dense
from .hybrid import VDNState, apply_vdn
from .retained import RuntimeLinearBranch

_log = logging.getLogger("comfy.vdn_fast")


def _validate_branch_shapes(path, branches, cfg, hidden, heads, head_dim):
    """Validate every enabled trained tensor on every block against the loaded base."""
    linear_dim = cfg["linear_head_dim"]
    if linear_dim != head_dim:
        raise RuntimeError(
            f"{path}: checkpoint linear_head_dim={linear_dim}, but this Comfy port shares "
            f"the base Q/K/V whose head_dim={head_dim}. The official architecture has "
            "no projection between them; refusing an incompatible base/checkpoint pair.")

    expected = {
        "to_out_linear.weight": (hidden, heads * linear_dim),
        "beta_proj.weight": (heads, hidden),
        "norm.weight": (linear_dim,),
        "alpha.A_log": (heads,),
        "alpha.dt_bias": (heads * linear_dim,),
        "alpha.down.weight": (linear_dim, hidden),
        "alpha.up.weight": (heads * linear_dim, linear_dim),
        "output_gate.down.weight": (linear_dim, hidden),
        "output_gate.up.weight": (heads * linear_dim, linear_dim),
        "output_gate.up.bias": (heads * linear_dim,),
    }
    if cfg["enable_softmax_gate"]:
        expected.update({
            "softmax_gate.up.weight": (heads, hidden),
            "softmax_gate.up.bias": (heads,),
        })
    channels = heads * linear_dim
    for target in cfg["short_conv"]:
        expected[f"short_conv.{target}_sp.weight"] = (channels, 1, 5, 5)
        expected[f"short_conv.{target}_tm.weight"] = (channels, 1, 5)

    errors = []
    for index, weights in enumerate(branches):
        for key, shape in expected.items():
            tensor = weights.get(key)
            if tensor is None:
                errors.append(f"block {index}: missing {key}")
            elif tuple(tensor.shape) != shape:
                errors.append(
                    f"block {index}: {key} has {tuple(tensor.shape)}, expected {shape}")
    if errors:
        preview = "; ".join(errors[:12])
        if len(errors) > 12:
            preview += f"; ... and {len(errors) - 12} more"
        raise RuntimeError(f"VDN checkpoint/base shape mismatch in {path}: {preview}")


class VDNH3Fast:
    @classmethod
    def INPUT_TYPES(cls):
        names = spec.list_vdn_checkpoints()
        return {"required": {
            "model": ("MODEL",),
            "vdn_checkpoint": (names or ["<place a VDN stage under models/vdn>"],),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patch/video"
    DESCRIPTION = ("Replace one native MiniMax-H3 MODEL with the released VDN 8-step "
                   "hybrid attention and tested fast kernels. Not a PDD/TM accelerator.")

    def apply(self, model, vdn_checkpoint):
        if any(key.endswith(".attn.forward") and getattr(patch, "_vdn_forward", False)
               for key, patch in model.object_patches.items()):
            raise RuntimeError("VDN is already applied. Connect VDN H3 Fast directly to the base H3 MODEL.")
        dm = model.get_model_object("diffusion_model")
        blocks = getattr(dm, "blocks", None)
        if not blocks or not hasattr(blocks[0].attn, "qkv_proj"):
            raise ValueError("VDN H3 Fast requires a native MiniMax-H3 MODEL")
        path = spec.resolve_vdn_checkpoint(vdn_checkpoint)
        branch_path = policy.select_branch_file(path, prefer_int8=True)
        cfg, weights, adapters, _ = policy.load_vdn_checkpoint(path, prefer_int8=True)
        if len(weights) != len(blocks):
            raise ValueError("VDN stage block count differs from the MiniMax-H3 backbone")
        if "default" not in adapters or "turbo" not in adapters:
            raise ValueError("VDN H3 Fast needs the stage's default and turbo adapters")
        heads, head_dim = blocks[0].attn.heads, blocks[0].attn.head_dim
        if cfg["linear_head_dim"] != head_dim:
            raise ValueError("VDN stage head dimension differs from the MiniMax-H3 backbone")
        _validate_branch_shapes(path, weights, cfg, dm.hidden_size, heads, head_dim)
        branches = [RuntimeLinearBranch(w, heads, head_dim,
                    delta_rule=cfg["delta_rule"], bridge=cfg["bridge"],
                    a_fp32=cfg["a_fp32"], short_conv=cfg["short_conv"],
                    enable_text_state=cfg["enable_text_state"]) for w in weights]
        for branch in branches:
            branch.fuse_epilogue = True
        state = VDNState(vdn_checkpoint, dict(cfg), branches, heads, head_dim,
                         retain_buffers=False)
        state.batch_nonlocal = True
        new_model = model.clone()
        apply_vdn(new_model, state)
        install_ck_dense(new_model)
        converted = {name: convert_adapter(*adapters[name]) for name in ("default", "turbo")}
        apply_adapters(new_model, converted, 1.0, mode="merge")
        _log.info("[vdn-fast] %s: %d blocks, branch=%s, CK INT8 windows, fused helpers",
                  vdn_checkpoint, len(branches), branch_path)
        return (new_model,)


NODE_CLASS_MAPPINGS = {"VDNH3Fast": VDNH3Fast}
NODE_DISPLAY_NAME_MAPPINGS = {"VDNH3Fast": "VDN H3 Fast"}
