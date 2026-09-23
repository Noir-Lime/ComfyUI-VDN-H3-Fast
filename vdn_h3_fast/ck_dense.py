"""Comfy Kitchen dense INT8 provider for VDN's trained grouped windows.

Derived from the isolated VDN CK-window experiment; see NOTICE.
"""
from comfy.patcher_extension import WrappersMP
from comfy_kitchen.sage_attention import int8_attention

from .softmax_provider import KEY_V4

def install(model):
    def outer(executor, *args, **kwargs):
        guider = executor.class_obj
        original = guider.model_options
        options = dict(original)
        options["transformer_options"] = dict(original.get("transformer_options", {}))
        if KEY_V4 in options["transformer_options"]:
            raise RuntimeError("VDN H3 Fast cannot replace an existing VDN attention provider")

        def provider(native, q, k, v, *, kind, scale, square_aligned=False,
                     sink_rows=0, query_position_map=None):
            if q.device.type != "cuda" or q.shape[-1] != 128:
                return native()
            return int8_attention(
                q.permute(1, 0, 2).unsqueeze(0),
                k.permute(1, 0, 2).unsqueeze(0),
                v.permute(1, 0, 2).unsqueeze(0), scale=scale,
            ).squeeze(0).permute(1, 0, 2)

        options["transformer_options"][KEY_V4] = provider
        guider.model_options = options
        try:
            return executor(*args, **kwargs)
        finally:
            guider.model_options = original

    model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "vdn_fast_ck_dense", outer)
