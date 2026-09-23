"""VDN H3 Fast: optimized VDN stage for native ComfyUI MiniMax-H3 models."""
import os
import sys

_PKG = os.path.dirname(__file__)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from vdn_h3_fast.compiler_guard import install_layout_guard
from vdn_h3_fast.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

install_layout_guard()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
