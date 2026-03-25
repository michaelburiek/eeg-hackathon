"""Custom EEG-AD pipeline components built on top of LaBraM."""

import os
import sys

# Make LaBraM submodule importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_labram_dir = os.path.join(_project_root, "LaBraM")
for _p in (_project_root, _labram_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Monkey-patch upstream Block.__init__ to handle init_values=None
# (upstream bug: `if init_values > 0` fails when init_values is None)
from modeling_finetune import Block as _Block

_orig_block_init = _Block.__init__


def _patched_block_init(self, *args, **kwargs):
    if "init_values" in kwargs and kwargs["init_values"] is None:
        kwargs["init_values"] = 0
    _orig_block_init(self, *args, **kwargs)


_Block.__init__ = _patched_block_init
