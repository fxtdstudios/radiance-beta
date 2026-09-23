"""Sampler speed defects found in the 3.5 live logs.

1. cfg 1.0 was replaced by the architecture's base CFG in the Auto preset
   whenever model_meta was absent, so a turbo / distilled checkpoint ran an
   extra unconditional forward pass on every step (and the wrong look).
2. Dynamic CFG boosted cfg 1.0 to 1.2 early on, switching that pass on.
3. Every stage called load_model_gpu() before sample_custom, which loads the
   model again itself; the live log shows each model "prepared" twice.
4. log_tensor built its stats (float copy + 4 GPU syncs) even with DEBUG off.
5. gc.collect() + empty_cache() before sampling cost ~0.5 s per run.
"""
import inspect
import json
import logging
import re
from unittest.mock import MagicMock

import torch

from _sampler_harness import run_sampler
import radiance.sampler_utils as su
from radiance.nodes.generate.sampler import RadianceSamplerPro


def test_auto_keeps_cfg_1_without_model_meta():
    _, rec = run_sampler(preset="Auto", model_type="z_image", cfg=1.0, steps=8)
    assert all(c["cfg"] == 1.0 for c in rec.calls)


def test_auto_applies_base_cfg_when_model_meta_names_the_checkpoint():
    meta = json.dumps({"arch": "z_image", "unet_file": "z_image_bf16.safetensors"})
    _, rec = run_sampler(preset="Auto", model_type="auto", model_meta=meta, cfg=1.0, steps=8)
    assert rec.calls[0]["cfg"] == 4.0


def test_auto_keeps_turbo_at_cfg_1_with_model_meta():
    meta = json.dumps({"arch": "z_image", "unet_file": "z_image_turbo_bf16.safetensors"})
    _, rec = run_sampler(preset="Auto", model_type="auto", model_meta=meta, cfg=1.0, steps=8)
    assert rec.calls[0]["cfg"] == 1.0


def test_dynamic_cfg_does_not_turn_on_the_uncond_pass():
    for step in range(0, 20):
        assert su.compute_dynamic_cfg(1.0, step, 20, 1.0) == 1.0
    assert su.compute_dynamic_cfg(5.0, 0, 20, 1.0) > 5.0      # still shapes real CFG


def _code_lines(fn):
    return [ln for ln in inspect.getsource(fn).splitlines() if not ln.strip().startswith("#")]


def test_no_second_model_load_per_stage():
    code = "\n".join(_code_lines(RadianceSamplerPro.sample))
    assert not re.search(r"\bload_model_gpu\s*\(", code)
    assert not re.search(r"\bgc\.collect\s*\(", code)


def test_log_tensor_costs_nothing_without_debug():
    su.logger.setLevel(logging.INFO)
    t = MagicMock()
    su.log_tensor("x", t)
    t.float.assert_not_called()
