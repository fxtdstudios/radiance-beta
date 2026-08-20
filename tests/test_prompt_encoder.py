import json

import pytest

from radiance.nodes.generate.prompt import (
    RadianceCinematicPromptEncoder,
    _detect_arch_from_clip,
    build_cinematic_prompt_v3,
)
from radiance.nodes.generate.loader import RadianceUnifiedLoader


class FakeClip:
    def __init__(self, keys):
        self.keys = tuple(keys)
        self.encoded_tokens = []

    def tokenize(self, text):
        words = [w for w in str(text).split() if w]
        tokens = []
        for index, _word in enumerate(words, 1):
            tokens.append((index, 1.0))

        chunks = []
        while tokens:
            chunk = tokens[:77]
            tokens = tokens[77:]
            chunks.append(self._pad_chunk(chunk))
        if not chunks:
            chunks.append(self._pad_chunk([]))

        return {key: list(chunks) for key in self.keys}

    @staticmethod
    def _pad_chunk(chunk):
        pad_id = 49407
        return list(chunk) + [(pad_id, 1.0)] * (77 - len(chunk))

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded_tokens.append(tokens)
        return [[f"conditioning_{len(self.encoded_tokens)}", {}]]


def test_model_meta_overrides_tokenizer_heuristic():
    clip = FakeClip(("t5xxl",))
    model_meta = json.dumps({"arch": "flux"})

    assert _detect_arch_from_clip(clip, "Auto", model_meta) == "flux"


def test_weak_negative_arch_downgrades_to_soft():
    _positive, negative, _tokens = build_cinematic_prompt_v3(
        base_prompt="a detective under neon rain",
        framing="Medium Shot (MS)",
        camera_type="ARRI Alexa 35",
        lens_focal="50mm Standard Prime",
        aperture_dof="f/2.8 (Cinematic Separation)",
        lighting="Cinematic Haze / Volumetric Fog",
        style_aesthetic="Photorealistic (Raw)",
        negative_strength="Aggressive",
        target_arch="flux",
    )

    assert "blur" in negative
    assert "low quality" in negative
    assert "deformed" not in negative
    assert "mutated" not in negative
    assert "cartoon" not in negative


# The encoder runs its token budget through torch; under conftest's stub the
# conditioning it returns is a MagicMock rather than the CLIP payload.
@pytest.mark.real_torch
def test_encoder_returns_debug_outputs_and_uses_model_meta():
    clip = FakeClip(("t5xxl",))
    encoder = RadianceCinematicPromptEncoder()

    result = encoder.encode_cinematic(
        clip,
        base_prompt="a hero crossing a rainy street at night",
        style_preset="None (Custom)",
        model_meta=json.dumps({"arch": "pixart"}),
    )

    positive, negative, positive_text, negative_text, resolved_arch, token_count = result["result"]

    assert positive == [["conditioning_1", {}]]
    assert negative == [["conditioning_2", {}]]
    assert "hero crossing" in positive_text
    assert isinstance(negative_text, str)
    assert resolved_arch == "pixart"
    assert token_count > 0


def test_loader_exposes_model_meta_output_contract():
    assert RadianceUnifiedLoader.RETURN_TYPES[-1] == "STRING"
    assert RadianceUnifiedLoader.RETURN_NAMES[-1] == "model_meta"


class TestMiniMaxArch:
    """MiniMax H3's Qwen3-VL-32B encoder: architecture detection, prose
    prompting, weak-negative handling, and its much higher token budget
    (comfy/text_encoders/qwen3vl.py's tokenizer has no practical limit,
    and MiniMax H3's own example prompts run several hundred words)."""

    def test_qwen3vl_32b_key_detected_as_minimax(self):
        clip = FakeClip(("qwen3vl_32b",))
        assert _detect_arch_from_clip(clip, "Auto", None) == "minimax"

    def test_minimax_is_a_prose_arch(self):
        from radiance.nodes.generate.prompt import PROSE_ARCHS
        assert "minimax" in PROSE_ARCHS

    def test_minimax_is_a_weak_negative_arch(self):
        from radiance.nodes.generate.prompt import _WEAK_NEG_ARCHS
        assert "minimax" in _WEAK_NEG_ARCHS

    @pytest.mark.real_torch
    def test_minimax_ui_channel_flags_weak_neg_arch(self):
        clip = FakeClip(("qwen3vl_32b",))
        encoder = RadianceCinematicPromptEncoder()
        result = encoder.encode_cinematic(clip, base_prompt="a rooftop chase at dusk")
        assert result["ui"]["weak_neg_arch"] == [True]

    @pytest.mark.real_torch
    def test_non_minimax_ui_channel_does_not_flag_weak_neg_arch(self):
        clip = FakeClip(("t5xxl", "g", "l"))  # sd3, not in _WEAK_NEG_ARCHS
        encoder = RadianceCinematicPromptEncoder()
        result = encoder.encode_cinematic(clip, base_prompt="a rooftop chase at dusk")
        assert result["ui"]["weak_neg_arch"] == [False]

    @pytest.mark.real_torch
    def test_minimax_does_not_truncate_a_long_prompt_that_flux_would(self):
        # 500 words needs 7 of FakeClip's 77-token chunks. flux's 256-token
        # limit truncates to 4 chunks (308); minimax's 2048-token limit (27
        # chunks) leaves all 7 untouched. Chunk-granular truncation means a
        # shorter prompt (e.g. 300 words, 4 chunks) would round up to fit
        # flux's limit too and not actually exercise this.
        long_prompt = "detail " * 500

        minimax_clip = FakeClip(("qwen3vl_32b",))
        minimax_result = RadianceCinematicPromptEncoder().encode_cinematic(
            minimax_clip, base_prompt=long_prompt,
        )
        _, _, _, _, _, minimax_token_count = minimax_result["result"]

        flux_clip = FakeClip(("t5xxl", "l"))
        flux_result = RadianceCinematicPromptEncoder().encode_cinematic(
            flux_clip, base_prompt=long_prompt,
        )
        _, _, _, _, _, flux_token_count = flux_result["result"]

        assert minimax_token_count > 400
        assert flux_token_count <= 308
