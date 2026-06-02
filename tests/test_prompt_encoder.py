import json

from radiance.nodes.generate.prompt import (
    RadianceCinematicPromptEncoder,
    _detect_arch_from_clip,
    build_cinematic_prompt_v3,
)
from radiance.nodes_loader import RadianceUnifiedLoader


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


def test_encoder_returns_debug_outputs_and_uses_model_meta():
    clip = FakeClip(("t5xxl",))
    encoder = RadianceCinematicPromptEncoder()

    result = encoder.encode_cinematic(
        clip,
        base_prompt="a hero crossing a rainy street at night",
        style_preset="None (Custom)",
        model_meta=json.dumps({"arch": "pixart"}),
    )

    positive, negative, positive_text, negative_text, resolved_arch, token_count = result

    assert positive == [["conditioning_1", {}]]
    assert negative == [["conditioning_2", {}]]
    assert "hero crossing" in positive_text
    assert isinstance(negative_text, str)
    assert resolved_arch == "pixart"
    assert token_count > 0


def test_loader_exposes_model_meta_output_contract():
    assert RadianceUnifiedLoader.RETURN_TYPES[-1] == "STRING"
    assert RadianceUnifiedLoader.RETURN_NAMES[-1] == "model_meta"
