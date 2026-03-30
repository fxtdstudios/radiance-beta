import math
import logging
from typing import Any, Tuple

logger = logging.getLogger("◎ Radiance.layout")


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR OPTIONS  (shared across reroute nodes)
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = [
    "Auto",      # inferred from connected type by radiance_layout.js
    "Gray", "Red", "Orange", "Yellow",
    "Green", "Cyan", "Blue", "Purple", "Magenta", "White",
]

# Pipeline-stage icons for Reroute+ — quick visual scan in dense graphs.
# Rendered as a prefix on the node pill by radiance_layout.js.
_ICONS = [
    "none",          # no prefix
    "→  Flow",       # generic data flow
    "⚡ Signal",     # latent / activation tensor
    "🖼  Image",     # image / frame data
    "🎞  Video",     # video frame sequence
    "🎨 Grade",      # color grading / LUT
    "🔊 Audio",      # audio waveform
    "🧠 Model",      # checkpoint / model weights
    "🗂  Config",    # metadata / settings / JSON
    "🔀 Branch",     # conditional / mux output
    "📤 Output",     # final export / save
    "📥 Input",      # source / loader
    "🔬 Debug",      # diagnostic / probe
    "⚙  Process",   # processing step
    "✅ Done",       # completed stage
    "⚠  Flag",      # attention / review needed
]

# Visual style variants for Reroute+ (rendered by radiance_layout.js).
_STYLES = [
    "Pill",     # compact rounded pill with color dot  (default)
    "Tag",      # rectangular tag with left color bar
    "Arrow",    # directional arrow shape — emphasizes flow direction
    "Dot",      # minimal color dot only — maximum compactness
    "Banner",   # full-width label bar — section divider
]



def _describe_type(data: Any) -> str:
    """
    Build a concise runtime type description for the type_info output.
    Called by RadianceAdvancedReroute.route() when show_type=True.
    """
    try:
        import torch
        if isinstance(data, torch.Tensor):
            shape_str = "×".join(str(d) for d in data.shape)
            return f"Tensor({shape_str}, {data.dtype})"
    except ImportError:
        pass

    if isinstance(data, str):
        return f"str({len(data)} chars)"
    if isinstance(data, (list, tuple)):
        return f"{type(data).__name__}[{len(data)}]"
    if isinstance(data, dict):
        keys = list(data.keys())[:3]
        return f"dict({len(data)} keys: {keys})"
    if isinstance(data, (int, float, bool)):
        return f"{type(data).__name__}={data}"
    return type(data).__name__


# ═══════════════════════════════════════════════════════════════════════════════
#  1. BASIC REROUTE  (backward compat — kept minimal)
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceReroute:
    """
    Lightweight pass-through reroute with optional label.

    Keeps complex graphs readable by letting you name every wire without
    adding visual weight. Backward compatible — existing workflows that
    connected only 'data' continue to work unchanged.

    When to use vs Reroute+:
    - Use this when you only need a named junction (no color, icon or style).
    - Use Reroute+ when you want color coding, icons, styles, group tags,
      type introspection, or the full annotation feature set.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("*", {"tooltip": "Any data — image, latent, model, string, etc."}),
            },
            "optional": {
                "label": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Short name for this wire. Shown on the node in the graph. "
                            "Examples: 'base latent', 'refined mask', 'cfg scale'."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("data",)
    FUNCTION = "route"
    CATEGORY = "FXTD Studios/Radiance/Layout"
    DESCRIPTION = (
        "Lightweight named reroute — label any wire without visual overhead. "
        "Backward compatible: the label is optional and purely informational. "
        "Use Reroute+ for color, icons, styles and group tags."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return math.nan  # always pass through — never cache

    def route(self, data, label=""):
        return (data,)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. ADVANCED REROUTE  (labeled, color-coded pill)
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceAdvancedReroute:
    """
    Professional labeled reroute with color, icon, style, group tag,
    description tooltip, and live type introspection.

    Renders as a styled node pill in the ComfyUI graph via the
    radiance_layout.js companion extension.

    Feature overview
    ────────────────
    label       Short name shown on the pill (e.g. "base latent").
    color       Pin / pill accent color. Auto = inferred from data type.
    icon        Pipeline-stage icon prefix for instant visual scan.
    style       Visual shape: Pill / Tag / Arrow / Dot / Banner.
    group       Group tag for logical sections (e.g. "Encode", "Decode").
                radiance_layout.js can highlight or fold by group.
    description Longer annotation shown as a hover tooltip on the node.
    show_type   When True, the Python type (or tensor shape) of the
                flowing data is appended to the pill label at runtime.
                Useful during development to confirm correct wiring.

    Outputs
    ───────
    data        The input data, unchanged.
    type_info   String describing the runtime type — wire to ShowText
                or DebugProbe for permanent type display in the graph.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("*", {"tooltip": "Any data — image, latent, model, string, etc."}),
            },
            "optional": {
                # ── Identity ──────────────────────────────────────────────────
                "label": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Short name shown on the pill. Keep it concise — "
                            "aim for 1-3 words. Examples: 'base latent', "
                            "'refined mask', 'cfg 7.5'."
                        ),
                    },
                ),
                "group": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Logical section tag. Reroutes with the same group name "
                            "can be highlighted or folded together by the JS extension. "
                            "Examples: 'Encode', 'Decode', 'ControlNet', 'Post'."
                        ),
                    },
                ),
                "description": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Longer annotation shown as a hover tooltip on the node. "
                            "Use to explain non-obvious wiring decisions, parameter "
                            "choices, or upstream context for collaborators."
                        ),
                    },
                ),
                # ── Appearance ────────────────────────────────────────────────
                "color": (
                    _COLORS,
                    {
                        "default": "Auto",
                        "tooltip": (
                            "Pill accent color. 'Auto' inherits from the connected "
                            "ComfyUI data type (IMAGE=green, LATENT=pink, MODEL=orange, "
                            "CLIP=yellow, VAE=red, STRING=gray). Set explicitly to "
                            "override — useful for visually grouping related wires."
                        ),
                    },
                ),
                "icon": (
                    _ICONS,
                    {
                        "default": "none",
                        "tooltip": (
                            "Pipeline-stage icon prefix. Renders before the label "
                            "for instant visual scan in dense graphs. Pick the icon "
                            "that describes the semantic role of this wire."
                        ),
                    },
                ),
                "style": (
                    _STYLES,
                    {
                        "default": "Pill",
                        "tooltip": (
                            "Visual shape rendered by radiance_layout.js:\n"
                            "• Pill — compact rounded pill with color dot (default)\n"
                            "• Tag  — rectangular with left color bar\n"
                            "• Arrow — directional arrow, emphasises flow\n"
                            "• Dot  — minimal color dot only, maximum compactness\n"
                            "• Banner — full-width label bar, section divider"
                        ),
                    },
                ),
                # ── Diagnostics ───────────────────────────────────────────────
                "show_type": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Append the runtime Python type (or tensor shape) to the "
                            "pill label. Useful during development to confirm the "
                            "correct data is flowing. Disable for clean production graphs."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("data", "type_info")
    OUTPUT_TOOLTIPS = (
        "Input data, passed through unchanged.",
        "Runtime type description — wire to ShowText or DebugProbe to display "
        "in the graph without breaking the pipeline.",
    )
    FUNCTION = "route"
    CATEGORY = "FXTD Studios/Radiance/Layout"
    DESCRIPTION = (
        "Professional labeled reroute — color, icon, style, group tag, "
        "description tooltip, and live type introspection. "
        "Renders as a styled pill via radiance_layout.js."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return math.nan  # always pass through — never cache

    def route(
        self,
        data: Any,
        label: str = "",
        group: str = "",
        description: str = "",
        color: str = "Auto",
        icon: str = "none",
        style: str = "Pill",
        show_type: bool = False,
    ) -> Tuple[Any, str]:

        # Build type_info string
        type_info = _describe_type(data)

        # Optionally append type to the displayed label (via ui dict → JS)
        display_label = label
        if show_type and type_info:
            display_label = f"{label} [{type_info}]" if label else type_info

        # Send all styling data to the JS companion via the ui dict.
        # radiance_layout.js reads these in onExecuted and applies them
        # as pill appearance / tooltip on the graph node.
        return {
            "ui": {
                "label":       [display_label],
                "group":       [group],
                "description": [description],
                "color":       [color],
                "icon":        [icon],
                "style":       [style],
                "type_info":   [type_info],
            },
            "result": (data, type_info),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. MUX — A/B/C/D Input Selector
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceMux:
    """
    Select one of up to 4 inputs by index (0=A, 1=B, 2=C, 3=D).

    Production use-cases:
    - Rapid A/B comparison: wire two samplers, toggle index without rewiring
    - Multi-preset testing: connect 4 VAE outputs, compare with a single flip
    - Optional overrides: index 0 = base image, index 1 = composited result
    - Video/Image branch: index 0 = still image path, index 1 = video path

    Unconnected slots are skipped — only the selected index is executed.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": (
                    "INT",
                    {
                        "default": 0, "min": 0, "max": 3, "step": 1,
                        "tooltip": "0 = A, 1 = B, 2 = C, 3 = D. "
                                   "Select which input to pass through.",
                    },
                ),
            },
            "optional": {
                "A": ("*", {"tooltip": "Input A (index 0)."}),
                "B": ("*", {"tooltip": "Input B (index 1)."}),
                "C": ("*", {"tooltip": "Input C (index 2)."}),
                "D": ("*", {"tooltip": "Input D (index 3)."}),
                "label": (
                    "STRING",
                    {"default": "", "multiline": False,
                     "tooltip": "Optional label for this mux node."},
                ),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("selected",)
    FUNCTION = "select"
    CATEGORY = "FXTD Studios/Radiance/Layout"
    DESCRIPTION = (
        "A/B/C/D input multiplexer — select which of up to 4 inputs "
        "to pass through using an index (0=A, 1=B, 2=C, 3=D). "
        "Perfect for rapid comparisons without rewiring."
    )

    @classmethod
    def IS_CHANGED(cls, index, **kwargs):
        # Cache key includes index so switching updates downstream immediately
        return index

    def select(
        self,
        index: int,
        A: Any = None,
        B: Any = None,
        C: Any = None,
        D: Any = None,
        label: str = "",
    ) -> Tuple[Any]:
        inputs = [A, B, C, D]
        names  = ["A", "B", "C", "D"]

        clamped = max(0, min(3, index))
        chosen  = inputs[clamped]

        if chosen is None:
            # Selected slot not connected — try adjacent slots gracefully
            fallback = next((v for v in inputs if v is not None), None)
            if fallback is not None:
                logger.warning(
                    f"[RadianceMux] Input {names[clamped]} (index={clamped}) is not connected. "
                    f"Falling back to first available input."
                )
                chosen = fallback
            else:
                raise ValueError(
                    f"[RadianceMux] All inputs are disconnected. "
                    f"Connect at least one input (index={clamped} selected)."
                )

        logger.debug(f"[RadianceMux] Selected: {names[clamped]}" +
                     (f" [{label}]" if label else ""))
        return (chosen,)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. GATE — Boolean pass/block
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceGate:
    """
    Pass data through when open=True, or return a configurable fallback
    when open=False.

    Production use-cases:
    - Optional watermark: gate(open=watermark_enabled) before overlay node
    - Skip heavy post-process on quick draft runs: wire enable_post to gate
    - A/B branch enable: two gates, one True one False, compare in one click
    - Debug bypass: temporarily disable a node chain without disconnecting

    When closed and no fallback is connected, returns None.
    Downstream nodes that don't handle None gracefully will error — this is
    intentional (equivalent to ComfyUI's native skip behavior).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data":  ("*", {"tooltip": "Data to pass through when gate is open."}),
                "open":  ("BOOLEAN", {"default": True,
                                      "tooltip": "True = pass data through. False = return fallback or None."}),
            },
            "optional": {
                "fallback": ("*", {"tooltip": "Returned when gate is closed. "
                                              "Leave disconnected to return None (block execution)."}),
                "label": (
                    "STRING",
                    {"default": "", "multiline": False,
                     "tooltip": "Optional label for this gate."},
                ),
            },
        }

    RETURN_TYPES = ("*", "BOOLEAN")
    RETURN_NAMES = ("data",   "is_open")
    OUTPUT_TOOLTIPS = (
        "The selected data (input when open, fallback when closed).",
        "Echoes the open state — wire to another gate or condition node.",
    )
    FUNCTION = "gate"
    CATEGORY = "FXTD Studios/Radiance/Layout"
    DESCRIPTION = (
        "Boolean gate — passes data when open=True, returns fallback (or None) "
        "when open=False. Use to enable/disable branches without rewiring."
    )

    @classmethod
    def IS_CHANGED(cls, open, **kwargs):
        return open  # recompute whenever state changes

    def gate(
        self,
        data: Any,
        open: bool,
        fallback: Any = None,
        label: str = "",
    ) -> Tuple[Any, bool]:
        state_str = "OPEN" if open else "CLOSED"
        logger.debug(f"[RadianceGate] {state_str}" + (f" [{label}]" if label else ""))

        if open:
            return (data, True)
        return (fallback, False)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. NOTE — Documentation sticky note (no connections)
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceNote:
    """
    A documentation-only node with no data connections.
    Renders as a styled sticky-note card in the graph via radiance_layout.js.

    Use to:
    - Document workflow sections ("--- VAE ENCODE ---")
    - Explain parameter choices for future collaborators
    - Mark TODO / WIP / REVIEW sections
    - Annotate presets or experimental branches

    The node is a pure OUTPUT_NODE that generates no outputs and never
    participates in the execution graph — it is display-only.
    """

    COLORS = [
        "Yellow",   # default — classic sticky note
        "Orange", "Red", "Green", "Cyan", "Blue",
        "Purple", "Gray", "White",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "📝 Note...",
                        "multiline": True,
                        "tooltip": "Documentation text. Markdown supported in the frontend renderer.",
                    },
                ),
            },
            "optional": {
                "title": (
                    "STRING",
                    {"default": "", "multiline": False,
                     "tooltip": "Optional bold title line shown above the body text."},
                ),
                "color": (
                    s.COLORS,
                    {"default": "Yellow",
                     "tooltip": "Note card background color."},
                ),
                "font_size": (
                    "INT",
                    {"default": 14, "min": 10, "max": 24, "step": 1,
                     "tooltip": "Text size in the rendered note card."},
                ),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "noop"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/Layout"
    DESCRIPTION = (
        "Sticky note for workflow documentation. No data connections — "
        "pure display. Supports a title, body text, and color selection."
    )

    def noop(self, text, title="", color="Yellow", font_size=14):
        # Display-only — nothing to execute
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
#  6. DEBUG PROBE — Non-destructive stats logger
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceDebugProbe:
    """
    Non-destructive pass-through that logs shape, dtype, value range,
    and other diagnostics for any data flowing through it.

    Supports:
    - torch.Tensor  → shape, dtype, min/max/mean, NaN/Inf count, device
    - str / int / float / bool → value printed directly
    - dict / list → type + length summary
    - anything else → type name + repr (truncated)

    The original data is returned unchanged — insert anywhere in a pipeline
    without breaking it. Enable/disable logging without disconnecting.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("*", {"tooltip": "Any data to inspect. Passed through unchanged."}),
            },
            "optional": {
                "label": (
                    "STRING",
                    {"default": "probe", "multiline": False,
                     "tooltip": "Identifier printed in the log line. Use to distinguish multiple probes."},
                ),
                "enabled": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Disable to silence logging without disconnecting the node."},
                ),
                "log_level": (
                    ["INFO", "DEBUG", "WARNING"],
                    {"default": "INFO",
                     "tooltip": "Python logging level for the probe output."},
                ),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("data",  "probe_report")
    OUTPUT_TOOLTIPS = (
        "Input data, passed through unchanged.",
        "Human-readable diagnostic string — wire to a text preview node.",
    )
    FUNCTION = "probe"
    CATEGORY = "FXTD Studios/Radiance/Layout"
    DESCRIPTION = (
        "Non-destructive debug probe. Logs tensor shape/range/dtype, "
        "string values, list/dict lengths. Data is passed through unchanged."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return math.nan  # always re-probe

    def probe(
        self,
        data: Any,
        label: str = "probe",
        enabled: bool = True,
        log_level: str = "INFO",
    ) -> Tuple[Any, str]:

        report = self._build_report(data, label)

        if enabled:
            log_fn = {
                "DEBUG":   logger.debug,
                "WARNING": logger.warning,
            }.get(log_level, logger.info)
            log_fn(f"[DebugProbe:{label}] {report}")

        return (data, report)

    @staticmethod
    def _build_report(data: Any, label: str) -> str:
        try:
            import torch

            if isinstance(data, torch.Tensor):
                t = data.float()
                has_nan = bool(torch.isnan(t).any())
                has_inf = bool(torch.isinf(t).any())
                flags   = []
                if has_nan: flags.append("NaN!")
                if has_inf: flags.append("Inf!")
                flag_str = f"  ⚠ {' '.join(flags)}" if flags else ""
                return (
                    f"Tensor | shape={list(data.shape)} | dtype={data.dtype} | "
                    f"device={data.device} | "
                    f"range=[{float(t.min()):.4f}, {float(t.max()):.4f}] | "
                    f"mean={float(t.mean()):.4f} | std={float(t.std()):.4f}"
                    f"{flag_str}"
                )
        except ImportError:
            pass

        if isinstance(data, str):
            preview = data[:120] + ("..." if len(data) > 120 else "")
            return f"str | len={len(data)} | {repr(preview)}"

        if isinstance(data, (int, float, bool)):
            return f"{type(data).__name__} | value={data}"

        if isinstance(data, dict):
            keys = list(data.keys())[:8]
            return f"dict | len={len(data)} | keys={keys}"

        if isinstance(data, (list, tuple)):
            return f"{type(data).__name__} | len={len(data)}"

        return f"{type(data).__name__} | {repr(data)[:200]}"


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceReroute":         RadianceReroute,
    "RadianceAdvancedReroute": RadianceAdvancedReroute,
    "RadianceMux":             RadianceMux,
    "RadianceGate":            RadianceGate,
    "RadianceNote":            RadianceNote,
    "RadianceDebugProbe":      RadianceDebugProbe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceReroute":         "◎ Radiance Reroute",
    "RadianceAdvancedReroute": "◎ Radiance Reroute+",
    "RadianceMux":             "◎ Radiance Mux",
    "RadianceGate":            "◎ Radiance Gate",
    "RadianceNote":            "◎ Radiance Note",
    "RadianceDebugProbe":      "◎ Radiance Debug Probe",
}
