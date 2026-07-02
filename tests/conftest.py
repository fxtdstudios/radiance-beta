"""
conftest.py — Radiance test suite shared fixtures and ComfyUI mocks.

ComfyUI is not installed in the test environment, so we stub the modules
that Radiance imports at the top of each nodes_*.py file.  The stubs expose
just enough surface to let the modules import cleanly; node logic that
actually calls into ComfyUI (e.g. comfy.sample.sample) is tested via
integration tests that mock the call-site, not here.
"""

import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np

RADIANCE_ROOT = Path(__file__).resolve().parent.parent
RADIANCE_PARENT = RADIANCE_ROOT.parent
if str(RADIANCE_PARENT) not in sys.path:
    sys.path.insert(0, str(RADIANCE_PARENT))

# Make `import radiance` resolve regardless of the checkout directory name.
# The package imports itself as `radiance`, which normally requires the repo
# directory to be named exactly "radiance". CI on a mirror/fork checked out as
# e.g. "radiance-beta" would otherwise fail every `import radiance` with
# ModuleNotFoundError → collection errors → exit code 2. Register the package
# from its __init__.py under the stable name so the dir name no longer matters.
if RADIANCE_ROOT.name != "radiance" and (RADIANCE_ROOT / "__init__.py").exists():
    # Install a lazy meta-path finder that maps the import name `radiance` to the
    # repo root. This makes the FIRST `import radiance` run __init__ through the
    # normal import machinery — exactly as it would if the checkout directory were
    # named "radiance" — instead of eagerly pre-importing here (which would cache
    # partial submodules under the test stubs and skew node-count tests).
    import importlib.abc as _iabc
    import importlib.util as _iutil

    class _RadianceNameFinder(_iabc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "radiance":
                return None
            return _iutil.spec_from_file_location(
                "radiance",
                str(RADIANCE_ROOT / "__init__.py"),
                submodule_search_locations=[str(RADIANCE_ROOT)],
            )

    if not any(type(f).__name__ == "_RadianceNameFinder" for f in sys.meta_path):
        sys.meta_path.insert(0, _RadianceNameFinder())


# ─────────────────────────────────────────────────────────────────────────────
#  Torch stub — installed BEFORE any radiance import so modules that do
#  `import torch` at the top level don't crash the collection phase.
#  Real torch tests are guarded with HAS_TORCH / pytest.skipif in each file.
# ─────────────────────────────────────────────────────────────────────────────

def _make_torch_stub():
    """
    Minimal torch stub that lets module-level `import torch` succeed.
    Actual tensor operations are NOT supported — those tests skip themselves.
    """
    try:
        import torch  # use the real thing if available
        return torch
    except ImportError:
        pass

    torch_mod = types.ModuleType("torch")

    # Basic sentinel class so isinstance checks don't crash
    class _FakeTensor:
        pass

    class _FakeModule:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return MagicMock()

        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

        def state_dict(self):
            return {}

    class _FakeSequential(_FakeModule):
        def __init__(self, *layers):
            super().__init__()
            self.layers = layers

    class _FakeNN(types.ModuleType):
        Module = _FakeModule
        Sequential = _FakeSequential
        Conv2d = MagicMock
        ReLU = MagicMock
        Upsample = MagicMock
        Linear = MagicMock
        SiLU = MagicMock
        AdaptiveAvgPool2d = MagicMock
        functional = MagicMock()

    torch_mod.Tensor      = _FakeTensor
    torch_mod.Generator   = MagicMock
    torch_mod.nn          = _FakeNN("torch.nn")
    torch_mod.device      = MagicMock()
    torch_mod.zeros       = MagicMock()
    torch_mod.ones        = MagicMock()
    torch_mod.randn       = MagicMock()
    torch_mod.from_numpy  = MagicMock()
    torch_mod.no_grad     = MagicMock()
    torch_mod.float32     = "float32"
    torch_mod.float16     = "float16"
    torch_mod.bfloat16    = "bfloat16"
    torch_mod.cuda        = MagicMock()
    torch_mod.tensor      = MagicMock(return_value=MagicMock())
    torch_mod.inverse     = MagicMock(return_value=MagicMock())
    torch_mod.diag        = MagicMock(return_value=MagicMock())
    torch_mod.stack       = MagicMock(return_value=MagicMock())
    torch_mod.cat         = MagicMock(return_value=MagicMock())
    torch_mod.clamp       = MagicMock(return_value=MagicMock())
    torch_mod.where       = MagicMock(return_value=MagicMock())
    torch_mod.det         = MagicMock(return_value=MagicMock())
    torch_mod.float64     = "float64"
    torch_mod.eye        = MagicMock(return_value=MagicMock())
    torch_mod.long        = MagicMock(return_value=MagicMock())
    torch_mod.arange      = MagicMock(return_value=MagicMock())
    torch_mod.max         = MagicMock(return_value=MagicMock())
    torch_mod.min         = MagicMock(return_value=MagicMock())
    torch_mod.abs         = MagicMock(return_value=MagicMock())
    torch_mod.sqrt        = MagicMock(return_value=MagicMock())
    torch_mod.exp         = MagicMock(return_value=MagicMock())
    torch_mod.pow         = MagicMock(return_value=MagicMock())
    torch_mod.clamp_      = MagicMock(return_value=MagicMock())
    torch_mod.zeros_like  = MagicMock(return_value=MagicMock())
    torch_mod.ones_like   = MagicMock(return_value=MagicMock())
    torch_mod.full        = MagicMock(return_value=MagicMock())
    torch_mod.load        = MagicMock(return_value=MagicMock())
    torch_mod.save        = MagicMock()
    torch_mod.inference_mode = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    return torch_mod


_torch_stub = _make_torch_stub()
_torch_mod = sys.modules.setdefault("torch", _torch_stub)
if not hasattr(_torch_mod, "Generator"):
    _torch_mod.Generator = MagicMock
if not hasattr(_torch_mod, "nn"):
    _torch_mod.nn = getattr(_torch_stub, "nn", MagicMock())
sys.modules.setdefault("torch.nn", getattr(_torch_mod, "nn", MagicMock()))
# Ensure torch.nn.functional exists so `import torch.nn.functional as F` succeeds
if "torch.nn.functional" not in sys.modules:
    _nn_functional = types.ModuleType("torch.nn.functional")
    for _fn in ("grid_sample", "pad", "interpolate", "conv2d", "relu",
                "softmax", "normalize", "avg_pool2d", "max_pool2d"):
        setattr(_nn_functional, _fn, MagicMock())
    sys.modules["torch.nn.functional"] = _nn_functional

if "node_helpers" not in sys.modules:
    _node_helpers = types.ModuleType("node_helpers")
    _node_helpers.conditioning_set_values = MagicMock(side_effect=lambda conditioning, values: conditioning)
    sys.modules["node_helpers"] = _node_helpers

if "aiohttp" not in sys.modules:
    _aiohttp = types.ModuleType("aiohttp")
    _aiohttp_web = types.ModuleType("aiohttp.web")
    _aiohttp.web = _aiohttp_web
    sys.modules["aiohttp"] = _aiohttp
    sys.modules["aiohttp.web"] = _aiohttp_web

if "server" not in sys.modules:
    _server = types.ModuleType("server")
    class _FakeRoutes:
        def get(self, path): return lambda fn: fn
        def post(self, path): return lambda fn: fn
    class _FakePromptServer:
        instance = type("PromptServerInstance", (), {"routes": _FakeRoutes()})()
    _server.PromptServer = _FakePromptServer
    sys.modules["server"] = _server

_radiance_ocio_stub = types.ModuleType("radiance.radiance_ocio")
_radiance_ocio_stub.get_ocio_manager = MagicMock(return_value=MagicMock())
_radiance_ocio_stub.HAS_OCIO = False
sys.modules.setdefault("radiance.radiance_ocio", _radiance_ocio_stub)

if not hasattr(sys.modules.get("torch"), "__version__"):
    _hdr_cs_stub = types.ModuleType("radiance.nodes_hdr_colorspace")
    for _attr in ("_EOTF_MAP", "_BRADFORD_CAT", "_PRIMARIES_MATRICES"):
        setattr(_hdr_cs_stub, _attr, {})
    _hdr_cs_stub._apply_matrix = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("radiance.nodes_hdr_colorspace", _hdr_cs_stub)


# ── radiance.image subpackage (used by nodes_qc via `from .image import defects`) ─
_image_pkg = types.ModuleType("radiance.image")
_defects_stub = types.ModuleType("radiance.image.defects")
_defects_stub.analyze_levels         = MagicMock(return_value={"crushed": 0.0, "clipped": 0.0})
_defects_stub.check_gamut            = MagicMock(return_value={"out_of_gamut_pct": 0.0})
_defects_stub.detect_banding         = MagicMock(return_value={"risk_pct": 0.0, "detected": False})
_defects_stub.analyze_noise          = MagicMock(return_value={"level": 0.0})
_defects_stub.detect_compression_artifacts = MagicMock(return_value={"detected": False})
_defects_stub.analyze_focus          = MagicMock(return_value={"sharpness": 1.0})
_image_pkg.defects = _defects_stub
sys.modules.setdefault("radiance.image",         _image_pkg)
sys.modules.setdefault("radiance.image.defects", _defects_stub)
# Also expose as bare `image` and `image.defects` for flat imports
_bare_image_pkg = sys.modules.get("image") or types.ModuleType("image")
_bare_image_pkg.defects = _defects_stub
sys.modules.setdefault("image",         _bare_image_pkg)
sys.modules.setdefault("image.defects", _defects_stub)


# ─────────────────────────────────────────────────────────────────────────────
#  ComfyUI stub modules
# ─────────────────────────────────────────────────────────────────────────────

def _make_comfy_stubs():
    """
    Build a minimal comfy.* module tree so Radiance nodes can be imported
    without a live ComfyUI installation.
    """
    # comfy (top-level)
    comfy = types.ModuleType("comfy")

    # comfy.samplers
    samplers = types.ModuleType("comfy.samplers")
    samplers.KSampler = MagicMock()
    samplers.SAMPLER_NAMES = [
        "euler", "euler_ancestral", "dpm_fast", "dpm_adaptive",
        "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "dpmpp_3m_sde",
        "ddim", "uni_pc",
    ]
    samplers.SCHEDULER_NAMES = [
        "normal", "karras", "exponential", "sgm_uniform",
        "simple", "ddim_uniform", "beta",
    ]
    comfy.samplers = samplers

    # comfy.sample
    sample = types.ModuleType("comfy.sample")
    sample.sample = MagicMock(return_value=MagicMock())
    comfy.sample = sample

    # comfy.model_management
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = MagicMock(return_value="cpu")
    mm.load_model_gpu = MagicMock()
    mm.soft_empty_cache = MagicMock()
    mm.unload_all_models = MagicMock()
    mm.get_free_memory = MagicMock(return_value=8 * 1024**3)
    mm.get_total_memory = MagicMock(return_value=16 * 1024**3)
    comfy.model_management = mm

    # comfy.utils
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = MagicMock()
    comfy.utils = utils

    # comfy.model_base
    model_base = types.ModuleType("comfy.model_base")
    ModelType = types.SimpleNamespace(
        FLOW=1, FLOW_LTX_AV=2, EPS=3, V_PREDICTION=4,
    )
    model_base.ModelType = ModelType
    comfy.model_base = model_base

    # comfy.sd
    sd_mod = types.ModuleType("comfy.sd")
    sd_mod.VAE = MagicMock()
    comfy.sd = sd_mod

    # comfy.latent_formats
    latent_formats = types.ModuleType("comfy.latent_formats")
    latent_formats.SD15 = MagicMock()
    latent_formats.SDXL = MagicMock()
    comfy.latent_formats = latent_formats

    # comfy.nested_tensor  (optional — some builds don't have it)
    nested_tensor = types.ModuleType("comfy.nested_tensor")
    nested_tensor.NestedTensor = None
    comfy.nested_tensor = nested_tensor

    # comfy.cldm.control_types
    cldm = types.ModuleType("comfy.cldm")
    control_types = types.ModuleType("comfy.cldm.control_types")
    control_types.UNION_CONTROLNET_TYPES = {}
    cldm.control_types = control_types
    comfy.cldm = cldm

    # folder_paths stub
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = MagicMock(return_value=[])
    folder_paths.get_full_path = MagicMock(return_value=None)
    folder_paths.get_input_directory  = MagicMock(return_value="/tmp")
    folder_paths.get_output_directory = MagicMock(return_value="/tmp/comfy_output")
    folder_paths.filter_files_content_types = MagicMock(return_value=[])
    folder_paths.get_annotated_filepath = MagicMock(side_effect=lambda name: name)
    folder_paths.exists_annotated_filepath = MagicMock(return_value=True)
    folder_paths.models_dir = "/tmp/comfy_models"
    folder_paths.output_directory = "/tmp/comfy_output"

    stubs = {
        "comfy": comfy,
        "comfy.samplers": samplers,
        "comfy.sample": sample,
        "comfy.model_management": mm,
        "comfy.utils": utils,
        "comfy.model_base": model_base,
        "comfy.sd": sd_mod,
        "comfy.latent_formats": latent_formats,
        "comfy.nested_tensor": nested_tensor,
        "comfy.cldm": cldm,
        "comfy.cldm.control_types": control_types,
        "folder_paths": folder_paths,
    }
    return stubs


# Install stubs before any radiance import can happen
_STUBS = _make_comfy_stubs()
for name, mod in _STUBS.items():
    if name not in sys.modules:
        sys.modules[name] = mod


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def np_rng():
    """Deterministic NumPy RNG for all tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def linear_gradient_1d():
    """Linear ramp [0, 2] with 1024 samples — covers SDR and HDR range."""
    return np.linspace(0.0, 2.0, 1024, dtype=np.float32)


@pytest.fixture(scope="session")
def linear_image_hwc(np_rng):
    """(64, 64, 3) float32 scene-linear image with values in [0, 4]."""
    img = np_rng.random((64, 64, 3), dtype=np.float32) * 4.0
    return img.astype(np.float32)
