[← Back to Radiance docs](README.md)

# Developer Notes

This page is for contributors and pipeline engineers extending Radiance.

## Runtime Shape

Radiance is loaded by `__init__.py`, which imports the grouped node catalog from `radiance.nodes`. The grouped catalog lives in `nodes/catalog.py` and points to these packages:

| Group | Package |
| :--- | :--- |
| Color | `radiance.nodes.color` |
| HDR | `radiance.nodes.hdr` |
| IO | `radiance.nodes.io` |
| VFX | `radiance.nodes.vfx` |
| Pipeline | `radiance.nodes.pipeline` |
| Review / Monitor | `radiance.nodes.monitor` |
| Upscale | `radiance.nodes.upscale` |
| Video | `radiance.nodes.video` |
| AI assist | `radiance.nodes.ai` |
| Generate | `radiance.nodes.generate` |
| Training | `radiance.nodes.training`, gated by `RADIANCE_DEV` |

The registry utility merges each package's `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`, then applies Radiance branding.

## Node Authoring Rules

Follow [CODE_STYLE.md](../CODE_STYLE.md). The short version:

| Rule | Requirement |
| :--- | :--- |
| Class names | PascalCase with `Radiance` prefix. |
| Display names | Use the `◎` prefix for user-facing Radiance nodes. |
| Categories | Use `FXTD STUDIOS/Radiance/...`. |
| Inputs | Put image/model inputs first, selections second, parameters last. |
| Returns | Always return tuples, even for one output. |
| Optional dependencies | Degrade gracefully and log a useful message. |

## Adding A Node

1. Put implementation in the correct grouped package or root compatibility module.
2. Add the class to that group's `NODE_CLASS_MAPPINGS`.
3. Add a user-facing name to `NODE_DISPLAY_NAME_MAPPINGS`.
4. Add or update tests in `tests/`.
5. Update [Node Reference](nodes.md) and [Coverage Ledger](coverage.md).
6. Run a focused smoke test for the group and any affected workflow.

## Testing

The project declares pytest settings in `pyproject.toml`.

Useful local checks:

```bash
python tools/check_release_ready.py
pytest
pytest tests/test_nodes_registry.py
pytest tests/test_node_smoke.py
```

Some tests require a real ComfyUI environment, optional dependencies, GPU support, or model weights. Mark or document those cases instead of hiding the dependency.

## Release Checks

Before publishing:

1. Confirm `pyproject.toml` version and Comfy Registry metadata.
2. Run the release readiness tool.
3. Run CI.
4. Verify the docs coverage ledger matches the registered node catalog.
5. Smoke test import inside the target ComfyUI environment.

## Dynamic Gizmos

Radiance loads dynamic gizmos after the static node catalog. A gizmo can expose a generated node name, display name, category, and wiring behavior. Because those nodes are user-created, this documentation covers the loader behavior and expects each studio to document its own gizmo workflows.

