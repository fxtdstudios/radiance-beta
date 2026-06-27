"""Declarative catalog of node groups exposed by Radiance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from radiance.config.env import ENV, get_env_bool
from radiance.nodes.registry import NodeModuleSpec


@dataclass(frozen=True)
class NodeGroupSpec:
    """A loadable node group and the feature flag that controls it."""

    module_path: str
    env_flag: Optional[str] = None

    def is_enabled(self) -> bool:
        if self.env_flag is None:
            return True
        return get_env_bool(self.env_flag, False)

    def as_module_spec(self) -> NodeModuleSpec:
        return NodeModuleSpec(self.module_path)


NODE_GROUPS: Tuple[NodeGroupSpec, ...] = (
    NodeGroupSpec("radiance.nodes.color"),
    NodeGroupSpec("radiance.nodes.hdr"),
    NodeGroupSpec("radiance.nodes.io"),
    NodeGroupSpec("radiance.nodes.vfx"),
    NodeGroupSpec("radiance.nodes.pipeline"),
    NodeGroupSpec("radiance.nodes.monitor"),
    NodeGroupSpec("radiance.nodes.upscale"),
    NodeGroupSpec("radiance.nodes.video"),
    NodeGroupSpec("radiance.nodes.ai"),
    NodeGroupSpec("radiance.nodes.generate"),
    NodeGroupSpec("radiance.nodes.training", env_flag=ENV.RADIANCE_DEV),
)


def enabled_node_group_specs() -> Tuple[NodeModuleSpec, ...]:
    """Return import specs for the currently enabled node groups."""

    return tuple(group.as_module_spec() for group in NODE_GROUPS if group.is_enabled())


__all__ = ["NODE_GROUPS", "NodeGroupSpec", "enabled_node_group_specs"]
