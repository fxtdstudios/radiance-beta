"""Immutable package-level constants."""

PACKAGE_NAME = "radiance"
PACKAGE_DISPLAY_NAME = "Radiance"
VERSION = "3.2.0"
AUTHOR = "Radiance"
WEB_DIRECTORY = "./js"

#: Floor for the number of nodes the entry point is expected to publish.
#:
#: `load_node_mappings` treats a failed optional import as non-fatal so a broken
#: dependency can never stop ComfyUI from starting. The cost is that a group
#: import failure silently removes every node in that group, and the startup
#: banner reported the surviving count as a success -- "successfully loaded 12
#: nodes" reads exactly like "successfully loaded 100 nodes" in a scrolling log.
#: Comparing against this floor turns that into a visible ERROR.
#:
#: Raise it when the catalog grows; a drop below it means something failed to
#: import, not that the catalog shrank.
EXPECTED_MIN_NODE_COUNT = 109
