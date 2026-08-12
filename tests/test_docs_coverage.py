"""The documentation must describe the nodes that actually exist.

`docs/coverage.md` is the ledger the rest of the docs are built against, and it
had drifted twice over in opposite directions: nine node classes were written
but never registered (so they were absent from both the menu and the ledger),
and four that WERE registered — the tone-map pair and the SDR→HDR pair — had
never been added to the ledger at all. The published total said 96 while
ComfyUI loaded 100.

A count in a README is the kind of thing nobody checks by hand and everybody
believes, so check it here.
"""
import importlib.util
import pathlib
import re
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(rel):
    return (_ROOT / rel).read_text(encoding="utf-8")


def _live_nodes():
    import radiance

    if radiance._LOAD_RESULT.failures:
        pytest.skip(
            "environment is short a runtime dependency, so the catalog is "
            "incomplete: "
            + ", ".join(f.source.label for f in radiance._LOAD_RESULT.failures)
        )
    return set(radiance.NODE_CLASS_MAPPINGS)


def _ledger_nodes():
    return set(re.findall(r"^- `(\w+)`", _src("docs/coverage.md"), re.M))


def _ledger_group_counts():
    rows = re.findall(r"\|\s*\[[^\]]+\]\([^)]+\)\s*\|\s*(\d+)\s*\|",
                      _src("docs/coverage.md"))
    return [int(r) for r in rows]


def test_the_ledger_lists_every_published_node():
    live, doc = _live_nodes(), _ledger_nodes()
    missing = sorted(live - doc)
    assert not missing, (
        f"{len(missing)} node(s) are published to ComfyUI but absent from "
        f"docs/coverage.md: {missing}"
    )


def test_the_ledger_lists_nothing_that_does_not_exist():
    live, doc = _live_nodes(), _ledger_nodes()
    stale = sorted(doc - live)
    assert not stale, (
        f"docs/coverage.md documents {len(stale)} node(s) that ComfyUI never "
        f"sees: {stale}"
    )


def test_the_group_counts_add_up():
    counts = _ledger_group_counts()
    assert counts, "the summary table did not parse"
    total = re.search(r"\*\*Total\*\* \| \*\*(\d+)\*\*", _src("docs/coverage.md"))
    assert total, "the ledger has no total row"
    assert sum(counts) == int(total.group(1)), (
        f"the group counts sum to {sum(counts)} but the total row says "
        f"{total.group(1)}"
    )


def test_the_total_matches_the_live_catalog():
    total = int(re.search(r"\*\*Total\*\* \| \*\*(\d+)\*\*",
                          _src("docs/coverage.md")).group(1))
    assert total == len(_live_nodes())


def test_the_readme_node_count_is_current():
    """Three places in the README quote the count, including a shields.io badge."""
    readme = _src("README.md")
    live = len(_live_nodes())
    quoted = set(re.findall(r"nodes-(\d+)-", readme))
    quoted |= set(re.findall(r"successfully loaded (\d+) nodes", readme))
    quoted |= set(re.findall(r"provides \*\*(\d+) nodes\*\*", readme))
    assert quoted, "the README no longer quotes a node count anywhere"
    wrong = {q for q in quoted if int(q) != live}
    assert not wrong, (
        f"the README says {sorted(wrong)} node(s); the catalog has {live}"
    )


def test_the_version_is_consistent_everywhere():
    """pyproject, the package constant and the README badge must agree."""
    tomllib = pytest.importorskip("tomllib")

    from radiance.config.constants import VERSION

    pyproject = tomllib.load(open(_ROOT / "pyproject.toml", "rb"))["project"]["version"]
    badge = re.search(r"version-([\d.]+)-", _src("README.md"))
    assert pyproject == VERSION, (
        f"pyproject says {pyproject}, config/constants.py says {VERSION}"
    )
    assert badge and badge.group(1) == VERSION, (
        f"the README badge says {badge.group(1) if badge else 'nothing'}, "
        f"the package says {VERSION}"
    )


def test_the_changelog_has_an_entry_for_this_version():
    from radiance.config.constants import VERSION

    changelog = _src("CHANGELOG.md")
    assert f"## [{VERSION}]" in changelog, (
        f"CHANGELOG.md has no entry for {VERSION}; the newest release is "
        "undocumented"
    )


def test_the_changelog_records_the_known_limitations():
    """Shipping a known-wrong tone scale silently would repeat the whole problem."""
    changelog = _src("CHANGELOG.md")
    section = changelog[changelog.index("## [3.2.0]"):]
    section = section[:section.index("\n## [")]
    assert "Known limitations" in section
    for topic in ("tone scale", "chromatic_adaptation", "laplacian_pyramid"):
        assert topic in section, f"the {topic!r} limitation is not documented"

# ── The generated reference must be current ────────────────────────────────

def test_the_generated_reference_is_up_to_date():
    """`tools/generate_docs.py --check` in test form.

    The node pages are generated from the live catalog. If someone adds a node,
    changes a default or edits a tooltip without regenerating, this fails and
    names the stale file.
    """
    import radiance

    if radiance._LOAD_RESULT.failures:
        pytest.skip("environment is short a runtime dependency")

    sys.path.insert(0, str(_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "_radiance_docgen", _ROOT / "tools" / "generate_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stale = []
    evidence = []
    for rel, text in sorted(mod.build(radiance).items()):
        path = _ROOT / rel
        if not path.exists():
            stale.append(rel)
            evidence.append(f"{rel}: file does not exist")
        elif path.read_text(encoding="utf-8") != text:
            stale.append(rel)
            # AUDIT (2026-08): a bare filename told us nothing when CI's
            # regeneration differed from a file that every local environment
            # reproduced byte-for-byte. Carry the actual difference in the
            # failure so the divergent environment identifies itself.
            import difflib
            diff = list(difflib.unified_diff(
                path.read_text(encoding="utf-8").splitlines(),
                text.splitlines(),
                f"{rel} (committed)", f"{rel} (regenerated here)",
                lineterm="", n=1))
            evidence.append("\n".join(diff[:60]))
    assert not stale, (
        f"{stale} are out of date. Run: python tools/generate_docs.py --stubs\n"
        "First divergence:\n" + "\n\n".join(evidence)
    )


@pytest.mark.parametrize("page", [
    "README.md", "quickstart.md", "concepts.md", "color-management.md",
    "viewer-and-delivery.md", "workflows.md", "nodes.md", "limitations.md",
    "troubleshooting.md", "glossary.md", "developer.md", "coverage.md",
])
def test_every_page_the_index_promises_exists(page):
    assert (_ROOT / "docs" / page).is_file(), f"docs/{page} is linked but missing"


def test_no_internal_doc_link_is_broken():
    """Relative markdown links inside docs/ must resolve."""
    broken = []
    docs = _ROOT / "docs"
    for md in sorted(docs.rglob("*.md")):
        for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", md.read_text(encoding="utf-8")):
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(_ROOT)} -> {target}")
    assert not broken, "broken relative links: " + "; ".join(broken)


def test_the_limitations_page_covers_what_the_changelog_admits():
    """The two must not disagree about what is broken.

    Whitespace is normalised before matching: a topic that happens to fall
    across a line break in the prose is still documented, and a test that says
    otherwise is testing the line wrapping.

    'display window' was on this list until 3.2.0, when the reader started
    conforming to it. What remains is that the overscan is discarded rather
    than carried as a bounding box, so 'overscan' is the topic now.
    """
    limitations = " ".join(_src("docs/limitations.md").split())
    for topic in ("Daniele Evo", "chromatic_adaptation", "laplacian_pyramid",
                  "Lucas", "overscan"):
        assert topic in limitations, f"{topic!r} is missing from the limitations page"
