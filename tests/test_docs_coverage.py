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
import pathlib
import re

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
