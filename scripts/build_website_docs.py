"""Build FXTD website-styled static documentation for Radiance."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DOCS = ROOT / "docs"
WEBSITE_ROOT = Path(r"D:\A.I\Devlopments\website")
OUT_DIR = WEBSITE_ROOT / "docs"


PAGES = [
    ("overview", "Introduction", SOURCE_DOCS / "index.md", "Get Started"),
    ("quickstart", "Quickstart", SOURCE_DOCS / "quickstart.md", "Get Started"),
    ("concepts", "Basic Concepts", SOURCE_DOCS / "concepts.md", "Get Started"),
    ("workflows", "Tutorials", SOURCE_DOCS / "workflows.md", "Learn"),
    ("nodes", "Built-in Nodes", SOURCE_DOCS / "nodes.md", "Reference"),
    ("nodes-io", "IO and Delivery", SOURCE_DOCS / "built-in-nodes" / "io-delivery.md", "Reference"),
    ("nodes-generate", "Generate", SOURCE_DOCS / "built-in-nodes" / "generate.md", "Reference"),
    ("nodes-color", "Color", SOURCE_DOCS / "built-in-nodes" / "color.md", "Reference"),
    ("nodes-hdr", "HDR and ACES", SOURCE_DOCS / "built-in-nodes" / "hdr-aces.md", "Reference"),
    ("nodes-vfx", "VFX", SOURCE_DOCS / "built-in-nodes" / "vfx.md", "Reference"),
    ("nodes-pipeline", "Pipeline", SOURCE_DOCS / "built-in-nodes" / "pipeline.md", "Reference"),
    ("nodes-review", "Review", SOURCE_DOCS / "built-in-nodes" / "review.md", "Reference"),
    ("nodes-upscale", "Upscale", SOURCE_DOCS / "built-in-nodes" / "upscale.md", "Reference"),
    ("nodes-video", "Video", SOURCE_DOCS / "built-in-nodes" / "video.md", "Reference"),
    ("nodes-ai", "AI Assist", SOURCE_DOCS / "built-in-nodes" / "ai-assist.md", "Reference"),
    ("coverage", "Coverage", SOURCE_DOCS / "coverage.md", "Reference"),
    ("developer", "Development", SOURCE_DOCS / "developer.md", "Development"),
    ("troubleshooting", "Troubleshooting", SOURCE_DOCS / "troubleshooting.md", "Support"),
]


def inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<span class=\"doc-link\">\1</span>", escaped)
    return escaped


def render_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        if line.replace("|", "").replace(":", "").replace("-", "").strip() == "":
            continue
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    if not rows:
        return ""
    head, *body = rows
    header = "".join(f"<th>{inline_markdown(cell)}</th>" for cell in head)
    body_html = "\n".join(
        "<tr>" + "".join(f"<td>{inline_markdown(cell)}</td>" for cell in row) + "</tr>"
        for row in body
    )
    return f"<div class=\"table-wrap\"><table><thead><tr>{header}</tr></thead><tbody>{body_html}</tbody></table></div>"


def markdown_to_html(markdown: str) -> str:
    lines = markdown.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    paragraph: list[str] = []
    bullets: list[str] = []
    table: list[str] = []
    code: list[str] = []
    in_code = False
    code_lang = ""

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{inline_markdown(' '.join(paragraph))}</p>")
            paragraph = []

    def flush_bullets() -> None:
        nonlocal bullets
        if bullets:
            out.append("<ul>" + "".join(f"<li>{inline_markdown(item)}</li>" for item in bullets) + "</ul>")
            bullets = []

    def flush_table() -> None:
        nonlocal table
        if table:
            out.append(render_table(table))
            table = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                body = html.escape(chr(10).join(code))
                if code_lang == "mermaid":
                    out.append(f'<div class="diagram"><pre class="mermaid">{body}</pre></div>')
                else:
                    out.append(f"<pre><code>{body}</code></pre>")
                code = []
                in_code = False
                code_lang = ""
            else:
                flush_paragraph()
                flush_bullets()
                flush_table()
                in_code = True
                code_lang = line[3:].strip().lower()
            continue
        if in_code:
            code.append(raw)
            continue
        if not line.strip():
            flush_paragraph()
            flush_bullets()
            flush_table()
            continue
        if re.match(r"^\|.+\|$", line):
            flush_paragraph()
            flush_bullets()
            table.append(line)
            continue
        flush_table()
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            flush_bullets()
            level = min(len(heading.group(1)), 3)
            title = inline_markdown(heading.group(2))
            out.append(f"<h{level}>{title}</h{level}>")
            continue
        bullet = re.match(r"^[-*]\s+(.*)$", line) or re.match(r"^\d+\.\s+(.*)$", line)
        if bullet:
            flush_paragraph()
            bullets.append(bullet.group(1))
            continue
        if line.startswith(">"):
            flush_paragraph()
            flush_bullets()
            out.append(f"<blockquote>{inline_markdown(line.lstrip('> '))}</blockquote>")
            continue
        paragraph.append(line.strip())

    flush_paragraph()
    flush_bullets()
    flush_table()
    return "\n".join(out)


def read_pages() -> list[dict[str, str]]:
    pages = []
    for page_id, title, path, group in PAGES:
        markdown = path.read_text(encoding="utf-8")
        pages.append(
            {
                "id": page_id,
                "title": title,
                "group": group,
                "html": markdown_to_html(markdown),
            }
        )
    return pages


def write_index() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = read_pages()
    data = json.dumps(pages, ensure_ascii=False)
    (OUT_DIR / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (OUT_DIR / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (OUT_DIR / "app.js").write_text(APP_JS.replace("__DOC_DATA__", data), encoding="utf-8")
    style_source = OUT_DIR / "_style-source-radiance.html"
    if style_source.exists():
        style_source.unlink()


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Radiance Documentation · FXTD Studios</title>
  <meta name="description" content="Radiance documentation for ComfyUI HDR, ACES, EXR, VFX, video, review, and DCC workflows." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <nav id="navbar">
    <a class="nav-left" href="#overview">
      <div class="logo-mark">R</div>
      <div class="nav-product">Radiance<span>/</span>Docs</div>
    </a>
    <div class="nav-right">
      <a href="#quickstart" class="nav-link">Start</a>
      <a href="#nodes" class="nav-link">Nodes</a>
      <a href="#workflows" class="nav-link">Workflows</a>
      <a href="../radiance.html" class="nav-link">Product</a>
      <a href="#nodes-io" class="nav-cta">Reference</a>
    </div>
  </nav>

  <header class="hero">
    <div class="hero-grid"></div>
    <div class="hero-glow"></div>
    <div class="hero-inner">
      <div class="hero-badge">ComfyUI Node Documentation</div>
      <h1 class="hero-title">Radiance<br><span class="hero-title-sub">Production Docs</span></h1>
      <p class="hero-desc">HDR imaging, ACES/OCIO color management, EXR delivery, VFX tools, video generation, review, upscaling, and studio handoff documentation in the FXTD site style.</p>
      <div class="hero-actions">
        <a href="#quickstart" class="btn-primary">Get Started →</a>
        <a href="#nodes" class="btn-ghost">Browse Nodes</a>
      </div>
      <div class="hero-meta">
        <div><div class="hero-meta-val">104</div><div class="hero-meta-label">Registered Nodes</div></div>
        <div><div class="hero-meta-val">10</div><div class="hero-meta-label">Node Groups</div></div>
        <div><div class="hero-meta-val">EXR</div><div class="hero-meta-label">First Pipeline</div></div>
      </div>
    </div>
  </header>

  <main class="docs-shell">
    <aside class="docs-sidebar">
      <label class="docs-search">
        <span>Search documentation</span>
        <input id="searchInput" type="search" placeholder="HDR, Nuke, sampler..." />
      </label>
      <div id="sideNav" class="side-nav"></div>
    </aside>
    <article class="docs-content">
      <div class="content-header">
        <p class="section-label" id="pageGroup">Documentation</p>
        <h2 class="section-title" id="pageTitle">Radiance Documentation</h2>
      </div>
      <div id="docContent" class="rendered-doc"></div>
    </article>
  </main>

  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      theme: 'base',
      fontFamily: "'Space Grotesk', sans-serif",
      flowchart: { curve: 'basis', useMaxWidth: false, htmlLabels: true, padding: 22, nodeSpacing: 70, rankSpacing: 80 },
      themeVariables: {
        background: '#0f0f0f',
        primaryColor: '#141414',
        primaryBorderColor: '#c8a96e',
        primaryTextColor: '#ffffff',
        secondaryColor: '#151515',
        tertiaryColor: '#0f0f0f',
        lineColor: '#888888',
        textColor: '#cfcfcf',
        edgeLabelBackground: '#080808',
        clusterBkg: '#0b0b0b',
        clusterBorder: 'rgba(255,255,255,0.12)',
        fontSize: '20px'
      }
    });
    window.__renderMermaid = async () => {
      const nodes = document.querySelectorAll('pre.mermaid:not([data-processed])');
      if (!nodes.length) return;
      try { await mermaid.run({ nodes: Array.from(nodes) }); }
      catch (e) { console.error('mermaid render failed', e); }
    };
    window.__renderMermaid();
  </script>
  <script src="app.js"></script>
</body>
</html>
"""


STYLES_CSS = r"""
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #080808;
  --bg-card: #0f0f0f;
  --bg-card-hover: #151515;
  --border: rgba(255,255,255,0.08);
  --border-hover: rgba(255,255,255,0.18);
  --text-primary: #ffffff;
  --text-secondary: #888888;
  --text-muted: #444444;
  --font-display: 'Space Grotesk', sans-serif;
  --font-body: 'Inter', sans-serif;
  --accent: #c8a96e;
}
html { scroll-behavior: smooth; }
body {
  background: var(--bg);
  color: var(--text-primary);
  font-family: var(--font-body);
  font-size: 15px;
  line-height: 1.6;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
}
a { color: inherit; text-decoration: none; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
nav {
  position: fixed; top: 0; left: 0; right: 0; z-index: 100;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 40px; height: 60px;
  background: rgba(8,8,8,0.95); backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
}
.nav-left { display: flex; align-items: center; gap: 16px; }
.logo-mark {
  width: 30px; height: 30px; border: 1.5px solid var(--text-primary);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-display); font-size: 14px; font-weight: 700;
}
.nav-product {
  font-family: var(--font-display); font-size: 13px; font-weight: 600;
  letter-spacing: 0.12em; text-transform: uppercase;
}
.nav-product span { color: var(--text-muted); margin: 0 8px; font-weight: 400; }
.nav-right { display: flex; align-items: center; gap: 24px; }
.nav-link {
  font-size: 11px; font-weight: 500; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--text-secondary);
  transition: color 0.2s;
}
.nav-link:hover { color: var(--text-primary); }
.nav-cta {
  font-size: 11px; font-weight: 500; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--text-primary);
  border: 1px solid rgba(255,255,255,0.25); padding: 8px 20px;
  background: transparent; transition: all 0.2s;
}
.nav-cta:hover { background: rgba(255,255,255,0.06); border-color: var(--text-primary); }
.hero {
  min-height: 88vh; display: flex; align-items: center;
  padding: 120px 40px 80px; position: relative; overflow: hidden;
  border-bottom: 1px solid var(--border);
}
.hero-grid {
  position: absolute; inset: 0; z-index: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
  background-size: 100px 100px;
}
.hero-glow {
  position: absolute; top: 18%; right: -12%; width: 64vw; height: 64vw;
  background: radial-gradient(ellipse at center, rgba(200,169,110,0.055) 0%, transparent 65%);
  pointer-events: none;
}
.hero-inner { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; width: 100%; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 10px; font-weight: 500; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--accent);
  border: 1px solid rgba(200,169,110,0.25); padding: 5px 14px;
  margin-bottom: 32px;
}
.hero-badge::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
.hero-title {
  font-family: var(--font-display);
  font-size: clamp(56px, 9vw, 128px);
  font-weight: 700; line-height: 0.9;
  letter-spacing: -0.025em; margin-bottom: 36px;
}
.hero-title-sub { display: block; color: var(--text-muted); font-size: clamp(36px, 6vw, 82px); }
.hero-desc { font-size: 16px; color: var(--text-secondary); max-width: 560px; line-height: 1.75; margin-bottom: 42px; }
.hero-actions { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.btn-primary, .btn-ghost {
  display: inline-flex; align-items: center; gap: 10px;
  font-size: 12px; font-weight: 500; letter-spacing: 0.15em;
  text-transform: uppercase; padding: 16px 32px;
  border: 1px solid var(--text-primary); transition: all 0.2s;
}
.btn-primary { background: var(--text-primary); color: var(--bg); }
.btn-primary:hover { background: rgba(255,255,255,0.9); }
.btn-ghost { border-color: rgba(255,255,255,0.2); color: var(--text-primary); background: transparent; }
.btn-ghost:hover { border-color: var(--text-primary); background: rgba(255,255,255,0.04); }
.hero-meta { display: flex; gap: 48px; margin-top: 72px; padding-top: 36px; border-top: 1px solid var(--border); }
.hero-meta-val { font-family: var(--font-display); font-size: 28px; font-weight: 700; line-height: 1; margin-bottom: 6px; }
.hero-meta-label { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-muted); }
.docs-shell {
  display: grid; grid-template-columns: 300px minmax(0, 1fr);
  max-width: 1440px; margin: 0 auto; padding: 0 40px 120px;
}
.docs-sidebar {
  position: sticky; top: 60px; height: calc(100vh - 60px);
  overflow-y: auto; padding: 40px 28px 40px 0;
  border-right: 1px solid var(--border);
}
.docs-search { display: block; margin-bottom: 28px; }
.docs-search span {
  display: block; font-size: 10px; font-weight: 500; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px;
}
.docs-search input {
  width: 100%; background: var(--bg-card); border: 1px solid var(--border);
  color: var(--text-primary); padding: 12px 14px; font: inherit; outline: none;
}
.docs-search input:focus { border-color: rgba(200,169,110,0.45); }
.side-group {
  font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--text-muted); margin: 28px 0 8px;
}
.side-nav a {
  display: flex; justify-content: space-between; gap: 12px;
  padding: 8px 0; color: var(--text-secondary);
  font-size: 13px; border-bottom: 1px solid rgba(255,255,255,0.035);
}
.side-nav a:hover, .side-nav a.active { color: var(--text-primary); }
.side-nav small { color: var(--text-muted); }
.docs-content { min-width: 0; padding: 52px 0 0 56px; }
.content-header { margin-bottom: 36px; }
.section-label {
  font-size: 11px; font-weight: 500; letter-spacing: 0.22em;
  text-transform: uppercase; color: var(--text-muted); margin-bottom: 20px;
}
.section-title {
  font-family: var(--font-display);
  font-size: clamp(32px, 4vw, 52px);
  font-weight: 700; line-height: 1.05; margin-bottom: 16px;
}
.rendered-doc { max-width: 920px; }
.rendered-doc h1 { display: none; }
.rendered-doc h2 {
  font-family: var(--font-display); font-size: 28px; line-height: 1.1;
  margin: 52px 0 18px; padding-top: 32px; border-top: 1px solid var(--border);
}
.rendered-doc h3 {
  font-family: var(--font-display); font-size: 17px; margin: 30px 0 12px;
}
.rendered-doc p, .rendered-doc li { color: var(--text-secondary); }
.rendered-doc p { margin: 0 0 18px; max-width: 760px; }
.rendered-doc ul { margin: 0 0 24px 18px; }
.rendered-doc li { margin: 8px 0; }
.table-wrap { overflow-x: auto; margin: 20px 0 32px; border: 1px solid var(--border); background: var(--bg-card); }
table { width: 100%; border-collapse: collapse; min-width: 680px; }
th, td { padding: 13px 14px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--border); }
th {
  font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--text-primary); font-weight: 500; background: #0b0b0b;
}
td { color: var(--text-secondary); font-size: 13px; }
tr:last-child td { border-bottom: 0; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.9em; color: var(--accent);
  background: rgba(200,169,110,0.07); border: 1px solid rgba(200,169,110,0.14);
  padding: 1px 5px;
}
pre {
  background: #050505; border: 1px solid var(--border);
  padding: 18px; margin: 18px 0 28px; overflow-x: auto;
}
pre code { background: transparent; border: 0; padding: 0; color: var(--text-primary); }
blockquote {
  border-left: 2px solid var(--accent); padding: 12px 18px;
  background: rgba(200,169,110,0.035); color: var(--text-secondary);
  margin: 20px 0;
}
mark { background: rgba(200,169,110,0.3); color: var(--text-primary); }
.doc-link { color: var(--accent); }
.diagram {
  margin: 32px 0; padding: 32px 28px; width: 100%;
  background: radial-gradient(120% 140% at 50% 0%, #101010 0%, var(--bg-card) 60%, #0b0b0b 100%);
  border: 1px solid var(--border); border-radius: 14px;
  overflow-x: auto; text-align: center;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 40px -28px rgba(0,0,0,0.9);
}
.diagram::before {
  content: "WORKFLOW"; display: block; margin-bottom: 16px;
  font-family: var(--font-display); font-size: 10px; font-weight: 600;
  letter-spacing: 0.22em; color: var(--accent); text-transform: uppercase; opacity: 0.7;
}
pre.mermaid {
  background: transparent !important; border: 0; padding: 0; margin: 0;
  display: inline-block; min-width: 0; line-height: normal;
  font-family: var(--font-display);
}
pre.mermaid:not([data-processed]) { color: var(--text-muted); font-size: 12px; }
.diagram svg { width: 100%; max-width: 960px; height: auto; }
.diagram .nodeLabel, .diagram .edgeLabel, .diagram .label { font-size: 16px; }
.diagram .edgeLabel { color: var(--text-secondary); }
@media (max-width: 900px) {
  nav { padding: 0 20px; }
  .nav-right .nav-link { display: none; }
  .hero { padding: 110px 20px 64px; }
  .hero-meta { flex-direction: column; gap: 20px; }
  .docs-shell { display: block; padding: 0 20px 80px; }
  .docs-sidebar { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--border); padding: 28px 0; }
  .docs-content { padding: 40px 0 0; }
}
"""


APP_JS = r"""
const pages = __DOC_DATA__;
const sideNav = document.getElementById('sideNav');
const content = document.getElementById('docContent');
const title = document.getElementById('pageTitle');
const group = document.getElementById('pageGroup');
const search = document.getElementById('searchInput');

function renderNav() {
  let current = '';
  sideNav.innerHTML = pages.map((page) => {
    const groupLabel = page.group !== current ? `<div class="side-group">${page.group}</div>` : '';
    current = page.group;
    return `${groupLabel}<a href="#${page.id}" data-page="${page.id}"><span>${page.title}</span><small>${page.group === 'Reference' ? 'ref' : ''}</small></a>`;
  }).join('');
}

function activeId() {
  return location.hash.replace('#', '') || 'overview';
}

function highlight(html, query) {
  const q = query.trim();
  if (!q) return html;
  return html.replace(new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'), '<mark>$1</mark>');
}

function renderPage() {
  const page = pages.find((item) => item.id === activeId()) || pages[0];
  title.textContent = page.title;
  group.textContent = page.group;
  content.innerHTML = highlight(page.html, search.value);
  document.querySelectorAll('.side-nav a').forEach((link) => {
    link.classList.toggle('active', link.dataset.page === page.id);
  });
  if (location.hash.replace('#', '') !== page.id) {
    history.replaceState(null, '', `#${page.id}`);
  }
  if (window.__renderMermaid) window.__renderMermaid();
}

renderNav();
renderPage();
window.addEventListener('hashchange', renderPage);
search.addEventListener('input', renderPage);
window.addEventListener('scroll', () => {
  document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 20);
});
"""


if __name__ == "__main__":
    write_index()
    print(f"Wrote {OUT_DIR}")
