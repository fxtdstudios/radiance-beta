const pages = [
  { id: "overview", title: "Introduction", file: "../index.md", short: "Start", group: "Get Started" },
  { id: "quickstart", title: "Quickstart", file: "../quickstart.md", short: "Setup", group: "Get Started" },
  { id: "concepts", title: "Basic Concepts", file: "../concepts.md", short: "Theory", group: "Get Started" },
  { id: "workflows", title: "Tutorials", file: "../workflows.md", short: "Recipes", group: "Learn" },
  { id: "nodes", title: "Built-in Nodes", file: "../nodes.md", short: "103", group: "Reference" },
  { id: "nodes-io", title: "IO and Delivery", file: "../built-in-nodes/io-delivery.md", short: "4", group: "Reference" },
  { id: "nodes-generate", title: "Generate", file: "../built-in-nodes/generate.md", short: "13", group: "Reference" },
  { id: "nodes-color", title: "Color", file: "../built-in-nodes/color.md", short: "15", group: "Reference" },
  { id: "nodes-hdr", title: "HDR and ACES", file: "../built-in-nodes/hdr-aces.md", short: "15", group: "Reference" },
  { id: "nodes-vfx", title: "VFX", file: "../built-in-nodes/vfx.md", short: "23", group: "Reference" },
  { id: "nodes-pipeline", title: "Pipeline", file: "../built-in-nodes/pipeline.md", short: "8", group: "Reference" },
  { id: "nodes-review", title: "Review", file: "../built-in-nodes/review.md", short: "7", group: "Reference" },
  { id: "nodes-upscale", title: "Upscale", file: "../built-in-nodes/upscale.md", short: "4", group: "Reference" },
  { id: "nodes-video", title: "Video", file: "../built-in-nodes/video.md", short: "12", group: "Reference" },
  { id: "nodes-ai", title: "AI Assist", file: "../built-in-nodes/ai-assist.md", short: "2", group: "Reference" },
  { id: "coverage", title: "Coverage Ledger", file: "../coverage.md", short: "Check", group: "Reference" },
  { id: "developer", title: "Development Guide", file: "../developer.md", short: "Dev", group: "Development" },
  { id: "troubleshooting", title: "Troubleshooting", file: "../troubleshooting.md", short: "Fixes", group: "Support" },
];

const fallbackMarkdown = {
  overview: `# Radiance Documentation

Radiance is a production-oriented ComfyUI custom node pack for HDR image handling, ACES/OCIO color management, VFX plate preparation, review, video workflows, upscaling, and DCC handoff.

## Getting Started

| Page | Use it for |
| :--- | :--- |
| Quickstart | Installing Radiance and building a first graph. |
| Basic Concepts | HDR, EXR, ACES/OCIO, tensors, and DCC handoff. |
| Built-in Nodes | End-user descriptions for the registered Radiance catalog. |
| Tutorials | Production recipes for HDR, VFX, video, and review. |

## Production Rules

- Use EXR for HDR and float master output.
- Keep color transforms explicit.
- Match paired HDR encode/decode settings.
- Review with diagnostics and scopes before delivery.`,
  quickstart: `# Quickstart

## Install

\`\`\`bash
cd ComfyUI/custom_nodes
git clone https://github.com/fxtdstudios/radiance.git
cd radiance
pip install -r requirements.txt
\`\`\`

## First Image Graph

\`\`\`text
Radiance Read -> Radiance Viewer -> Radiance Grade -> Radiance Write
\`\`\`

## First HDR Graph

\`\`\`text
Radiance Read -> HDR Auto Log Select -> HDR Color Pipeline -> HDR Monitor -> Radiance Write
\`\`\`

Use EXR for master output and PNG/JPEG only for review proxies.`,
  concepts: `# Basic Concepts

## Image Range

| Term | Meaning |
| :--- | :--- |
| Scene-linear | Pixel values represent scene light linearly. |
| Display-referred | Pixel values are shaped for a display. |
| HDR | Values may carry detail above display white. |
| SDR | Output is usually constrained to display range. |

## ACES and OCIO

Use Radiance color nodes to make source, working, and display spaces explicit. Do not rely on viewport appearance as the master color transform.

## DCC Handoff

Nuke handoff uses a local listener. Resolve handoff is folder based by default.`,
  workflows: `# Tutorials

## HDR EXR Roundtrip

\`\`\`text
Radiance Read -> HDR Auto Log Select -> HDR Color Pipeline -> Generate/VFX -> HDR Diagnostics -> Radiance Write
\`\`\`

## VFX Plate Prep

\`\`\`text
Radiance Read -> Subpixel Plate Stabilizer -> Depth Map Generator / Optical Flow / SAM Mask Generator
\`\`\`

## Video Generation

\`\`\`text
Video Model Info -> Video Latent Noise -> Video Cond Merge -> Video Sampler -> Video Batch Decode -> Video Export
\`\`\`

## DCC Export

Write EXR or high-quality sequence output before sending to Nuke or Resolve.`,
  nodes: `# Built-in Nodes

The full Markdown reference documents 103 registered Radiance nodes across IO, Generate, Color, HDR, VFX, Pipeline, Review, Upscale, Video, and AI Assist.

## Main Groups

| Group | Examples |
| :--- | :--- |
| IO and Delivery | Radiance Read, Radiance Write, EXR Multi-Part |
| Generate | Sampler Pro, Read Models, LoRA Stack, Cinematic Prompt Encoder |
| Color | Grade, CDL, Curves, White Balance, OCIO Context |
| HDR | HDR Color Pipeline, ACES 2.0 Output Transform, HDR Diagnostics |
| VFX | Depth, Optical Flow, SAM, Multipass, Inpaint Crop/Stitch |
| Video | T2V, I2V, Video Sampler, Video Export |

Open this site through the local server to load the complete generated node table.`,
  coverage: `# Coverage Ledger

The documentation coverage check verified 103 registered node keys.

| Result | Count |
| :--- | ---: |
| Registered nodes | 103 |
| Missing from node reference | 0 |
| Missing from coverage ledger | 0 |`,
  developer: `# Development Guide

Radiance loads grouped node packages through the registry in \`nodes/catalog.py\`.

## Add a Node

1. Add the node class.
2. Register it in \`NODE_CLASS_MAPPINGS\`.
3. Add a display name.
4. Add tests.
5. Update node docs and coverage.`,
  troubleshooting: `# Troubleshooting

## Common Fixes

| Symptom | Fix |
| :--- | :--- |
| Very few nodes appear | Install requirements in the ComfyUI Python environment. |
| EXR save fails | Install OpenEXR and Imath. |
| Highlights clip | Use EXR and HDR diagnostics. |
| Nuke send fails | Start the local Nuke listener and verify host/port/token. |`,
};

const nav = document.getElementById("nav");
const content = document.getElementById("content");
const pageTitle = document.getElementById("pageTitle");
const pageLabel = document.getElementById("pageLabel");
const sourceLink = document.getElementById("sourceLink");
const searchInput = document.getElementById("searchInput");

const cache = new Map();

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function inlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
}

function renderTable(lines) {
  const rows = lines
    .filter((line) => line.replace(/[|\s:-]/g, "") !== "")
    .map((line) => line.trim().slice(1, -1).split("|").map((cell) => cell.trim()));
  if (!rows.length) return "";
  const [head, ...body] = rows;
  const header = `<thead><tr>${head.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead>`;
  const rowsHtml = body
    .map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`)
    .join("");
  return `<table>${header}<tbody>${rowsHtml}</tbody></table>`;
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let list = [];
  let table = [];
  let code = [];
  let inCode = false;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (!list.length) return;
    html.push(`<ul>${list.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>`);
    list = [];
  };

  const flushTable = () => {
    if (!table.length) return;
    html.push(renderTable(table));
    table = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();

    if (line.startsWith("```")) {
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
        code = [];
        inCode = false;
      } else {
        flushParagraph();
        flushList();
        flushTable();
        inCode = true;
      }
      continue;
    }

    if (inCode) {
      code.push(rawLine);
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      flushTable();
      continue;
    }

    if (/^\|.+\|$/.test(line)) {
      flushParagraph();
      flushList();
      table.push(line);
      continue;
    }

    flushTable();

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length, 3);
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = line.match(/^[-*]\s+(.*)$/);
    if (bullet) {
      flushParagraph();
      list.push(bullet[1]);
      continue;
    }

    const numbered = line.match(/^\d+\.\s+(.*)$/);
    if (numbered) {
      flushParagraph();
      list.push(numbered[1]);
      continue;
    }

    if (line.startsWith(">")) {
      flushParagraph();
      flushList();
      html.push(`<blockquote>${inlineMarkdown(line.replace(/^>\s?/, ""))}</blockquote>`);
      continue;
    }

    paragraph.push(line.trim());
  }

  flushParagraph();
  flushList();
  flushTable();
  return html.join("\n");
}

function highlight(html, query) {
  const q = query.trim();
  if (!q) return html;
  const escaped = q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return html.replace(new RegExp(`(${escaped})`, "gi"), "<mark>$1</mark>");
}

function setActive(id) {
  document.querySelectorAll(".nav a").forEach((link) => {
    link.classList.toggle("active", link.dataset.page === id);
  });
}

async function loadPage(id, query = "") {
  const page = pages.find((item) => item.id === id) || pages[0];
  setActive(page.id);
  pageTitle.textContent = page.title;
  pageLabel.textContent = page.id === "nodes" ? "Full Catalog" : "Documentation";
  sourceLink.href = page.file;
  sourceLink.textContent = "View Markdown";

  try {
    if (location.protocol === "file:") {
      cache.set(page.id, fallbackMarkdown[page.id]);
    } else if (!cache.has(page.id)) {
      const response = await fetch(page.file);
      if (!response.ok) throw new Error(`Could not load ${page.file}`);
      cache.set(page.id, await response.text());
    }
    const markdown = cache.get(page.id);
    content.innerHTML = highlight(renderMarkdown(markdown), query);
  } catch (error) {
    const fallback = fallbackMarkdown[page.id] || `# ${page.title}

This page is part of the full Radiance documentation website.

Open the local server version to load the complete Markdown reference:

\`\`\`text
http://127.0.0.1:8787/site/#${page.id}
\`\`\`

The source file for this page is \`${page.file.replace("../", "docs/")}\`.`;
    if (fallback) {
      content.innerHTML = highlight(renderMarkdown(fallback), query);
      return;
    }
    content.innerHTML = `
      <div class="empty-state">
        <strong>This website should be opened through a local web server.</strong>
        <p>From the repository, run <code>python -m http.server 8787 -d docs</code>, then open <code>http://localhost:8787/site/</code>.</p>
      </div>
    `;
  }

  if (location.hash.slice(1) !== page.id) {
    history.replaceState(null, "", `#${page.id}`);
  }
}

function renderNav() {
  let currentGroup = "";
  nav.innerHTML = pages.map((page) => {
    const group = page.group !== currentGroup ? `<div class="nav-group">${page.group}</div>` : "";
    currentGroup = page.group;
    return `${group}<a href="#${page.id}" data-page="${page.id}"><span>${page.title}</span><small>${page.short}</small></a>`;
  }).join("");
}

function currentPageId() {
  return location.hash.replace("#", "") || "overview";
}

renderNav();
loadPage(currentPageId());

window.addEventListener("hashchange", () => loadPage(currentPageId(), searchInput.value));

searchInput.addEventListener("input", () => {
  loadPage(currentPageId(), searchInput.value);
});

window.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    searchInput.focus();
  }
});
