#!/usr/bin/env node
/**
 * scripts/build_js.js — Radiance v3.1 JavaScript build pipeline
 *
 * Minifies all JS files in js/ into build/js/ using esbuild.
 * The build/ directory is outside js/ so ComfyUI won't load the
 * built files as extensions (it recursively loads all .js files
 * from the custom node's js/ directory).
 *
 * Usage
 * -----
 *   node scripts/build_js.js           # one-shot build
 *   node scripts/build_js.js --watch   # rebuild on file change
 *   npm run build                       # alias for one-shot
 *   npm run build:watch                 # alias for watch mode
 *
 * Output
 * ------
 *   build/js/<name>.min.js   — minified bundle for each source file
 *   build/js/<name>.min.js.map — source map for debugging
 *
 * Notes
 * -----
 *   - Source files are ES modules loaded directly by ComfyUI via
 *     dynamic import(). Built files stay as browser ES modules so
 *     ComfyUI can load them without CommonJS shims.
 *   - Local Radiance imports are bundled; ComfyUI's own ../../scripts/*
 *     imports stay external because they are provided by the host app.
 *   - esbuild preserves string literals that start with "use strict"
 *     and legal comments (/*! ... ) for licence compliance.
 *   - Run `npm install` once to install esbuild before building.
 */

const esbuild = require("esbuild");
const path    = require("path");
const fs      = require("fs");

const ROOT    = path.resolve(__dirname, "..");
const SRC_DIR = path.join(ROOT, "js");
const OUT_DIR = path.join(ROOT, "build", "js");

const watchMode = process.argv.includes("--watch");

const COMFY_EXTERNAL_IMPORTS = [
  "../../scripts/*",
  "../../../scripts/*",
];

// Collect all .js files in js/ (not recursively — skip js/dist/ itself)
const entryPoints = fs
  .readdirSync(SRC_DIR)
  .filter(f => f.endsWith(".js") && !f.startsWith("."))
  .map(f => path.join(SRC_DIR, f));

if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

const buildOptions = {
  entryPoints,
  outdir: OUT_DIR,
  outExtension: { ".js": ".min.js" },
  minify: true,
  sourcemap: true,
  legalComments: "inline",   // preserve /*! licence blocks
  target: ["chrome110", "firefox110", "safari16"],
  logLevel: "info",
  bundle: true,              // inline local Radiance modules per entry point
  format: "esm",             // browser-native modules; no require() wrappers
  external: COMFY_EXTERNAL_IMPORTS,
};

if (watchMode) {
  esbuild.context(buildOptions).then(ctx => {
    ctx.watch();
    console.log("[radiance] Watching js/ for changes …");
  }).catch(() => process.exit(1));
} else {
  esbuild.build(buildOptions)
    .then(result => {
      if (result.errors.length === 0) {
        console.log(`[radiance] Build complete — ${entryPoints.length} files → build/js/`);
      }
    })
    .catch(() => process.exit(1));
}
