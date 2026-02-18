/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                         FXTD RADIANCE VIEWER
 *                  VFX Industry-Standard Image Viewer
 *                        FXTD Studios © 2024-2026
 * 
 * Full Feature Set:
 * - Fullscreen (native browser), keyboard shortcuts
 * - Pixel probe with float values + copy to clipboard
 * - A/B comparison (wipe, side-by-side, difference)
 * - Professional scopes (Histogram, Waveform, Vectorscope) + overlay mode
 * - Annotations (circle, arrow, rectangle)
 * - Grid overlay (rule of thirds, center)
 * - Export snapshot, reset controls
 * ═══════════════════════════════════════════════════════════════════════════════
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { RadianceWebGLRenderer } from "./radiance_webgl.js?v=2.6.0";


class RadianceViewer {
    constructor(node, container) {
        this.node = node;
        this.container = container;

        // v3.0: Cinema Scope Fonts
        if (!document.getElementById('radiance-fonts')) {
            const fontLink = document.createElement('link');
            fontLink.id = 'radiance-fonts';
            fontLink.rel = 'stylesheet';
            fontLink.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap';
            document.head.appendChild(fontLink);
        }

        this.theme = {
            bg: '#0a0a0f',
            panel: 'rgba(16, 16, 24, 0.95)',
            panelBorder: 'rgba(60, 70, 100, 0.25)',
            accent: '#00a8ff',
            text: '#e8e8f0',
            textDim: '#707088',
            font: "'Inter', system-ui, -apple-system, sans-serif",
            mono: "'JetBrains Mono', monospace"
        };

        // State
        this.image = null;
        this.compareImage = null;
        this.imageWidth = 0;
        this.imageHeight = 0;
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.isPanning = false;
        this.exposure = 0.0;
        this.lift = [0, 0, 0];
        this.gamma = [1, 1, 1];
        this.gain = [1, 1, 1];
        this.channel = 'rgb';

        // DoF / Lens Settings
        this.dofEnabled = false;
        this.focusDistance = 0.5;
        this.aperture = 0.0;
        this.apertureBlades = 0;
        this.apertureRotation = 0.0;
        this.apertureAnamorphic = 1.0;
        this.lensDistortion = 0.0;
        this.lensFringe = 0.0;
        this.vignetteIntensity = 0.0;
        this.vignetteFalloff = 0.5;


        this.saturation = 1.0;
        this.zebraThreshold = 0.95;
        this.lutIntensity = 1.0;

        this.falseColor = false;
        this.zebra = false;
        this.colorspace = 'sRGB';

        // Batch Navigation
        this.currentFrame = 0;
        this.totalFrames = 1;
        this.frameImages = [];
        this.frameCompareImages = [];
        this.frameZdepthImages = [];  // Z-Depth frames
        this.zdepthImage = null;       // Current zdepth image
        this.showZdepth = false;       // Toggle for zdepth display

        // Safe Area Guides
        this.safeAreaMode = 'none'; // none, action, title, both

        // Focus Peaking
        this.focusPeaking = false;
        this.focusPeakingColor = '#ff0000';
        this.focusPeakingThreshold = 30;

        // Pixel Loupe
        this.showLoupe = true;
        this.loupeSize = 80;
        this.loupeMagnification = 8;

        // Comparison
        this.compareMode = 'none';
        this.wipePosition = 0.5;
        this.isDraggingWipe = false;

        // Scopes
        this.showHistogram = false;
        this.showWaveform = false;
        this.showVectorscope = false;
        this.scopeOverlay = false;
        this.waveformParadeMode = false; // false = overlay, true = RGB parade
        this.scopeMode = localStorage.getItem('radiance_scope_mode') || 'parade'; // parade|waveform|histogram|vectorscope|falsecolor

        // Annotations
        this.annotations = [];
        this.isAnnotating = false;
        this.annotationTool = 'pen'; // pen, circle, arrow, rect, text
        this.annotationStart = null;
        this.annotationColor = '#ff4444';
        this.annotationLineWidth = 3;
        this.currentPath = null; // For pen drawing

        // Grid & Safe Areas
        this.showGrid = false;
        this.gridMode = 0; // 0=off, 1=thirds, 2=safe areas, 3=center

        // Measurement Tools
        this.measurementMode = 'none'; // none, distance, angle
        this.measurements = [];
        this.currentMeasurement = null;

        // Fullscreen
        this.isFullscreen = false;

        // Pixel data
        this.imageData = null;
        this.lastPixelColor = null;

        this.initialized = false; // Track if we've set initial size

        // Run & Prompt
        this.showPromptPanel = false;
        this.promptPanel = null;
        this.isQueueing = false;

        // Progress
        this.progressBar = null;
        this.progressText = null;
        this.progressStart = 0;

        this.progressHistory = [];

        // Color Space / LUT
        this.displayLut = 'None';
        this.lutOptions = ['None', 'sRGB', 'Rec.709', 'LogC3', 'ACEScg'];
        this.denoise = 0.0;
        this.grain = 0.0;

        // Help Screen
        this.showHelp = false;
        this.helpPanel = null;

        // A/B Grading Bypass
        this._gradingBypassed = false;
        this._savedGrading = null;

        // Undo / Redo Stack
        this._undoStack = [];
        this._redoStack = [];
        this._undoMaxSize = 50;

        // HUD / Controls Panel
        this.showControls = true;
        this.controlsPanel = null;

        // HUD Panel Sizing (persisted)
        const savedHudWidth = localStorage.getItem('radiance_hud_width');
        this.hudPanelWidth = savedHudWidth ? parseInt(savedHudWidth) : 580;
        this.hudPanelMinWidth = 480;
        this.hudPanelMaxWidth = 960;

        // Scope update debouncing (for performance)
        this.scopeUpdateTimer = null;
        this.scopeDebounceMs = 150; // Reduced for snappier feedback (industry standard: 100-200ms)

        // Scope Panel Sizing
        // Load saved width from localStorage, default to 280px
        const savedWidth = localStorage.getItem('radiance_scope_width');
        this.scopePanelWidth = savedWidth ? parseInt(savedWidth) : 280;
        this.scopePanelMinWidth = 200;
        this.scopePanelMaxWidth = 600;
        this.isResizingScopePanel = false;

        // v2.5: Qualifier State Initialization
        this.qualifierState = {
            enabled: false,
            showMask: false,
            h: 0.0, hW: 0.1, hS: 0.05,
            s: 0.5, sW: 0.5, sS: 0.1,
            l: 0.5, lW: 0.5, lS: 0.1
        };

        this.init();
    }

    init() {
        this.createUI();

        this.setupProgressUI();
        this.setupEventListeners();
        this.setupKeyboardShortcuts();
        requestAnimationFrame(() => this.resize());

        // Immersive Mode: Disabled by user request
        // (UI remains permanently visible)
    }

    showUI(visible) {
        const opacity = visible ? '1' : '0';
        const pointerEvents = visible ? 'all' : 'none';
        const transform = visible ? 'translateX(-50%)' : 'translate(-50%, 20px)';

        if (this.controlsPanel) {
            this.controlsPanel.style.opacity = opacity;
            this.controlsPanel.style.transform = transform;
            this.controlsPanel.style.pointerEvents = pointerEvents;
        }
        if (this.toolbar) {
            this.toolbar.style.opacity = opacity;
            this.toolbar.style.pointerEvents = pointerEvents;
            this.toolbar.style.transition = 'opacity 0.4s';
        }
    }

    createUI() {
        const t = this.theme;

        this.container.style.cssText = `
            position: relative;
            width: 100%;
            height: 100%;
            min-height: 300px;
            background: linear-gradient(180deg, #0f0f14 0%, #08080c 100%);
            border-radius: 8px;
            overflow: hidden;
            user-select: none;
            border: 1px solid ${t.panelBorder};
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            display: flex;
            flex-direction: column;
        `;

        // Toolbar
        this.toolbar = document.createElement('div');
        this.toolbar.style.cssText = `
            flex: 0 0 32px;
            display: flex;
            align-items: center;
            gap: 2px;
            padding: 0 6px;
            background: ${t.panel};
            border-bottom: 1px solid ${t.panelBorder};
            overflow-x: auto;
        `;
        this.container.appendChild(this.toolbar);
        this.createToolbar();

        // Main area
        this.mainArea = document.createElement('div');
        this.mainArea.style.cssText = `flex: 1; display: flex; position: relative; overflow: hidden;`;
        this.container.appendChild(this.mainArea);

        // Canvas wrapper
        this.canvasWrapper = document.createElement('div');
        this.canvasWrapper.style.cssText = `flex: 1; position: relative; overflow: hidden;`;
        this.mainArea.appendChild(this.canvasWrapper);

        // Create HUD (floating controls)
        this.createHUD();

        // Main canvas (2D fallback)
        this.canvas = document.createElement('canvas');
        this.canvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair;`;
        this.canvas.tabIndex = 0;
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false });
        this.canvasWrapper.appendChild(this.canvas);

        // WebGL Canvas (Primary renderer - GPU accelerated)
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;`;
        this.canvasWrapper.appendChild(this.glCanvas); // Insert after 2D canvas so it overlays

        // Initialize WebGL Renderer
        this.useWebGL = true; // Enable by default
        try {
            if (RadianceWebGLRenderer) {
                this.renderer = new RadianceWebGLRenderer(this.glCanvas);
                console.log("[Radiance] WebGL Renderer Initialized (GPU acceleration enabled)");
            } else {
                console.error("[Radiance] RadianceWebGLRenderer class is missing.");
                this.renderer = null;
                this.useWebGL = false;
            }
        } catch (e) {
            console.error("[Radiance] WebGL init failed:", e);
            this.renderer = null;
            this.useWebGL = false;
        }

        // If WebGL failed, hide the canvas and use 2D fallback
        if (!this.renderer || !this.useWebGL) {
            this.glCanvas.style.display = 'none';
            console.warn("[Radiance] Falling back to 2D Canvas rendering");
        }
        // v3.0 FIX: Do NOT hide the 2D canvas. Architecture:
        //   glCanvas = off-screen WebGL render buffer (hidden in render())
        //   canvas   = display surface (2D context draws glCanvas with pan/zoom)
        // The old code hid the 2D canvas, making BOTH canvases invisible → black.



        // Overlay canvas
        this.overlayCanvas = document.createElement('canvas');
        this.overlayCanvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;`;
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        this.canvasWrapper.appendChild(this.overlayCanvas);

        // Scope panel
        this.scopePanel = document.createElement('div');
        this.scopePanel.style.cssText = `
            display: none;
            flex: 0 0 ${this.scopePanelWidth}px;
            flex-direction: column;
            background: rgba(0,0,0,0.9);
            border-left: 1px solid ${t.panelBorder};
            padding: 4px;
            overflow-y: auto;
        `;
        this.mainArea.appendChild(this.scopePanel);
        this.createScopes();

        // Resize handle for scope panel
        this.scopeResizeHandle = document.createElement('div');
        this.scopeResizeHandle.style.cssText = `
            position: absolute;
            left: 0;
            top: 0;
            width: 4px;
            height: 100%;
            cursor: ew-resize;
            background: transparent;
            transition: background 0.2s;
            z-index: 10;
        `;
        this.scopeResizeHandle.onmouseenter = () => {
            this.scopeResizeHandle.style.background = this.theme.accent;
        };
        this.scopeResizeHandle.onmouseleave = () => {
            if (!this.isResizingScopePanel) {
                this.scopeResizeHandle.style.background = 'transparent';
            }
        };
        this.scopePanel.appendChild(this.scopeResizeHandle);
        this.setupScopePanelResize();



        // Pixel probe tooltip
        this.probeTooltip = document.createElement('div');
        this.probeTooltip.style.cssText = `
            position: absolute;
            display: none;
            background: rgba(0,0,0,0.92);
            border: 1px solid ${t.panelBorder};
            border-radius: 4px;
            padding: 6px 8px;
            font-size: 9px;
            color: ${t.text};
            pointer-events: none;
            z-index: 1000;
            white-space: pre;
            cursor: pointer;
        `;
        this.canvasWrapper.appendChild(this.probeTooltip);

        // Metadata Info Panel
        this.metadataPanel = document.createElement('div');
        this.metadataPanel.style.cssText = `
            position: absolute;
            top: 45px;
            right: 8px;
            width: 220px;
            background: rgba(0,0,0,0.92);
            border: 1px solid ${t.panelBorder};
            border-radius: 4px;
            padding: 8px;
            font-size: 9px;
            color: ${t.text};
            z-index: 90;
            display: none;
        `;

        const metaTitle = document.createElement('div');
        metaTitle.style.cssText = `font-weight: 600; font-size: 10px; margin-bottom: 8px; color: ${t.accent}; padding-bottom: 4px; border-bottom: 1px solid ${t.panelBorder};`;
        metaTitle.textContent = 'ℹ️ Image Info';
        this.metadataPanel.appendChild(metaTitle);

        this.metadataContent = document.createElement('div');
        this.metadataContent.style.cssText = 'line-height: 1.5; font-family: monospace;';
        this.metadataPanel.appendChild(this.metadataContent);

        this.canvasWrapper.appendChild(this.metadataPanel);

        // Status bar
        this.statusBar = document.createElement('div');
        this.statusBar.style.cssText = `
            flex: 0 0 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 8px;
            background: ${t.panel};
            border-top: 1px solid ${t.panelBorder};
            font-size: 9px;
            color: ${t.textDim};
        `;
        this.container.appendChild(this.statusBar);

        this.cursorInfo = document.createElement('span');
        this.cursorInfo.textContent = 'X: — Y: —';
        this.statusBar.appendChild(this.cursorInfo);

        this.colorInfo = document.createElement('span');
        this.colorInfo.textContent = 'RGB: — — —';
        this.colorInfo.style.cursor = 'pointer';
        this.colorInfo.title = 'Click to copy';
        this.colorInfo.onclick = () => this.copyColor();
        this.statusBar.appendChild(this.colorInfo);

        this.colorspaceInfo = document.createElement('span');
        this.colorspaceInfo.textContent = 'sRGB / HDR';
        this.colorspaceInfo.style.color = t.accent;
        this.statusBar.appendChild(this.colorspaceInfo);

        this.dimensionInfo = document.createElement('span');
        this.dimensionInfo.textContent = '—×—';
        this.statusBar.appendChild(this.dimensionInfo);

        this.zoomInfo = document.createElement('span');
        this.zoomInfo.textContent = '100%';
        this.zoomInfo.style.color = t.accent;
        this.statusBar.appendChild(this.zoomInfo);

        // Bit depth indicator
        this.bitDepthInfo = document.createElement('span');
        this.bitDepthInfo.textContent = '—';
        this.bitDepthInfo.style.cssText = `
            padding: 1px 6px;
            border-radius: 3px;
            font-weight: 600;
            font-size: 8px;
            letter-spacing: 0.5px;
        `;
        this.statusBar.appendChild(this.bitDepthInfo);
    }

    createScopes() {
        // Histogram
        this.histogramCanvas = document.createElement('canvas');
        this.histogramCanvas.width = 256;
        this.histogramCanvas.height = 120; // Increased from 50px for better distribution visibility
        this.histogramCanvas.style.cssText = 'width: 100%; height: 120px; display: none;';
        this.histogramCtx = this.histogramCanvas.getContext('2d');
        this.histogramLabel = this.createLabel('Histogram', true);
        this.scopePanel.appendChild(this.histogramLabel);
        this.scopePanel.appendChild(this.histogramCanvas);

        // Waveform
        this.waveformCanvas = document.createElement('canvas');
        this.waveformCanvas.width = 256;
        this.waveformCanvas.height = 150; // Increased from 50px to broadcast standard
        this.waveformCanvas.style.cssText = 'width: 100%; height: 150px; display: none;';
        this.waveformCtx = this.waveformCanvas.getContext('2d');
        this.waveformLabel = this.createLabel('Waveform', true);
        this.scopePanel.appendChild(this.waveformLabel);
        this.scopePanel.appendChild(this.waveformCanvas);

        // Vectorscope
        this.vectorscopeCanvas = document.createElement('canvas');
        this.vectorscopeCanvas.width = 256; // Increased from 140px
        this.vectorscopeCanvas.height = 256; // Increased from 140px
        this.vectorscopeCanvas.style.cssText = 'width: 100%; aspect-ratio: 1; display: none;';
        this.vectorscopeCtx = this.vectorscopeCanvas.getContext('2d');
        this.vectorscopeLabel = this.createLabel('Vectorscope', true);
        this.scopePanel.appendChild(this.vectorscopeLabel);
        this.scopePanel.appendChild(this.vectorscopeCanvas);
    }

    setupScopePanelResize() {
        let startX, startWidth;

        this.scopeResizeHandle.addEventListener('mousedown', (e) => {
            this.isResizingScopePanel = true;
            startX = e.clientX;
            startWidth = this.scopePanelWidth;
            this.scopeResizeHandle.style.background = this.theme.accent;
            e.preventDefault();
            e.stopPropagation();
        });

        document.addEventListener('mousemove', (e) => {
            if (!this.isResizingScopePanel) return;

            const delta = startX - e.clientX; // Reversed: dragging left makes wider
            const newWidth = Math.max(
                this.scopePanelMinWidth,
                Math.min(this.scopePanelMaxWidth, startWidth + delta)
            );

            this.scopePanelWidth = newWidth;
            this.scopePanel.style.flex = `0 0 ${newWidth}px`;
        });

        document.addEventListener('mouseup', () => {
            if (this.isResizingScopePanel) {
                this.isResizingScopePanel = false;
                this.scopeResizeHandle.style.background = 'transparent';
                // Save width to localStorage
                localStorage.setItem('radiance_scope_width', this.scopePanelWidth);
            }
        });
    }

    // v2.2: Full cleanup — prevents memory leaks on node deletion
    destroy() {
        // Remove global event listeners
        if (this._docMoveHandler) document.removeEventListener('mousemove', this._docMoveHandler);
        if (this._docUpHandler) document.removeEventListener('mouseup', this._docUpHandler);
        if (this._docKeyHandler) document.removeEventListener('keydown', this._docKeyHandler);
        if (this._winUpHandler) window.removeEventListener('mouseup', this._winUpHandler);

        // Disconnect ResizeObserver
        if (this.resizeObserver) this.resizeObserver.disconnect();

        // Destroy WebGL renderer
        if (this.renderer) this.renderer.destroy();

        // Remove DOM
        if (this.container) this.container.innerHTML = '';

        console.log('[Radiance] Viewer destroyed');
    }

    createLabel(text, hidden = false) {
        const label = document.createElement('div');
        label.textContent = text;
        label.style.cssText = `
            color: ${this.theme.textDim};
            font-size: 11px;
            font-family: ${this.theme.font};
            min-width: 50px;
            font-weight: 500;
            display: ${hidden ? 'none' : 'block'};
        `;
        return label;
    }

    createToolbar() {
        // Fullscreen & Export
        this.addButton('⛶', () => this.toggleFullscreen(), 'Fullscreen');
        this.addButton('💾', () => this.exportSnapshot(), 'Export');
        this.addButton('↺', () => this.resetControls(), 'Reset');
        this.addButton('?', () => this.toggleHelp(), 'Keyboard Shortcuts (?)');
        this.addSep();

        // Run & Editor
        this.runButton = this.addButton('▶', () => this.runWorkflow(), 'Run (Shift+Enter)');
        this.runButton.style.color = '#4f4'; // Green hue

        this.promptButton = this.addButton('P', () => this.togglePromptPanel(), 'Prompt Editor (P)');
        this.addSep();

        // Controls Toggle
        this.controlsToggle = this.addButton('🎛️', () => this.toggleControls(), 'Toggle Grading Controls');
        this.controlsToggle.style.color = this.theme.accent; // Active by default
        this.addSep();

        // Removed EV/Gamma sliders from toolbar
        // They are now in HUD

        // Removed separate Lens controls from toolbar
        // They are now in HUD

        // Zoom
        this.addButton('Fit', () => this.fitToView(), 'F');
        this.addButton('1:1', () => this.setZoom(1.0), '1');
        this.addSep();

        // LUT Dropdown
        this.addLbl('LUT');
        const lutSel = document.createElement('select');
        lutSel.style.cssText = `
            background: #181820;
            color: ${this.theme.text};
            border: 1px solid ${this.theme.panelBorder};
            border-radius: 4px;
            padding: 2px 4px;
            font-size: 11px;
            outline: none;
            cursor: pointer;
        `;
        this.lutOptions.forEach(opt => {
            const el = document.createElement('option');
            el.value = opt;
            el.textContent = opt;
            lutSel.appendChild(el);
        });
        lutSel.value = this.displayLut;
        lutSel.onchange = (e) => { this.displayLut = e.target.value; this.render(); };
        this.toolbar.appendChild(lutSel);

        this.addSep();

        // Channels (added Alpha)
        ['RGB', 'R', 'G', 'B', 'A', 'L'].forEach(ch => {
            this.addButton(ch, () => {
                this.channel = ch.toLowerCase() === 'l' ? 'luma' : ch.toLowerCase();
                this.render();
            }, ch === 'A' ? 'Alpha Channel' : '');
        });
        this.addSep();

        // Batch Navigation
        this.prevFrameBtn = this.addButton('◀', () => this.prevFrame(), 'Previous Frame (←)');
        this.frameDisplay = document.createElement('span');
        this.frameDisplay.textContent = '1/1';
        this.frameDisplay.style.cssText = `color: ${this.theme.text}; font-size: 9px; min-width: 32px; text-align: center;`;
        this.toolbar.appendChild(this.frameDisplay);
        this.nextFrameBtn = this.addButton('▶', () => this.nextFrame(), 'Next Frame (→)');
        this.addSep();

        // View
        this.addButton('FC', () => { this.falseColor = !this.falseColor; this.zebra = false; this.focusPeaking = false; this.showZdepth = false; this.render(); }, 'False Color (E)');
        this.addButton('Z', () => this.toggleZdepth(), 'Z-Depth / Zebra (Z)');
        this.focusPeakingBtn = this.addButton('FP', () => { this.focusPeaking = !this.focusPeaking; this.falseColor = false; this.zebra = false; this.render(); }, 'Focus Peaking (K)');
        this.gridBtn = this.addButton('▦', () => this.cycleGridMode(), 'Grid / Safe Areas (G)');
        this.addButton('📺', () => this.cycleSafeAreas(), 'Safe Areas (S)');
        this.loupeBtn = this.addButton('🔍', () => { this.showLoupe = !this.showLoupe; }, 'Pixel Loupe (Q)');
        this.addSep();

        // Compare
        this.addButton('A|B', () => this.cycleCompareMode(), 'Compare (A)');
        this.addSep();

        // Scopes
        this.addButton('H', () => this.toggleScope('histogram'), 'Histogram');
        this.addButton('W', () => this.toggleScope('waveform'), 'Waveform');
        this.addButton('V', () => this.toggleScope('vectorscope'), 'Vectorscope');
        this.addButton('ℹ️', () => this.toggleMetadata(), 'Image Info');
        this.addSep();
        this.addButton('◫', () => { this.scopeOverlay = !this.scopeOverlay; this.renderOverlay(); }, 'Overlay');
        this.addSep();

        // Annotations
        this.annotBtns = {};

        // Color Picker
        const colorInput = document.createElement('input');
        colorInput.type = 'color';
        colorInput.value = this.annotationColor;
        colorInput.style.cssText = `width: 20px; height: 18px; border: none; padding: 0; background: none; cursor: pointer;`;
        colorInput.oninput = (e) => this.annotationColor = e.target.value;
        this.toolbar.appendChild(colorInput);

        // Size Selector
        const sizeSel = document.createElement('select');
        [1, 2, 3, 5, 8, 12].forEach(s => {
            const opt = document.createElement('option');
            opt.value = s; opt.text = s + 'px';
            if (s === this.annotationLineWidth) opt.selected = true;
            sizeSel.appendChild(opt);
        });
        sizeSel.style.cssText = `
            background: #1a1a28; color: ${this.theme.textDim}; border: 1px solid ${this.theme.panelBorder};
            border-radius: 3px; font-size: 9px; margin-right: 2px; height: 18px;
        `;
        sizeSel.onchange = (e) => this.annotationLineWidth = parseInt(e.target.value);
        this.toolbar.appendChild(sizeSel);

        this.annotBtns.pen = this.addButton('✎', () => this.setAnnotationTool('pen'), 'Pen');
        this.annotBtns.arrow = this.addButton('→', () => this.setAnnotationTool('arrow'), 'Arrow');
        this.annotBtns.rect = this.addButton('□', () => this.setAnnotationTool('rect'), 'Rectangle');
        this.annotBtns.circle = this.addButton('○', () => this.setAnnotationTool('circle'), 'Circle');
        this.annotBtns.text = this.addButton('T', () => this.setAnnotationTool('text'), 'Text');

        this.addButton('↩', () => this.undoAnnotation(), 'Undo');
        this.addButton('⟲', () => this.clearAnnotations(), 'Clear All');
        this.addSep();

        // Measurement Tools
        this.measureBtns = {};
        this.measureBtns.distance = this.addButton('📏', () => this.setMeasurementTool('distance'), 'Distance (D)');
        this.measureBtns.angle = this.addButton('∠', () => this.setMeasurementTool('angle'), 'Angle (Ø)');
        this.addButton('⌫', () => this.clearMeasurements(), 'Clear Measurements');
    }

    addLbl(text) {
        const lbl = document.createElement('span');
        lbl.textContent = text;
        lbl.style.cssText = `color: ${this.theme.textDim}; font-size: 9px;`;
        this.toolbar.appendChild(lbl);
    }

    addButton(text, onClick, title = '') {
        const btn = document.createElement('button');
        btn.textContent = text;
        btn.title = title;
        btn.style.cssText = `
            padding: 2px 5px;
            background: linear-gradient(180deg, #1a1a28 0%, #12121a 100%);
            border: 1px solid ${this.theme.panelBorder};
            border-radius: 3px;
            color: ${this.theme.textDim};
            cursor: pointer;
            font-size: 9px;
        `;
        btn.onmouseenter = () => btn.style.color = this.theme.text;
        btn.onmouseleave = () => { if (!btn.classList.contains('active')) btn.style.color = this.theme.textDim; };
        btn.onclick = onClick;
        this.toolbar.appendChild(btn);
        return btn;
    }

    // v2.2: Removed dead createCustomSlider() — replaced by createControlRow() HUD sliders

    addSep() {
        const s = document.createElement('div');
        s.style.cssText = `width: 1px; height: 12px; background: ${this.theme.panelBorder}; margin: 0 2px;`;
        this.toolbar.appendChild(s);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          RESET & COPY
    // ═══════════════════════════════════════════════════════════════════════════

    resetControls() {
        // v2.2: Complete reset — all grading + analysis state
        this.exposure = 0.0;
        this.lift = [0, 0, 0];
        this.gamma = [1, 1, 1];
        this.gain = [1, 1, 1];
        this.saturation = 1.0;
        this.channel = 'rgb';
        this.falseColor = false;
        this.zebra = false;
        this.showZdepth = false;
        this.focusPeaking = false;
        this.displayLut = 'None';

        // Reset HUD slider inputs (v2.2: replaces crashed evSlider/gammaSlider refs)
        const resetSlider = (controlRow, value) => {
            if (!controlRow) return;
            const input = controlRow.querySelector('input[type="range"]');
            if (input) { input.value = value; input.dispatchEvent(new Event('input')); }
        };
        resetSlider(this.evControl, 0);
        resetSlider(this.gammaControl, 1.0);
        resetSlider(this.satControl, 1.0);

        if (this.renderer) {
            this.renderer.setExposure(0);
            this.renderer.setLift(0, 0, 0);
            this.renderer.setGamma(1, 1, 1);
            this.renderer.setGain(1, 1, 1);
            this.renderer.setSaturation(1.0);
            this.renderer.setChannelMode(0);
            this.renderer.setFocusPeaking(false);
            this.renderer.setDisplayLutMode(0);
        }
        this.render();
    }

    copyColor() {
        if (this.lastPixelColor) {
            const { r, g, b } = this.lastPixelColor;
            const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
            navigator.clipboard.writeText(hex).then(() => {
                this.colorInfo.textContent = `Copied: ${hex}`;
                setTimeout(() => {
                    if (this.lastPixelColor) {
                        this.colorInfo.textContent = `RGB: ${this.lastPixelColor.r} ${this.lastPixelColor.g} ${this.lastPixelColor.b}`;
                    }
                }, 1000);
            });
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          ANNOTATIONS
    // ═══════════════════════════════════════════════════════════════════════════


    setAnnotationTool(tool) {
        this.annotationTool = tool;
        this.isAnnotating = true;
        this.measurementMode = 'none'; // Disable measurement
        this.canvas.style.cursor = 'crosshair';

        // Clear measurement tool selection
        if (this.measureBtns) {
            Object.keys(this.measureBtns).forEach(k => {
                const btn = this.measureBtns[k];
                btn.style.background = '';
                btn.style.color = this.theme.textDim;
                btn.classList.remove('active');
            });
        }


        // Highlight active
        Object.keys(this.annotBtns).forEach(k => {
            const btn = this.annotBtns[k];
            if (k === tool) {
                btn.style.background = this.theme.accent;
                btn.style.color = '#fff';
                btn.classList.add('active');
            } else {
                btn.style.background = '';
                btn.style.color = this.theme.textDim;
                btn.classList.remove('active');
            }
        });
    }

    undoAnnotation() {
        if (this.annotations.length > 0) {
            this.annotations.pop();
            this.renderOverlay();
        }
    }

    addAnnotation(type, x1, y1, x2, y2, extra = {}) {
        this.annotations.push({
            type, x1, y1, x2, y2,
            color: this.annotationColor,
            width: this.annotationLineWidth,
            ...extra
        });
        this.renderOverlay();
    }

    clearAnnotations() {
        this.annotations = [];
        this.renderOverlay();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          MEASUREMENTS
    // ═══════════════════════════════════════════════════════════════════════════

    setMeasurementTool(mode) {
        this.measurementMode = mode;
        this.isAnnotating = false; // Disable annotation
        this.annotationTool = null;
        this.canvas.style.cursor = mode === 'none' ? 'default' : 'crosshair';

        // Highlight active measurement button
        if (this.measureBtns) {
            Object.keys(this.measureBtns).forEach(k => {
                const btn = this.measureBtns[k];
                if (k === mode) {
                    btn.style.background = this.theme.accent;
                    btn.style.color = '#fff';
                    btn.classList.add('active');
                } else {
                    btn.style.background = '';
                    btn.style.color = this.theme.textDim;
                    btn.classList.remove('active');
                }
            });
        }

        // Clear annotation tool selection
        if (this.annotBtns && mode !== 'none') {
            Object.keys(this.annotBtns).forEach(k => {
                const btn = this.annotBtns[k];
                btn.style.background = '';
                btn.style.color = this.theme.textDim;
                btn.classList.remove('active');
            });
        }
    }

    clearMeasurements() {
        this.measurements = [];
        this.currentMeasurement = null;
        this.renderOverlay();
    }

    drawMeasurement(ctx, m, scale) {
        const color = '#ffff00'; // Yellow
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2 * scale;
        ctx.beginPath();

        if (m.type === 'distance') {
            ctx.moveTo(m.x1, m.y1);
            ctx.lineTo(m.x2, m.y2);
            ctx.stroke();

            // Draw T-ends
            const ang = Math.atan2(m.y2 - m.y1, m.x2 - m.x1);
            const perp = ang + Math.PI / 2;
            const len = 5 * scale;

            ctx.beginPath();
            ctx.moveTo(m.x1 + Math.cos(perp) * len, m.y1 + Math.sin(perp) * len);
            ctx.lineTo(m.x1 - Math.cos(perp) * len, m.y1 - Math.sin(perp) * len);
            ctx.moveTo(m.x2 + Math.cos(perp) * len, m.y2 + Math.sin(perp) * len);
            ctx.lineTo(m.x2 - Math.cos(perp) * len, m.y2 - Math.sin(perp) * len);
            ctx.stroke();

            // Text
            const dist = Math.sqrt(Math.pow(m.x2 - m.x1, 2) + Math.pow(m.y2 - m.y1, 2));
            const cx = (m.x1 + m.x2) / 2;
            const cy = (m.y1 + m.y2) / 2;

            ctx.font = `bold ${Math.max(10, 12 * scale)}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            // Draw background for text
            const text = `${dist.toFixed(1)} px`;
            const tm = ctx.measureText(text);
            ctx.globalAlpha = 0.7;
            ctx.fillStyle = '#000';
            ctx.fillRect(cx - tm.width / 2 - 2, cy - 14 * scale, tm.width + 4, 16 * scale);
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = color;
            ctx.fillText(text, cx, cy - 2 * scale);

        } else if (m.type === 'angle') {
            ctx.moveTo(m.x1, m.y1);
            ctx.lineTo(m.x2, m.y2);
            ctx.stroke();

            const dx = m.x2 - m.x1;
            const dy = m.y2 - m.y1;
            let ang = Math.atan2(dy, dx) * 180 / Math.PI;

            // Normalize angle for display (e.g., relative to horizontal)
            // Or just show positive angle
            if (ang < 0) ang += 360;

            // Draw arc
            ctx.beginPath();
            ctx.arc(m.x1, m.y1, 20 * scale, 0, Math.atan2(dy, dx), dy < 0);
            ctx.stroke();

            ctx.font = `bold ${Math.max(10, 12 * scale)}px sans-serif`;
            ctx.textAlign = 'left';
            const text = `${ang.toFixed(1)}°`;

            // Background
            const mx = m.x1 + 25 * scale;
            const my = m.y1;
            const tm = ctx.measureText(text);
            ctx.globalAlpha = 0.7;
            ctx.fillStyle = '#000';
            ctx.fillRect(mx - 2, my - 10 * scale, tm.width + 4, 14 * scale);
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = color;
            ctx.fillText(text, mx, my);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          BATCH NAVIGATION
    // ═══════════════════════════════════════════════════════════════════════════

    prevFrame() {
        if (this.totalFrames <= 1) return;
        this.currentFrame = (this.currentFrame - 1 + this.totalFrames) % this.totalFrames;
        this.loadCurrentFrame();
    }

    nextFrame() {
        if (this.totalFrames <= 1) return;
        this.currentFrame = (this.currentFrame + 1) % this.totalFrames;
        this.loadCurrentFrame();
    }

    loadCurrentFrame() {
        if (this.frameImages[this.currentFrame]) {
            this.image = this.frameImages[this.currentFrame];
            this.imageWidth = this.image.width;
            this.imageHeight = this.image.height;

            // Update imageData for CPU processing
            if (!this._frameDataCanvas) {
                this._frameDataCanvas = document.createElement('canvas');
            }
            this._frameDataCanvas.width = this.image.width;
            this._frameDataCanvas.height = this.image.height;
            const ctx = this._frameDataCanvas.getContext('2d');
            ctx.drawImage(this.image, 0, 0);
            this.imageData = ctx.getImageData(0, 0, this.image.width, this.image.height).data;

            // Update Z-Depth
            if (this.frameZdepthImages && this.frameZdepthImages[this.currentFrame]) {
                this.zdepthImage = this.frameZdepthImages[this.currentFrame];
                if (this.renderer) this.renderer.loadDepthTexture(this.zdepthImage);
            } else {
                this.zdepthImage = null;
            }

            // Update compare image if batch has per-frame compare
            if (this.frameCompareImages && this.frameCompareImages[this.currentFrame]) {
                this.compareImage = this.frameCompareImages[this.currentFrame];
                this.diffCanvas = null; // Clear cached diff
            }

            // Update HDR Data
            if (this.frameHDRData && this.frameHDRData[this.currentFrame]) {
                this.hdrData = this.frameHDRData[this.currentFrame];
            } else {
                this.hdrData = null;
            }

            // Update WebGL Renderer Main Texture
            if (this.renderer) {
                let loadedFloat = false;
                if (this.hdrData && this.hdrData.data) {
                    // v2.2: Prefer fp16 upload for .rhdr format (half VRAM)
                    let tex;
                    if (this.hdrData.fp16data) {
                        tex = this.renderer.loadFloat16Texture(
                            this.hdrData.fp16data,
                            this.hdrData.width,
                            this.hdrData.height,
                            this.hdrData.channels
                        );
                    } else {
                        tex = this.renderer.loadFloat32Texture(
                            this.hdrData.data,
                            this.hdrData.width,
                            this.hdrData.height,
                            this.hdrData.channels
                        );
                    }
                    if (tex) {
                        loadedFloat = true;
                        // console.log("[Radiance] Loaded float HDR texture for frame", this.currentFrame);
                    }
                }

                if (!loadedFloat) {
                    this.renderer.loadImageTexture(this.image);
                }
            }

            // v2.2: Refit if resolution changed between frames
            if (this._lastFrameW !== this.imageWidth || this._lastFrameH !== this.imageHeight) {
                this.fitToView();
            }
            this._lastFrameW = this.imageWidth;
            this._lastFrameH = this.imageHeight;

            this.updateFrameDisplay();
            this.render();
            this.updateScopes();
        }
    }

    updateFrameDisplay() {
        if (this.frameDisplay) {
            this.frameDisplay.textContent = `${this.currentFrame + 1}/${this.totalFrames}`;
        }
    }

    // v2.2: Check if all frames in current batch have loaded
    _allFramesReady() {
        if (!this.totalFrames) return true;
        for (let i = 0; i < this.totalFrames; i++) {
            if (!this.frameImages[i] && !this.frameHDRData[i]) return false;
        }
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          GRID & SAFE AREAS
    // ═══════════════════════════════════════════════════════════════════════════

    cycleGridMode() {
        // 0=off, 1=thirds, 2=safe areas, 3=center, 4=both
        this.gridMode = (this.gridMode + 1) % 5;
        this.showGrid = this.gridMode > 0;

        const labels = ['Off', 'Thirds', 'Safe', 'Center', 'All'];
        if (this.gridBtn) {
            this.gridBtn.title = `Grid: ${labels[this.gridMode]} (G)`;
        }
        this.renderOverlay();
    }

    cycleSafeAreas() {
        const modes = ['none', 'action', 'title', 'both'];
        const idx = modes.indexOf(this.safeAreaMode);
        this.safeAreaMode = modes[(idx + 1) % modes.length];
        this.renderOverlay();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          KEYBOARD SHORTCUTS
    // ═══════════════════════════════════════════════════════════════════════════

    setupKeyboardShortcuts() {
        const handler = (e) => {
            if (e.target.tagName === 'INPUT') return;
            this.handleKey(e);
        };
        this.canvas.addEventListener('keydown', handler);
        document.addEventListener('keydown', (e) => { if (this.isFullscreen) handler(e); });
    }

    handleKey(e) {
        const key = e.key.toLowerCase();
        switch (key) {
            case '?': case '/': if (e.shiftKey) this.toggleHelp(); break;
            case 'f': this.fitToView(); break;
            case '1': this.setZoom(1.0); break;
            case 'r': this.channel = 'r'; this.render(); break;
            case 'g': if (e.shiftKey) { this.cycleGridMode(); } else if (!e.ctrlKey) { this.channel = 'g'; this.render(); } break;
            case 'b': this.channel = 'b'; this.render(); break;
            case 'l': this.channel = 'luma'; this.render(); break;
            case 'c': this.channel = 'rgb'; this.render(); break;
            case 'h': this.toggleScope('histogram'); break;
            case 'w': this.toggleScope('waveform'); break;
            case 'm': this.toggleParadeMode(); break;
            case 'v': this.toggleScope('vectorscope'); break;
            case 'e': this.falseColor = !this.falseColor; this.zebra = false; this.focusPeaking = false; this.showZdepth = false; this.render(); break;
            case 'k': this.focusPeaking = !this.focusPeaking; this.falseColor = false; this.zebra = false; this.showZdepth = false; this.render(); break;
            case 'z': this.toggleZdepth(); break;
            case 'q': this.showLoupe = !this.showLoupe; break;
            case 'a':
                if (e.shiftKey) { this.channel = 'a'; this.render(); }
                else { this.cycleCompareMode(); }
                break;
            case 's': if (!e.ctrlKey) this.cycleSafeAreas(); break;
            case 'arrowleft': this.prevFrame(); break;
            case 'arrowright': this.nextFrame(); break;
            case 'escape':
                if (this.showHelp) { this.toggleHelp(); }
                else if (this.isFullscreen) { this.exitFullscreen(); }
                break;
            case '=': case '+': this.adjustEV(0.5); break;
            case '-': this.adjustEV(-0.5); break;
            case '0': this.resetControls(); break;
            case 'p': if (!e.ctrlKey) this.togglePromptPanel(); break;
            case 'enter': if (e.shiftKey) this.runWorkflow(); break;
            case 'd': if (!e.ctrlKey) this.setMeasurementTool('distance'); break;
            case 'ø': this.setMeasurementTool('angle'); break;
            case 'delete': case 'backspace': if (!e.target.tagName || e.target.tagName !== 'INPUT') this.clearMeasurements(); break;
        }
    }

    adjustEV(delta) {
        this.exposure = Math.max(-5, Math.min(5, this.exposure + delta));
        // v2.2: Update HUD slider (replaces crashed evSlider.setValue)
        if (this.evControl) {
            const input = this.evControl.querySelector('input[type="range"]');
            if (input) { input.value = this.exposure; input.dispatchEvent(new Event('input')); }
        }
        if (this.renderer) this.renderer.setExposure(this.exposure);
        this.render();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          FULLSCREEN
    // ═══════════════════════════════════════════════════════════════════════════

    toggleFullscreen() {
        if (!this.isFullscreen) {
            const elem = this.container;
            if (elem.requestFullscreen) elem.requestFullscreen();
            else if (elem.webkitRequestFullscreen) elem.webkitRequestFullscreen();
            this.isFullscreen = true;

            const handler = () => {
                if (!document.fullscreenElement && !document.webkitFullscreenElement) {
                    this.isFullscreen = false;
                    document.removeEventListener('fullscreenchange', handler);
                    document.removeEventListener('webkitfullscreenchange', handler);
                }
                requestAnimationFrame(() => { this.resize(); this.render(); });
            };
            document.addEventListener('fullscreenchange', handler);
            document.addEventListener('webkitfullscreenchange', handler);
        } else {
            this.exitFullscreen();
        }
    }

    exitFullscreen() {
        if (document.exitFullscreen) document.exitFullscreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
        this.isFullscreen = false;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          EXPORT
    // ═══════════════════════════════════════════════════════════════════════════

    exportSnapshot() {
        if (!this.image) return;
        // v2.2: Export from WebGL canvas (WYSIWYG) instead of CPU renderImage
        const exp = document.createElement('canvas');
        exp.width = this.imageWidth;
        exp.height = this.imageHeight;
        const ctx = exp.getContext('2d');

        if (this.useWebGL && this.renderer && this.renderer.textures.image) {
            // Ensure WebGL renders at full image resolution
            const prevW = this.glCanvas.width, prevH = this.glCanvas.height;
            this.glCanvas.width = this.imageWidth;
            this.glCanvas.height = this.imageHeight;
            this.renderer.render(this.lutIntensity || 1.0);
            ctx.drawImage(this.glCanvas, 0, 0);
            // Restore
            this.glCanvas.width = prevW;
            this.glCanvas.height = prevH;
        } else {
            this.renderImage(ctx, this.image);
        }
        this.annotations.forEach(a => this.drawAnnotation(ctx, a, 1));

        // v3.2: Export EXR if available (Requested by users)
        // Checks current HDR data object or fallback image for metadata
        const currentData = this.hdrData || this.image;
        if (currentData && currentData.exr_filename) {
            const url = api.apiURL(`/view?filename=${encodeURIComponent(currentData.exr_filename)}&subfolder=${encodeURIComponent(currentData.subfolder || '')}&type=${encodeURIComponent(currentData.type || 'temp')}`);
            const link = document.createElement('a');
            link.href = url;
            link.download = currentData.exr_filename;
            link.click();
            return;
        }

        const link = document.createElement('a');
        link.download = `radiance_${Date.now()}.png`;
        link.href = exp.toDataURL('image/png');
        link.click();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          KEYBOARD HELP
    // ═══════════════════════════════════════════════════════════════════════════

    toggleHelp() {
        this.showHelp = !this.showHelp;
        if (this.showHelp) {
            this.showHelpScreen();
        } else {
            if (this.helpPanel) {
                this.helpPanel.remove();
                this.helpPanel = null;
            }
        }
    }

    showHelpScreen() {
        if (this.helpPanel) return;

        this.helpPanel = document.createElement('div');
        this.helpPanel.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(10, 10, 20, 0.98);
            border: 2px solid ${this.theme.accent};
            border-radius: 12px;
            padding: 24px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            z-index: 10000;
            box-shadow: 0 8px 32px rgba(0,0,0,0.8);
        `;

        const shortcuts = [
            {
                category: 'Navigation', items: [
                    ['F', 'Fit to view'],
                    ['1', '1:1 pixel zoom'],
                    ['Mouse Wheel', 'Zoom in/out'],
                    ['Shift+Drag', 'Pan image'],
                    ['← →', 'Previous/Next frame']
                ]
            },
            {
                category: 'Display', items: [
                    ['R/G/B/L', 'View R/G/B/Luma channel'],
                    ['C', 'RGB (color) view'],
                    ['Shift+A', 'Alpha channel'],
                    ['+/−', 'Adjust exposure'],
                    ['0', 'Reset all controls'],
                    ['E', 'False color'],
                    ['K', 'Focus peaking (GPU)'],
                    ['Q', 'Pixel loupe'],
                    ['Z', 'Z-Depth / Zebra']
                ]
            },
            {
                category: 'Analysis', items: [
                    ['H', 'Histogram (HDR-aware)'],
                    ['W', 'Waveform'],
                    ['M', 'Toggle Parade mode'],
                    ['V', 'Vectorscope'],
                    ['Shift+G', 'Cycle grid modes'],
                    ['G', 'Green channel'],
                    ['S', 'Safe areas'],
                    ['A', 'A/B compare'],
                    ['D', 'Distance measure'],
                    ['Del', 'Clear measurements']
                ]
            },
            {
                category: 'General', items: [
                    ['⛶', 'Fullscreen'],
                    ['Shift+Enter', 'Run workflow'],
                    ['P', 'Prompt editor'],
                    ['?', 'This help'],
                    ['Esc', 'Close help/Exit fullscreen']
                ]
            }
        ];

        let html = `<div style="color: ${this.theme.accent}; font-size: 18px; font-weight: bold; margin-bottom: 16px; text-align: center;">⌨️ Keyboard Shortcuts</div>`;

        shortcuts.forEach(section => {
            html += `<div style="margin-bottom: 16px;">`;
            html += `<div style="color: ${this.theme.accent}; font-size: 12px; font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid ${this.theme.panelBorder}; padding-bottom: 4px;">${section.category}</div>`;
            section.items.forEach(([key, desc]) => {
                html += `<div style="display: flex; justify-content: space-between; padding: 3px 0; font-size: 11px;">`;
                html += `<span style="background: #1a1a28; padding: 2px 8px; border-radius: 3px; font-family: monospace; color: ${this.theme.text};">${key}</span>`;
                html += `<span style="color: ${this.theme.textDim}; margin-left: 12px;">${desc}</span>`;
                html += `</div>`;
            });
            html += `</div>`;
        });

        html += `<div style="text-align: center; margin-top: 16px; padding-top: 12px; border-top: 1px solid ${this.theme.panelBorder}; color: ${this.theme.textDim}; font-size: 9px;">Press ? or Esc to close</div>`;

        this.helpPanel.innerHTML = html;
        this.canvasWrapper.appendChild(this.helpPanel);

        // Click to close
        this.helpPanel.onclick = () => this.toggleHelp();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          SCOPES
    // ═══════════════════════════════════════════════════════════════════════════

    toggleScope(scope) {
        if (scope === 'histogram') {
            this.showHistogram = !this.showHistogram;
            this.histogramCanvas.style.display = this.showHistogram ? 'block' : 'none';
            this.histogramLabel.style.display = this.showHistogram ? 'block' : 'none';
        } else if (scope === 'waveform') {
            this.showWaveform = !this.showWaveform;
            this.waveformCanvas.style.display = this.showWaveform ? 'block' : 'none';
            this.waveformLabel.style.display = this.showWaveform ? 'block' : 'none';
        } else if (scope === 'vectorscope') {
            this.showVectorscope = !this.showVectorscope;
            this.vectorscopeCanvas.style.display = this.showVectorscope ? 'block' : 'none';
            this.vectorscopeLabel.style.display = this.showVectorscope ? 'block' : 'none';
        }

        const any = this.showHistogram || this.showWaveform || this.showVectorscope;
        this.scopePanel.style.display = any ? 'flex' : 'none';
        if (this.image) this.updateScopes();
    }

    // v2.2: HDR-aware histogram — reads float32 data, maps to 256 bins (log scale for >1.0)
    _updateHistogramHDR() {
        const fd = this.hdrData.data;
        const ch = this.hdrData.channels || 3;
        const total = fd.length / ch;
        const hR = new Uint32Array(256), hG = new Uint32Array(256), hB = new Uint32Array(256);

        // Map: [0,1] → bins 0-200, [1,∞) → bins 200-255 (log scale)
        const toBin = (v) => {
            if (v <= 0) return 0;
            if (v <= 1.0) return Math.floor(v * 200);
            // log2 scale for >1.0: each stop = ~10 bins
            return Math.min(255, 200 + Math.floor(Math.log2(v) * 10));
        };

        for (let i = 0; i < total; i++) {
            const r = fd[i * ch];
            const g = ch > 1 ? fd[i * ch + 1] : r;
            const b = ch > 2 ? fd[i * ch + 2] : r;
            hR[toBin(r)]++;
            hG[toBin(g)]++;
            hB[toBin(b)]++;
        }

        let max = 1;
        for (let i = 0; i < 256; i++) max = Math.max(max, hR[i], hG[i], hB[i]);
        this.histogramData = { hR, hG, hB, max };

        const hCtx = this.histogramCtx;
        const w = this.histogramCanvas.width, h = this.histogramCanvas.height;
        hCtx.fillStyle = '#0a0a0f';
        hCtx.fillRect(0, 0, w, h);

        // Draw HDR reference line at bin 200 (= 1.0)
        hCtx.strokeStyle = 'rgba(255,255,255,0.15)';
        hCtx.lineWidth = 1;
        hCtx.beginPath();
        const refX = (200 / 255) * w;
        hCtx.moveTo(refX, 0); hCtx.lineTo(refX, h);
        hCtx.stroke();
        hCtx.fillStyle = 'rgba(255,255,255,0.3)';
        hCtx.font = '7px monospace';
        hCtx.fillText('1.0', refX + 2, 8);

        const draw = (hist, color) => {
            hCtx.strokeStyle = color; hCtx.lineWidth = 1; hCtx.beginPath();
            for (let i = 0; i < 256; i++) {
                const y = h - (hist[i] / max) * h;
                if (i === 0) hCtx.moveTo(i, y); else hCtx.lineTo(i, y);
            }
            hCtx.stroke();
        };
        hCtx.globalAlpha = 0.7;
        draw(hR, '#ff4444'); draw(hG, '#44ff44'); draw(hB, '#4444ff');
        hCtx.globalAlpha = 1;
    }

    toggleZdepth() {
        // If Z-Depth image is available, toggle it
        if (this.zdepthImage || (this.frameZdepthImages && this.frameZdepthImages.length > 0)) {
            this.showZdepth = !this.showZdepth;
            // Disable other overlays
            if (this.showZdepth) {
                this.zebra = false;
                this.falseColor = false;
                this.focusPeaking = false;
            }
        }
        // If no Z-Depth, toggle Zebra as fallback
        else {
            this.zebra = !this.zebra;
            if (this.zebra) {
                this.falseColor = false;
                this.focusPeaking = false;
                this.showZdepth = false;
            }
        }
        this.render();
    }

    cycleCompareMode() {
        if (!this.compareImage) { this.compareMode = 'none'; return; }
        const modes = ['none', 'wipe', 'sidebyside', 'difference'];
        this.compareMode = modes[(modes.indexOf(this.compareMode) + 1) % modes.length];
        this.render();
    }

    updateScopes() {
        // Debounce scope updates for better performance
        if (this.scopeUpdateTimer) {
            clearTimeout(this.scopeUpdateTimer);
        }
        this.scopeUpdateTimer = setTimeout(() => {
            if (this.showHistogram) this.updateHistogram();
            if (this.showWaveform) this.updateWaveform();
            if (this.showVectorscope) this.updateVectorscope();
            if (this.scopeOverlay) this.renderOverlay();
        }, this.scopeDebounceMs);
    }

    updateHistogram() {
        if (!this.image || !this.imageData) return;

        // v2.2: Use HDR float data for histogram when available
        if (this.hdrData && this.hdrData.data) {
            this._updateHistogramHDR();
            return;
        }

        const data = this.imageData;
        const hR = new Uint32Array(256), hG = new Uint32Array(256), hB = new Uint32Array(256);
        for (let i = 0; i < data.length; i += 4) { hR[data[i]]++; hG[data[i + 1]]++; hB[data[i + 2]]++; }

        let max = 1;
        for (let i = 0; i < 256; i++) max = Math.max(max, hR[i], hG[i], hB[i]);

        this.histogramData = { hR, hG, hB, max };

        const hCtx = this.histogramCtx;
        const w = this.histogramCanvas.width, h = this.histogramCanvas.height;
        hCtx.fillStyle = '#0a0a0f';
        hCtx.fillRect(0, 0, w, h);

        const draw = (hist, color) => {
            hCtx.strokeStyle = color; hCtx.lineWidth = 1; hCtx.beginPath();
            for (let i = 0; i < 256; i++) {
                const y = h - (hist[i] / max) * h;
                if (i === 0) hCtx.moveTo(i, y); else hCtx.lineTo(i, y);
            }
            hCtx.stroke();
        };
        hCtx.globalAlpha = 0.7;
        draw(hR, '#ff4444'); draw(hG, '#44ff44'); draw(hB, '#4444ff');
        hCtx.globalAlpha = 1;
    }

    toggleParadeMode() {
        this.waveformParadeMode = !this.waveformParadeMode;
        if (this.showWaveform) {
            this.updateWaveform();
        }
    }

    updateWaveform() {
        if (!this.image || !this.imageData) return;
        const data = this.imageData;

        if (this.waveformParadeMode) {
            this.updateWaveformParade(data);
        } else {
            this.updateWaveformOverlay(data);
        }
    }

    updateWaveformOverlay(data) {
        const wCtx = this.waveformCtx;
        const w = this.waveformCanvas.width, h = this.waveformCanvas.height;
        wCtx.fillStyle = '#0a0a0f';
        wCtx.fillRect(0, 0, w, h);

        const step = Math.max(1, Math.floor(this.imageWidth / w));
        wCtx.globalAlpha = 0.1;
        for (let col = 0; col < this.imageWidth; col += step) {
            const x = Math.floor((col / this.imageWidth) * w);
            for (let row = 0; row < this.imageHeight; row += 3) {
                const idx = (row * this.imageWidth + col) * 4;
                const luma = data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
                const y = h - (luma / 255) * h;
                wCtx.fillStyle = `rgb(${data[idx]},${data[idx + 1]},${data[idx + 2]})`;
                wCtx.fillRect(x, y, 1, 1);
            }
        }
        wCtx.globalAlpha = 1;
    }

    updateWaveformParade(data) {
        const wCtx = this.waveformCtx;
        const w = this.waveformCanvas.width, h = this.waveformCanvas.height;
        wCtx.fillStyle = '#0a0a0f';
        wCtx.fillRect(0, 0, w, h);

        const paradeWidth = Math.floor(w / 3);
        const step = Math.max(1, Math.floor(this.imageWidth / paradeWidth));

        // Draw separators
        wCtx.strokeStyle = this.theme.panelBorder;
        wCtx.lineWidth = 1;
        wCtx.beginPath();
        wCtx.moveTo(paradeWidth, 0);
        wCtx.lineTo(paradeWidth, h);
        wCtx.moveTo(paradeWidth * 2, 0);
        wCtx.lineTo(paradeWidth * 2, h);
        wCtx.stroke();

        // Labels
        wCtx.fillStyle = '#ff4444';
        wCtx.font = '10px monospace';
        wCtx.fillText('R', 4, 12);
        wCtx.fillStyle = '#44ff44';
        wCtx.fillText('G', paradeWidth + 4, 12);
        wCtx.fillStyle = '#4444ff';
        wCtx.fillText('B', paradeWidth * 2 + 4, 12);

        wCtx.globalAlpha = 0.15;

        // Red parade
        for (let col = 0; col < this.imageWidth; col += step) {
            const x = Math.floor((col / this.imageWidth) * paradeWidth);
            for (let row = 0; row < this.imageHeight; row += 3) {
                const idx = (row * this.imageWidth + col) * 4;
                const r = data[idx];
                const y = h - (r / 255) * h;
                wCtx.fillStyle = `rgb(${r},0,0)`;
                wCtx.fillRect(x, y, 1, 1);
            }
        }

        // Green parade
        for (let col = 0; col < this.imageWidth; col += step) {
            const x = paradeWidth + Math.floor((col / this.imageWidth) * paradeWidth);
            for (let row = 0; row < this.imageHeight; row += 3) {
                const idx = (row * this.imageWidth + col) * 4;
                const g = data[idx + 1];
                const y = h - (g / 255) * h;
                wCtx.fillStyle = `rgb(0,${g},0)`;
                wCtx.fillRect(x, y, 1, 1);
            }
        }

        // Blue parade
        for (let col = 0; col < this.imageWidth; col += step) {
            const x = paradeWidth * 2 + Math.floor((col / this.imageWidth) * paradeWidth);
            for (let row = 0; row < this.imageHeight; row += 3) {
                const idx = (row * this.imageWidth + col) * 4;
                const b = data[idx + 2];
                const y = h - (b / 255) * h;
                wCtx.fillStyle = `rgb(0,0,${b})`;
                wCtx.fillRect(x, y, 1, 1);
            }
        }

        wCtx.globalAlpha = 1;
    }

    updateVectorscope() {
        if (!this.image || !this.imageData) return;
        const data = this.imageData; // Use cached data

        const vCtx = this.vectorscopeCtx;
        const size = this.vectorscopeCanvas.width;
        const cx = size / 2, cy = size / 2, rad = size / 2 - 6;

        vCtx.fillStyle = '#0a0a0f';
        vCtx.fillRect(0, 0, size, size);

        vCtx.strokeStyle = '#222'; vCtx.lineWidth = 1;
        for (let r = 0.25; r <= 1; r += 0.25) { vCtx.beginPath(); vCtx.arc(cx, cy, rad * r, 0, Math.PI * 2); vCtx.stroke(); }

        const targets = [{ a: 103, c: '#f00' }, { a: 167, c: '#ff0' }, { a: 241, c: '#0f0' }, { a: 283, c: '#0ff' }, { a: 347, c: '#00f' }, { a: 61, c: '#f0f' }];
        targets.forEach(t => {
            const ang = (t.a - 90) * Math.PI / 180;
            vCtx.fillStyle = t.c;
            vCtx.beginPath();
            vCtx.arc(cx + Math.cos(ang) * rad * 0.75, cy + Math.sin(ang) * rad * 0.75, 2, 0, Math.PI * 2);
            vCtx.fill();
        });

        vCtx.globalAlpha = 0.03;
        const step = Math.max(1, Math.floor(data.length / 4 / 30000));
        for (let i = 0; i < data.length; i += 4 * step) {
            const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
            const y = r * 0.2126 + g * 0.7152 + b * 0.0722;
            const u = (b - y) * 0.492, v = (r - y) * 0.877;
            vCtx.fillStyle = `rgb(${data[i]},${data[i + 1]},${data[i + 2]})`;
            vCtx.fillRect(cx + u * rad * 2, cy - v * rad * 2, 1, 1);
        }
        vCtx.globalAlpha = 1;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          OVERLAY RENDERING
    // ═══════════════════════════════════════════════════════════════════════════

    renderOverlay() {
        const ctx = this.overlayCtx;
        const w = this.overlayCanvas.width, h = this.overlayCanvas.height;
        ctx.clearRect(0, 0, w, h);

        // Grid
        if (this.showGrid) this.drawGrid(ctx, w, h);

        // Annotations
        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);
        this.annotations.forEach(a => this.drawAnnotation(ctx, a, 1 / this.zoom));

        // Draw current stroke if active
        if (this.isAnnotating && this.currentPath && this.annotationTool === 'pen') {
            this.drawAnnotation(ctx, {
                type: 'pen',
                points: this.currentPath,
                color: this.annotationColor,
                width: this.annotationLineWidth
            }, 1 / this.zoom);
        } else if (this.isAnnotating && this.annotationStart && this.annotationTool !== 'text') {
            // Preview shapes (simplified logic: we don't have mouse pos here easily without storing it,
            // but `renderOverlay` is called usually when done or panning.
            // For drag interaction, we need mouse move to trigger render.
        }

        ctx.restore();

        // Measurements
        if (this.measurements || this.currentMeasurement) {
            ctx.save();
            ctx.translate(this.panX, this.panY);
            ctx.scale(this.zoom, this.zoom);
            const scale = 1 / this.zoom;

            if (this.measurements) {
                this.measurements.forEach(m => this.drawMeasurement(ctx, m, scale));
            }
            if (this.currentMeasurement) {
                this.drawMeasurement(ctx, this.currentMeasurement, scale);
            }
            ctx.restore();
        }

        // Scope overlay
        if (this.scopeOverlay && this.histogramData) this.drawHistogramOverlay(ctx, w, h);
    }

    drawGrid(ctx, w, h) {
        // Grid mode: 1=thirds, 2=safe areas, 3=center, 4=all

        // Rule of thirds (mode 1 or 4)
        if (this.gridMode === 1 || this.gridMode === 4) {
            ctx.strokeStyle = 'rgba(255,255,255,0.2)';
            ctx.lineWidth = 1;
            for (let i = 1; i <= 2; i++) {
                ctx.beginPath();
                ctx.moveTo(w * i / 3, 0); ctx.lineTo(w * i / 3, h);
                ctx.moveTo(0, h * i / 3); ctx.lineTo(w, h * i / 3);
                ctx.stroke();
            }
        }

        // Safe areas (mode 2 or 4, or via safeAreaMode)
        const showSafeFromGrid = this.gridMode === 2 || this.gridMode === 4;
        const showActionSafe = showSafeFromGrid || this.safeAreaMode === 'action' || this.safeAreaMode === 'both';
        const showTitleSafe = showSafeFromGrid || this.safeAreaMode === 'title' || this.safeAreaMode === 'both';

        // Action Safe (93% - broadcast safe)
        if (showActionSafe) {
            ctx.strokeStyle = 'rgba(0, 200, 255, 0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([8, 4]);
            const actionMargin = 0.035; // 3.5% margin = 93% visible
            ctx.beginPath();
            ctx.rect(
                w * actionMargin, h * actionMargin,
                w * (1 - 2 * actionMargin), h * (1 - 2 * actionMargin)
            );
            ctx.stroke();
            ctx.setLineDash([]);

            // Label
            ctx.fillStyle = 'rgba(0, 200, 255, 0.6)';
            ctx.font = '9px sans-serif';
            ctx.fillText('Action Safe 93%', w * actionMargin + 4, h * actionMargin + 12);
        }

        // Title Safe (80% - text safe)
        if (showTitleSafe) {
            ctx.strokeStyle = 'rgba(255, 200, 0, 0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            const titleMargin = 0.10; // 10% margin = 80% visible
            ctx.beginPath();
            ctx.rect(
                w * titleMargin, h * titleMargin,
                w * (1 - 2 * titleMargin), h * (1 - 2 * titleMargin)
            );
            ctx.stroke();
            ctx.setLineDash([]);

            // Label
            ctx.fillStyle = 'rgba(255, 200, 0, 0.6)';
            ctx.font = '9px sans-serif';
            ctx.fillText('Title Safe 80%', w * titleMargin + 4, h * titleMargin + 12);
        }

        // Center cross (mode 3 or 4)
        if (this.gridMode === 3 || this.gridMode === 4) {
            ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(w / 2 - 30, h / 2); ctx.lineTo(w / 2 + 30, h / 2);
            ctx.moveTo(w / 2, h / 2 - 30); ctx.lineTo(w / 2, h / 2 + 30);
            ctx.stroke();

            // Center circle
            ctx.beginPath();
            ctx.arc(w / 2, h / 2, 5, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    drawAnnotation(ctx, a, lineScale) {
        ctx.strokeStyle = a.color;
        ctx.fillStyle = a.color;
        ctx.lineWidth = (a.width || 3) * lineScale;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();

        if (a.type === 'pen') {
            if (a.points && a.points.length > 0) {
                ctx.moveTo(a.points[0].x, a.points[0].y);
                for (let i = 1; i < a.points.length; i++) {
                    ctx.lineTo(a.points[i].x, a.points[i].y);
                }
            }
            ctx.stroke();
            return;
        }

        if (a.type === 'text') {
            ctx.font = `bold ${Math.max(10, (a.width || 3) * 5)}px sans-serif`;
            // Draw background for readability
            ctx.save();
            ctx.scale(1 / this.zoom, 1 / this.zoom); // Keep text constant size or scale? Let's scale with image so it stays fixed in place
            ctx.restore();
            // Actually, keep it simple: text scales with image

            ctx.shadowColor = 'black';
            ctx.shadowBlur = 4 * lineScale;
            ctx.lineWidth = 1 * lineScale; // reset for text stroke

            ctx.fillText(a.text, a.x1, a.y1);
            ctx.shadowBlur = 0; // v2.2: Reset shadow to prevent leak to subsequent draws
            return;
        }

        if (a.type === 'circle') {
            const rx = Math.abs(a.x2 - a.x1) / 2, ry = Math.abs(a.y2 - a.y1) / 2;
            ctx.ellipse((a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2, rx, ry, 0, 0, Math.PI * 2);
        } else if (a.type === 'arrow') {
            ctx.moveTo(a.x1, a.y1); ctx.lineTo(a.x2, a.y2);
            const ang = Math.atan2(a.y2 - a.y1, a.x2 - a.x1);
            const hl = (12 + (a.width || 3)) * lineScale;
            ctx.lineTo(a.x2 - hl * Math.cos(ang - 0.4), a.y2 - hl * Math.sin(ang - 0.4));
            ctx.moveTo(a.x2, a.y2);
            ctx.lineTo(a.x2 - hl * Math.cos(ang + 0.4), a.y2 - hl * Math.sin(ang + 0.4));
        } else if (a.type === 'rect') {
            ctx.rect(a.x1, a.y1, a.x2 - a.x1, a.y2 - a.y1);
        }
        ctx.stroke();
    }

    drawHistogramOverlay(ctx, w, h) {
        const { hR, hG, hB, max } = this.histogramData;
        const hx = w - 130, hy = 8, hw = 120, hh = 40;

        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(hx - 4, hy - 4, hw + 8, hh + 8);

        const draw = (hist, color) => {
            ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const x = hx + (i / 255) * hw, y = hy + hh - (hist[i] / max) * hh;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        };
        ctx.globalAlpha = 0.6;
        draw(hR, '#f44'); draw(hG, '#4f4'); draw(hB, '#44f');
        ctx.globalAlpha = 1;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    setupEventListeners() {
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = this.canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left, my = e.clientY - rect.top;
            const factor = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.max(0.02, Math.min(100, this.zoom * factor));
            this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
            this.panY = my - (my - this.panY) * (newZoom / this.zoom);
            this.zoom = newZoom;
            this.updateInfo();
            this.render();
        });

        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left, my = e.clientY - rect.top;
            const x = (mx - this.panX) / this.zoom;
            const y = (my - this.panY) / this.zoom;

            if (this.compareMode === 'wipe' && Math.abs(mx - rect.width * this.wipePosition) < 10) {
                this.isDraggingWipe = true; return;
            }

            if (this.isAnnotating && e.button === 0) {
                if (this.annotationTool === 'text') {
                    // v2.2: Inline text input instead of blocking prompt()
                    const input = document.createElement('input');
                    input.type = 'text';
                    input.placeholder = 'Type annotation...';
                    input.style.cssText = `
                        position: absolute; z-index: 1000;
                        left: ${e.clientX - this.canvas.getBoundingClientRect().left}px;
                        top: ${e.clientY - this.canvas.getBoundingClientRect().top}px;
                        background: rgba(0,0,0,0.9); color: #fff;
                        border: 1px solid ${this.theme.accent};
                        padding: 4px 8px; font-size: 12px; outline: none;
                        border-radius: 3px; min-width: 120px;
                    `;
                    const finalize = () => {
                        const text = input.value.trim();
                        if (text) this.addAnnotation('text', x, y, x, y, { text });
                        input.remove();
                    };
                    input.onkeydown = (ke) => { if (ke.key === 'Enter') finalize(); if (ke.key === 'Escape') input.remove(); };
                    input.onblur = finalize;
                    this.canvasWrapper.appendChild(input);
                    input.focus();
                    return;
                }

                if (this.annotationTool === 'pen') {
                    this.currentPath = [{ x, y }];
                }

                this.annotationStart = { x, y };
                return;
            }

            if (this.measurementMode !== 'none' && e.button === 0) {
                const x = (mx - this.panX) / this.zoom;
                const y = (my - this.panY) / this.zoom;
                this.measurementStart = { x, y };
                this.currentMeasurement = {
                    type: this.measurementMode,
                    x1: x, y1: y,
                    x2: x, y2: y
                };
                return;
            }

            if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
                this.isPanning = true;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                this.canvas.style.cursor = 'grabbing';
            }
        });

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left, my = e.clientY - rect.top;

            if (this.isDraggingWipe) {
                this.wipePosition = Math.max(0.02, Math.min(0.98, mx / rect.width));
                this.render(); return;
            }

            if (this.isPanning) {
                this.panX += e.clientX - this.lastMouseX;
                this.panY += e.clientY - this.lastMouseY;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                this.render();
            }

            // Handle Drawing Preview
            if (this.isAnnotating && this.annotationStart) {
                const x = (mx - this.panX) / this.zoom;
                const y = (my - this.panY) / this.zoom;

                if (this.annotationTool === 'pen' && this.currentPath) {
                    this.currentPath.push({ x, y });
                    this.renderOverlay();
                } else if (['circle', 'arrow', 'rect'].includes(this.annotationTool)) {
                    // We need to render a preview.
                    // The most efficient way is to just redraw overlay with a temporary shape.
                    this.renderOverlay();

                    // Draw preview directly here to avoid state complexity?
                    // Or better, add a "preview" param to renderOverlay or just use a temp field.
                    // Let's use direct drawing on overlay context for performance?
                    // Actually `renderOverlay` clears everything. So we must redraw all annotations + preview.

                    // Let's do a manual preview draw after `renderOverlay` call?
                    // No, `renderOverlay` is called above. Let's make `renderOverlay` aware of drag.
                    const ctx = this.overlayCtx;
                    ctx.save();
                    ctx.translate(this.panX, this.panY);
                    ctx.scale(this.zoom, this.zoom);
                    this.drawAnnotation(ctx, {
                        type: this.annotationTool,
                        x1: this.annotationStart.x,
                        y1: this.annotationStart.y,
                        x2: x,
                        y2: y,
                        color: this.annotationColor,
                        width: this.annotationLineWidth
                    }, 1 / this.zoom);
                    ctx.restore();
                }
            }

            // Handle Measurement Preview
            if (this.measurementStart && this.currentMeasurement) {
                const x = (mx - this.panX) / this.zoom;
                const y = (my - this.panY) / this.zoom;
                this.currentMeasurement.x2 = x;
                this.currentMeasurement.y2 = y;
                this.renderOverlay();
            }

            this.updateCursor(e);
            this.updateProbe(e);
        });

        // Click-to-Focus for DoF

        window.addEventListener('mouseup', (e) => {
            // Finalize annotation shapes on mouseup
            if (this.isAnnotating && this.annotationStart) {
                const rect = this.canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;
                const x = (mx - this.panX) / this.zoom;
                const y = (my - this.panY) / this.zoom;

                if (this.annotationTool === 'pen' && this.currentPath && this.currentPath.length > 1) {
                    this.annotations.push({
                        type: 'pen',
                        points: this.currentPath,
                        color: this.annotationColor,
                        width: this.annotationLineWidth
                    });
                } else if (['circle', 'arrow', 'rect'].includes(this.annotationTool)) {
                    const dx = x - this.annotationStart.x;
                    const dy = y - this.annotationStart.y;
                    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
                        this.addAnnotation(
                            this.annotationTool,
                            this.annotationStart.x,
                            this.annotationStart.y,
                            x, y
                        );
                    }
                }

                this.annotationStart = null;
                this.currentPath = null;
                this.renderOverlay();
            }

            // Finalize measurement on mouseup
            if (this.measurementStart) {
                if (this.currentMeasurement) {
                    // Only add if non-zero length
                    const dx = this.currentMeasurement.x2 - this.currentMeasurement.x1;
                    const dy = this.currentMeasurement.y2 - this.currentMeasurement.y1;
                    if (Math.abs(dx) > 0.1 || Math.abs(dy) > 0.1) {
                        this.measurements.push(this.currentMeasurement);
                    }
                }
                this.measurementStart = null;
                this.currentMeasurement = null;
                this.renderOverlay();
            }

            this.isPanning = false;
            this.isDraggingWipe = false;
            if (!this.isAnnotating) this.canvas.style.cursor = 'crosshair';
        });

        this.canvas.addEventListener('click', (e) => {
            if (this.dofEnabled && !this.isAnnotating && !this.isPanning && !this.isDraggingWipe) {
                const rect = this.canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;
                const imgX = Math.floor((mx - this.panX) / this.zoom);
                const imgY = Math.floor((my - this.panY) / this.zoom);

                if (this.imageData && imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
                    if (this.zdepthImage) {
                        // Sample Z-depth
                        const off = document.createElement('canvas'); off.width = 1; off.height = 1;
                        const ctx = off.getContext('2d');
                        ctx.drawImage(this.zdepthImage, imgX, imgY, 1, 1, 0, 0, 1, 1);
                        const d = ctx.getImageData(0, 0, 1, 1).data[0] / 255;

                        this.focusDistance = d;
                        // v2.2: Update HUD focus control (replaces undefined focusSlider)
                        if (this.focusControl) {
                            const finput = this.focusControl.querySelector('input[type="range"]');
                            if (finput) { finput.value = d; finput.dispatchEvent(new Event('input')); }
                        }
                        if (this.renderer) {
                            this.renderer.setFocusDistance(d);
                            this.render();
                        }
                    }
                }
            }
        });

        this.canvas.addEventListener('mouseleave', () => { this.probeTooltip.style.display = 'none'; });

        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => this.resize());
            this.resizeObserver.observe(this.canvasWrapper);
        }
    }

    resize() {
        const rect = this.canvasWrapper.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        this.canvas.width = Math.floor(rect.width);
        this.canvas.height = Math.floor(rect.height);
        this.overlayCanvas.width = this.canvas.width;
        this.overlayCanvas.height = this.canvas.height;
        this.render();
    }

    updateCursor(e) {
        if (!this.image) return;
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const imgX = Math.floor((mx - this.panX) / this.zoom);
        const imgY = Math.floor((my - this.panY) / this.zoom);
        if (imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
            this.cursorInfo.textContent = `X: ${imgX} Y: ${imgY}`;
        } else {
            this.cursorInfo.textContent = 'X: — Y: —';
        }
    }

    updateProbe(e) {
        if (!this.image || !this.imageData) { this.probeTooltip.style.display = 'none'; return; }
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const imgX = Math.floor((mx - this.panX) / this.zoom);
        const imgY = Math.floor((my - this.panY) / this.zoom);

        if (imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
            const idx = (imgY * this.imageWidth + imgX) * 4;
            let r = this.imageData[idx], g = this.imageData[idx + 1], b = this.imageData[idx + 2], a = this.imageData[idx + 3];

            // Check for HDR float data
            let floatR, floatG, floatB;

            if (this.hdrData && this.hdrData.data) {
                // NPY is usually HWC or CHW?
                // Our backend saves as (H, W, C).
                // shape[0]=H, shape[1]=W, shape[2]=C.
                const channels = this.hdrData.shape[2] || 3;
                const hIdx = (imgY * this.imageWidth + imgX) * channels;

                floatR = this.hdrData.data[hIdx];
                if (channels === 1) {
                    floatG = floatR; floatB = floatR;
                } else {
                    floatG = this.hdrData.data[hIdx + 1];
                    floatB = channels >= 3 ? this.hdrData.data[hIdx + 2] : 0;
                }
            } else {
                // Fallback to 8-bit normalized
                floatR = r / 255;
                floatG = g / 255;
                floatB = b / 255;
            }

            this.lastPixelColor = { r, g, b, a };
            const luma = (r * 0.2126 + g * 0.7152 + b * 0.0722).toFixed(0);
            const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;

            this.colorInfo.textContent = `RGB: ${r} ${g} ${b}`;
            this.probeTooltip.innerHTML = `
                <div style="margin-bottom:2px; border-bottom:1px solid #444; padding-bottom:2px"><b>X:${imgX} Y:${imgY}</b></div>
                <div style="color:${this.theme.accent}">Float: ${(floatR).toFixed(4)} ${(floatG).toFixed(4)} ${(floatB).toFixed(4)}</div>
                <div style="color:#888; font-size: 0.9em">8-bit: ${r} ${g} ${b} ${a !== 255 ? 'A:' + a : ''}</div>
                <div style="color:#666; font-size: 0.8em">Hex: ${hex} | L: ${luma}</div>
            `;
            this.probeTooltip.style.display = 'block';
            this.probeTooltip.style.left = `${mx + 12}px`;
            this.probeTooltip.style.top = `${my + 12}px`;

            // Draw pixel loupe on overlay
            if (this.showLoupe) {
                this.drawLoupe(mx, my, imgX, imgY);
            }
        } else {
            this.probeTooltip.style.display = 'none';
            this.colorInfo.textContent = 'RGB: — — —';
            this.lastPixelColor = null;
        }
    }

    drawLoupe(mx, my, imgX, imgY) {
        const ctx = this.overlayCtx;
        const size = this.loupeSize;
        const mag = this.loupeMagnification;
        const halfPixels = Math.floor(size / mag / 2);

        // Position loupe in corner opposite to cursor
        let lx = mx > this.canvas.width / 2 ? 10 : this.canvas.width - size - 10;
        let ly = my > this.canvas.height / 2 ? 10 : this.canvas.height - size - 10;

        // Draw loupe background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(lx - 2, ly - 2, size + 4, size + 4, 4);
        ctx.fill();
        ctx.stroke();

        // Draw magnified pixels
        for (let py = -halfPixels; py <= halfPixels; py++) {
            for (let px = -halfPixels; px <= halfPixels; px++) {
                const sx = imgX + px, sy = imgY + py;
                if (sx >= 0 && sx < this.imageWidth && sy >= 0 && sy < this.imageHeight) {
                    let r, g, b;
                    // v2.2: Read HDR float data for loupe when available
                    if (this.hdrData && this.hdrData.data) {
                        const ch = this.hdrData.channels || 3;
                        const hIdx = (sy * this.imageWidth + sx) * ch;
                        // Tonemap for display: simple Reinhard per-channel
                        const fr = this.hdrData.data[hIdx];
                        const fg = ch > 1 ? this.hdrData.data[hIdx + 1] : fr;
                        const fb = ch > 2 ? this.hdrData.data[hIdx + 2] : fr;
                        // Apply exposure
                        const em = Math.pow(2, this.exposure || 0);
                        r = Math.min(255, Math.max(0, Math.round((fr * em / (fr * em + 1)) * 255)));
                        g = Math.min(255, Math.max(0, Math.round((fg * em / (fg * em + 1)) * 255)));
                        b = Math.min(255, Math.max(0, Math.round((fb * em / (fb * em + 1)) * 255)));
                    } else {
                        const idx = (sy * this.imageWidth + sx) * 4;
                        r = this.imageData[idx]; g = this.imageData[idx + 1]; b = this.imageData[idx + 2];
                    }
                    ctx.fillStyle = `rgb(${r},${g},${b})`;
                } else {
                    ctx.fillStyle = '#222';
                }

                const dx = lx + (px + halfPixels) * mag;
                const dy = ly + (py + halfPixels) * mag;
                ctx.fillRect(dx, dy, mag, mag);

                // Highlight center pixel
                if (px === 0 && py === 0) {
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(dx, dy, mag, mag);
                }
            }
        }

        // Draw crosshair
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(lx + size / 2, ly); ctx.lineTo(lx + size / 2, ly + size);
        ctx.moveTo(lx, ly + size / 2); ctx.lineTo(lx + size, ly + size / 2);
        ctx.stroke();
    }

    toggleMetadata() {
        const isVisible = this.metadataPanel.style.display !== 'none';
        this.metadataPanel.style.display = isVisible ? 'none' : 'block';
        if (!isVisible) this.updateInfo();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          IMAGE
    // ═══════════════════════════════════════════════════════════════════════════

    async setImage(src, data = {}) {
        // Support both URL string and Image element
        if (src instanceof HTMLImageElement || src instanceof HTMLCanvasElement) {
            // Direct image element passed
            this.image = src;
            this.imageWidth = src.width;
            this.imageHeight = src.height;

            // Create ImageData for CPU processing
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = src.width;
            tempCanvas.height = src.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(src, 0, 0);
            this.imageData = tempCtx.getImageData(0, 0, src.width, src.height).data;

            // Load to WebGL renderer
            if (this.renderer) {
                this.renderer.loadImageTexture(src);
            }

            this.imageSrc = src.src || '';
            this.viewerData = data;
        } else {
            // URL string path
            this.imageSrc = src;
            this.viewerData = data;
            // Standard 8/16-bit image loading
            await this.loadStandardImage(src);
        }

        // Load depth map if provided
        if (data.depth_path) {
            const depthImg = new Image();
            depthImg.crossOrigin = 'anonymous';
            depthImg.onload = () => {
                this.zdepthImage = depthImg;
                if (this.renderer) this.renderer.loadDepthTexture(depthImg);
            };
            depthImg.src = data.depth_path;
        }

        this.fitToView();
        this.render();
        this.updateInfo();
        this.updateScopes();

        // Update Curve Editor Histogram
        if (this.curveEditor && this.image) {
            this.curveEditor.updateHistogram(this.image);
        }
    }

    async loadStandardImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';

            img.onload = () => {
                this.image = img;
                this.imageWidth = img.width;
                this.imageHeight = img.height;

                // Create ImageData for CPU processing
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = img.width;
                tempCanvas.height = img.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(img, 0, 0);
                this.imageData = tempCtx.getImageData(0, 0, img.width, img.height).data;

                // Load to WebGL renderer
                if (this.renderer) {
                    this.renderer.loadImageTexture(img);
                }

                resolve();
            };

            img.onerror = reject;
            img.src = src;
        });
    }

    // v3.1: Load HDR data from .rhdr (compressed fp16) or legacy .npy
    // Consolidated: now uses _parseHDRBuffer (which delegates to _parseRHDR)
    async loadHDRData(hdr_path) {
        const response = await fetch(hdr_path);
        const arrayBuffer = await response.arrayBuffer();

        const parsed = await this._parseHDRBuffer(arrayBuffer);

        if (!parsed) throw new Error('Failed to parse HDR sidecar');

        const height = parsed.shape[0];
        const width = parsed.shape[1];
        const channels = parsed.shape[2] || 3;

        const hdrData = {
            data: parsed.data,       // Float32Array (CPU reads)
            fp16data: parsed.fp16data, // Uint16Array (GPU upload, RHDR only)
            width, height, channels,
            shape: parsed.shape,
            format: parsed.format || 'npy'
        };

        // v2.2: Prefer HALF_FLOAT upload for .rhdr (2x faster upload, half VRAM)
        if (hdrData.fp16data && this.renderer) {
            this.renderer.loadFloat16Texture(
                hdrData.fp16data, width, height, channels
            );
        } else if (this.renderer) {
            this.renderer.loadFloat32Texture(
                hdrData.data, width, height, channels
            );
        }

        this.imageWidth = width;
        this.imageHeight = height;
        this.hdrData = hdrData;

        this.createPlaceholderImage(width, height);
    }

    // v3.1: _loadRHDR removed — consolidated into _parseRHDR (single implementation)
    // The loadHDRData method now uses _parseHDRBuffer → _parseRHDR.



    createPlaceholderImage(width, height) {
        // Create a small placeholder for 2D canvas operations
        const placeholder = document.createElement('canvas');
        placeholder.width = width;
        placeholder.height = height;
        const ctx = placeholder.getContext('2d');
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#fff';
        ctx.font = '24px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('WebGL2 HDR Mode', width / 2, height / 2);

        this.image = placeholder;
        // v3.0 FIX: Store .data (Uint8ClampedArray), not ImageData object.
        // The rest of the code indexes this.imageData[i] for pixel values.
        this.imageData = ctx.getImageData(0, 0, width, height).data;
    }

    setCompareImage(img) {
        this.compareImage = img;
        this.diffCanvas = null; // Clear difference cache
        if (this.compareMode === 'none') this.compareMode = 'wipe';
        this.render();
    }

    togglePlayback() {
        this.isPlaying = !this.isPlaying;
        if (this.playBtn) this.playBtn.textContent = this.isPlaying ? '⏸' : '▶';

        if (this.isPlaying) {
            this.lastFrameTime = performance.now();
            this.playbackLoop();
        }
    }

    playbackLoop() {
        if (!this.isPlaying) return;

        const now = performance.now();
        const fps = 24; // Target FPS, could be configurable
        const interval = 1000 / fps;

        if (now - this.lastFrameTime > interval) {
            this.nextFrame();
            this.lastFrameTime = now - ((now - this.lastFrameTime) % interval);
        }

        requestAnimationFrame(() => this.playbackLoop());
    }

    nextFrame() {
        if (!this.totalFrames || this.totalFrames <= 1) return;

        let next = this.currentFrame + 1;
        if (next >= this.totalFrames) {
            if (this.loop) {
                next = 0;
            } else {
                next = this.totalFrames - 1;
                this.isPlaying = false;
                if (this.playBtn) this.playBtn.textContent = '▶';
            }
        }

        this.setFrame(next);
    }

    prevFrame() {
        if (!this.totalFrames || this.totalFrames <= 1) return;

        let prev = this.currentFrame - 1;
        if (prev < 0) {
            if (this.loop) {
                prev = this.totalFrames - 1;
            } else {
                prev = 0;
            }
        }

        this.setFrame(prev);
    }

    setFrame(idx) {
        if (idx === this.currentFrame) return;
        this.currentFrame = idx;

        // Update Display
        if (this.frameHDRData[idx]) {
            // We have HDR data for this frame
            const npy = this.frameHDRData[idx];
            this.hdrData = npy;
            this.imageWidth = npy.width;
            this.imageHeight = npy.height;

            if (this.renderer) {
                if (npy.fp16data) {
                    this.renderer.loadFloat16Texture(npy.fp16data, npy.width, npy.height, npy.channels);
                } else {
                    this.renderer.loadFloat32Texture(npy.data, npy.width, npy.height, npy.channels);
                }
            }
        } else if (this.frameImages[idx]) {
            // Fallback to PNG
            this.hdrData = null;
            this.image = this.frameImages[idx];
            if (this.renderer) this.renderer.loadImageTexture(this.image);
        }

        this.render();
        this.updateInfo();
        this.updateFrameDisplay();

        // Update Scopes
        // Note: Real-time scopes update from displayed texture, so just calling updateScopes() is enough
        // assuming updateScopes pulls from renderer's texture.
        if (this.activeTab === 'scopes') {
            this.renderScopesTab(this.tabContentContainer);
        }
    }

    updateFrameDisplay() {
        if (this.frameCounter) {
            this.frameCounter.textContent = `${this.currentFrame + 1} / ${this.totalFrames}`;
        }
        // Also show transport if frames > 1
        if (this.totalFrames > 1 && this.transportPanel) {
            // ensure it's visible if it was hidden
        }
    }

    // Existing fitToView...
    fitToView() {
        if (!this.image) return;
        const w = this.canvas.width, h = this.canvas.height;
        if (w < 10 || h < 10) {
            // Retry if canvas is somehow not ready
            requestAnimationFrame(() => this.fitToView());
            return;
        }

        let z = Math.min(w / this.imageWidth, h / this.imageHeight);
        // Add a small margin (5%) if it's tight
        z = z * 0.95;

        this.zoom = z;
        this.panX = (w - this.imageWidth * this.zoom) / 2;
        this.panY = (h - this.imageHeight * this.zoom) / 2;
        this.updateInfo();
        this.render();
    }

    setZoom(z) {
        const cx = this.canvas.width / 2, cy = this.canvas.height / 2;
        // Zoom towards center of view, not 0,0
        const oldZ = this.zoom;

        // Calculate world point at center
        const wx = (cx - this.panX) / oldZ;
        const wy = (cy - this.panY) / oldZ;

        this.zoom = z;
        // Recalculate pan to keep world point at center
        this.panX = cx - wx * this.zoom;
        this.panY = cy - wy * this.zoom;

        this.updateInfo();
        this.render();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          RENDERING
    // ═══════════════════════════════════════════════════════════════════════════

    render() {
        if (this.canvas.width === 0 || this.canvas.height === 0) return;

        // ═══════════════════════════════════════════════════════════════
        // GPU-ACCELERATED WEBGL RENDERING PATH (Primary - Always On)
        // ═══════════════════════════════════════════════════════════════
        const useWebGL = this.useWebGL && this.renderer && this.renderer.textures.image;

        if (useWebGL) {
            // Ensure WebGL canvas is backend-only (hidden)
            this.glCanvas.style.visibility = 'hidden';

            // 1. Resize/Init WebGL Canvas to Image Size (Texture size)
            if (this.glCanvas.width !== this.imageWidth || this.glCanvas.height !== this.imageHeight) {
                this.glCanvas.width = this.imageWidth;
                this.glCanvas.height = this.imageHeight;
            }

            // 2. Update renderer state from UI controls (GPU parameters)
            this.renderer.setExposure(this.exposure || 0.0);
            const gArr = Array.isArray(this.gamma) ? this.gamma : [this.gamma || 1, this.gamma || 1, this.gamma || 1];
            this.renderer.setGamma(gArr[0], gArr[1], gArr[2]);
            this.renderer.setSaturation(this.saturation !== undefined ? this.saturation : 1.0);

            // Analytics State
            this.renderer.setFalseColor(this.falseColor || false);
            this.renderer.setZebra(this.zebra || false);
            this.renderer.setZebraThreshold(this.zebraThreshold || 0.95);

            // v2.2: Channel isolation + focus peaking + display LUT on GPU
            const chMap = { 'rgb': 0, 'r': 1, 'g': 2, 'b': 3, 'luma': 4, 'a': 5 };
            this.renderer.setChannelMode(chMap[this.channel] || 0);
            this.renderer.setFocusPeaking(this.focusPeaking || false);
            const lutMap = { 'None': 0, 'sRGB': 1, 'Rec.709': 2, 'LogC3': 3, 'ACEScg': 4 };
            this.renderer.setDisplayLutMode(lutMap[this.displayLut] || 0);

            // v2.3: Denoise & Depth Eval
            this.renderer.setDenoise(this.denoise || 0.0);
            this.renderer.setShowDepth(this.showZdepth || false);

            // DoF controls
            if (this.dofEnabled && this.renderer.textures.depth) {
                this.renderer.setDoFEnabled(true);
                this.renderer.setFocusDistance(this.focusDistance || 0.5);
                this.renderer.setAperture(this.aperture || 0.0);
            } else {
                this.renderer.setDoFEnabled(false);
            }

            // Render to WebGL canvas (GPU)
            const lutStrength = this.lutIntensity !== undefined ? this.lutIntensity : 1.0;
            this.renderer.render(lutStrength);

            // 3. Draw WebGL output to Main Canvas via 2D context with transform
            const ctx = this.ctx;
            const w = this.canvas.width, h = this.canvas.height;

            ctx.fillStyle = this.theme.bg;
            ctx.fillRect(0, 0, w, h);

            ctx.save();
            ctx.translate(this.panX, this.panY);
            ctx.scale(this.zoom, this.zoom);

            // Draw the GPU-rendered image
            ctx.drawImage(this.glCanvas, 0, 0);

            ctx.restore();

            this.renderOverlay();
            return; // WebGL path complete
        } else {
            if (this.glCanvas) this.glCanvas.style.visibility = 'hidden';
        }

        // ═══════════════════════════════════════════════════════════════
        // 2D CANVAS FALLBACK (CPU rendering - legacy/fallback only)
        // ═══════════════════════════════════════════════════════════════
        const ctx = this.ctx;
        const w = this.canvas.width, h = this.canvas.height;
        if (w === 0 || h === 0) return;

        ctx.fillStyle = this.theme.bg;
        ctx.fillRect(0, 0, w, h);

        if (!this.image) {
            ctx.fillStyle = this.theme.textDim;
            ctx.font = '12px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No image loaded', w / 2, h / 2);
            return;
        }

        // Ensure high quality scaling
        ctx.imageSmoothingEnabled = this.zoom < 1.0; // Smooth when downscaling, pixelated when upscaling?
        if (this.zoom > 2.0) ctx.imageSmoothingEnabled = false; // Pixel art look for high zoom
        else ctx.imageSmoothingQuality = 'high';

        if (this.compareMode === 'sidebyside' && this.compareImage) {
            this.renderSideBySide(ctx, w, h);
        } else if (this.compareMode === 'difference' && this.compareImage) {
            this.renderDifference(ctx, w, h);
        } else {
            ctx.save();
            ctx.translate(this.panX, this.panY);
            ctx.scale(this.zoom, this.zoom);
            this.renderImage(ctx, this.image);
            ctx.restore();

            if (this.compareMode === 'wipe' && this.compareImage) this.renderWipe(ctx, w, h);
        }

        this.renderOverlay();
    }

    renderImage(ctx, img) {
        // Z-Depth override
        if (this.showZdepth && this.zdepthImage) {
            ctx.drawImage(this.zdepthImage, 0, 0);
            return;
        }

        const gammaScalar = Array.isArray(this.gamma) ? (this.gamma[0] + this.gamma[1] + this.gamma[2]) / 3 : (this.gamma || 1.0);
        if (this.exposure !== 0 || gammaScalar !== 1.0 || this.channel !== 'rgb' || this.falseColor || this.zebra || this.focusPeaking || this.displayLut !== 'None') {
            this.renderProcessed(ctx, img);
        } else {
            ctx.drawImage(img, 0, 0);
        }
    }

    renderWipe(ctx, w, h) {
        const wipeX = w * this.wipePosition;
        ctx.save();
        ctx.beginPath(); ctx.rect(wipeX, 0, w - wipeX, h); ctx.clip();
        ctx.translate(this.panX, this.panY); ctx.scale(this.zoom, this.zoom); // Zoom needs adjustment for half width? No, keep relative
        // Actually for SxS usually we behave as two separate viewports or just cropped
        // Let's implement cropped view for better comparison
        ctx.drawImage(this.compareImage, 0, 0); ctx.restore();

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(wipeX, 0); ctx.lineTo(wipeX, h); ctx.stroke();

        // Handle
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.beginPath(); ctx.arc(wipeX, h / 2, 10, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.beginPath(); ctx.arc(wipeX, h / 2, 4, 0, Math.PI * 2); ctx.fill();
    }

    renderSideBySide(ctx, w, h) {
        const hw = w / 2;
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, hw, h); ctx.clip();
        ctx.translate(this.panX * 0.5, this.panY); ctx.scale(this.zoom * 0.5, this.zoom);
        ctx.drawImage(this.image, 0, 0); ctx.restore();

        ctx.save(); ctx.beginPath(); ctx.rect(hw, 0, hw, h); ctx.clip();
        ctx.translate(hw + this.panX * 0.5, this.panY); ctx.scale(this.zoom * 0.5, this.zoom);
        ctx.drawImage(this.compareImage, 0, 0); ctx.restore();

        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(hw, 0); ctx.lineTo(hw, h); ctx.stroke();
    }

    renderDifference(ctx, w, h) {
        if (!this.diffCanvas) {
            this.diffCanvas = document.createElement('canvas');
            this.diffCanvas.width = this.imageWidth;
            this.diffCanvas.height = this.imageHeight;
            const dCtx = this.diffCanvas.getContext('2d');

            // Draw A
            dCtx.drawImage(this.image, 0, 0);
            const dA = dCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);

            // Draw B to temporary canvas to get data
            const tmp = document.createElement('canvas');
            tmp.width = this.imageWidth; tmp.height = this.imageHeight;
            const tCtx = tmp.getContext('2d');
            tCtx.drawImage(this.compareImage, 0, 0);
            const dB = tCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);

            // Compute Difference
            for (let i = 0; i < dA.data.length; i += 4) {
                const r = Math.abs(dA.data[i] - dB.data[i]);
                const g = Math.abs(dA.data[i + 1] - dB.data[i + 1]);
                const b = Math.abs(dA.data[i + 2] - dB.data[i + 2]);

                // Visualization: Boost difference
                dA.data[i] = Math.min(255, r * 4);
                dA.data[i + 1] = Math.min(255, g * 4);
                dA.data[i + 2] = Math.min(255, b * 4);
                dA.data[i + 3] = 255;
            }
            dCtx.putImageData(dA, 0, 0);
        }

        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);
        ctx.drawImage(this.diffCanvas, 0, 0);
        ctx.restore();
    }

    renderProcessed(ctx, img) {
        // Reuse offscreen canvas to avoid severe GC pressure (33MB+ per 4K frame)
        if (!this._processCanvas) {
            this._processCanvas = document.createElement('canvas');
        }
        const off = this._processCanvas;
        if (off.width !== this.imageWidth || off.height !== this.imageHeight) {
            off.width = this.imageWidth; off.height = this.imageHeight;
        }
        const offCtx = off.getContext('2d');
        offCtx.drawImage(img, 0, 0);

        const imageData = offCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);
        const data = imageData.data;
        const expMult = Math.pow(2, this.exposure);
        const gammaAvg = Array.isArray(this.gamma) ? (this.gamma[0] + this.gamma[1] + this.gamma[2]) / 3 : (this.gamma || 1.0);
        const invGamma = 1.0 / gammaAvg;

        for (let i = 0; i < data.length; i += 4) {
            let r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;

            // v2.2: Linearize sRGB PNG data before grading (fixes double-gamma bug)
            const srgb2lin = (c) => c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
            r = srgb2lin(r); g = srgb2lin(g); b = srgb2lin(b);

            // Apply exposure in linear space
            r *= expMult; g *= expMult; b *= expMult;

            // Apply LUT / Color Space
            if (this.displayLut !== 'None') {
                [r, g, b] = this.applyColorTransform(r, g, b, this.displayLut);
            }

            // Apply Gamma
            if (gammaAvg !== 1.0) {
                r = Math.pow(Math.max(0, r), invGamma);
                g = Math.pow(Math.max(0, g), invGamma);
                b = Math.pow(Math.max(0, b), invGamma);
            }

            if (this.channel === 'r') { g = r; b = r; }
            else if (this.channel === 'g') { r = g; b = g; }
            else if (this.channel === 'b') { r = b; g = b; }
            else if (this.channel === 'a') {
                // Alpha channel - show as grayscale with checkerboard for transparency
                const a = data[i + 3] / 255;
                const x = (i / 4) % this.imageWidth, y = Math.floor((i / 4) / this.imageWidth);
                const checker = ((Math.floor(x / 8) + Math.floor(y / 8)) % 2) === 0 ? 0.3 : 0.5;
                r = g = b = a * 1.0 + (1 - a) * checker;
            }
            else if (this.channel === 'luma') { const l = r * 0.2126 + g * 0.7152 + b * 0.0722; r = g = b = l; }

            if (this.falseColor) {
                const l = r * 0.2126 + g * 0.7152 + b * 0.0722;
                const fc = this.getFalseColor(l);
                r = fc.r; g = fc.g; b = fc.b;
            }

            if (this.zebra) {
                const l = r * 0.2126 + g * 0.7152 + b * 0.0722;
                const x = (i / 4) % this.imageWidth, y = Math.floor((i / 4) / this.imageWidth);
                if (l > 0.95 && (x + y) % 8 < 4) { r = 1; g = 0; b = 0; }
                if (l < 0.02 && (x + y) % 8 < 4) { r = 0; g = 0; b = 1; }
            }

            // Clip final values
            data[i] = Math.min(255, Math.max(0, r * 255));
            data[i + 1] = Math.min(255, Math.max(0, g * 255));
            data[i + 2] = Math.min(255, Math.max(0, b * 255));
        }

        // Focus peaking - apply as post-process for edge detection
        if (this.focusPeaking) {
            this.applyFocusPeaking(imageData, this.imageWidth, this.imageHeight);
        }

        offCtx.putImageData(imageData, 0, 0);
        ctx.drawImage(off, 0, 0);
    }

    applyFocusPeaking(imageData, width, height) {
        const data = imageData.data;
        const threshold = this.focusPeakingThreshold;

        // Create edge buffer
        const edges = new Uint8Array(width * height);

        // Sobel edge detection on luminance
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;

                // Get luminance of surrounding pixels
                const getLuma = (ox, oy) => {
                    const i = ((y + oy) * width + (x + ox)) * 4;
                    return data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722;
                };

                // Sobel operators
                const gx = -getLuma(-1, -1) + getLuma(1, -1) +
                    -2 * getLuma(-1, 0) + 2 * getLuma(1, 0) +
                    -getLuma(-1, 1) + getLuma(1, 1);

                const gy = -getLuma(-1, -1) - 2 * getLuma(0, -1) - getLuma(1, -1) +
                    getLuma(-1, 1) + 2 * getLuma(0, 1) + getLuma(1, 1);

                const magnitude = Math.sqrt(gx * gx + gy * gy);
                edges[y * width + x] = magnitude > threshold ? 1 : 0;
            }
        }

        // Apply peaks as colored overlay
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                if (edges[y * width + x]) {
                    const idx = (y * width + x) * 4;
                    // Red highlight for sharp edges
                    data[idx] = 255;
                    data[idx + 1] = Math.floor(data[idx + 1] * 0.3);
                    data[idx + 2] = Math.floor(data[idx + 2] * 0.3);
                }
            }
        }
        // ═══════════════════════════════════════════════════════════════════════════
        //                          METADATA OVERLAY
        // ═══════════════════════════════════════════════════════════════════════════

        this.metadataOverlay = document.createElement('div');
        this.metadataOverlay.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            padding: 10px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            z-index: 90;
        `;

        // Top Left: Resolution + Bit Depth
        this.metaTL = document.createElement('div');
        this.metaTL.style.cssText = 'color: rgba(255,255,255,0.7); font-family: monospace; font-size: 11px; text-shadow: 1px 1px 2px #000;';
        this.metadataOverlay.appendChild(this.metaTL);

        // This info will be populated by updateInfo()

        // Top Right: Zoom / Pan (optional, maybe just Zoom)
        this.metaTR = document.createElement('div');
        this.metaTR.style.cssText = 'color: rgba(255,255,255,0.7); font-family: monospace; font-size: 11px; text-shadow: 1px 1px 2px #000; align-self: flex-end; position: absolute; top: 10px; right: 10px;';
        this.metadataOverlay.appendChild(this.metaTR);

        // Add to container (behind controls, above image)
        this.container.appendChild(this.metadataOverlay);

        this.container.onwheel = (e) => {
            // Update Zoom display on scroll
            requestAnimationFrame(() => this.updateInfo());
        };
    }

    getFalseColor(l) {
        if (l < 0.01) return { r: 0.1, g: 0, b: 0.3 };
        if (l < 0.08) return { r: 0, g: 0, b: 0.8 };
        if (l < 0.20) return { r: 0.3, g: 0.3, b: 0.3 };
        if (l < 0.40) return { r: 0, g: 0.7, b: 0 };
        if (l < 0.60) return { r: 0.8, g: 0.8, b: 0 };
        if (l < 0.80) return { r: 1, g: 0.5, b: 0 };
        if (l < 0.95) return { r: 1, g: 0, b: 0 };
        return { r: 1, g: 0, b: 1 };
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          RUN & PROMPT
    // ═══════════════════════════════════════════════════════════════════════════

    runWorkflow() {
        if (this.isQueueing) return;
        this.isQueueing = true;
        this.runButton.textContent = '⏳';

        // Use ComfyUI's queue prompt function
        app.queuePrompt(0).then(() => {
            this.isQueueing = false;
            this.runButton.textContent = '▶';
        }).catch(() => {
            this.isQueueing = false;
            this.runButton.textContent = '❌';
            setTimeout(() => this.runButton.textContent = '▶', 1000);
        });
    }

    updateInfo() {
        if (!this.image) {
            this.dimensionInfo.textContent = '—×—';
            this.zoomInfo.textContent = '—';
            return;
        }

        const z = (this.zoom * 100).toFixed(0);
        this.dimensionInfo.textContent = `${this.imageWidth}×${this.imageHeight}`;
        this.zoomInfo.textContent = `${z}%`;

        // Always update bit depth indicator in status bar
        if (this.bitDepthInfo) {
            let bitDepth = 'Int 8-bit';
            let bitColor = this.theme.textDim;
            if (this.hdrData) {
                if (this.hdrData.format === 'rhdr') {
                    bitDepth = 'Float 16-bit';
                    bitColor = '#60a5fa';
                } else {
                    bitDepth = 'Float 32-bit';
                    bitColor = '#4ade80';
                }
            }
            this.bitDepthInfo.textContent = bitDepth;
            this.bitDepthInfo.style.color = bitColor;
            this.bitDepthInfo.style.background = `${bitColor}15`;
            this.bitDepthInfo.style.border = `1px solid ${bitColor}30`;
        }

        // Update metadata panel if visible
        if (this.metadataContent && this.metadataPanel.style.display !== 'none') {
            let hdrStatus = '—';
            let formatStr = 'PNG 8-bit';
            let bitDepth = 'Int 8-bit';
            let bitColor = this.theme.textDim;
            if (this.hdrData) {
                const hasHighValues = this.hdrData.data &&
                    Array.from(this.hdrData.data.slice(0, 1000)).some(v => v > 1.0 || v < 0.0);
                hdrStatus = hasHighValues ? '✓ HDR Content' : '✗ Standard Range';
                if (this.hdrData.format === 'rhdr') {
                    formatStr = 'RHDR fp16 (primary)';
                    bitDepth = 'Float 16-bit';
                    bitColor = '#60a5fa'; // blue
                } else {
                    formatStr = 'Float32 HDR';
                    bitDepth = 'Float 32-bit';
                    bitColor = '#4ade80'; // green
                }
            } else if (this.imageData) {
                hdrStatus = '✗ Standard Range';
            }

            // Update status bar bit depth
            if (this.bitDepthInfo) {
                this.bitDepthInfo.textContent = bitDepth;
                this.bitDepthInfo.style.color = bitColor;
                this.bitDepthInfo.style.background = `${bitColor}15`;
                this.bitDepthInfo.style.border = `1px solid ${bitColor}30`;
            }

            this.metadataContent.innerHTML = `
                <div style="margin-bottom: 6px;">
                    <span style="color:${this.theme.textDim}">Resolution:</span><br/>
                    <span style="color:${this.theme.accent}">${this.imageWidth} × ${this.imageHeight}</span>
                </div>
                <div style="margin-bottom: 6px;">
                    <span style="color:${this.theme.textDim}">Format:</span><br/>
                    <span style="color:${formatStr.includes('RHDR') ? '#4ade80' : this.theme.text}">${formatStr}</span>
                </div>
                <div style="margin-bottom: 6px;">
                    <span style="color:${this.theme.textDim}">Channels:</span><br/>
                    <span>${this.hdrData ? `${this.hdrData.channels || 3}ch fp16` : 'RGBA (4)'}</span>
                </div>
                <div style="margin-bottom: 6px;">
                    <span style="color:${this.theme.textDim}">HDR Status:</span><br/>
                    <span style="color:${hdrStatus.includes('✓') ? '#4ade80' : '#94a3b8'}">${hdrStatus}</span>
                </div>
                <div>
                    <span style="color:${this.theme.textDim}">Zoom:</span><br/>
                    <span>${z}%</span>
                </div>
            `;
        }
    }

    togglePromptPanel() {
        this.showPromptPanel = !this.showPromptPanel;

        if (this.showPromptPanel) {
            this.promptButton.classList.add('active');
            this.promptButton.style.background = this.theme.accent;
            this.promptButton.style.color = '#fff';
            this.createPromptPanel();
        } else {
            this.promptButton.classList.remove('active');
            this.promptButton.style.background = '';
            this.promptButton.style.color = this.theme.textDim;
            if (this.promptPanel) this.promptPanel.remove();
            this.promptPanel = null;
        }
    }

    createPromptPanel() {
        if (this.promptPanel) this.promptPanel.remove();

        const t = this.theme;
        this.promptPanel = document.createElement('div');
        this.promptPanel.style.cssText = `
            position: absolute;
            top: 40px;
            right: 10px;
            width: 300px;
            max-height: 80%;
            background: rgba(16, 16, 24, 0.95);
            border: 1px solid ${t.panelBorder};
            border-radius: 4px;
            padding: 8px;
            overflow-y: auto;
            z-index: 100;
            display: flex;
            flex-direction: column;
            gap: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        `;

        // Heuristic: Scan for ANY node that looks like it has a text/prompt widget
        const nodes = app.graph._nodes.filter(n => {
            if (!n.widgets) return false;

            // Check specific types first
            if (n.type && (n.type.includes("CLIPTextEncode") || n.type.includes("Prompt"))) return true;

            // Heuristic checking of widgets
            const hasTextWidget = n.widgets.some(w => {
                // Check widget names likely to be prompts
                const name = (w.name || "").toLowerCase();
                // check if value is string and not a small config string (like "enable")
                const isString = typeof w.value === "string";

                return isString && (
                    name === "text" ||
                    name === "string" ||
                    name.includes("prompt") ||
                    w.type === "customtext"
                );
            });

            return hasTextWidget;
        });

        // Sort by vertical position
        nodes.sort((a, b) => a.pos[1] - b.pos[1]);

        if (nodes.length === 0) {
            const msg = document.createElement('div');
            msg.textContent = "No prompt nodes found.";
            msg.style.color = t.textDim;
            msg.style.fontSize = "11px";
            this.promptPanel.appendChild(msg);
        } else {
            nodes.forEach(node => {
                const wrapper = document.createElement('div');
                wrapper.style.display = 'flex';
                wrapper.style.flexDirection = 'column';
                wrapper.style.gap = '4px';

                const label = document.createElement('div');
                label.textContent = node.title || node.type;
                label.style.cssText = `color: ${t.accent}; font-size: 11px; font-weight: bold; cursor: pointer;`;
                label.title = "Jump to node";
                label.onclick = () => {
                    app.canvas.centerOnNode(node);
                    app.canvas.selectNode(node);
                };

                wrapper.appendChild(label);

                // Render ALL widgets
                if (node.widgets) {
                    node.widgets.forEach(w => {
                        // Skip converted or hidden widgets
                        if (w.type === 'converted-widget' || w.name === '_temp') return;

                        const wContainer = document.createElement('div');
                        wContainer.style.cssText = 'display: flex; flex-direction: column; gap: 2px; margin-bottom: 4px;';

                        // Label for parameters (skip for main prompt text to save space, or keep small?)
                        const isMainText = (w.type === 'customtext' || w.name === 'text');
                        if (!isMainText) {
                            const wl = document.createElement('div');
                            wl.textContent = w.name;
                            wl.style.cssText = `color: ${t.textDim}; font-size: 9px;`;
                            wContainer.appendChild(wl);
                        }

                        let input;

                        // 1. Text / String
                        if (w.type === 'customtext' || w.type === 'text' || (!w.type && typeof w.value === 'string')) {
                            input = document.createElement('textarea');
                            input.value = w.value;
                            input.style.cssText = `
                                width: 100%;
                                height: ${isMainText ? '60px' : '30px'};
                                background: #0a0a0f;
                                border: 1px solid ${t.panelBorder};
                                color: ${t.text};
                                font-size: 11px;
                                padding: 4px;
                                resize: vertical;
                                font-family: inherit;
                            `;
                            input.addEventListener('input', (e) => {
                                w.value = e.target.value;
                                if (w.callback) w.callback(w.value);
                            });
                            // Focus helpers
                            input.addEventListener('focus', () => { app.canvas.selectNode(node); wrapper.style.borderLeft = `2px solid ${t.accent}`; });
                            input.addEventListener('blur', () => { wrapper.style.borderLeft = 'none'; });
                        }
                        // 2. Number
                        else if (w.type === 'number' || typeof w.value === 'number') {
                            input = document.createElement('input');
                            input.type = 'number';
                            input.value = w.value;
                            if (w.options) {
                                if (w.options.min !== undefined) input.min = w.options.min;
                                if (w.options.max !== undefined) input.max = w.options.max;
                                if (w.options.step !== undefined) input.step = w.options.step;
                            }
                            input.style.cssText = `
                                width: 100%;
                                background: #0a0a0f;
                                border: 1px solid ${t.panelBorder};
                                color: ${t.text};
                                font-size: 11px;
                                padding: 2px 4px;
                            `;
                            input.addEventListener('input', (e) => {
                                let val = parseFloat(e.target.value);
                                if (w.options) {
                                    if (w.options.min !== undefined) val = Math.max(w.options.min, val);
                                    if (w.options.max !== undefined) val = Math.min(w.options.max, val);
                                }
                                w.value = val;
                                if (w.callback) w.callback(w.value);
                            });
                        }
                        // 3. Combo
                        else if (w.type === 'combo') {
                            input = document.createElement('select');
                            input.style.cssText = `
                                width: 100%;
                                background: #0a0a0f;
                                border: 1px solid ${t.panelBorder};
                                color: ${t.text};
                                font-size: 11px;
                                padding: 2px;
                            `;
                            if (w.options && w.options.values) {
                                w.options.values.forEach(v => {
                                    const opt = document.createElement('option');
                                    opt.value = v;
                                    opt.textContent = v;
                                    input.appendChild(opt);
                                });
                            }
                            input.value = w.value;
                            input.addEventListener('change', (e) => {
                                w.value = e.target.value;
                                if (w.callback) w.callback(w.value);
                            });
                        }
                        // 4. Toggle / Boolean
                        else if (w.type === 'toggle' || typeof w.value === 'boolean') {
                            const row = document.createElement('div');
                            row.style.cssText = 'display: flex; align-items: center; gap: 6px;';
                            input = document.createElement('input');
                            input.type = 'checkbox';
                            input.checked = w.value;
                            input.addEventListener('change', (e) => {
                                w.value = e.target.checked;
                                if (w.callback) w.callback(w.value);
                            });
                            row.appendChild(input);

                            // Move label here for checkboxes
                            if (wContainer.firstChild) wContainer.firstChild.remove(); // Remove top label
                            const lbl = document.createElement('span');
                            lbl.textContent = w.name;
                            lbl.style.cssText = `color: ${t.textDim}; font-size: 11px;`;
                            row.appendChild(lbl);

                            wContainer.appendChild(row);
                            input = null; // Handled wrappers
                        }

                        if (input) wContainer.appendChild(input);
                        wrapper.appendChild(wContainer);
                    });
                }

                this.promptPanel.appendChild(wrapper);
            });

            // Add Run Button Inside Panel
            const runBtnPanel = document.createElement('button');
            runBtnPanel.textContent = 'Apply & Queue (Shift+Enter)';
            runBtnPanel.style.textTransform = 'uppercase';
            runBtnPanel.style.cssText = `
                margin-top: 8px;
                padding: 6px;
                background: ${t.accent};
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
                font-weight: bold;
            `;
            runBtnPanel.onclick = () => this.runWorkflow();
            this.promptPanel.appendChild(runBtnPanel);
        }

        this.container.appendChild(this.promptPanel);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          HUD / CONTROLS
    // ═══════════════════════════════════════════════════════════════════════════

    createHUD() {
        if (this.controlsPanel) this.controlsPanel.remove();

        const t = this.theme;
        this.controlsPanel = document.createElement('div');
        this.controlsPanel.className = 'radiance-glass-dock';
        this.controlsPanel.style.cssText = `
            position: absolute;
            bottom: 32px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(10, 10, 14, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 12px;
            z-index: 50;
            display: flex;
            flex-direction: column;
            gap: 12px;
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            box-shadow: 0 16px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05) inset;
            transition: opacity 0.4s cubic-bezier(0.2, 0.8, 0.2, 1.0), transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1.0);
            width: ${this.hudPanelWidth}px;
            min-width: ${this.hudPanelMinWidth}px;
            max-width: ${this.hudPanelMaxWidth}px;
            resize: horizontal;
            overflow: hidden;
            font-family: ${t.font};
            opacity: 1;
            pointer-events: auto;
        `;

        // Observe resize (from CSS resize handle) and persist width
        const hudResizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const newWidth = Math.round(entry.contentRect.width + 24); // +padding
                if (newWidth >= this.hudPanelMinWidth && newWidth <= this.hudPanelMaxWidth) {
                    this.hudPanelWidth = newWidth;
                    localStorage.setItem('radiance_hud_width', newWidth);
                }
            }
        });
        hudResizeObserver.observe(this.controlsPanel);

        // 1. Tabs Header
        const tabsHeader = document.createElement('div');
        tabsHeader.style.cssText = 'display: flex; gap: 4px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 4px;';

        const activeTabStyle = `background: rgba(255,255,255,0.1); color: ${t.text}; border-bottom: 2px solid ${t.accent}`;
        const inactiveTabStyle = `background: transparent; color: ${t.textDim}; border-bottom: 2px solid transparent`;
        const baseTabStyle = `flex: 1; text-align: center; padding: 6px 0; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; cursor: pointer; border-radius: 4px; transition: all 0.2s;`;

        let activeTab = 'primaries';
        const tabContentContainer = document.createElement('div');
        this.tabContentContainer = tabContentContainer;

        const tabs = [
            { id: 'primaries', label: 'PRIMARIES' },
            { id: 'curves', label: 'CURVES' },
            { id: 'film', label: 'FILM' },
            { id: 'lens', label: 'LENS' },
            { id: 'qualifiers', label: 'QUALIFIER' },
            { id: 'scopes', label: 'SCOPES' }
        ];

        const renderTabs = () => {
            tabsHeader.innerHTML = '';
            tabs.forEach(tab => {
                const btn = document.createElement('div');
                btn.textContent = tab.label;
                btn.style.cssText = baseTabStyle + (activeTab === tab.id ? activeTabStyle : inactiveTabStyle);
                btn.onmouseover = () => { if (activeTab !== tab.id) btn.style.background = 'rgba(255,255,255,0.05)'; };
                btn.onmouseout = () => { if (activeTab !== tab.id) btn.style.background = 'transparent'; };
                btn.onclick = () => {
                    activeTab = tab.id;
                    renderTabs();
                    renderContent();
                };
                tabsHeader.appendChild(btn);
            });
        };

        this.controlsPanel.appendChild(tabsHeader);
        this.controlsPanel.appendChild(tabContentContainer);

        // 2. Tab Content Renderer
        const renderContent = () => {
            tabContentContainer.innerHTML = '';
            tabContentContainer.style.cssText = 'min-height: 140px; display: flex; flex-direction: column; justify-content: center;';

            if (activeTab === 'primaries') {
                this.renderPrimariesTab(tabContentContainer);
            } else if (activeTab === 'curves') {
                this.renderCurvesTab(tabContentContainer);
            } else if (activeTab === 'film') {
                this.renderEffectsTab(tabContentContainer);
            } else if (activeTab === 'lens') {
                this.renderLensTab(tabContentContainer);
            } else if (activeTab === 'qualifiers') {
                this.renderQualifiersTab(tabContentContainer);
            } else if (activeTab === 'scopes') {
                this.renderScopesTab(tabContentContainer);
            }
        };

        // Initialize
        renderTabs();
        renderContent();
        // Save reference for undo/redo panel refresh
        this._lastRenderContent = renderContent;

        // Footer: A/B Bypass + Reset All
        const footer = document.createElement('div');
        footer.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 4px;';

        // A/B Bypass Toggle
        const bypassBtn = document.createElement('div');
        bypassBtn.textContent = this._gradingBypassed ? '● BYPASSED' : 'A/B';
        bypassBtn.title = 'Toggle grading bypass (compare original)';
        bypassBtn.style.cssText = `font-size: 9px; color: ${this._gradingBypassed ? '#ff6b6b' : '#666'}; cursor: pointer; letter-spacing: 1px; padding: 2px 6px; border: 1px solid ${this._gradingBypassed ? 'rgba(255,100,100,0.3)' : 'rgba(255,255,255,0.08)'}; border-radius: 4px;`;
        bypassBtn.onclick = () => {
            this._gradingBypassed = !this._gradingBypassed;
            if (this._gradingBypassed) {
                // Save current state and set identity
                this._savedGrading = {
                    exposure: this.exposure, lift: [...(this.lift || [0, 0, 0])], gamma: [...(this.gamma || [1, 1, 1])], gain: [...(this.gain || [1, 1, 1])],
                    temperature: this.temperature, tint: this.tint, contrast: this.contrast, pivot: this.pivot, saturation: this.saturation,
                    grain: this.grain, denoise: this.denoise,
                    lensDistortion: this.lensDistortion, lensFringe: this.lensFringe,
                    vignetteIntensity: this.vignetteIntensity, vignetteFalloff: this.vignetteFalloff,
                };
                if (this.renderer) {
                    this.renderer.setExposure(0); this.renderer.setLift(0, 0, 0); this.renderer.setGamma(1, 1, 1); this.renderer.setGain(1, 1, 1);
                    this.renderer.setTemperature(0); this.renderer.setTint(0); this.renderer.setContrast(1); this.renderer.setPivot(0.5); this.renderer.setSaturation(1);
                    this.renderer.setGrain(0); this.renderer.setDenoise(0);
                    this.renderer.setLensDistortion(0, 0); this.renderer.setVignette(0, 0.5);
                }
                bypassBtn.textContent = '● BYPASSED'; bypassBtn.style.color = '#ff6b6b'; bypassBtn.style.borderColor = 'rgba(255,100,100,0.3)';
            } else {
                // Restore saved state
                const s = this._savedGrading;
                if (s && this.renderer) {
                    this.renderer.setExposure(s.exposure); this.renderer.setLift(s.lift[0], s.lift[1], s.lift[2]);
                    this.renderer.setGamma(s.gamma[0], s.gamma[1], s.gamma[2]); this.renderer.setGain(s.gain[0], s.gain[1], s.gain[2]);
                    this.renderer.setTemperature(s.temperature); this.renderer.setTint(s.tint); this.renderer.setContrast(s.contrast); this.renderer.setPivot(s.pivot); this.renderer.setSaturation(s.saturation);
                    this.renderer.setGrain(s.grain); this.renderer.setDenoise(s.denoise);
                    this.renderer.setLensDistortion(s.lensDistortion, s.lensFringe); this.renderer.setVignette(s.vignetteIntensity, s.vignetteFalloff);
                }
                bypassBtn.textContent = 'A/B'; bypassBtn.style.color = '#666'; bypassBtn.style.borderColor = 'rgba(255,255,255,0.08)';
            }
            this.render();
        };
        footer.appendChild(bypassBtn);

        // Undo / Redo Buttons
        const undoRedoGroup = document.createElement('div');
        undoRedoGroup.style.cssText = 'display: flex; gap: 4px;';

        const undoBtn = document.createElement('div');
        undoBtn.textContent = '↶';
        undoBtn.title = 'Undo (Ctrl+Z)';
        undoBtn.style.cssText = 'font-size: 14px; color: #555; cursor: pointer; padding: 0 4px; user-select: none; transition: color 0.15s;';
        undoBtn.onmouseenter = () => { undoBtn.style.color = this._undoStack.length > 0 ? this.theme.accent : '#555'; };
        undoBtn.onmouseleave = () => { undoBtn.style.color = '#555'; };
        undoBtn.onclick = () => { this.undo(); };

        const redoBtn = document.createElement('div');
        redoBtn.textContent = '↷';
        redoBtn.title = 'Redo (Ctrl+Shift+Z)';
        redoBtn.style.cssText = 'font-size: 14px; color: #555; cursor: pointer; padding: 0 4px; user-select: none; transition: color 0.15s;';
        redoBtn.onmouseenter = () => { redoBtn.style.color = this._redoStack.length > 0 ? this.theme.accent : '#555'; };
        redoBtn.onmouseleave = () => { redoBtn.style.color = '#555'; };
        redoBtn.onclick = () => { this.redo(); };

        undoRedoGroup.appendChild(undoBtn);
        undoRedoGroup.appendChild(redoBtn);
        footer.appendChild(undoRedoGroup);

        // Reset All Button
        const resetBtn = document.createElement('div');
        resetBtn.textContent = 'RESET ALL';
        resetBtn.style.cssText = 'font-size: 9px; color: #666; cursor: pointer; letter-spacing: 1px;';
        resetBtn.onclick = () => {
            // Push undo before resetting
            this._pushUndo();
            // Primaries
            this.exposure = 0.0;
            this.lift = [0, 0, 0]; this.gamma = [1, 1, 1]; this.gain = [1, 1, 1];
            this.temperature = 0.0; this.tint = 0.0;
            this.contrast = 1.0; this.pivot = 0.5; this.saturation = 1.0;
            // Film / Effects
            this.grain = 0.0; this.denoise = 0.0;
            // Lens
            this.focusDistance = 0.5; this.aperture = 0.0; this.dofEnabled = false;
            this.apertureBlades = 0; this.apertureRotation = 0.0; this.apertureAnamorphic = 1.0;
            this.lensDistortion = 0.0; this.lensFringe = 0.0;
            this.vignetteIntensity = 0.0; this.vignetteFalloff = 0.5;
            // Curves
            if (this.curveEditor) this.curveEditor.resetAllChannels?.();
            // Qualifier
            if (this.qualifierState) {
                this.qualifierState.enabled = false; this.qualifierState.showMask = false;
            }

            if (this.renderer) {
                this.renderer.setExposure(0);
                this.renderer.setLift(0, 0, 0); this.renderer.setGamma(1, 1, 1); this.renderer.setGain(1, 1, 1);
                this.renderer.setTemperature(0); this.renderer.setTint(0);
                this.renderer.setContrast(1); this.renderer.setPivot(0.5); this.renderer.setSaturation(1);
                this.renderer.setGrain(0); this.renderer.setDenoise(0);
                this.renderer.setDoFEnabled(false); this.renderer.setFocusDistance(0.5); this.renderer.setAperture(0);
                this.renderer.setApertureShape(0, 0, 1.0);
                this.renderer.setLensDistortion(0, 0); this.renderer.setVignette(0, 0.5);
                this.renderer.setCurveMix(0);
                if (this.qualifierState && this.renderer.setQualifier) {
                    this.renderer.setQualifier(this.qualifierState);
                }
            }
            this._gradingBypassed = false;
            this.render();
            renderContent();
        };
        footer.appendChild(resetBtn);
        this.controlsPanel.appendChild(footer);

        // Keyboard shortcuts for undo/redo (scoped to document when HUD is visible)
        if (!this._undoKeyListener) {
            this._undoKeyListener = (e) => {
                // Only respond when HUD is visible and not in a text input
                if (!this.showControls) return;
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
                    e.preventDefault();
                    this.undo();
                } else if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
                    e.preventDefault();
                    this.redo();
                } else if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
                    e.preventDefault();
                    this.redo();
                }
            };
            document.addEventListener('keydown', this._undoKeyListener);
        }

        // ═══════════════════════════════════════════════════════════════════════════
        //                          TRANSPORT CONTROLS
        // ═══════════════════════════════════════════════════════════════════════════

        this.transportPanel = document.createElement('div');
        this.transportPanel.style.cssText = `
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(10, 10, 12, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 8px 16px;
            backdrop-filter: blur(8px);
            z-index: 100;
            transition: opacity 0.2s;
            opacity: 0; 
            pointer-events: none;
        `;

        // Show transport on hover over container
        this.container.addEventListener('mouseenter', () => {
            if (this.totalFrames > 1) {
                this.transportPanel.style.opacity = '1';
                this.transportPanel.style.pointerEvents = 'auto';
            }
        });
        this.container.addEventListener('mouseleave', () => {
            this.transportPanel.style.opacity = '0';
            this.transportPanel.style.pointerEvents = 'none';
        });

        // Loop Toggle
        const loopBtn = document.createElement('div');
        loopBtn.innerHTML = '∞'; // Infinite symbol
        loopBtn.title = 'Toggle Loop';
        loopBtn.style.cssText = 'color: #888; cursor: pointer; font-size: 16px; width: 20px; text-align: center;';
        loopBtn.onclick = () => {
            this.loop = !this.loop;
            loopBtn.style.color = this.loop ? this.theme.accent : '#888';
        };
        this.transportPanel.appendChild(loopBtn);

        // Prev Frame
        const prevBtn = document.createElement('div');
        prevBtn.textContent = '⏮';
        prevBtn.style.cssText = 'color: #ccc; cursor: pointer; font-size: 14px;';
        prevBtn.onclick = () => this.prevFrame();
        this.transportPanel.appendChild(prevBtn);

        // Play/Pause
        const playBtn = document.createElement('div');
        playBtn.textContent = '▶';
        playBtn.style.cssText = 'color: #fff; cursor: pointer; font-size: 18px; width: 24px; text-align: center;';
        playBtn.onclick = () => this.togglePlayback();
        this.playBtn = playBtn; // Save ref to update icon
        this.transportPanel.appendChild(playBtn);

        // Next Frame
        const nextBtn = document.createElement('div');
        nextBtn.textContent = '⏭';
        nextBtn.style.cssText = 'color: #ccc; cursor: pointer; font-size: 14px;';
        nextBtn.onclick = () => this.nextFrame();
        this.transportPanel.appendChild(nextBtn);

        // Frame Counter
        this.frameCounter = document.createElement('div');
        this.frameCounter.textContent = '1 / 1';
        this.frameCounter.style.cssText = 'color: #888; font-size: 11px; font-family: monospace; margin-left: 8px; min-width: 50px; text-align: center;';
        this.transportPanel.appendChild(this.frameCounter);

        this.container.appendChild(this.transportPanel);

        this.canvasWrapper.appendChild(this.controlsPanel);
    }

    renderPrimariesTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; gap: 12px; padding: 10px 4px;';

        const createMini = (lbl, min, max, val, step, cb) => {
            const k = this.createKnob(lbl, min, max, val, step, cb);
            k.style.transform = 'scale(0.9)';
            return k;
        };

        // ═════════════════════════════════════════════════════════════════════
        // 1. TOP BAR: Temp | Tint | Contrast | Pivot | Mid/Detail
        // ═════════════════════════════════════════════════════════════════════
        const topBar = document.createElement('div');
        topBar.style.cssText = 'display: grid; grid-template-columns: repeat(5, 1fr); gap: 2px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; justify-items: center;';

        topBar.appendChild(createMini('TEMP', -2.0, 2.0, this.temperature || 0.0, 0.05, v => {
            this.temperature = v;
            if (this.renderer) this.renderer.setTemperature(v);
            this.render();
        }));

        topBar.appendChild(createMini('TINT', -2.0, 2.0, this.tint || 0.0, 0.05, v => {
            this.tint = v;
            if (this.renderer) this.renderer.setTint(v);
            this.render();
        }));

        topBar.appendChild(createMini('CONTRAST', 0.5, 2.0, this.contrast || 1.0, 0.02, v => {
            this.contrast = v;
            if (this.renderer) this.renderer.setContrast(v);
            this.render();
        }));

        topBar.appendChild(createMini('PIVOT', 0.0, 1.0, this.pivot || 0.5, 0.05, v => {
            this.pivot = v;
            if (this.renderer) this.renderer.setPivot(v);
            this.render();
        }));

        // Midtone Detail
        topBar.appendChild(createMini('M.DETAIL', -1.0, 1.0, this.midDetail || 0.0, 0.05, v => {
            this.midDetail = v;
            if (this.renderer) this.renderer.setMidDetail(v);
            this.render();
        }));

        container.appendChild(topBar);

        // ═════════════════════════════════════════════════════════════════════
        // 2. COLOR WHEELS: Lift | Gamma | Gain | Offset
        // ═════════════════════════════════════════════════════════════════════
        const wheelsRow = document.createElement('div');
        wheelsRow.style.cssText = 'display: flex; justify-content: space-between; gap: 4px; padding: 4px 0;';

        // Lift
        wheelsRow.appendChild(this.createColorWheel('LIFT', -0.2, 0.2, this.lift || [0, 0, 0], 0.005, (r, g, b) => {
            this.lift = [r, g, b];
            if (this.renderer) this.renderer.setLift(r, g, b);
            this.render();
        }));

        // Gamma
        wheelsRow.appendChild(this.createColorWheel('GAMMA', -0.5, 0.5, this.gamma ? this.gamma.map(x => x - 1.0) : [0, 0, 0], 0.01, (r, g, b) => {
            this.gamma = [Math.max(0.1, 1.0 + r), Math.max(0.1, 1.0 + g), Math.max(0.1, 1.0 + b)];
            if (this.renderer) this.renderer.setGamma(this.gamma[0], this.gamma[1], this.gamma[2]);
            this.render();
        }));

        // Gain
        wheelsRow.appendChild(this.createColorWheel('GAIN', -0.5, 1.5, this.gain ? this.gain.map(x => x - 1.0) : [0, 0, 0], 0.01, (r, g, b) => {
            this.gain = [Math.max(0, 1.0 + r), Math.max(0, 1.0 + g), Math.max(0, 1.0 + b)];
            if (this.renderer) this.renderer.setGain(this.gain[0], this.gain[1], this.gain[2]);
            this.render();
        }));

        // Offset (Global)
        // Global Offset is usually additive.
        wheelsRow.appendChild(this.createColorWheel('OFFSET', -0.5, 0.5, this.offset || [0, 0, 0], 0.005, (r, g, b) => {
            this.offset = [r, g, b];
            if (this.renderer) this.renderer.setOffset(r, g, b);
            this.render();
        }));

        container.appendChild(wheelsRow);


        // ═════════════════════════════════════════════════════════════════════
        // 3. BOTTOM BAR: Boost | Shadows | Highlights | Sat | Hue | Luma Mix
        // ═════════════════════════════════════════════════════════════════════
        const botBar = document.createElement('div');
        botBar.style.cssText = 'display: grid; grid-template-columns: repeat(6, 1fr); gap: 2px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px; justify-items: center;';

        // Color Boost
        botBar.appendChild(createMini('C.BOOST', 0.0, 2.0, this.colorBoost || 0.0, 0.05, v => {
            this.colorBoost = v;
            if (this.renderer) this.renderer.setColorBoost(v);
            this.render();
        }));

        // Shadows
        botBar.appendChild(createMini('SHADOWS', -1.0, 1.0, this.shadows || 0.0, 0.05, v => {
            this.shadows = v;
            if (this.renderer) this.renderer.setShadows(v);
            this.render();
        }));

        // Highlights
        botBar.appendChild(createMini('HILIGHT', -1.0, 1.0, this.highlights || 0.0, 0.05, v => {
            this.highlights = v;
            if (this.renderer) this.renderer.setHighlights(v);
            this.render();
        }));

        // Saturation
        botBar.appendChild(createMini('SAT', 0.0, 3.0, this.saturation || 1.0, 0.05, v => {
            this.saturation = v;
            if (this.renderer) this.renderer.setSaturation(v);
            this.render();
        }));

        // Hue
        botBar.appendChild(createMini('HUE', 0.0, 360.0, this.hueShift || 0.0, 1.0, v => {
            this.hueShift = v;
            if (this.renderer) this.renderer.setHueShift(v);
            this.render();
        }));

        // Luma Mix
        botBar.appendChild(createMini('LUMA MIX', 0.0, 1.0, this.lumaMix !== undefined ? this.lumaMix : 1.0, 0.05, v => {
            this.lumaMix = v;
            if (this.renderer) this.renderer.setLumaMix(v);
            this.render();
        }));

        container.appendChild(botBar);
    }


    renderEffectsTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; gap: 10px; padding: 10px; max-height: 280px; overflow-y: auto;';

        // ─── Grain Knobs Row ──────────────────────────────
        const grid = document.createElement('div');
        grid.style.cssText = 'display: flex; gap: 16px; justify-content: flex-start;';

        // Grain Amount
        grid.appendChild(this.createKnob('GRAIN', 0.0, 1.0, this.grain || 0.0, 0.05, v => {
            this.grain = v;
            if (this.renderer) this.renderer.setGrain(v);
            this.render();
        }));

        // Grain Size
        grid.appendChild(this.createKnob('SIZE', 1.0, 4.0, this.grainSize || 1.0, 0.1, v => {
            this.grainSize = v;
            if (this.renderer) this.renderer.setGrainSize(v);
            this.render();
        }));

        // Color Grain
        grid.appendChild(this.createKnob('COLOR', 0.0, 1.0, this.grainColor || 0.0, 0.05, v => {
            this.grainColor = v;
            if (this.renderer) this.renderer.setGrainColor(v);
            this.render();
        }));

        // Denoise
        grid.appendChild(this.createKnob('DENOISE', 0.0, 1.0, this.denoise || 0.0, 0.05, v => {
            this.denoise = v;
            if (this.renderer) this.renderer.setDenoise(v);
            this.render();
        }));

        container.appendChild(grid);

        // ─── Film Stock Presets ───────────────────────────
        const presetLabel = document.createElement('div');
        presetLabel.style.cssText = 'color: #888; font-size: 10px; text-transform: uppercase; margin-top: 6px;';
        presetLabel.textContent = 'Film Stock Presets';
        container.appendChild(presetLabel);

        const presets = document.createElement('div');
        presets.style.cssText = 'display: flex; flex-wrap: wrap; gap: 6px;';

        const btnStyle = `background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #aaa; padding: 4px 8px; border-radius: 4px; font-size: 10px; cursor: pointer; transition: background 0.15s;`;

        const addPreset = (lbl, gVal, sVal, cVal) => {
            const b = document.createElement('div');
            b.textContent = lbl;
            b.style.cssText = btnStyle;
            b.onmouseenter = () => b.style.background = 'rgba(255,255,255,0.12)';
            b.onmouseleave = () => b.style.background = 'rgba(255,255,255,0.05)';
            b.onclick = () => {
                this.grain = gVal;
                this.grainSize = sVal;
                this.grainColor = cVal;
                if (this.renderer) {
                    this.renderer.setGrain(gVal);
                    this.renderer.setGrainSize(sVal);
                    this.renderer.setGrainColor(cVal);
                }
                this.render();
                this._lastRenderContent();
            };
            presets.appendChild(b);
        };

        //                    Label              Grain  Size   Color
        addPreset('No Grain', 0.00, 1.0, 0.0);
        addPreset('Super 8', 0.50, 3.5, 0.4);
        addPreset('16mm', 0.35, 2.5, 0.3);
        addPreset('Super 16', 0.28, 2.0, 0.25);
        addPreset('35mm', 0.15, 1.5, 0.15);
        addPreset('65mm / IMAX', 0.06, 1.0, 0.05);
        addPreset('Kodak 500T', 0.12, 1.3, 0.1);
        addPreset('Kodak 50D', 0.05, 1.0, 0.05);
        addPreset('Fuji Eterna', 0.10, 1.2, 0.08);
        addPreset('CineStill 800T', 0.18, 1.8, 0.2);
        addPreset('Tri-X 400', 0.25, 2.0, 0.0);
        addPreset('Digital Noise', 0.08, 1.0, 0.0);

        container.appendChild(presets);
    }

    renderLensTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; gap: 10px; padding: 10px; max-height: 340px; overflow-y: auto;';

        // 1. Focus & DoF Top Row
        const topRow = document.createElement('div');
        topRow.style.cssText = 'display: flex; gap: 10px; align-items: center; margin-bottom: 5px;';

        const dofCheck = document.createElement('div');
        dofCheck.innerHTML = `
            <input type="checkbox" id="dof-enable" ${this.dofEnabled ? 'checked' : ''}>
            <label for="dof-enable" style="color: #ccc; font-size: 11px; margin-left: 4px;">Enable DoF</label>
        `;
        dofCheck.querySelector('input').onchange = (e) => {
            this.dofEnabled = e.target.checked;
            if (this.renderer) this.renderer.setDoFEnabled(this.dofEnabled);
            this.render();
        };
        topRow.appendChild(dofCheck);
        container.appendChild(topRow);

        // 2. Main Knobs Row (Focus, Aperture, Distortion)
        const knobsRow = document.createElement('div');
        knobsRow.style.cssText = 'display: flex; gap: 16px; justify-content: space-around;';

        knobsRow.appendChild(this.createKnob('FOCUS', 0.0, 1.0, this.focusDistance || 0.5, 0.01, v => {
            this.focusDistance = v;
            if (this.renderer) this.renderer.setFocusDistance(v);
            this.render();
        }));

        knobsRow.appendChild(this.createKnob('SIZE', 0.0, 1.0, this.aperture || 0.0, 0.01, v => {
            this.aperture = v;
            if (this.renderer) this.renderer.setAperture(v);
            this.render();
        }));

        knobsRow.appendChild(this.createKnob('DISTORT', -0.5, 0.5, this.lensDistortion || 0.0, 0.01, v => {
            this.lensDistortion = v;
            if (this.renderer) this.renderer.setLensDistortion(v, this.lensFringe || 0.0);
            this.render();
        }));

        container.appendChild(knobsRow);

        // 3. Bokeh Shape Presets
        const shapeGroup = document.createElement('div');
        shapeGroup.style.marginTop = '10px';
        shapeGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 6px; text-transform: uppercase;">Bokeh Shape</div>';

        // ─── Preset Buttons Row ──────────────────────────
        const presetRow = document.createElement('div');
        presetRow.style.cssText = 'display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap;';

        const presetBtnStyle = `background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #aaa; padding: 4px 10px; border-radius: 4px; font-size: 10px; cursor: pointer; transition: background 0.15s;`;
        const presetBtnActiveStyle = `background: rgba(100,140,255,0.2); border: 1px solid rgba(100,140,255,0.4); color: #9cf;`;

        const bokehPresets = [
            { label: '● Circle', blades: 0, angle: 0 },
            { label: '⬠ Pentagon', blades: 5, angle: 0 },
            { label: '⬡ Hexagon', blades: 6, angle: 0 },
            { label: '⬡ Heptagon', blades: 7, angle: 0 },
            { label: '⬡ Octagon', blades: 8, angle: 0 },
        ];

        const currentBlades = this.apertureBlades || 0;
        bokehPresets.forEach(p => {
            const btn = document.createElement('div');
            btn.textContent = p.label;
            const isActive = currentBlades === p.blades;
            btn.style.cssText = presetBtnStyle + (isActive ? presetBtnActiveStyle : '');
            btn.onmouseenter = () => { if (!isActive) btn.style.background = 'rgba(255,255,255,0.12)'; };
            btn.onmouseleave = () => { if (!isActive) btn.style.background = 'rgba(255,255,255,0.05)'; };
            btn.onclick = () => {
                this.apertureBlades = p.blades;
                this.apertureRotation = p.angle;
                if (this.renderer) this.renderer.setApertureShape(p.blades, p.angle, this.apertureAnamorphic || 1.0);
                this.render();
                this._lastRenderContent();
            };
            presetRow.appendChild(btn);
        });

        shapeGroup.appendChild(presetRow);

        // ─── Manual Shape Knobs ──────────────────────────
        const shapeGrid = document.createElement('div');
        shapeGrid.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;';

        shapeGrid.appendChild(this.createKnob('BLADES', 0, 9, this.apertureBlades || 0, 1, v => {
            this.apertureBlades = Math.round(v);
            if (this.renderer) this.renderer.setApertureShape(this.apertureBlades, this.apertureRotation || 0, this.apertureAnamorphic || 1.0);
            this.render();
        }));

        shapeGrid.appendChild(this.createKnob('ANGLE', 0, 360, this.apertureRotation || 0, 1, v => {
            this.apertureRotation = v;
            if (this.renderer) this.renderer.setApertureShape(this.apertureBlades || 0, this.apertureRotation, this.apertureAnamorphic || 1.0);
            this.render();
        }));

        shapeGrid.appendChild(this.createKnob('RATIO', 1.0, 2.0, this.apertureAnamorphic || 1.0, 0.05, v => {
            this.apertureAnamorphic = v;
            if (this.renderer) this.renderer.setApertureShape(this.apertureBlades || 0, this.apertureRotation || 0, this.apertureAnamorphic);
            this.render();
        }));

        shapeGroup.appendChild(shapeGrid);
        container.appendChild(shapeGroup);

        // 4. Optical Filters
        const filterGroup = document.createElement('div');
        filterGroup.style.marginTop = '10px';
        filterGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 4px; text-transform: uppercase;">Optical Filters</div>';

        const filterGrid = document.createElement('div');
        filterGrid.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;';

        filterGrid.appendChild(this.createKnob('FRINGE', 0.0, 2.0, this.lensFringe || 0.0, 0.05, v => {
            this.lensFringe = v;
            if (this.renderer) this.renderer.setLensDistortion(this.lensDistortion || 0.0, v);
            this.render();
        }));

        filterGrid.appendChild(this.createKnob('VIG INT', 0.0, 1.0, this.vignetteIntensity || 0.0, 0.01, v => {
            this.vignetteIntensity = v;
            if (this.renderer) this.renderer.setVignette(v, this.vignetteFalloff || 0.5);
            this.render();
        }));

        filterGrid.appendChild(this.createKnob('VIG FALL', 0.1, 1.0, this.vignetteFalloff || 0.5, 0.05, v => {
            this.vignetteFalloff = v;
            if (this.renderer) this.renderer.setVignette(this.vignetteIntensity || 0.0, v);
            this.render();
        }));

        filterGroup.appendChild(filterGrid);
        container.appendChild(filterGroup);

        // 5. Lens Effects (Bloom, Halation, Diffusion)
        const fxGroup = document.createElement('div');
        fxGroup.style.marginTop = '10px';
        fxGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 4px; text-transform: uppercase;">Lens Effects</div>';

        const fxGrid = document.createElement('div');
        fxGrid.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;';

        fxGrid.appendChild(this.createKnob('BLOOM', 0.0, 1.0, this.bloom || 0.0, 0.01, v => {
            this.bloom = v;
            if (this.renderer) this.renderer.setBloom(v);
            this.render();
        }));

        fxGrid.appendChild(this.createKnob('HALATION', 0.0, 1.0, this.halation || 0.0, 0.01, v => {
            this.halation = v;
            if (this.renderer) this.renderer.setHalation(v);
            this.render();
        }));

        fxGrid.appendChild(this.createKnob('DIFFUSION', 0.0, 1.0, this.diffusion || 0.0, 0.01, v => {
            this.diffusion = v;
            if (this.renderer) this.renderer.setDiffusion(v);
            this.render();
        }));

        fxGroup.appendChild(fxGrid);
        container.appendChild(fxGroup);
    }

    renderCurvesTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 10px; max-height: 400px; overflow-y: auto;';

        // 1. Create Editor Container
        const editorContainer = document.createElement('div');
        editorContainer.style.cssText = 'position: relative; width: 280px; height: 280px; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); border-radius: 4px;';
        container.appendChild(editorContainer);

        // 2. Initialize Curve Editor
        if (!this.curveEditor) {
            this.curveEditor = new RadianceCurveEditor(280, 280, (data) => {
                if (this.renderer) {
                    this.renderer.updateCurveLut(data);
                    this.renderer.setCurveMix(this.curveMix !== undefined ? this.curveMix : 1.0);
                    this.render();
                }
            });
            if (this.image) this.curveEditor.updateHistogram(this.image);
        }

        editorContainer.appendChild(this.curveEditor.canvas);

        // 3. Channel Selectors (top-left overlay)
        const channels = document.createElement('div');
        channels.style.cssText = 'position: absolute; top: 10px; left: 36px; display: flex; gap: 3px;';

        ['RGB', 'R', 'G', 'B'].forEach(ch => {
            const btn = document.createElement('div');
            btn.textContent = ch;
            const isActive = this.curveEditor.activeChannel === ch;
            const color = ch === 'R' ? '#ff5555' : ch === 'G' ? '#55ff55' : ch === 'B' ? '#5555ff' : '#ffffff';

            btn.style.cssText = `
                width: 28px; height: 20px;
                background: ${isActive ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.5)'};
                color: ${color};
                border: 1px solid ${isActive ? color : 'rgba(255,255,255,0.1)'};
                border-radius: 3px;
                display: flex; align-items: center; justify-content: center;
                font-size: 10px; font-weight: bold; cursor: pointer;
                transition: background 0.15s;
            `;
            btn.onmouseenter = () => { if (!isActive) btn.style.background = 'rgba(255,255,255,0.1)'; };
            btn.onmouseleave = () => { if (!isActive) btn.style.background = 'rgba(0,0,0,0.5)'; };
            btn.onclick = () => {
                this.curveEditor.setActiveChannel(ch);
                this._lastRenderContent();
            };
            channels.appendChild(btn);
        });
        editorContainer.appendChild(channels);

        // 4. Reset Buttons (top-right overlay)
        const resetGroup = document.createElement('div');
        resetGroup.style.cssText = 'position: absolute; top: 10px; right: 10px; display: flex; gap: 4px;';

        const resetChBtn = document.createElement('div');
        resetChBtn.textContent = '↺';
        resetChBtn.title = 'Reset Channel';
        resetChBtn.style.cssText = 'color: #888; cursor: pointer; font-size: 13px; padding: 2px 4px; border-radius: 3px; transition: color 0.15s;';
        resetChBtn.onmouseenter = () => resetChBtn.style.color = '#fff';
        resetChBtn.onmouseleave = () => resetChBtn.style.color = '#888';
        resetChBtn.onclick = () => { this.curveEditor.resetActiveChannel(); };
        resetGroup.appendChild(resetChBtn);

        const resetAllBtn = document.createElement('div');
        resetAllBtn.textContent = '⟲';
        resetAllBtn.title = 'Reset All Channels';
        resetAllBtn.style.cssText = 'color: #888; cursor: pointer; font-size: 13px; padding: 2px 4px; border-radius: 3px; transition: color 0.15s;';
        resetAllBtn.onmouseenter = () => resetAllBtn.style.color = '#f55';
        resetAllBtn.onmouseleave = () => resetAllBtn.style.color = '#888';
        resetAllBtn.onclick = () => {
            this.curveEditor.resetAll();
            this.curveMix = 1.0;
            if (this.renderer) this.renderer.setCurveMix(1.0);
            this._lastRenderContent();
        };
        resetGroup.appendChild(resetAllBtn);

        editorContainer.appendChild(resetGroup);

        // 5. Mix Slider
        const mixRow = document.createElement('div');
        mixRow.style.cssText = 'display: flex; align-items: center; gap: 8px; width: 280px;';

        const mixLabel = document.createElement('div');
        mixLabel.style.cssText = 'color: #888; font-size: 10px; text-transform: uppercase; min-width: 28px;';
        mixLabel.textContent = 'MIX';
        mixRow.appendChild(mixLabel);

        const mixSlider = document.createElement('input');
        mixSlider.type = 'range';
        mixSlider.min = '0'; mixSlider.max = '100'; mixSlider.step = '1';
        mixSlider.value = String(Math.round((this.curveMix !== undefined ? this.curveMix : 1.0) * 100));
        mixSlider.style.cssText = 'flex: 1; accent-color: #6a8aff; height: 4px; cursor: pointer;';
        mixSlider.oninput = (e) => {
            this.curveMix = parseInt(e.target.value) / 100;
            if (this.renderer) this.renderer.setCurveMix(this.curveMix);
            mixValue.textContent = e.target.value + '%';
            this.render();
        };
        mixRow.appendChild(mixSlider);

        const mixValue = document.createElement('div');
        mixValue.style.cssText = 'color: #aaa; font-size: 10px; min-width: 32px; text-align: right;';
        mixValue.textContent = Math.round((this.curveMix !== undefined ? this.curveMix : 1.0) * 100) + '%';
        mixRow.appendChild(mixValue);

        container.appendChild(mixRow);

        // 6. Curve Presets
        const presetLabel = document.createElement('div');
        presetLabel.style.cssText = 'color: #888; font-size: 10px; text-transform: uppercase; width: 280px;';
        presetLabel.textContent = 'Presets';
        container.appendChild(presetLabel);

        const presetRow = document.createElement('div');
        presetRow.style.cssText = 'display: flex; flex-wrap: wrap; gap: 5px; width: 280px;';

        const btnStyle = `background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #aaa; padding: 3px 8px; border-radius: 4px; font-size: 10px; cursor: pointer; transition: background 0.15s;`;

        const presets = [
            { label: 'Linear', data: { RGB: [{ x: 0, y: 0 }, { x: 1, y: 1 }] } },
            { label: 'S-Curve Soft', data: { RGB: [{ x: 0, y: 0 }, { x: 0.25, y: 0.20 }, { x: 0.75, y: 0.80 }, { x: 1, y: 1 }] } },
            { label: 'S-Curve Hard', data: { RGB: [{ x: 0, y: 0 }, { x: 0.20, y: 0.10 }, { x: 0.80, y: 0.90 }, { x: 1, y: 1 }] } },
            { label: 'Lift Shadows', data: { RGB: [{ x: 0, y: 0.08 }, { x: 0.5, y: 0.52 }, { x: 1, y: 1 }] } },
            { label: 'Crush Blacks', data: { RGB: [{ x: 0, y: 0 }, { x: 0.15, y: 0.02 }, { x: 0.5, y: 0.45 }, { x: 1, y: 1 }] } },
            { label: 'High Key', data: { RGB: [{ x: 0, y: 0.05 }, { x: 0.4, y: 0.55 }, { x: 1, y: 1 }] } },
            { label: 'Low Key', data: { RGB: [{ x: 0, y: 0 }, { x: 0.6, y: 0.40 }, { x: 1, y: 0.90 }] } },
            {
                label: 'Cross Process', data: {
                    RGB: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    R: [{ x: 0, y: 0.05 }, { x: 0.4, y: 0.50 }, { x: 1, y: 0.95 }],
                    G: [{ x: 0, y: 0 }, { x: 0.5, y: 0.45 }, { x: 1, y: 1 }],
                    B: [{ x: 0, y: 0.10 }, { x: 0.6, y: 0.55 }, { x: 1, y: 0.85 }]
                }
            },
        ];

        presets.forEach(p => {
            const btn = document.createElement('div');
            btn.textContent = p.label;
            btn.style.cssText = btnStyle;
            btn.onmouseenter = () => btn.style.background = 'rgba(255,255,255,0.12)';
            btn.onmouseleave = () => btn.style.background = 'rgba(255,255,255,0.05)';
            btn.onclick = () => {
                this.curveEditor.applyPreset(p.data);
                this.curveMix = 1.0;
                if (this.renderer) this.renderer.setCurveMix(1.0);
                this._lastRenderContent();
            };
            presetRow.appendChild(btn);
        });

        container.appendChild(presetRow);
    }

    renderQualifiersTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; gap: 10px; padding: 10px; max-height: 280px; overflow-y: auto;';

        // Initialize state if missing
        if (!this.qualifierState) {
            this.qualifierState = {
                enabled: false,
                showMask: false,
                h: 0.0, hW: 0.1, hS: 0.05,
                s: 0.5, sW: 0.5, sS: 0.1,
                l: 0.5, lW: 0.5, lS: 0.1
            };
        }

        const update = () => {
            if (this.renderer) {
                this.renderer.setQualifier(this.qualifierState);
                this.render();
            }
        };

        // 1. Top Controls (Enable, Show Mask, Eyedropper)
        const topRow = document.createElement('div');
        topRow.style.cssText = 'display: flex; gap: 10px; align-items: center; justify-content: space-between; margin-bottom: 5px;';

        // Enable Toggle
        const enableCheck = document.createElement('div');
        enableCheck.innerHTML = `
            <input type="checkbox" id="qual-enable" ${this.qualifierState.enabled ? 'checked' : ''}>
            <label for="qual-enable" style="color: #ccc; font-size: 11px; margin-left: 4px;">Active</label>
        `;
        enableCheck.querySelector('input').onchange = (e) => {
            this.qualifierState.enabled = e.target.checked;
            update();
        };
        topRow.appendChild(enableCheck);

        // Show Mask
        const maskCheck = document.createElement('div');
        maskCheck.innerHTML = `
            <input type="checkbox" id="qual-mask" ${this.qualifierState.showMask ? 'checked' : ''}>
            <label for="qual-mask" style="color: #ccc; font-size: 11px; margin-left: 4px;">Show Mask</label>
        `;
        maskCheck.querySelector('input').onchange = (e) => {
            this.qualifierState.showMask = e.target.checked;
            update();
        };
        topRow.appendChild(maskCheck);

        // Eyedropper
        const pickerBtn = document.createElement('button');
        pickerBtn.textContent = '🖌 Pick';
        pickerBtn.style.cssText = 'background: #333; color: #ccc; border: 1px solid #555; padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 11px;';
        pickerBtn.onclick = () => this.activateEyedropper(update);
        topRow.appendChild(pickerBtn);

        container.appendChild(topRow);

        // Helper to create knob row
        const createRow = (label, param, labelColor) => {
            const row = document.createElement('div');
            row.style.cssText = 'display: flex; gap: 8px; align-items: center; background: rgba(0,0,0,0.2); padding: 4px; border-radius: 4px;';

            const title = document.createElement('div');
            title.textContent = label;
            title.style.cssText = `color: ${labelColor}; font-size: 10px; width: 30px; font-weight: bold;`;
            row.appendChild(title);

            // Center
            row.appendChild(this.createKnob('Center', 0, 1, this.qualifierState[param], 0.01, (v) => {
                this.qualifierState[param] = v;
                update();
            }));

            // Width
            row.appendChild(this.createKnob('Width', 0, 1, this.qualifierState[param + 'W'], 0.01, (v) => {
                this.qualifierState[param + 'W'] = v;
                update();
            }));

            // Soft
            row.appendChild(this.createKnob('Soft', 0, 0.5, this.qualifierState[param + 'S'], 0.01, (v) => {
                this.qualifierState[param + 'S'] = v;
                update();
            }));

            return row;
        };

        container.appendChild(createRow('HUE', 'h', '#f55'));
        container.appendChild(createRow('SAT', 's', '#5f5'));
        container.appendChild(createRow('LUM', 'l', '#aaa'));
    }

    activateEyedropper(callback) {
        if (this.isPicking) return;
        this.isPicking = true;

        // Show overlay helper
        const overlay = document.createElement('div');
        overlay.textContent = 'Click to Pick Color';
        overlay.style.cssText = 'position: absolute; top: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); color: #fff; padding: 5px 10px; border-radius: 4px; pointer-events: none; z-index: 200;';
        this.container.appendChild(overlay);

        this.container.style.cursor = 'crosshair';

        const clickHandler = (e) => {
            // Get click coordinate relative to image
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Map to image coordinates
            // Need inverse fitToView transform... 
            // Simplified: if we click on canvas, we read displayed pixel? 
            // Or read from source data? Source data is better.

            // Calculate UV
            // Pan/Zoom aware mapping
            // x_img_scaled = (uv.x * width * zoom) + pan.x
            const imgX = (x - this.panX) / this.zoom;
            const imgY = (y - this.panY) / this.zoom;

            if (imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
                // Determine color
                let r = 0, g = 0, b = 0;

                if (this.hdrData) {
                    // Float sample
                    const ix = Math.floor(imgX);
                    const iy = Math.floor(imgY);
                    const idx = (iy * this.imageWidth + ix) * this.hdrData.channels;
                    const d = this.hdrData.fp16data ? this.hdrData.data : this.hdrData.data; // both are float32 in js wrapper usually
                    // Wait, .data is Float32Array
                    r = d[idx];
                    g = d[idx + 1];
                    b = d[idx + 2];
                } else if (this.imageData) {
                    // 8-bit sample from canvas/image? 
                    // We can't easily read image.data without context usually.
                    // But we created a temp context in Scopes...
                    // Let's assume we can read from renderer's buffer? 
                    // Or just use the temp canvas approach if needed. 
                    // For now, if no HDR, maybe skip or use approximation.
                    // Actually, if we have Image object, we can draw to canvas.
                }

                // Convert to HSL
                // r,g,b are linear or sRGB? HDR data is likely linear? 
                // Let's assume linear.

                // RGB to HSL logic (JS version)
                const max = Math.max(r, g, b);
                const min = Math.min(r, g, b);
                let h, s, l = (max + min) / 2;

                if (max === min) {
                    h = s = 0; // achromatic
                } else {
                    const d = max - min;
                    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
                    switch (max) {
                        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                        case g: h = (b - r) / d + 2; break;
                        case b: h = (r - g) / d + 4; break;
                    }
                    h /= 6;
                }

                // Set Qualifiers
                this.qualifierState.h = h;
                this.qualifierState.s = s;
                this.qualifierState.l = l;
                this.qualifierState.enabled = true;

                // Refresh UI sliders? They need to re-render or update value
                // Since createKnob sets initial value, re-rendering tab is easiest
                // But we passed 'update' callback which just renders WebGL.
                // We need to re-render controls.
                this.renderQualifiersTab(this.tabContentContainer);

                if (this.renderer) {
                    this.renderer.setQualifier(this.qualifierState);
                    this.render();
                }
            }

            // Cleanup
            this.container.style.cursor = 'default';
            overlay.remove();
            this.container.removeEventListener('click', clickHandler);
            this.isPicking = false;
        };

        this.container.addEventListener('click', clickHandler);
    }

    renderScopesTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 8px;';

        // ─── Mode Selector Bar ─────────────────────────────
        const modeBar = document.createElement('div');
        modeBar.style.cssText = 'display: flex; gap: 3px; width: 280px;';

        const modes = [
            { id: 'parade', label: 'Parade' },
            { id: 'waveform', label: 'Waveform' },
            { id: 'histogram', label: 'Histogram' },
            { id: 'vectorscope', label: 'Vector' },
            { id: 'falsecolor', label: 'False Color' },
        ];

        modes.forEach(m => {
            const btn = document.createElement('div');
            btn.textContent = m.label;
            const isActive = this.scopeMode === m.id;
            btn.style.cssText = `
                flex: 1; text-align: center; padding: 3px 0;
                background: ${isActive ? 'rgba(106,138,255,0.2)' : 'rgba(255,255,255,0.04)'};
                color: ${isActive ? '#8aafff' : '#777'};
                border: 1px solid ${isActive ? 'rgba(106,138,255,0.4)' : 'rgba(255,255,255,0.08)'};
                border-radius: 3px; font-size: 9px; cursor: pointer;
                transition: background 0.15s;
            `;
            btn.onmouseenter = () => { if (!isActive) btn.style.background = 'rgba(255,255,255,0.08)'; };
            btn.onmouseleave = () => { if (!isActive) btn.style.background = 'rgba(255,255,255,0.04)'; };
            btn.onclick = () => {
                this.scopeMode = m.id;
                localStorage.setItem('radiance_scope_mode', m.id);
                this._lastRenderContent();
            };
            modeBar.appendChild(btn);
        });
        container.appendChild(modeBar);

        // ─── Scope Canvas ──────────────────────────────────
        const isSquare = this.scopeMode === 'vectorscope';
        const cW = 280, cH = isSquare ? 240 : 180;

        const canvas = document.createElement('canvas');
        canvas.width = cW;
        canvas.height = cH;
        canvas.style.cssText = `background: #050508; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; width: 280px; height: ${cH}px;`;
        container.appendChild(canvas);

        // ─── Extract Pixel Data ─────────────────────────────
        if (!this.image) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#444'; ctx.font = '10px monospace';
            ctx.fillText('No Image', cW / 2 - 25, cH / 2);
            return;
        }

        const sampleW = 280;
        const sampleH = Math.round(this.image.height * (sampleW / this.image.width));
        const tmp = document.createElement('canvas');
        tmp.width = sampleW; tmp.height = sampleH;
        const tctx = tmp.getContext('2d');
        tctx.drawImage(this.image, 0, 0, sampleW, sampleH);
        const imgData = tctx.getImageData(0, 0, sampleW, sampleH);
        const pixels = imgData.data;

        const ctx = canvas.getContext('2d');

        // ─── Render Based on Mode ───────────────────────────
        switch (this.scopeMode) {
            case 'parade': this._drawScopeParade(ctx, pixels, sampleW, sampleH, cW, cH); break;
            case 'waveform': this._drawScopeWaveform(ctx, pixels, sampleW, sampleH, cW, cH); break;
            case 'histogram': this._drawScopeHistogram(ctx, pixels, cW, cH); break;
            case 'vectorscope': this._drawScopeVectorscope(ctx, pixels, cW, cH); break;
            case 'falsecolor': this._drawScopeFalseColor(ctx, pixels, sampleW, sampleH, cW, cH); break;
        }
    }

    // ─── RGB Parade ──────────────────────────────────────────
    _drawScopeParade(ctx, data, imgW, imgH, w, h) {
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, w, h);

        const secW = Math.floor(w / 3);
        const channels = [
            { idx: 0, color: 'rgba(255,80,80,', label: 'R', x: 0 },
            { idx: 1, color: 'rgba(80,255,80,', label: 'G', x: secW },
            { idx: 2, color: 'rgba(80,120,255,', label: 'B', x: secW * 2 }
        ];

        const step = Math.max(1, Math.floor(imgW / secW));

        channels.forEach(ch => {
            // Plot dots
            ctx.globalAlpha = 1.0;
            for (let col = 0; col < imgW; col += step) {
                const x = ch.x + Math.floor((col / imgW) * secW);
                const hist = new Uint32Array(256);
                for (let row = 0; row < imgH; row++) {
                    hist[data[(row * imgW + col) * 4 + ch.idx]]++;
                }
                for (let v = 0; v < 256; v++) {
                    if (hist[v] > 0) {
                        const intensity = Math.min(hist[v] / (imgH * 0.08), 1.0);
                        const alpha = intensity * 0.7 + 0.15;
                        ctx.fillStyle = ch.color + alpha + ')';
                        ctx.fillRect(x, h - (v / 255) * h, 1, 2);
                    }
                }
            }

            // Label
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = ch.color + '0.8)';
            ctx.font = '10px monospace';
            ctx.fillText(ch.label, ch.x + 4, 12);
        });

        // Separators
        ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(secW, 0); ctx.lineTo(secW, h);
        ctx.moveTo(secW * 2, 0); ctx.lineTo(secW * 2, h);
        ctx.stroke();

        // 50% line
        ctx.strokeStyle = '#333'; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(0, h * 0.5); ctx.lineTo(w, h * 0.5); ctx.stroke();
        ctx.setLineDash([]);
    }

    // ─── Luma Waveform ───────────────────────────────────────
    _drawScopeWaveform(ctx, data, imgW, imgH, w, h) {
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, w, h);

        // IRE guide lines
        ctx.strokeStyle = '#2a2a2a'; ctx.lineWidth = 1;
        ctx.font = '8px monospace'; ctx.fillStyle = '#444';
        [0, 25, 50, 75, 100].forEach(ire => {
            const y = h - (ire / 100) * h;
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
            ctx.fillText(ire + '', 2, y - 2);
        });

        // Plot luma dots
        const step = Math.max(1, Math.floor(imgW / w));
        ctx.globalAlpha = 0.08;
        for (let col = 0; col < imgW; col += step) {
            const x = Math.floor((col / imgW) * w);
            for (let row = 0; row < imgH; row += 2) {
                const idx = (row * imgW + col) * 4;
                const luma = data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
                const y = h - (luma / 255) * h;
                const bright = Math.floor(40 + luma * 0.6);
                ctx.fillStyle = `rgb(${bright},${Math.floor(bright * 1.4)},${bright})`;
                ctx.fillRect(x, y, 1, 1);
            }
        }
        ctx.globalAlpha = 1.0;

        // Label
        ctx.fillStyle = '#5a5'; ctx.font = '10px monospace';
        ctx.fillText('LUMA', 4, 12);
    }

    // ─── Histogram ───────────────────────────────────────────
    _drawScopeHistogram(ctx, data, w, h) {
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, w, h);

        const hR = new Uint32Array(256), hG = new Uint32Array(256), hB = new Uint32Array(256);
        for (let i = 0; i < data.length; i += 4) {
            hR[data[i]]++;
            hG[data[i + 1]]++;
            hB[data[i + 2]]++;
        }

        let max = 1;
        for (let i = 0; i < 256; i++) max = Math.max(max, hR[i], hG[i], hB[i]);

        // Grid
        ctx.strokeStyle = '#222'; ctx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
            const x = (i / 4) * w;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }

        const drawCurve = (hist, color) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const x = (i / 255) * w;
                const y = h - (hist[i] / max) * h * 0.95;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        };

        // Fill under curves
        const fillCurve = (hist, color) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(0, h);
            for (let i = 0; i < 256; i++) {
                const x = (i / 255) * w;
                const y = h - (hist[i] / max) * h * 0.95;
                ctx.lineTo(x, y);
            }
            ctx.lineTo(w, h);
            ctx.fill();
        };

        ctx.globalAlpha = 0.15;
        fillCurve(hR, '#ff4444');
        fillCurve(hG, '#44ff44');
        fillCurve(hB, '#4488ff');
        ctx.globalAlpha = 0.8;
        drawCurve(hR, '#ff4444');
        drawCurve(hG, '#44ff44');
        drawCurve(hB, '#4488ff');
        ctx.globalAlpha = 1.0;

        // Labels
        ctx.fillStyle = '#666'; ctx.font = '8px monospace';
        ctx.fillText('0', 2, h - 3);
        ctx.fillText('255', w - 20, h - 3);
    }

    // ─── Vectorscope ─────────────────────────────────────────
    _drawScopeVectorscope(ctx, data, w, h) {
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, w, h);

        const cx = w / 2, cy = h / 2;
        const rad = Math.min(cx, cy) - 10;

        // Graticule rings
        ctx.strokeStyle = '#1a1a1a'; ctx.lineWidth = 1;
        [0.25, 0.5, 0.75, 1.0].forEach(r => {
            ctx.beginPath(); ctx.arc(cx, cy, rad * r, 0, Math.PI * 2); ctx.stroke();
        });

        // Crosshair
        ctx.strokeStyle = '#1a1a1a';
        ctx.beginPath(); ctx.moveTo(cx, cy - rad); ctx.lineTo(cx, cy + rad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx - rad, cy); ctx.lineTo(cx + rad, cy); ctx.stroke();

        // Rec.709 color targets
        const targets = [
            { a: 103, c: '#f33', l: 'R' },
            { a: 167, c: '#ff0', l: 'Yl' },
            { a: 241, c: '#0f0', l: 'G' },
            { a: 283, c: '#0ff', l: 'Cy' },
            { a: 347, c: '#33f', l: 'B' },
            { a: 61, c: '#f0f', l: 'Mg' },
        ];
        targets.forEach(t => {
            const ang = (t.a - 90) * Math.PI / 180;
            const tx = cx + Math.cos(ang) * rad * 0.75;
            const ty = cy + Math.sin(ang) * rad * 0.75;
            ctx.fillStyle = t.c;
            ctx.beginPath(); ctx.arc(tx, ty, 3, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = '#555'; ctx.font = '8px monospace';
            ctx.fillText(t.l, tx + 5, ty + 3);
        });

        // Plot pixels
        ctx.globalAlpha = 0.04;
        const step = Math.max(1, Math.floor(data.length / 4 / 25000));
        for (let i = 0; i < data.length; i += 4 * step) {
            const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
            const y = r * 0.2126 + g * 0.7152 + b * 0.0722;
            const u = (b - y) * 0.492;
            const v = (r - y) * 0.877;
            ctx.fillStyle = `rgb(${data[i]},${data[i + 1]},${data[i + 2]})`;
            ctx.fillRect(cx + u * rad * 2.2, cy - v * rad * 2.2, 1, 1);
        }
        ctx.globalAlpha = 1.0;
    }

    // ─── False Color ─────────────────────────────────────────
    _drawScopeFalseColor(ctx, data, imgW, imgH, w, h) {
        // Create false color image from luminance
        const outImg = ctx.createImageData(w, h);
        const out = outImg.data;

        // Scale factors
        const sx = imgW / w, sy = imgH / h;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const srcX = Math.floor(x * sx), srcY = Math.floor(y * sy);
                const idx = (srcY * imgW + srcX) * 4;
                const luma = data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
                const ire = luma / 255; // 0..1

                let r, g, b;
                if (ire < 0.02) {
                    // Under black — purple
                    r = 80; g = 0; b = 120;
                } else if (ire < 0.10) {
                    // Deep shadows — blue
                    r = 20; g = 40; b = 180;
                } else if (ire < 0.25) {
                    // Shadows — cyan
                    r = 0; g = 140; b = 180;
                } else if (ire < 0.40) {
                    // Low mid — teal
                    r = 0; g = 160; b = 100;
                } else if (ire < 0.55) {
                    // Mid — green (proper exposure)
                    r = 40; g = 180; b = 40;
                } else if (ire < 0.68) {
                    // Upper mid — yellow-green
                    r = 160; g = 180; b = 0;
                } else if (ire < 0.80) {
                    // Highlights — yellow
                    r = 220; g = 200; b = 0;
                } else if (ire < 0.90) {
                    // Hot highlights — orange
                    r = 240; g = 120; b = 0;
                } else if (ire < 0.97) {
                    // Near clipping — red
                    r = 230; g = 30; b = 30;
                } else {
                    // Clipped — hot pink/white
                    r = 255; g = 50; b = 150;
                }

                const oi = (y * w + x) * 4;
                out[oi] = r; out[oi + 1] = g; out[oi + 2] = b; out[oi + 3] = 255;
            }
        }

        ctx.putImageData(outImg, 0, 0);

        // Legend
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(w - 72, 2, 70, 12);
        ctx.fillStyle = '#aaa'; ctx.font = '9px monospace';
        ctx.fillText('FALSE COLOR', w - 70, 11);

        // IRE scale bar
        const barX = w - 14, barH = h - 20, barY = 18;
        const ireColors = [
            [80, 0, 120], [20, 40, 180], [0, 140, 180], [0, 160, 100],
            [40, 180, 40], [160, 180, 0], [220, 200, 0], [240, 120, 0],
            [230, 30, 30], [255, 50, 150]
        ];
        const segH = barH / ireColors.length;
        ireColors.forEach((c, i) => {
            ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
            ctx.fillRect(barX, barY + (ireColors.length - 1 - i) * segH, 10, segH);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         UNDO / REDO STACK
    // ═══════════════════════════════════════════════════════════════════════════

    _captureGradingState() {
        return {
            exposure: this.exposure || 0.0,
            lift: this.lift ? [...this.lift] : [0, 0, 0],
            gamma: this.gamma ? (Array.isArray(this.gamma) ? [...this.gamma] : [this.gamma, this.gamma, this.gamma]) : [1, 1, 1],
            gain: this.gain ? [...this.gain] : [1, 1, 1],
            temperature: this.temperature || 0.0,
            tint: this.tint || 0.0,
            contrast: this.contrast || 1.0,
            pivot: this.pivot ?? 0.5,
            saturation: this.saturation || 1.0,
            grain: this.grain || 0.0,
            denoise: this.denoise || 0.0,
            lensDistortion: this.lensDistortion || 0.0,
            lensFringe: this.lensFringe || 0.0,
            vignetteIntensity: this.vignetteIntensity || 0.0,
            vignetteFalloff: this.vignetteFalloff ?? 0.5,
        };
    }

    _restoreGradingState(snapshot) {
        this.exposure = snapshot.exposure;
        this.lift = [...snapshot.lift];
        this.gamma = [...snapshot.gamma];
        this.gain = [...snapshot.gain];
        this.temperature = snapshot.temperature;
        this.tint = snapshot.tint;
        this.contrast = snapshot.contrast;
        this.pivot = snapshot.pivot;
        this.saturation = snapshot.saturation;
        this.grain = snapshot.grain;
        this.denoise = snapshot.denoise;
        this.lensDistortion = snapshot.lensDistortion;
        this.lensFringe = snapshot.lensFringe;
        this.vignetteIntensity = snapshot.vignetteIntensity;
        this.vignetteFalloff = snapshot.vignetteFalloff;

        if (this.renderer) {
            this.renderer.setExposure(this.exposure);
            this.renderer.setLift(this.lift[0], this.lift[1], this.lift[2]);
            this.renderer.setGamma(this.gamma[0], this.gamma[1], this.gamma[2]);
            this.renderer.setGain(this.gain[0], this.gain[1], this.gain[2]);
            this.renderer.setTemperature(this.temperature);
            this.renderer.setTint(this.tint);
            this.renderer.setContrast(this.contrast);
            this.renderer.setPivot(this.pivot);
            this.renderer.setSaturation(this.saturation);
            this.renderer.setGrain(this.grain);
            this.renderer.setDenoise(this.denoise);
            this.renderer.setLensDistortion(this.lensDistortion, this.lensFringe);
            this.renderer.setVignette(this.vignetteIntensity, this.vignetteFalloff);
        }
        this.render();
    }

    _pushUndo() {
        const state = this._captureGradingState();
        this._undoStack.push(state);
        if (this._undoStack.length > this._undoMaxSize) {
            this._undoStack.shift(); // Drop oldest
        }
        this._redoStack = []; // Clear redo on new action
    }

    undo() {
        if (this._undoStack.length === 0) return;
        // Save current state to redo before restoring
        this._redoStack.push(this._captureGradingState());
        const prev = this._undoStack.pop();
        this._restoreGradingState(prev);
        // Re-render the controls panel to update knob visuals
        if (this.controlsPanel && this._lastRenderContent) this._lastRenderContent();
    }

    redo() {
        if (this._redoStack.length === 0) return;
        // Save current state to undo before restoring
        this._undoStack.push(this._captureGradingState());
        const next = this._redoStack.pop();
        this._restoreGradingState(next);
        if (this.controlsPanel && this._lastRenderContent) this._lastRenderContent();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         KNOB CONTROL WIDGET
    // ═══════════════════════════════════════════════════════════════════════════

    createKnob(label, min, max, initial, step, callback) {
        const container = document.createElement('div');
        container.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 6px; width: 64px; position: relative;';

        // 1. Label
        const lbl = document.createElement('div');
        lbl.textContent = label;
        lbl.style.cssText = `color: ${this.theme.textDim}; font-size: 10px; font-weight: 500; font-family: ${this.theme.font}; letter-spacing: 0.5px; text-transform: uppercase;`;

        // 2. Knob SVG
        const size = 48;
        const strokeWidth = 3;
        const radius = (size - strokeWidth) / 2;
        const circumference = 2 * Math.PI * radius;

        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("width", size);
        svg.setAttribute("height", size);
        svg.style.cssText = "cursor: ns-resize; transform: rotate(-90deg); touch-action: none;";

        // Track (Background Ring)
        const track = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        track.setAttribute("cx", size / 2);
        track.setAttribute("cy", size / 2);
        track.setAttribute("r", radius);
        track.setAttribute("stroke", "rgba(255, 255, 255, 0.1)");
        track.setAttribute("stroke-width", strokeWidth);
        track.setAttribute("fill", "transparent");

        // Progress Arc
        const progress = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        progress.setAttribute("cx", size / 2);
        progress.setAttribute("cy", size / 2);
        progress.setAttribute("r", radius);
        progress.setAttribute("stroke", this.theme.accent);
        progress.setAttribute("stroke-width", strokeWidth);
        progress.setAttribute("fill", "transparent");
        progress.setAttribute("stroke-dasharray", circumference);
        progress.setAttribute("stroke-dashoffset", circumference);
        progress.style.transition = "stroke-dashoffset 0.05s linear";

        svg.appendChild(track);
        svg.appendChild(progress);

        // 3. Value Display
        const valDisplay = document.createElement('div');
        valDisplay.textContent = initial.toFixed(2);
        valDisplay.style.cssText = `
            font-family: ${this.theme.mono};
            font-size: 10px;
            color: ${this.theme.text};
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -20%);
            pointer-events: none;
            text-shadow: 0 1px 2px rgba(0,0,0,0.8);
        `;

        container.appendChild(lbl);
        container.appendChild(svg);
        container.appendChild(valDisplay);

        // Logic
        let currentValue = initial;
        let startY = 0;
        let startVal = 0;

        const updateVisuals = (val) => {
            // Normalized 0..1
            const t = (val - min) / (max - min);
            // Arc length (leave gap at bottom? No, full circle for now or 270deg is standard)
            // Let's do full circle for simplicity, or 270deg industry standard.
            // Industry standard: -135deg to +135deg (270deg range).
            // Current SVG implementation is full 360 ring. Let's keep 360 ring for "infinite" feel or 0-100% fill.

            const offset = circumference - (t * circumference);
            progress.setAttribute("stroke-dashoffset", offset);

            // Color change if non-default?
            progress.setAttribute("stroke", val === initial ? "rgba(255,255,255,0.3)" : this.theme.accent);
            lbl.style.color = val === initial ? this.theme.textDim : this.theme.accent;

            valDisplay.textContent = val.toFixed(2);
        };

        updateVisuals(initial);

        // Expose method to update externally (for presets)
        container.updateValue = (val) => {
            currentValue = val;
            updateVisuals(val);
        };

        // Interaction
        svg.onpointerdown = (e) => {
            e.preventDefault();
            // Capture undo state before this drag
            this._pushUndo();
            startY = e.clientY;
            startVal = currentValue;

            svg.setPointerCapture(e.pointerId);

            const onMove = (em) => {
                const delta = startY - em.clientY; // Standard drag delta
                // Sensitivity: Ctrl = fine (0.1x), Shift = coarse (3x)
                const range = max - min;
                let sensitivity = range * 0.005;
                if (em.ctrlKey || em.metaKey) sensitivity *= 0.1;  // Fine mode
                if (em.shiftKey) sensitivity *= 3.0;  // Coarse mode

                let newVal = startVal + (delta * sensitivity);
                newVal = Math.max(min, Math.min(newVal, max));

                // Snap to step
                if (step) newVal = Math.round(newVal / step) * step;

                currentValue = newVal;
                updateVisuals(currentValue);
                callback(currentValue);
            };

            const onUp = () => {
                svg.removeEventListener('pointermove', onMove);
                svg.removeEventListener('pointerup', onUp);
                svg.releasePointerCapture(e.pointerId);
            };

            svg.addEventListener('pointermove', onMove);
            svg.addEventListener('pointerup', onUp);
        };

        svg.ondblclick = () => {
            this._pushUndo();
            currentValue = initial;
            updateVisuals(initial);
            callback(initial);
        };

        // Right-click numeric entry
        svg.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();

            // Remove any existing input
            const existingInput = container.querySelector('.knob-numeric-input');
            if (existingInput) { existingInput.remove(); return; }

            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'knob-numeric-input';
            input.value = currentValue.toFixed(Math.max(0, -Math.log10(step || 0.01)));
            input.style.cssText = `
                position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                width: 48px; text-align: center; font-size: 10px;
                font-family: ${this.theme.mono}; color: ${this.theme.text};
                background: rgba(0,0,0,0.9); border: 1px solid ${this.theme.accent};
                border-radius: 4px; padding: 2px 4px; outline: none; z-index: 100;
            `;
            container.appendChild(input);
            input.focus();
            input.select();

            const apply = () => {
                let val = parseFloat(input.value);
                if (!isNaN(val)) {
                    this._pushUndo();
                    val = Math.max(min, Math.min(val, max));
                    if (step) val = Math.round(val / step) * step;
                    currentValue = val;
                    updateVisuals(currentValue);
                    callback(currentValue);
                }
                input.remove();
            };

            input.onkeydown = (ke) => {
                if (ke.key === 'Enter') { ke.preventDefault(); apply(); }
                else if (ke.key === 'Escape') { ke.preventDefault(); input.remove(); }
                ke.stopPropagation();
            };
            input.onblur = () => { setTimeout(() => { if (input.parentNode) apply(); }, 50); };
        });

        return container;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                    PER-CHANNEL COLOR WHEEL WIDGET
    // ═══════════════════════════════════════════════════════════════════════════

    createColorWheel(label, min, max, defaults, step, callback) {
        // defaults = [r, g, b] (Relative offsets usually 0,0,0)
        // The wheel controls Tint (Hue) and Saturation (distance from center).
        // The Slider below controls Luma (Master).
        // callback returns (r, g, b) combined.

        const container = document.createElement('div');
        container.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 4px; min-width: 100px; flex: 1; position: relative;';

        // Label
        const lbl = document.createElement('div');
        lbl.textContent = label;
        lbl.style.cssText = `color: ${this.theme.textDim}; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px;`;
        container.appendChild(lbl);

        // 1. Wheel (Hue/Sat)
        const size = 100; // slightly smaller to fit 4
        const center = size / 2;
        const radius = 36;
        const knobRadius = 4;

        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("width", size);
        svg.setAttribute("height", size);
        svg.style.cssText = "cursor: default; touch-action: none; background: radial-gradient(circle, #222 0%, #111 60%, #08080c 100%); border-radius: 50%; box-shadow: 0 4px 8px rgba(0,0,0,0.3) inset;";

        // Color Spectrum Ring (Visual)
        const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        ring.setAttribute("cx", center);
        ring.setAttribute("cy", center);
        ring.setAttribute("r", radius);
        ring.setAttribute("fill", "none");
        ring.setAttribute("stroke", "rgba(255,255,255,0.05)");
        ring.setAttribute("stroke-width", "1");
        svg.appendChild(ring);

        // Crosshair
        const crossH = document.createElementNS("http://www.w3.org/2000/svg", "line");
        crossH.setAttribute("x1", center - 4); crossH.setAttribute("x2", center + 4);
        crossH.setAttribute("y1", center); crossH.setAttribute("y2", center);
        crossH.setAttribute("stroke", "rgba(255,255,255,0.1)");
        svg.appendChild(crossH);
        const crossV = document.createElementNS("http://www.w3.org/2000/svg", "line");
        crossV.setAttribute("x1", center); crossV.setAttribute("x2", center);
        crossV.setAttribute("y1", center - 4); crossV.setAttribute("y2", center + 4);
        crossV.setAttribute("stroke", "rgba(255,255,255,0.1)");
        svg.appendChild(crossV);

        // Handle (Puck)
        const puck = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        puck.setAttribute("r", knobRadius);
        puck.setAttribute("fill", "rgba(255,255,255,0.8)");
        puck.setAttribute("stroke", "#000");
        puck.setAttribute("stroke-width", "1");
        puck.style.cursor = "grab";
        svg.appendChild(puck);

        // State
        // We store RGB offsets. Convert to HSL for wheel position?
        // Let's store internal H and S.
        // H = angle, S = distance.
        // Current defaults [r, g, b].
        let currentRGB = [...defaults];
        let masterLuma = (defaults[0] + defaults[1] + defaults[2]) / 3.0;

        // Helper: RGB -> Hue/Sat (ignoring Luma)
        // Simply: project RGB point to 2D plane perpendicular to (1,1,1).
        // Or just standard RGB->HSL.
        // Simplified: 
        // x = (R - G) * cos(30) - (G - B) * cos(30)? 
        // Let's use standard angle logic.

        // Initial puck position
        // We'll reset puck to center initially as parsing RGB back to wheel pos is complex and 'defaults' might be just 0,0,0
        let px = center, py = center;

        const updatePuck = (x, y) => {
            puck.setAttribute("cx", x);
            puck.setAttribute("cy", y);
            // Draw line to center
            // (Optional, cleaner without)
        };
        updatePuck(px, py);

        container.appendChild(svg);

        // 2. Master Slider (Luma)
        const sliderWrapper = document.createElement('div');
        sliderWrapper.style.cssText = 'width: 100%; height: 16px; position: relative; margin-top: 4px; background: #08080c; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);';

        const sliderFill = document.createElement('div');
        sliderFill.style.cssText = `position: absolute; left: 50%; top: 0; bottom: 0; width: 0%; background: ${this.theme.accent}; opacity: 0.5; transition: none;`;
        sliderWrapper.appendChild(sliderFill);

        const sliderThumb = document.createElement('div');
        sliderThumb.style.cssText = `position: absolute; left: 50%; top: -2px; bottom: -2px; width: 4px; background: #ccc; border-radius: 2px; cursor: ew-resize; transform: translateX(-50%);`;
        sliderWrapper.appendChild(sliderThumb);

        // Slider Logic
        // Sensitivity range: +/- 1.0?
        let sliderVal = masterLuma; // -1 to 1?

        const updateSliderVisuals = (v) => {
            // v is -1 to 1 approx?
            // Clamp visualization -1..1
            const p = Math.max(-1, Math.min(1, v));
            const pct = (p + 1) / 2 * 100; // 0..100

            sliderThumb.style.left = `${pct}%`;

            // Fill from center (50%)
            if (p > 0) {
                sliderFill.style.left = '50%';
                sliderFill.style.width = `${p * 50}%`;
            } else {
                sliderFill.style.left = `${(p + 1) * 50}%`;
                sliderFill.style.width = `${-p * 50}%`;
            }
        };
        updateSliderVisuals(sliderVal);

        container.appendChild(sliderWrapper);

        // Wheel Logic
        svg.onpointerdown = (e) => {
            e.preventDefault();
            this._pushUndo();
            svg.setPointerCapture(e.pointerId);

            const onMove = (em) => {
                const rect = svg.getBoundingClientRect();
                let dx = em.clientX - (rect.left + center);
                let dy = em.clientY - (rect.top + center);

                // Limit to radius
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist > radius) {
                    dx = (dx / dist) * radius;
                    dy = (dy / dist) * radius;
                }

                updatePuck(center + dx, center + dy);

                // Convert dx, dy to Color Balance (RGB offsets)
                // Angle 0 (Right) = Red? 
                // Standard: Top=Red? Resolve: Top=Red, Left=Green? 
                // Let's use:
                // angle 0 = Red
                // angle 120 = Green
                // angle 240 = Blue
                // (This is standard vectorscope orientation usually)

                const ang = Math.atan2(dy, dx); // radians
                const mag = dist / radius; // 0..1 normal usage

                // Boost magnitude for effect (wheel isn't limited to 0.1 shift)
                const strength = mag * 0.5; // Max shift 0.5

                // Convert polar to RGB
                // Simple approx:
                // R = cos(ang)
                // G = cos(ang - 120deg)
                // B = cos(ang - 240deg)

                const deg = ang * 180 / Math.PI;
                const r = Math.cos(ang) * strength;
                const g = Math.cos(ang - 2 * Math.PI / 3) * strength;
                const b = Math.cos(ang - 4 * Math.PI / 3) * strength;

                // Combine with Master Luma
                // currentRGB = [r + sliderVal, g + sliderVal, b + sliderVal]; 
                // But wait, sliderVal changes luma, wheel changes balance (chroma).
                // They overlap. 
                // Let's store wheelRGB and sliderLuma separately?
                // No, we need to return combined R,G,B to callback.

                // Keep wheel state
                container.wheelR = r; container.wheelG = g; container.wheelB = b;

                const totalR = r + (container.sliderLuma || 0);
                const totalG = g + (container.sliderLuma || 0);
                const totalB = b + (container.sliderLuma || 0);

                callback(totalR, totalG, totalB);
            };

            const onUp = () => {
                svg.removeEventListener('pointermove', onMove);
                svg.removeEventListener('pointerup', onUp);
                svg.releasePointerCapture(e.pointerId);
            };
            svg.addEventListener('pointermove', onMove);
            svg.addEventListener('pointerup', onUp);

            // Immediate update on click
            onMove(e);
        };

        // Reset wheel on double click
        svg.ondblclick = () => {
            this._pushUndo();
            updatePuck(center, center);
            container.wheelR = 0; container.wheelG = 0; container.wheelB = 0;
            const totalR = 0 + (container.sliderLuma || 0);
            const totalG = 0 + (container.sliderLuma || 0);
            const totalB = 0 + (container.sliderLuma || 0);
            callback(totalR, totalG, totalB);
        };

        // Slider Interactions
        sliderWrapper.onpointerdown = (e) => {
            e.preventDefault();
            this._pushUndo();
            sliderWrapper.setPointerCapture(e.pointerId);
            const rect = sliderWrapper.getBoundingClientRect();

            const updateS = (cx) => {
                let x = cx - rect.left;
                let pct = x / rect.width; // 0..1
                // Map to -1..1 (or range)
                let val = (pct - 0.5) * 2.0;
                // Range scale? usually -0.5 to 0.5 is enough for lift/gamma
                // Let's allow -1 to 1.

                updateSliderVisuals(val);
                container.sliderLuma = val; // Store luma

                // Combine
                const r = (container.wheelR || 0) + val;
                const g = (container.wheelG || 0) + val;
                const b = (container.wheelB || 0) + val;
                callback(r, g, b);
            };

            const onMove = (em) => { updateS(em.clientX); };
            const onUp = () => {
                sliderWrapper.removeEventListener('pointermove', onMove);
                sliderWrapper.removeEventListener('pointerup', onUp);
                sliderWrapper.releasePointerCapture(e.pointerId);
            };
            sliderWrapper.addEventListener('pointermove', onMove);
            sliderWrapper.addEventListener('pointerup', onUp);
            updateS(e.clientX);
        };

        sliderWrapper.ondblclick = () => {
            this._pushUndo();
            updateSliderVisuals(0);
            container.sliderLuma = 0;
            const r = (container.wheelR || 0);
            const g = (container.wheelG || 0);
            const b = (container.wheelB || 0);
            callback(r, g, b);
        };

        // Initialize internal state from defaults
        // This is tricky because we can't easily inverse RGB -> (Angle, Mag, Luma) uniquely.
        // For now, assume started at 0. If reloading with existing values, visuals might not match exactly.
        // Best effort: set Slider to average, and Wheel to difference?

        const avg = (defaults[0] + defaults[1] + defaults[2]) / 3;
        container.sliderLuma = avg;
        updateSliderVisuals(avg);

        // Wheel diff
        const dr = defaults[0] - avg;
        const dg = defaults[1] - avg;
        const db = defaults[2] - avg;
        // Project back to wheel? 
        // Not strictly necessary for functionality, just visual sync.
        // Let's skip precise wheel feedback for now to avoid complexity.
        // Centered wheel is fine, user adjusts from there.
        // (Resolve typically has encoders that don't have absolute position match)

        return container;
    }



    createControlRow(label, min, max, initial, step, callback) {
        // v3.0: Compact horizontal layout (Label + Value on top, Slider below)
        const row = document.createElement('div');
        row.style.cssText = 'display: flex; flex-direction: column; gap: 4px; width: 100%;';
        // ... (Legacy code maintained for Lens/other controls if needed)
        // Actually, we are replacing all grading controls with knobs.
        // Keeping this for backward compatibility or non-grading controls.

        const metaRow = document.createElement('div');
        metaRow.style.cssText = 'display: flex; justify-content: space-between; align-items: center; width: 100%;';

        const lbl = document.createElement('div');
        lbl.textContent = label;
        lbl.style.cssText = `
            color: ${this.theme.textDim};
            font-size: 11px;
            font-family: ${this.theme.font};
            min-width: 50px;
            font-weight: 500;
        `;

        const value = document.createElement('div');
        value.textContent = initial.toFixed(2);
        value.style.cssText = `
            font-family: ${this.theme.mono};
            font-size: 11px;
            color: ${this.theme.accent};
            min-width: 40px;
            text-align: right;
            font-variant-numeric: tabular-nums;
        `;

        metaRow.appendChild(lbl);
        metaRow.appendChild(value);

        const sliderContainer = document.createElement('div');
        sliderContainer.style.cssText = 'position: relative; height: 4px; background: #333; border-radius: 2px; margin-top: 2px;';

        const sliderFill = document.createElement('div');
        sliderFill.style.cssText = `
            position: absolute; left: 0; top: 0; height: 100%; background: #445;
            width: 50%; pointer-events: none; border-radius: 2px;
        `;

        const sliderInput = document.createElement('input');
        sliderInput.type = 'range';
        sliderInput.min = min; sliderInput.max = max; sliderInput.step = step; sliderInput.value = initial;
        sliderInput.style.cssText = `
            position: absolute; left: 0; top: -6px; width: 100%; height: 16px; opacity: 0; cursor: ew-resize; margin: 0;
        `;

        const updateVisuals = (val) => {
            const pct = ((val - min) / (max - min)) * 100;
            sliderFill.style.width = pct + '%';
            sliderFill.style.background = this.theme.accent;
            value.textContent = parseFloat(val).toFixed(2);
        };
        updateVisuals(initial);

        sliderInput.oninput = (e) => {
            const v = parseFloat(e.target.value);
            updateVisuals(v);
            callback(v);
        };

        // Reset on double click
        sliderInput.ondblclick = () => {
            sliderInput.value = initial;
            updateVisuals(initial);
            callback(initial);
        };

        sliderContainer.appendChild(sliderFill);
        sliderContainer.appendChild(sliderInput);

        row.appendChild(metaRow);
        row.appendChild(sliderContainer);

        return row;
    }

    toggleControls() {
        this.showControls = !this.showControls;
        if (this.controlsPanel) {
            this.controlsPanel.style.display = this.showControls ? 'flex' : 'none';
        }
        this.controlsToggle.style.color = this.showControls ? this.theme.accent : this.theme.textDim;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          PROGRESS & STATUS
    // ═══════════════════════════════════════════════════════════════════════════

    setupProgressUI() {
        const t = this.theme;
        this.progressContainer = document.createElement('div');
        this.progressContainer.style.cssText = `
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: rgba(0,0,0,0.5);
            z-index: 50;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;

        this.progressBar = document.createElement('div');
        this.progressBar.style.cssText = `
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, ${t.accent}, #4f4);
            transition: width 0.1s linear;
            box-shadow: 0 0 10px ${t.accent};
        `;

        this.progressText = document.createElement('div');
        this.progressText.style.cssText = `
            position: absolute;
            bottom: 6px;
            right: 10px;
            font-size: 10px;
            font-family: monospace;
            color: rgba(255,255,255,0.8);
            text-shadow: 0 1px 2px black;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
            background: rgba(0,0,0,0.6);
            padding: 2px 6px;
            border-radius: 4px;
        `;

        this.progressContainer.appendChild(this.progressBar);
        this.container.appendChild(this.progressContainer);
        this.container.appendChild(this.progressText);

        // API Events
        api.addEventListener("execution_start", () => {
            this.progressStart = Date.now();
            this.progressHistory = [];
            this.showProgress(true);
        });

        api.addEventListener("progress", ({ detail }) => {
            const { value, max } = detail;
            const pct = (value / max) * 100;
            this.progressBar.style.width = `${pct}%`;

            // ETA Calculation
            const now = Date.now();
            if (value > 0) {
                const elapsed = (now - this.progressStart) / 1000;
                const timePerStep = elapsed / value;
                const remaining = (max - value) * timePerStep;

                // Simple formatting
                const eta = remaining < 60 ? `${remaining.toFixed(1)}s` : `${Math.floor(remaining / 60)}m ${Math.floor(remaining % 60)}s`;
                this.progressText.textContent = `Step ${value}/${max} | ETA: ${eta}`;
            }
        });

        api.addEventListener("executed", ({ detail }) => {
            // Hide progress eventually if queue empty, but ComfyUI usually handles global progress
        });

        api.addEventListener("status", ({ detail }) => {
            if (!detail || detail.exec_info.queue_remaining === 0) {
                this.showProgress(false);
            }
        });
    }

    showProgress(show) {
        this.progressContainer.style.opacity = show ? 1 : 0;
        this.progressText.style.opacity = show ? 1 : 0;
        if (!show) {
            setTimeout(() => {
                this.progressBar.style.width = '0%';
                this.progressText.textContent = '';
            }, 300); // Wait for fade out
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          COLOR TRANSFORMS
    // ═══════════════════════════════════════════════════════════════════════════

    applyColorTransform(r, g, b, mode) {
        // Helper: Linear to sRGB
        const lin2srgb = (c) => c > 0.0031308 ? 1.055 * Math.pow(c, 1 / 2.4) - 0.055 : 12.92 * c;
        // Helper: Linear to Rec.709
        const lin2rec709 = (c) => c < 0.018 ? 4.5 * c : 1.099 * Math.pow(c, 0.45) - 0.099;

        switch (mode) {
            case 'sRGB':
                // Assume Input is Linear, Output sRGB
                return [lin2srgb(r), lin2srgb(g), lin2srgb(b)];
            case 'Rec.709':
                // Assume Input is Linear, Output Rec.709
                return [lin2rec709(r), lin2rec709(g), lin2rec709(b)];
            case 'LogC3':
                // ARRI LogC3 to Rec709 LUT approximation
                r = (r > 0.1496582 ? (Math.pow(10.0, (r - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (r / 0.1496582) * (Math.pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272));
                g = (g > 0.1496582 ? (Math.pow(10.0, (g - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (g / 0.1496582) * (Math.pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272));
                b = (b > 0.1496582 ? (Math.pow(10.0, (b - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (b / 0.1496582) * (Math.pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272));
                return [lin2rec709(Math.max(0, r)), lin2rec709(Math.max(0, g)), lin2rec709(Math.max(0, b))];
            case 'ACEScg':
                // ACEScg -> Rec.709 (Simple Tonemap)
                // Matrix (AP1 -> Rec.709)
                let rr = r * 1.70485868 - g * 0.62171602 - b * 0.08329937;
                let gg = -r * 0.19644612 + g * 1.26432540 + b * 0.03212072;
                let bb = -r * 0.01776686 - g * 0.00403754 + b * 1.02179971;
                // Simple Tonemap
                rr = rr / (rr + 1); gg = gg / (gg + 1); bb = bb / (bb + 1);
                return [lin2rec709(Math.max(0, rr)), lin2rec709(Math.max(0, gg)), lin2rec709(Math.max(0, bb))];
            default:
                return [r, g, b];
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          NPY PARSER
    // ═══════════════════════════════════════════════════════════════════════════

    // v2.2: Async HDR buffer parser (for fetch chain in onExecuted)
    async _parseHDRBuffer(buffer) {
        const magic = String.fromCharCode(...new Uint8Array(buffer.slice(0, 4)));
        if (magic === 'RHDR') {
            return await this._parseRHDR(buffer);
        }
        return this._parseNumpy(buffer);
    }

    // v2.2: Parse .rhdr (compressed float16) or legacy .npy (float32)
    parseHDRSidecar(buffer) {
        try {
            const view = new DataView(buffer);
            // Check for RHDR magic
            const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));

            if (magic === 'RHDR') {
                return this._parseRHDR(buffer);
            }
            // Fallback: try legacy .npy format
            return this._parseNumpy(buffer);
        } catch (e) {
            console.error("[Radiance] Failed to parse HDR sidecar:", e);
            return null;
        }
    }

    // Parse .rhdr: zlib-compressed float16 with 12-byte header
    async _parseRHDR(buffer) {
        const view = new DataView(buffer);
        const width = view.getUint16(4, true);
        const height = view.getUint16(6, true);
        const channels = view.getUint16(8, true);
        // reserved = view.getUint16(10, true);

        // Decompress zlib payload
        const compressed = new Uint8Array(buffer, 12);
        const decompressed = await this._zlibInflateAsync(compressed);
        if (!decompressed) return null;

        // v3.0 FIX: Validate decompressed size (was missing — _loadRHDR had it, this didn't)
        const expectedSize = width * height * channels * 2; // 2 bytes per float16
        if (decompressed.byteLength !== expectedSize) {
            console.error(`[Radiance] RHDR integrity failure: expected ${expectedSize} bytes, got ${decompressed.byteLength} (${width}×${height}×${channels}ch)`);
            return null;
        }

        // Raw float16 as Uint16Array (for WebGL HALF_FLOAT upload)
        const fp16Raw = new Uint16Array(decompressed.buffer, decompressed.byteOffset, decompressed.byteLength / 2);

        // Also create Float32Array for CPU-side reads (probe, scopes)
        const fp32 = new Float32Array(fp16Raw.length);
        for (let i = 0; i < fp16Raw.length; i++) {
            fp32[i] = this._halfToFloat(fp16Raw[i]);
        }

        return {
            data: fp32,           // Float32Array for CPU reads
            fp16data: fp16Raw,    // Uint16Array for GPU HALF_FLOAT upload
            shape: [height, width, channels],
            format: 'rhdr'
        };
    }

    // Legacy .npy parser (backward compatibility)
    _parseNumpy(buffer) {
        const headerLengthBlock = new DataView(buffer.slice(8, 10)).getUint16(0, true);
        const headerStr = new TextDecoder("ascii").decode(buffer.slice(10, 10 + headerLengthBlock));

        const shapeMatch = headerStr.match(/\'shape\':\s*\(([^)]+)\)/);
        if (!shapeMatch) return null;

        const shape = shapeMatch[1].split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
        const dtypeMatch = headerStr.match(/\'descr\':\s*\'([^']+)\'/);
        const dtype = dtypeMatch ? dtypeMatch[1] : '<f4';

        const isFloat32 = dtype.includes('f4');
        const offset = 10 + headerLengthBlock;

        if (isFloat32) {
            return {
                data: new Float32Array(buffer.slice(offset)),
                shape: shape,
                format: 'npy'
            };
        }
        return null;
    }

    // IEEE 754 half-float (16-bit) → single-precision float (32-bit)
    _halfToFloat(h) {
        const s = (h >> 15) & 0x1;
        const e = (h >> 10) & 0x1f;
        const m = h & 0x3ff;

        if (e === 0) {
            if (m === 0) return s ? -0.0 : 0.0;
            // Subnormal
            let val = m / 1024.0 * Math.pow(2, -14);
            return s ? -val : val;
        }
        if (e === 31) {
            return m ? NaN : (s ? -Infinity : Infinity);
        }
        let val = Math.pow(2, e - 15) * (1 + m / 1024.0);
        return s ? -val : val;
    }

    // v3.1: Robust zlib inflate with error handling and fallback
    // Python zlib.compress() outputs zlib-wrapped deflate.
    // DecompressionStream('deflate') handles zlib wrapper in modern browsers.
    async _zlibInflateAsync(compressed) {
        if (typeof DecompressionStream === 'undefined') {
            console.warn('[Radiance] DecompressionStream API not supported. Cannot load compressed RHDR.');
            return null;
        }

        try {
            const ds = new DecompressionStream('deflate');
            const writer = ds.writable.getWriter();
            const reader = ds.readable.getReader();

            writer.write(compressed);
            writer.close();

            const chunks = [];
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
            }

            let totalLen = chunks.reduce((s, c) => s + c.byteLength, 0);
            if (totalLen === 0) {
                console.error('[Radiance] _zlibInflateAsync: decompressed to 0 bytes');
                return null;
            }
            const result = new Uint8Array(totalLen);
            let offset = 0;
            for (const chunk of chunks) {
                result.set(new Uint8Array(chunk), offset);
                offset += chunk.byteLength;
            }
            return result;
        } catch (e) {
            console.error('[Radiance] _zlibInflateAsync failed:', e);
            return null;
        }
    }

    // Synchronous zlib inflate wrapper (calls async internally)
    _zlibInflate(compressed) {
        // For sync context, we use a pre-resolved promise approach
        // Actually, parseHDRSidecar callers are already async — 
        // we'll make this truly async in the fetch chain.
        // This stub exists for structure; actual decompression happens in loadHDRData.
        return null; // Will be replaced by async path
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                          NODE REGISTRATION
// ═══════════════════════════════════════════════════════════════════════════════

app.registerExtension({
    name: "FXTD.RadianceViewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "FXTD_RadianceViewer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const container = document.createElement('div');
            container.id = `radiance-viewer-${this.id}`;
            this.addDOMWidget("viewer", "viewer", container, { serialize: false, hideOnZoom: false });

            // Force container properties to ensure it expands
            container.style.display = 'flex';
            container.style.width = '100%';
            container.style.height = '100%';

            this.radianceViewer = new RadianceViewer(this, container);
        };
        // ═══════════════════════════════════════════════════════════════════════════════
        //                           CURVE EDITOR MOVED TO END
        // ═══════════════════════════════════════════════════════════════════════════════

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            onResize?.apply(this, arguments);
            if (this.radianceViewer) {
                // Debounce resize to prevent thrashing
                if (this._resizeTimer) clearTimeout(this._resizeTimer);
                this._resizeTimer = setTimeout(() => {
                    this.radianceViewer.resize();
                }, 20);
            }
        };

        nodeType.prototype.onExecuted = function (message) {
            // v2.3: Error handling - show backend errors instead of silent failure
            if (message?.error?.length) {
                alert("Radiance Viewer Error:\n" + message.error.join("\n"));
                return;
            }

            if (!message?.radiance_images?.length) return;

            const viewer = this.radianceViewer;
            if (!viewer) return;

            // Separate main images from compare and zdepth images
            const mainImages = message.radiance_images.filter(img => !img.is_compare && !img.is_zdepth);
            const compareImages = message.radiance_images.filter(img => img.is_compare);
            const zdepthImages = message.radiance_images.filter(img => img.is_zdepth);

            // Reset frame arrays
            viewer.frameImages = [];
            viewer.frameCompareImages = [];
            viewer.frameZdepthImages = [];
            viewer.frameHDRData = [];
            viewer.totalFrames = mainImages.length;
            viewer.currentFrame = 0;
            // FIX: Clear stale active HDR data to prevent display of previous run's data
            viewer.hdrData = null;

            // v2.2: HDR-PRIMARY architecture — .rhdr is the display source, PNG is just a placeholder.
            // Like DJV/RV loading EXR: GPU tonemaps float data in real-time.
            mainImages.forEach((imgData, idx) => {
                const hasHDRPrimary = imgData.hdr_sidecar && imgData.hdr_primary;
                const hdrUrl = imgData.hdr_sidecar ?
                    api.apiURL(`/view?filename=${encodeURIComponent(imgData.hdr_sidecar)}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&type=${imgData.type || 'temp'}`) : null;
                const pngUrl = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&type=${imgData.type || 'temp'}`);

                // Load tiny PNG thumbnail as placeholder (shows instantly while RHDR loads)
                const img = new Image();
                img.crossOrigin = 'anonymous';
                // Store metadata on image object for fallback access
                img.exr_filename = imgData.exr_filename;
                img.subfolder = imgData.subfolder;
                img.type = imgData.type;

                img.onload = () => {
                    viewer.frameImages[idx] = img;
                    // Only set as display if no HDR primary, or as placeholder while HDR loads
                    // If HDR load eventually fails, this will remain as the display image
                    if (idx === 0 && !viewer.frameHDRData[0]) {
                        viewer.image = img;
                        viewer.imageWidth = img.width;
                        viewer.imageHeight = img.height;
                        if (viewer.renderer) viewer.renderer.loadImageTexture(img);
                        viewer.fitToView();
                        viewer.render();
                    }
                    if (viewer._allFramesReady()) viewer.updateFrameDisplay();
                };
                img.onerror = (e) => {
                    console.error("[Radiance] Failed to load thumbnail:", imgData.filename, e);
                };
                img.src = pngUrl;

                // Load .rhdr as PRIMARY display source (like DJV loading EXR)
                if (hdrUrl) {
                    fetch(hdrUrl)
                        .then(r => r.arrayBuffer())
                        .then(async (buffer) => {
                            // If parse fails (e.g. no DecompressionStream), it returns null
                            const npy = await viewer._parseHDRBuffer(buffer);

                            if (npy) {
                                npy.height = npy.shape[0];
                                npy.width = npy.shape[1];
                                npy.channels = npy.shape.length > 2 ? npy.shape[2] : 1;

                                // Propagate metadata
                                npy.exr_filename = imgData.exr_filename;
                                npy.subfolder = imgData.subfolder;
                                npy.type = imgData.type;

                                viewer.frameHDRData[idx] = npy;

                                // Set as PRIMARY display immediately
                                if (idx === viewer.currentFrame) {
                                    viewer.hdrData = npy;
                                    viewer.imageWidth = npy.width;
                                    viewer.imageHeight = npy.height;

                                    if (viewer.renderer) {
                                        let tex;
                                        try {
                                            if (npy.fp16data) {
                                                tex = viewer.renderer.loadFloat16Texture(
                                                    npy.fp16data, npy.width, npy.height, npy.channels
                                                );
                                            } else {
                                                tex = viewer.renderer.loadFloat32Texture(
                                                    npy.data, npy.width, npy.height, npy.channels
                                                );
                                            }
                                        } catch (e) {
                                            console.warn("[Radiance] HDR Texture creation error:", e);
                                            tex = null;
                                        }

                                        // v3.1: Robust fallback — reload full-res PNG as display source
                                        // This handles case where texture creation fails (e.g. OOM or invalid dimensions)
                                        if (!tex) {
                                            console.warn("[Radiance] HDR texture load failed. Falling back to full-res PNG.");
                                            viewer.hdrData = null;
                                            viewer.frameHDRData[idx] = null;
                                            throw new Error("Texture creation failed"); // Trigger catch block for consistency
                                        }
                                    }

                                    viewer.createPlaceholderImage(npy.width, npy.height);
                                    viewer.fitToView();
                                    viewer.render();
                                    viewer.updateScopes();
                                    viewer.updateInfo();
                                }
                            } else {
                                throw new Error("RHDR parsing failed (returned null)");
                            }
                        })
                        .catch(e => {
                            console.warn("[Radiance] Failed to load RHDR primary:", e);
                            // v3.1: On RHDR fetch/parse failure, ensure PNG is loaded as fallback
                            // FIX: Race condition handled by checking if frameImages[idx] is available.
                            // If available, force-update display to use it if we are on that frame.
                            // If not available yet, img.onload will handle it (since frameHDRData[idx] is unset).

                            // If we already have the PNG and we are on this frame, ensure it's displayed
                            if (idx === viewer.currentFrame && viewer.frameImages[idx]) {
                                viewer.hdrData = null; // Explicitly clear any partial state
                                viewer.image = viewer.frameImages[idx];
                                viewer.imageWidth = viewer.image.width;
                                viewer.imageHeight = viewer.image.height;
                                if (viewer.renderer) viewer.renderer.loadImageTexture(viewer.image);
                                viewer.fitToView();
                                viewer.render();
                                viewer.updateInfo();
                            }
                        });
                }
            });

            // Load compare images
            compareImages.forEach((imgData, idx) => {
                const cmp = new Image();
                cmp.crossOrigin = 'anonymous';
                cmp.onload = () => {
                    viewer.frameCompareImages[idx] = cmp;
                    if (idx === 0) viewer.setCompareImage(cmp);
                };
                cmp.onerror = (e) => console.warn("[Radiance] Failed to load compare image:", imgData.filename);
                cmp.src = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&type=${imgData.type || 'temp'}`);
            });

            // Load Z-depth images
            zdepthImages.forEach((imgData, idx) => {
                const zImg = new Image();
                zImg.crossOrigin = 'anonymous';
                zImg.onload = () => {
                    viewer.frameZdepthImages[idx] = zImg;
                    if (idx === viewer.currentFrame) {
                        viewer.zdepthImage = zImg;
                        if (viewer.renderer) viewer.renderer.loadDepthTexture(zImg);
                        // Force re-render if we are already displaying this frame
                        viewer.render();
                    }
                };
                zImg.onerror = (e) => console.warn("[Radiance] Failed to load zdepth image:", imgData.filename);
                zImg.src = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&type=${imgData.type || 'temp'}`);
            });
        };
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
//                           CURVE EDITOR
// ═══════════════════════════════════════════════════════════════════════════════

class RadianceCurveEditor {
    constructor(width, height, onChange) {
        this.width = width;
        this.height = height;
        this.onChange = onChange;
        this.padding = { left: 28, bottom: 18, top: 6, right: 6 };

        this.canvas = document.createElement('canvas');
        this.canvas.width = width;
        this.canvas.height = height;
        this.canvas.style.cursor = 'crosshair';
        this.ctx = this.canvas.getContext('2d');

        this.histograms = { R: null, G: null, B: null, L: null };

        this.curves = {
            'RGB': [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            'R': [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            'G': [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            'B': [{ x: 0, y: 0 }, { x: 1, y: 1 }]
        };

        this.activeChannel = 'RGB';
        this.hoverPoint = null;
        this.draggingPoint = null;

        this.channelColors = { 'RGB': '#fff', 'R': '#f55', 'G': '#5f5', 'B': '#55f' };

        this.setupEvents();
        this.draw();
    }

    // ─── Coordinate helpers (account for padding) ─────────────
    get plotX() { return this.padding.left; }
    get plotY() { return this.padding.top; }
    get plotW() { return this.width - this.padding.left - this.padding.right; }
    get plotH() { return this.height - this.padding.top - this.padding.bottom; }

    normToCanvas(nx, ny) {
        return { cx: this.plotX + nx * this.plotW, cy: this.plotY + (1 - ny) * this.plotH };
    }
    canvasToNorm(cx, cy) {
        return { x: (cx - this.plotX) / this.plotW, y: 1.0 - (cy - this.plotY) / this.plotH };
    }

    updateHistogram(img) {
        if (!img) return;
        const scale = Math.min(1.0, 256 / Math.max(img.width, img.height));
        const w = Math.floor(img.width * scale);
        const h = Math.floor(img.height * scale);

        const temp = document.createElement('canvas');
        temp.width = w; temp.height = h;
        const tctx = temp.getContext('2d');
        tctx.drawImage(img, 0, 0, w, h);
        const data = tctx.getImageData(0, 0, w, h).data;

        const buckets = 256;
        const R = new Uint32Array(buckets), G = new Uint32Array(buckets);
        const B = new Uint32Array(buckets), L = new Uint32Array(buckets);

        for (let i = 0; i < data.length; i += 4) {
            R[data[i]]++;
            G[data[i + 1]]++;
            B[data[i + 2]]++;
            const luma = Math.min(255, Math.floor(data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722));
            L[luma]++;
        }

        let max = 0;
        for (let i = 0; i < buckets; i++) max = Math.max(max, L[i], R[i], G[i], B[i]);
        if (max === 0) max = 1;

        const norm = (arr) => { const r = new Float32Array(buckets); for (let i = 0; i < buckets; i++) r[i] = arr[i] / max; return r; };
        this.histograms = { R: norm(R), G: norm(G), B: norm(B), L: norm(L) };
        this.draw();
    }

    setActiveChannel(ch) { this.activeChannel = ch; this.draw(); }

    resetActiveChannel() {
        this.curves[this.activeChannel] = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
        this.notifyChange();
        this.draw();
    }

    resetAll() {
        for (const ch of ['RGB', 'R', 'G', 'B']) {
            this.curves[ch] = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
        }
        this.notifyChange();
        this.draw();
    }

    applyPreset(preset) {
        // Preset is { RGB?: [...pts], R?: [...], G?: [...], B?: [...] }
        for (const ch of ['RGB', 'R', 'G', 'B']) {
            if (preset[ch]) {
                this.curves[ch] = preset[ch].map(p => ({ x: p.x, y: p.y }));
            } else {
                this.curves[ch] = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
            }
        }
        this.notifyChange();
        this.draw();
    }

    // ─── Events ───────────────────────────────────────────────
    setupEvents() {
        const cvs = this.canvas;

        cvs.onmousedown = (e) => {
            const pt = this.getMousePos(e);
            const existing = this.findPoint(pt);

            if (existing) {
                this.draggingPoint = existing;
                cvs.style.cursor = 'grabbing';
            } else {
                const pts = this.curves[this.activeChannel];
                const newPt = { x: Math.max(0, Math.min(1, pt.x)), y: Math.max(0, Math.min(1, pt.y)) };
                pts.push(newPt);
                pts.sort((a, b) => a.x - b.x);
                this.draggingPoint = newPt;
                cvs.style.cursor = 'grabbing';
            }
            this.draw();
        };

        cvs.onmousemove = (e) => {
            if (!this.draggingPoint) {
                const pt = this.getMousePos(e);
                const hover = this.findPoint(pt);
                if (hover !== this.hoverPoint) {
                    this.hoverPoint = hover;
                    cvs.style.cursor = hover ? 'grab' : 'crosshair';
                    this.draw();
                }
            }
        };

        window.addEventListener('mousemove', (e) => {
            if (this.draggingPoint) {
                const rect = cvs.getBoundingClientRect();
                const pos = this.canvasToNorm(e.clientX - rect.left, e.clientY - rect.top);
                let x = Math.max(0, Math.min(1, pos.x));
                let y = Math.max(0, Math.min(1, pos.y));

                const pts = this.curves[this.activeChannel];
                const idx = pts.indexOf(this.draggingPoint);

                if (idx === 0) x = 0;
                else if (idx === pts.length - 1) x = 1;
                else {
                    if (x <= pts[idx - 1].x + 0.005) x = pts[idx - 1].x + 0.005;
                    if (x >= pts[idx + 1].x - 0.005) x = pts[idx + 1].x - 0.005;
                }

                this.draggingPoint.x = x;
                this.draggingPoint.y = y;
                this.notifyChange();
                this.draw();
            }
        });

        window.addEventListener('mouseup', () => {
            if (this.draggingPoint) {
                this.draggingPoint = null;
                cvs.style.cursor = 'crosshair';
                this.draw();
            }
        });

        cvs.ondblclick = (e) => {
            const pt = this.getMousePos(e);
            const target = this.findPoint(pt);
            if (target) {
                const pts = this.curves[this.activeChannel];
                const idx = pts.indexOf(target);
                if (idx > 0 && idx < pts.length - 1) {
                    this.curves[this.activeChannel] = pts.filter(p => p !== target);
                    this.notifyChange();
                    this.draw();
                }
            }
        };
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return this.canvasToNorm(e.clientX - rect.left, e.clientY - rect.top);
    }

    findPoint(pos) {
        const pts = this.curves[this.activeChannel];
        const threshX = 0.04, threshY = 0.04;
        return pts.find(p => Math.abs(p.x - pos.x) < threshX && Math.abs(p.y - pos.y) < threshY);
    }

    // ─── Drawing ──────────────────────────────────────────────
    draw() {
        const ctx = this.ctx;
        const w = this.width, h = this.height;
        const pX = this.plotX, pY = this.plotY, pW = this.plotW, pH = this.plotH;

        // Clear
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, w, h);

        // Plot area background
        ctx.fillStyle = '#222';
        ctx.fillRect(pX, pY, pW, pH);

        // ─── Grid Labels ─────────────────────────────────
        ctx.fillStyle = '#555';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const labels = [0, 25, 50, 75, 100];
        labels.forEach(v => {
            const nx = v / 100;
            const { cx } = this.normToCanvas(nx, 0);
            ctx.fillText(v + '', cx, pY + pH + 3);
        });
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        labels.forEach(v => {
            const ny = v / 100;
            const { cy } = this.normToCanvas(0, ny);
            ctx.fillText(v + '', pX - 4, cy);
        });

        // ─── Grid Lines ──────────────────────────────────
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 1; i < 4; i++) {
            const n = i * 0.25;
            const { cx: gx } = this.normToCanvas(n, 0);
            const { cy: gy } = this.normToCanvas(0, n);
            ctx.moveTo(gx, pY); ctx.lineTo(gx, pY + pH);
            ctx.moveTo(pX, gy); ctx.lineTo(pX + pW, gy);
        }
        ctx.stroke();

        // Diagonal reference
        ctx.strokeStyle = '#2a2a2a';
        ctx.lineWidth = 1;
        ctx.beginPath();
        const d0 = this.normToCanvas(0, 0), d1 = this.normToCanvas(1, 1);
        ctx.moveTo(d0.cx, d0.cy); ctx.lineTo(d1.cx, d1.cy);
        ctx.stroke();

        // ─── Histogram ───────────────────────────────────
        if (this.histograms.L) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(pX, pY, pW, pH);
            ctx.clip();

            ctx.globalAlpha = 0.25;
            const drawHist = (hist, color) => {
                if (!hist) return;
                ctx.fillStyle = color;
                ctx.beginPath();
                const b0 = this.normToCanvas(0, 0);
                ctx.moveTo(b0.cx, b0.cy);
                for (let i = 0; i < 256; i++) {
                    const { cx, cy } = this.normToCanvas(i / 255, hist[i]);
                    ctx.lineTo(cx, cy);
                }
                const bEnd = this.normToCanvas(1, 0);
                ctx.lineTo(bEnd.cx, bEnd.cy);
                ctx.fill();
            };

            if (this.activeChannel === 'RGB') drawHist(this.histograms.L, '#888');
            else if (this.activeChannel === 'R') drawHist(this.histograms.R, '#f44');
            else if (this.activeChannel === 'G') drawHist(this.histograms.G, '#4f4');
            else if (this.activeChannel === 'B') drawHist(this.histograms.B, '#44f');

            ctx.globalAlpha = 1.0;
            ctx.restore();
        }

        // ─── Ghost Curves (inactive channels) ────────────
        ctx.save();
        ctx.beginPath();
        ctx.rect(pX, pY, pW, pH);
        ctx.clip();

        const ghostChannels = this.activeChannel === 'RGB'
            ? ['R', 'G', 'B']
            : ['RGB'];

        ghostChannels.forEach(ch => {
            const ghostPts = this.curves[ch];
            const ghostLut = this.solveCatmullRom(ghostPts);
            const ghostColor = this.channelColors[ch];

            ctx.globalAlpha = 0.15;
            ctx.strokeStyle = ghostColor;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const { cx, cy } = this.normToCanvas(i / 255, ghostLut[i]);
                if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
            }
            ctx.stroke();
        });
        ctx.globalAlpha = 1.0;
        ctx.restore();

        // ─── Active Curve ─────────────────────────────────
        const pts = this.curves[this.activeChannel];
        const lut = this.solveCatmullRom(pts);
        const color = this.channelColors[this.activeChannel];

        ctx.save();
        ctx.beginPath();
        ctx.rect(pX, pY, pW, pH);
        ctx.clip();

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < 256; i++) {
            const { cx, cy } = this.normToCanvas(i / 255, lut[i]);
            if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
        }
        ctx.stroke();

        ctx.restore();

        // ─── Clipping Indicators ──────────────────────────
        const clipTop = lut[255] >= 0.995;    // whites clipped
        const clipBot = lut[0] <= 0.005;       // blacks crushed
        const clipHighMid = lut.some((v, i) => i > 128 && v >= 0.995);
        const clipLowMid = lut.some((v, i) => i < 128 && v <= 0.005);

        if (clipTop || clipHighMid) {
            ctx.fillStyle = 'rgba(255,60,60,0.3)';
            ctx.fillRect(pX, pY, pW, 3);
        }
        if (!clipBot || clipLowMid) {
            // Only show bottom indicator if blacks are lifted (y > 0 at x=0)
            if (lut[0] > 0.01) {
                ctx.fillStyle = 'rgba(60,60,255,0.3)';
                ctx.fillRect(pX, pY + pH - 3, pW, 3);
            }
        }

        // ─── Control Points ───────────────────────────────
        pts.forEach(p => {
            const { cx, cy } = this.normToCanvas(p.x, p.y);
            const isHover = (p === this.hoverPoint);
            const isDrag = (p === this.draggingPoint);
            const radius = (isHover || isDrag) ? 6 : 5;

            // Glow
            if (isHover || isDrag) {
                ctx.beginPath();
                ctx.arc(cx, cy, radius + 4, 0, Math.PI * 2);
                ctx.fillStyle = color.replace(')', ',0.15)').replace('rgb', 'rgba').replace('#', '');
                // Use a simpler glow approach
                ctx.shadowColor = color;
                ctx.shadowBlur = 8;
                ctx.fill();
                ctx.shadowBlur = 0;
            }

            // Point ring
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, Math.PI * 2);
            ctx.fillStyle = '#111';
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();
        });
    }

    // ─── Catmull-Rom Spline ───────────────────────────────
    solveCatmullRom(points) {
        const n = points.length;
        const xs = points.map(p => p.x);
        const ys = points.map(p => p.y);
        const result = new Float32Array(256);

        for (let i = 0; i < 256; i++) {
            const val = i / 255;

            if (val <= xs[0]) { result[i] = Math.max(0, Math.min(1, ys[0])); continue; }
            if (val >= xs[n - 1]) { result[i] = Math.max(0, Math.min(1, ys[n - 1])); continue; }

            // Find segment
            let k = 0;
            while (k < n - 2 && xs[k + 1] < val) k++;

            const t = (val - xs[k]) / (xs[k + 1] - xs[k]);

            // Catmull-Rom needs 4 points: p0, p1, p2, p3
            const y0 = (k > 0) ? ys[k - 1] : 2 * ys[0] - ys[1]; // mirror
            const y1 = ys[k];
            const y2 = ys[k + 1];
            const y3 = (k + 2 < n) ? ys[k + 2] : 2 * ys[n - 1] - ys[n - 2]; // mirror

            // Catmull-Rom formula (tension = 0.5)
            const t2 = t * t, t3 = t2 * t;
            const v = 0.5 * (
                (2 * y1) +
                (-y0 + y2) * t +
                (2 * y0 - 5 * y1 + 4 * y2 - y3) * t2 +
                (-y0 + 3 * y1 - 3 * y2 + y3) * t3
            );

            result[i] = Math.max(0, Math.min(1, v));
        }

        return result;
    }

    // Keep old name as alias for backward compat
    solveSpline(points) { return this.solveCatmullRom(points); }

    notifyChange() {
        if (!this.onChange) return;

        const lut = new Uint8Array(256 * 4);

        const master = this.solveCatmullRom(this.curves['RGB']);
        const rCurve = this.solveCatmullRom(this.curves['R']);
        const gCurve = this.solveCatmullRom(this.curves['G']);
        const bCurve = this.solveCatmullRom(this.curves['B']);

        for (let i = 0; i < 256; i++) {
            const mVal = master[i];

            const lookup = (curve, u) => {
                const idxF = u * 255;
                const idx = Math.floor(idxF);
                const fract = idxF - idx;
                if (idx >= 255) return curve[255];
                return curve[idx] * (1 - fract) + curve[idx + 1] * fract;
            };

            const rVal = lookup(rCurve, mVal);
            const gVal = lookup(gCurve, mVal);
            const bVal = lookup(bCurve, mVal);

            lut[i * 4 + 0] = Math.min(255, Math.max(0, rVal * 255));
            lut[i * 4 + 1] = Math.min(255, Math.max(0, gVal * 255));
            lut[i * 4 + 2] = Math.min(255, Math.max(0, bVal * 255));
            lut[i * 4 + 3] = 255;
        }

        this.onChange(lut);
    }
}

