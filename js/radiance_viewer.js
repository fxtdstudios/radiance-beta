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

console.log("[Radiance] Viewer Script Loading...");

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { RadianceWebGLRenderer } from "./radiance_webgl.js?v=2.1.0";


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
        this.bokehHighlightBias = 0.0;
        this.bokehSoapBubble = 0.0;
        this.bokehOpticalVig = 0.0;

        this.saturation = 1.0;
        this.zebraThreshold = 0.95;
        this.lutIntensity = 1.0;

        this.falseColor = false;
        this.zebra = false;
        this.gamutWarning = false;
        this.clippingMonitor = false;
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
        this.waveformParadeMode = true; // true = RGB parade
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
        this.lutOptions = [
            'None',
            'sRGB (Display)',
            'Rec.709 (Broadcast)',
            'Filmic (Cinematic)',
            'Log C3 (ARRI)',
            'Log C4 (ARRI)',
            'Canon Log',
            'Fuji F-Log',
            'Blackmagic Gen5',
            'Linear to Log',
            'Reinhard Tonemap',
            'ACES Filmic'
        ];
        this.denoise = 0.0;
        this.grain = 0.0;

        // Grading State
        this.exposure = 0.0;
        this.temperature = 0.0;
        this.tint = 0.0;
        this.contrast = 1.0;
        this.pivot = 0.5;
        this.saturation = 1.0;
        this.lift = [0, 0, 0];
        this.gamma = [1, 1, 1];
        this.gain = [1, 1, 1];
        this.offset = [0, 0, 0];

        // v2.2 Pro features
        this.displayLutMode = 0;
        this.displayLutStrength = 1.0;
        this.lumaMix = 1.0;
        this.wipe = 0.5;
        this.wipeEnabled = false;
        this.wipeRefEnabled = false;
        this.gridMode = 0;
        this.gridColor = [0.0, 0.7, 1.0, 0.5]; // Light blue guide color

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

        // HUD Panel Sizing and Position (persisted)
        const savedHudWidth = localStorage.getItem('radiance_hud_width');
        this.hudPanelWidth = savedHudWidth ? parseInt(savedHudWidth) : 580;
        this.hudPanelMinWidth = 480;
        this.hudPanelMaxWidth = 960;

        const savedHudX = localStorage.getItem('radiance_hud_x');
        const savedHudY = localStorage.getItem('radiance_hud_y');
        this.hudX = savedHudX !== null ? parseFloat(savedHudX) : null; // null means default (bottom center)
        this.hudY = savedHudY !== null ? parseFloat(savedHudY) : null;

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

        // v3.1: Masking (Power Windows) state
        this.maskState = {
            type: 0, // 0=None, 1=Circle, 2=Box
            center: [0.5, 0.5],
            scale: [0.3, 0.3],
            feather: 0.2,
            rotation: 0.0,
            invert: false,
            showOverlay: false
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

        if (this.controlsPanel) {
            this.controlsPanel.style.opacity = opacity;
            this.controlsPanel.style.pointerEvents = pointerEvents;
            this.controlsPanel.style.transform = visible ? 'translateX(-50%)' : 'translate(-50%, 20px)';
        }
        if (this.toolbar) {
            this.toolbar.style.opacity = opacity;
            this.toolbar.style.pointerEvents = pointerEvents;
            this.toolbar.style.transition = 'opacity 0.4s';
        }
        if (this.bottomInfoBar) {
            this.bottomInfoBar.style.display = visible ? 'flex' : 'none';
        }
        if (this.transportPanel) {
            this.transportPanel.style.opacity = opacity;
            this.transportPanel.style.pointerEvents = pointerEvents;
            this.transportPanel.style.transform = visible ? 'translateX(-50%)' : 'translate(-50%, 10px)';
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
        this.useWebGL = true;
        try {
            if (typeof RadianceWebGLRenderer !== 'undefined') {
                this.renderer = new RadianceWebGLRenderer(this.glCanvas);
                if (this.renderer.init()) {
                    console.log("[Radiance] WebGL Renderer Initialized");
                }
            } else {
                console.warn("[Radiance] WebGL Renderer class not found, falling back to 2D.");
                this.useWebGL = false;
            }
        } catch (e) {
            console.warn("[Radiance] WebGL init failed:", e);
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



        // Bottom Info Bar (HUD)
        this.bottomInfoBar = document.createElement('div');
        this.bottomInfoBar.style.cssText = `
            flex: 0 0 22px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 10px;
            background: rgba(10, 10, 14, 0.95);
            border-top: 1px solid ${t.panelBorder};
            font-size: 10.5px;
            font-family: monospace;
            color: #aaa;
            white-space: nowrap;
            overflow: hidden;
            z-index: 1000;
        `;

        this.infoLeft = document.createElement('div');
        this.infoLeft.style.cssText = 'display: flex; gap: 16px; align-items: center;';
        this.bottomInfoBar.appendChild(this.infoLeft);

        this.infoRight = document.createElement('div');
        this.infoRight.style.cssText = 'display: flex; gap: 16px; align-items: center;';
        this.bottomInfoBar.appendChild(this.infoRight);

        // False Color Legend (Overlay within Bottom Bar)
        this.fcLegend = document.createElement('div');
        this.fcLegend.style.cssText = `
            position: absolute; left: 50%; transform: translateX(-50%);
            display: none; align-items: center; gap: 4px; height: 100%;
        `;
        const fcMap = [
            { s: '-4', c: '#0d0d66' }, { s: '-3', c: '#1a668c' }, { s: '-2', c: '#1a7359' },
            { s: '-1', c: '#268c26' }, { s: 'MD', c: '#666666' }, { s: '+1', c: '#b3a61a' },
            { s: '+2', c: '#bf660d' }, { s: '+3', c: '#b31a1a' }, { s: 'CLIP', c: '#cc2699' }
        ];
        fcMap.forEach(item => {
            const block = document.createElement('div');
            block.style.cssText = `display: flex; align-items: center; gap: 3px;`;
            block.innerHTML = `<div style="width:8px; height:8px; background:${item.c}; border:1px solid rgba(255,255,255,0.2)"></div><span style="font-size:8.5px; color:#888">${item.s}</span>`;
            this.fcLegend.appendChild(block);
        });
        this.bottomInfoBar.appendChild(this.fcLegend);

        this.container.appendChild(this.bottomInfoBar);

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

        // Create metadata overlay (once, not per-render)
        this.createMetadataOverlay();
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
        const waveformHeader = document.createElement('div');
        waveformHeader.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-top: 5px;';

        this.waveformLabel = this.createLabel('Waveform', true);

        this.paradeToggle = document.createElement('div');
        this.paradeToggle.textContent = 'PARADE';
        this.paradeToggle.style.cssText = `
            font-size: 8px;
            padding: 1px 4px;
            border: 1px solid ${this.theme.accent};
            color: ${this.theme.accent};
            cursor: pointer;
            border-radius: 2px;
            display: none;
            opacity: ${this.waveformParadeMode ? 1 : 0.4};
        `;
        this.paradeToggle.onclick = () => {
            this.toggleParadeMode();
            this.paradeToggle.style.opacity = this.waveformParadeMode ? 1 : 0.4;
        };

        waveformHeader.appendChild(this.waveformLabel);
        waveformHeader.appendChild(this.paradeToggle);
        this.scopePanel.appendChild(waveformHeader);

        this.waveformCanvas = document.createElement('canvas');
        this.waveformCanvas.width = 256;
        this.waveformCanvas.height = 150; // Increased from 50px to broadcast standard
        this.waveformCanvas.style.cssText = 'width: 100%; height: 150px; display: none;';
        this.waveformCtx = this.waveformCanvas.getContext('2d');
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
        this.addButton('💾', (e) => this.showExportMenu(e), 'Export');
        this.addButton('↺', () => this.resetControls(), 'Reset');
        this.addButton('?', () => this.toggleHelp(), 'Keyboard Shortcuts (?)');
        this.addSep();

        // Run & Editor
        // this.runButton = this.addButton('▶', () => this.runWorkflow(), 'Run (Shift+Enter)');
        // this.runButton.style.color = '#4f4'; // Green hue

        // Removed separate Prompt button (v2.4: Integration into HUD complete)
        // this.promptButton = this.addButton('P', () => this.togglePromptPanel(), 'Prompt Editor (P)');
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
        this.addButton('FC', () => { this.falseColor = !this.falseColor; this.zebra = false; this.focusPeaking = false; this.showZdepth = false; this.gamutWarning = false; this.clippingMonitor = false; this.render(); }, 'False Color (E)');
        this.addButton('GW', () => { this.gamutWarning = !this.gamutWarning; this.clippingMonitor = false; this.falseColor = false; this.render(); }, 'Gamut Warning');
        this.addButton('CLP', () => { this.clippingMonitor = !this.clippingMonitor; this.gamutWarning = false; this.falseColor = false; this.render(); }, 'Clipping Monitor');
        this.addButton('Z', () => this.toggleZdepth(), 'Z-Depth / Zebra (Z)');
        this.focusPeakingBtn = this.addButton('FP', () => { this.focusPeaking = !this.focusPeaking; this.falseColor = false; this.zebra = false; this.render(); }, 'Focus Peaking (K)');
        this.gridBtn = this.addButton('▦', () => this.cycleGridMode(), 'Grid / Safe Areas (G)');
        this.addButton('📺', () => this.cycleSafeAreas(), 'Safe Areas (S)');
        this.loupeBtn = this.addButton('🔍', () => {
            this.showLoupe = !this.showLoupe;
            this.loupeBtn.classList.toggle('active', this.showLoupe);
            this.loupeBtn.style.color = this.showLoupe ? this.theme.accent : this.theme.textDim;
            this.renderOverlay();
        }, 'Pixel Loupe (Q)');
        if (this.showLoupe) {
            this.loupeBtn.classList.add('active');
            this.loupeBtn.style.color = this.theme.accent;
        }
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
        // v2.5: Global document listener for standard pipeline reliability
        this._docKeyHandler = (e) => {
            // Ignore if in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            // Only handle if viewer is the active component or in fullscreen
            if (this.isFullscreen) {
                this.handleKey(e);
                return;
            }

            // Simple heuristic: if container is visible and not hidden behind other Comfy nodes
            if (this.container.style.display !== 'none') {
                this.handleKey(e);
            }
        };
        document.addEventListener('keydown', this._docKeyHandler);
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
            case 'q':
                this.showLoupe = !this.showLoupe;
                if (this.loupeBtn) {
                    this.loupeBtn.classList.toggle('active', this.showLoupe);
                    this.loupeBtn.style.color = this.showLoupe ? this.theme.accent : this.theme.textDim;
                }
                this.renderOverlay();
                break;
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
        this._pushUndoDebounced();
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

    showExportMenu(e) {
        if (this.exportMenu) this.exportMenu.remove();

        const menu = document.createElement('div');
        this.exportMenu = menu;
        menu.style.cssText = `
            position: absolute;
            background: rgba(15, 15, 20, 0.95);
            border: 1px solid rgba(100, 110, 150, 0.4);
            border-radius: 6px;
            padding: 4px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 2px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.5);
            backdrop-filter: blur(10px);
        `;

        // Position menu above/near the button
        const rect = e.target.getBoundingClientRect();
        menu.style.left = rect.left + 'px';
        menu.style.top = (rect.bottom + 5) + 'px';

        const addOption = (label, icon, onClick, disabled = false) => {
            const opt = document.createElement('div');
            opt.innerHTML = `<span style="margin-right: 8px;">${icon}</span> ${label}`;
            opt.style.cssText = `
                padding: 6px 12px;
                color: ${disabled ? '#555' : '#ccc'};
                font-size: 11px;
                cursor: ${disabled ? 'default' : 'pointer'};
                border-radius: 4px;
                white-space: nowrap;
                transition: 0.2s;
            `;
            if (!disabled) {
                opt.onmouseenter = () => opt.style.background = 'rgba(255,255,255,0.08)';
                opt.onmouseleave = () => opt.style.background = 'transparent';
                opt.onclick = () => {
                    onClick();
                    menu.remove();
                };
            }
            menu.appendChild(opt);
        };

        // v2.3: Foolproof metadata retrieval - use the last known message data
        // This prevents metadata loss when 'this.image' is replaced by a canvas placeholder
        const imgData = (this.lastResult || []).find(d => d.frame === this.currentFrame && !d.is_compare && !d.is_zdepth);
        const hasEXR = !!(imgData && imgData.exr_filename);

        addOption('Save PNG (Result)', '🖼️', () => this.exportSnapshot('png'));
        addOption('Save EXR (Source)', '🏗️', () => this.exportSnapshot('exr'), !hasEXR);

        document.body.appendChild(menu);

        // Close on click outside
        const closeMenu = (ev) => {
            if (!menu.contains(ev.target) && ev.target !== e.target) {
                menu.remove();
                document.removeEventListener('mousedown', closeMenu);
            }
        };
        setTimeout(() => document.addEventListener('mousedown', closeMenu), 10);
    }

    exportSnapshot(format = 'png') {
        if (!this.image) return;

        if (format === 'exr') {
            const imgData = (this.lastResult || []).find(d => d.frame === this.currentFrame && !d.is_compare && !d.is_zdepth);
            if (imgData && imgData.exr_filename) {
                const url = api.apiURL(`/view?filename=${encodeURIComponent(imgData.exr_filename)}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&type=${encodeURIComponent(imgData.type || 'temp')}`);
                const link = document.createElement('a');
                link.href = url;
                link.download = imgData.exr_filename;
                link.click();
            }
            return;
        }

        // Default PNG Export
        const exp = document.createElement('canvas');
        exp.width = this.imageWidth;
        exp.height = this.imageHeight;
        const ctx = exp.getContext('2d');

        if (this.useWebGL && this.renderer && this.renderer.textures.image) {
            const prevW = this.glCanvas.width, prevH = this.glCanvas.height;
            this.glCanvas.width = this.imageWidth;
            this.glCanvas.height = this.imageHeight;
            this.renderer.render(this.lutIntensity || 1.0);
            ctx.drawImage(this.glCanvas, 0, 0);
            this.glCanvas.width = prevW;
            this.glCanvas.height = prevH;
        } else {
            this.renderImage(ctx, this.image);
        }
        this.annotations.forEach(a => this.drawAnnotation(ctx, a, 1));

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
            this.paradeToggle.style.display = this.showWaveform ? 'block' : 'none';
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
        if (!this.image || !this.renderer) return;

        // v2.5: GPU-Accelerated HDR Histogram
        const tex = this.renderer.textures.image;
        if (tex) {
            this.renderer.renderScope('histogram', this.histogramCanvas, tex, this.renderer.isLinearTexture);
        }
    }

    toggleParadeMode() {
        this.waveformParadeMode = !this.waveformParadeMode;
        if (this.showWaveform) {
            this.updateWaveform();
        }
    }

    updateWaveform() {
        if (!this.image || !this.renderer) return;

        // v2.5: GPU-Accelerated Waveform (32-bit HDR)
        const tex = this.renderer.textures.image;
        if (tex) {
            this.renderer.renderScope('waveform', this.waveformCanvas, tex, this.renderer.isLinearTexture, this.waveformParadeMode);
        }
    }


    updateVectorscope() {
        if (!this.image || !this.renderer) return;

        // v2.5: GPU-Accelerated Vectorscope
        const tex = this.renderer.textures.image;
        if (tex) {
            this.renderer.renderScope('vectorscope', this.vectorscopeCanvas, tex, this.renderer.isLinearTexture);

            // Draw Pro Overlays (Skin Tone Line, Targets) on top of GPU result
            const vCtx = this.vectorscopeCtx;
            const size = this.vectorscopeCanvas.width;
            const cx = size / 2, cy = size / 2, rad = size / 2 - 10;

            // Skin Tone Line (I-axis in YIQ, approx 123 deg)
            vCtx.strokeStyle = 'rgba(255, 120, 80, 0.4)';
            vCtx.lineWidth = 1.5;
            vCtx.setLineDash([4, 4]);
            const skinAng = (123 - 90) * Math.PI / 180;
            vCtx.beginPath();
            vCtx.moveTo(cx, cy);
            vCtx.lineTo(cx + Math.cos(skinAng) * rad, cy + Math.sin(skinAng) * rad);
            vCtx.stroke();
            vCtx.setLineDash([]);

            // Rec.709 Targets
            const targets = [
                { a: 103, c: '#f44', n: 'R' }, { a: 167, c: '#ff4', n: 'Y' },
                { a: 241, c: '#4f4', n: 'G' }, { a: 283, c: '#4ff', n: 'C' },
                { a: 347, c: '#44f', n: 'B' }, { a: 61, c: '#f4f', n: 'M' }
            ];
            targets.forEach(t => {
                const ang = (t.a - 90) * Math.PI / 180;
                vCtx.strokeStyle = t.c; vCtx.lineWidth = 1;
                vCtx.strokeRect(cx + Math.cos(ang) * rad * 0.75 - 3, cy + Math.sin(ang) * rad * 0.75 - 3, 6, 6);
            });
        }
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

            // v2.5: Exponential zoom for smoother response at all scales (Nuke/Resolve style)
            const sensitivity = 0.001;
            const factor = Math.exp(-e.deltaY * sensitivity);
            const newZoom = Math.max(0.01, Math.min(200, this.zoom * factor));

            this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
            this.panY = my - (my - this.panY) * (newZoom / this.zoom);
            this.zoom = newZoom;
            this.updateBottomBar();
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

        this.canvas.addEventListener('mouseleave', () => {
            this.infoLeft.innerHTML = '';
            // Don't clear infoRight as it contains static info
        });

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
        if (!this.image || !this.imageData) { this.infoLeft.innerHTML = ''; return; }
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

            let floatVals = '';
            let evVal = '';
            const luma_f = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 255;

            if (this.hdrData && this.hdrData.data) {
                const ch = this.hdrData.channels || 3;
                const hIdx = (imgY * this.imageWidth + imgX) * ch;
                const fr = this.hdrData.data[hIdx];
                const fg = ch > 1 ? this.hdrData.data[hIdx + 1] : fr;
                const fb = ch > 2 ? this.hdrData.data[hIdx + 2] : fr;
                floatVals = `<span style="color:${this.theme.accent}">F: ${(fr).toFixed(4)} ${(fg).toFixed(4)} ${(fb).toFixed(4)}</span> | `;

                const curLuma = fr * 0.2126 + fg * 0.7152 + fb * 0.0722;
                const stops = Math.log2(Math.max(1e-6, curLuma) / 0.18);
                evVal = `<span style="color:#d49dff">EV: ${stops > 0 ? '+' : ''}${stops.toFixed(1)}</span> | `;
            } else {
                const stops = Math.log2(Math.max(1e-6, luma_f) / 0.18);
                evVal = `<span style="color:#d49dff">EV: ${stops > 0 ? '+' : ''}${stops.toFixed(1)}</span> | `;
            }

            this.lastPixelColor = { r, g, b, a };
            const luma = (r * 0.2126 + g * 0.7152 + b * 0.0722).toFixed(0);
            const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;

            this.colorInfo.textContent = `RGB: ${r} ${g} ${b}`;

            // v2.5 Pro Probing UI
            this.infoLeft.innerHTML = `
                <span style="color:#888; margin-right:8px;">X:${imgX.toString().padStart(4, '0')} Y:${imgY.toString().padStart(4, '0')}</span>
                ${evVal}${floatVals}
                <span style="margin-right:8px; border-left:1px solid #333; padding-left:8px;">8b: ${r} ${g} ${b}</span>
                <span style="color:#777">Hx: ${hex} | L: ${luma}</span>
            `;

            // Draw pixel loupe on overlay
            if (this.showLoupe) {
                this.renderOverlay(); // Clear and redraw annotations/grid first
                this.drawLoupe(mx, my, imgX, imgY);
            }
        } else {
            this.infoLeft.innerHTML = '';
            this.colorInfo.textContent = 'RGB: — — —';
            this.lastPixelColor = null;
        }

        // Update fixed right info stats continuously
        const zoomPct = Math.round(this.zoom * 100);
        const depth = (this.hdrData && this.hdrData.data) ? '32b FP' : '8b INT';

        let indicators = '';
        if (this.gamutWarning) indicators += '<span style="color:#f0f; font-weight:bold">GW</span> ';
        if (this.clippingMonitor) indicators += '<span style="color:#f00; font-weight:bold">CLP</span> ';
        if (this.falseColor) indicators += '<span style="color:#fc0; font-weight:bold">FC</span> ';

        this.infoRight.innerHTML = `
            ${indicators}
            <span>${depth}</span>
            <span>CH: ${this.channel.toUpperCase()}</span>
            <span>RES: ${this.imageWidth}x${this.imageHeight}</span>
            <span>ZOOM: ${zoomPct}%</span>
            <span>FRM: ${this.currentFrame + 1}/${this.totalFrames}</span>
        `;

        // Toggle Legend visibility
        if (this.fcLegend) {
            this.fcLegend.style.display = this.falseColor ? 'flex' : 'none';
        }

    }

    updateBottomBar() {
        if (!this.infoRight) return;
        const zoomPct = Math.round((this.zoom || 1) * 100);
        const w = this.imageWidth || 0;
        const h = this.imageHeight || 0;
        this.infoRight.innerHTML = `
            <span>CH: ${(this.channel || 'rgb').toUpperCase()}</span>
            <span>RES: ${w}x${h}</span>
            <span>ZOOM: ${zoomPct}%</span>
            <span>FRM: ${(this.currentFrame || 0) + 1}/${this.totalFrames || 1}</span>
        `;
    }

    drawLoupe(mx, my, imgX, imgY) {
        if (!this.overlayCtx) return;
        const ctx = this.overlayCtx;
        const size = this.loupeSize;
        const mag = this.loupeMagnification;
        const numPixels = Math.floor(size / mag);
        const halfPixels = Math.floor(numPixels / 2);

        // Position loupe in corner opposite to cursor
        let lx = mx > this.canvas.width / 2 ? 10 : this.canvas.width - size - 10;
        let ly = my > this.canvas.height / 2 ? 10 : this.canvas.height - size - 10;

        // Draw loupe background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(lx - 1, ly - 1, size + 2, size + 2, 4);
        ctx.fill();
        ctx.stroke();

        // Draw magnified pixels
        for (let py = -halfPixels; py <= halfPixels; py++) {
            for (let px = -halfPixels; px <= halfPixels; px++) {
                const sx = imgX + px, sy = imgY + py;
                const dx = lx + (px + halfPixels) * mag;
                const dy = ly + (py + halfPixels) * mag;

                // Stop if we are outside the loupe box bounds
                if (px + halfPixels >= numPixels || py + halfPixels >= numPixels) continue;

                if (sx >= 0 && sx < this.imageWidth && sy >= 0 && sy < this.imageHeight) {
                    let r, g, b;
                    // v2.2: Read HDR float data for loupe when available
                    if (this.hdrData && this.hdrData.data) {
                        const channels = this.hdrData.shape ? (this.hdrData.shape[2] || 3) : (this.hdrData.channels || 3);
                        const hIdx = (sy * this.imageWidth + sx) * channels;

                        // Tonemap for display: simple Reinhard per-channel
                        const fr = this.hdrData.data[hIdx];
                        const fg = channels > 1 ? this.hdrData.data[hIdx + 1] : fr;
                        const fb = channels > 2 ? this.hdrData.data[hIdx + 2] : fr;

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
                    ctx.fillStyle = '#111';
                }

                ctx.fillRect(dx, dy, mag, mag);

                // Highlight center pixel
                if (px === 0 && py === 0) {
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(dx + 0.5, dy + 0.5, mag - 1, mag - 1);
                }
            }
        }

        // Draw border
        ctx.strokeStyle = this.theme.accent;
        ctx.lineWidth = 1;
        ctx.strokeRect(lx, ly, size, size);

        // Draw crosshair
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
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
        this.updateBottomBar();
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
            this.renderer.setGamutWarning(this.gamutWarning || false);
            this.renderer.setClippingMonitor(this.clippingMonitor || false);

            // v3.1: Masking (Power Windows)
            this.renderer.setMask(this.maskState);

            // v2.2: Channel isolation + focus peaking + display LUT on GPU
            const chMap = { 'rgb': 0, 'r': 1, 'g': 2, 'b': 3, 'luma': 4, 'a': 5 };
            this.renderer.setChannelMode(chMap[this.channel] || 0);
            this.renderer.setFocusPeaking(this.focusPeaking || false);
            const lutMap = {
                'None': 0,
                'sRGB (Display)': 1,
                'Rec.709 (Broadcast)': 2,
                'Filmic (Cinematic)': 3,
                'Log C3 (ARRI)': 4,
                'Log C4 (ARRI)': 11,
                'Canon Log': 12,
                'Fuji F-Log': 14,
                'Blackmagic Gen5': 15,
                'Linear to Log': 6,
                'Reinhard Tonemap': 8,
                'ACES Filmic': 9
            };
            this.renderer.setDisplayLutMode(lutMap[this.displayLut] || 0);

            // v2.3: Denoise & Depth Eval
            this.renderer.setDenoise(this.denoise || 0.0);
            this.renderer.setShowDepth(this.showZdepth || false);

            // DoF controls
            if (this.dofEnabled && this.renderer.textures.depth) {
                this.renderer.setDoFEnabled(true);
                this.renderer.setFocusDistance(this.focusDistance || 0.5);
                this.renderer.setAperture(this.aperture || 0.0);
                this.renderer.setBokehPhysics(this.bokehHighlightBias || 0.0, this.bokehSoapBubble || 0.0, this.bokehOpticalVig || 0.0);

                // v3.1: Optical shape
                this.renderer.setApertureShape(this.apertureBlades || 0, this.apertureRotation || 0.0, this.apertureAnamorphic || 1.0);

                this.renderer.setFrame(this.currentFrame || 0);
                this.renderer.setTime(performance.now() / 1000.0);
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

            this.updateBottomBar();
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

        this.updateBottomBar();
        this.renderOverlay();
        this._updateAnalysisIndicator();
    }

    // P1: Visual viewport border for active analysis modes
    _updateAnalysisIndicator() {
        let borderColor = 'transparent';
        let label = '';
        if (this.falseColor) { borderColor = '#FFD600'; label = 'FALSE COLOR'; }
        else if (this.zebra) { borderColor = '#FF4444'; label = 'ZEBRA'; }
        else if (this.focusPeaking) { borderColor = '#FF6600'; label = 'FOCUS PEAK'; }
        else if (this.showZdepth) { borderColor = '#00BBFF'; label = 'Z-DEPTH'; }
        else if (this.gamutWarning) { borderColor = '#FF00FF'; label = 'GAMUT WARN'; }
        else if (this.clippingMonitor) { borderColor = '#FF2222'; label = 'CLIPPING'; }

        this.canvasWrapper.style.boxShadow = borderColor !== 'transparent'
            ? `inset 0 0 0 2px ${borderColor}`
            : 'none';

        // Floating label in top-right of viewport
        if (!this._analysisLabel) {
            this._analysisLabel = document.createElement('div');
            this._analysisLabel.style.cssText = `
                position: absolute; top: 6px; right: 6px; z-index: 95;
                padding: 2px 8px; border-radius: 3px; font-size: 10px;
                font-family: ${this.theme?.mono || 'monospace'}; font-weight: 600;
                letter-spacing: 0.5px; pointer-events: none; transition: opacity 0.15s;
            `;
            this.canvasWrapper.appendChild(this._analysisLabel);
        }
        if (label) {
            this._analysisLabel.textContent = label;
            this._analysisLabel.style.background = borderColor;
            this._analysisLabel.style.color = '#000';
            this._analysisLabel.style.opacity = '1';
        } else {
            this._analysisLabel.style.opacity = '0';
        }

        // Auto-tonemap indicator (shows when HDR + no explicit display LUT)
        if (!this._autoTonemapBadge) {
            this._autoTonemapBadge = document.createElement('div');
            this._autoTonemapBadge.style.cssText = `
                position: absolute; top: 6px; left: 6px; z-index: 95;
                padding: 2px 8px; border-radius: 3px; font-size: 10px;
                font-family: ${this.theme?.mono || 'monospace'}; font-weight: 600;
                letter-spacing: 0.5px; pointer-events: none; transition: opacity 0.15s;
                background: rgba(255, 160, 0, 0.85); color: #000;
            `;
            this.canvasWrapper.appendChild(this._autoTonemapBadge);
        }
        const isAutoACES = this.renderer && this.renderer.isLinearTexture &&
            (this.displayLut === 'None' || !this.displayLut);
        if (isAutoACES) {
            this._autoTonemapBadge.textContent = 'ACES (Auto)';
            this._autoTonemapBadge.style.opacity = '1';
        } else {
            this._autoTonemapBadge.style.opacity = '0';
        }
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
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          METADATA OVERLAY
    // ═══════════════════════════════════════════════════════════════════════════

    createMetadataOverlay() {
        if (this.metadataOverlay) return; // Already created — prevent duplicates

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

        this.container.addEventListener('wheel', () => {
            // Update Zoom display on scroll
            requestAnimationFrame(() => this.updateInfo());
        }, { passive: true });
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

        const oldText = this.runButton ? this.runButton.innerHTML : '▶ RUN';
        if (this.runButton) this.runButton.innerHTML = '⏳ QUEUING...';

        app.queuePrompt(0).then(() => {
            this.isQueueing = false;
            if (this.runButton) this.runButton.innerHTML = oldText;
        }).catch(() => {
            this.isQueueing = false;
            if (this.runButton) {
                this.runButton.innerHTML = '❌ ERROR';
                setTimeout(() => this.runButton.innerHTML = oldText, 1500);
            }
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
            if (this.promptButton) {
                this.promptButton.classList.add('active');
                this.promptButton.style.background = this.theme.accent;
                this.promptButton.style.color = '#fff';
            }
            // v2.4: Integrate with HUD
            this.showControls = true;
            if (this.controlsPanel) {
                this.controlsPanel.style.display = 'flex';
                this.controlsPanel.style.opacity = '1';
                this.controlsPanel.style.transform = 'translateX(-50%) translateY(0)';
                this.controlsPanel.style.pointerEvents = 'auto';
            }
            if (this.controlsToggle) {
                this.controlsToggle.style.color = this.theme.accent;
            }

            // Switch to prompt tab
            const promptTab = this._hudTabs?.find(t => t.id === 'prompt');
            if (promptTab) {
                promptTab.click();
            }
        } else {
            if (this.promptButton) {
                this.promptButton.classList.remove('active');
                this.promptButton.style.background = '';
                this.promptButton.style.color = this.theme.textDim;
            }
            // If we were toggling prompt specifically, maybe we hide HUD?
            // Actually, stay consistent: if promptPanel is false, but showControls is true, keep HUD open but prompt tab is still active.
            // But usually 'P' toggles the prompt UI. If prompt is now in HUD, 'P' should toggle HUD.
            this.toggleControls(); // Toggle entire HUD for 'P' key consistency
            if (this.promptPanel) this.promptPanel.remove();
            this.promptPanel = null;
        }
    }

    renderPromptTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; gap: 10px; padding: 10px; max-height: 380px; overflow-y: auto; color: #fff;';
        const t = this.theme;

        // v2.4: Move Run Button to top of prompt tab
        const runBtnWrapper = document.createElement('div');
        runBtnWrapper.style.cssText = 'padding: 5px 0 15px 0; display: flex; justify-content: center; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 5px;';

        const runBtn = document.createElement('button');
        runBtn.innerHTML = '▶ RUN WORKFLOW';
        runBtn.style.cssText = `
        background: #1a4a1a;
        color: #4f4;
        border: 1px solid #2a6a2a;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 12px;
        font-weight: 800;
        cursor: pointer;
        letter-spacing: 1px;
        transition: all 0.2s;
        width: 100%;
        font-family: ${t.font};
    `;
        runBtn.onmouseover = () => {
            runBtn.style.background = '#226222';
            runBtn.style.boxShadow = '0 0 15px rgba(79, 255, 79, 0.2)';
        };
        runBtn.onmouseout = () => {
            runBtn.style.background = '#1a4a1a';
            runBtn.style.boxShadow = 'none';
        };
        this.runButton = runBtn;
        runBtn.onclick = () => this.runWorkflow();

        runBtnWrapper.appendChild(runBtn);
        container.appendChild(runBtnWrapper);

        // Get Generation Settings Nodes
        const nodes = app.graph._nodes.filter(n => {
            const nodeType = n.type || "";
            const comfyClass = n.comfyClass || nodeType;

            const isEncoder = comfyClass.includes("CinematicPromptEncoder");
            const isUnet = comfyClass === "CheckpointLoaderSimple" || comfyClass === "UNETLoader" || comfyClass.includes("DualCLIPLoader");
            const isLatent = comfyClass === "EmptyLatentImage" || comfyClass === "EmptySD3LatentImage";

            return isEncoder || isUnet || isLatent;
        });

        console.log("[Radiance] Found", nodes.length, "Generation settings nodes.");

        if (nodes.length === 0) {
            const msg = document.createElement('div');
            msg.textContent = "No Generation Nodes found.";
            msg.style.color = t.textDim;
            msg.style.fontSize = "11px";
            container.appendChild(msg);
            return;
        }

        nodes.forEach(node => {
            const wrapper = document.createElement('div');
            wrapper.style.cssText = `display: flex; flex-direction: column; gap: 8px; padding: 10px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px;`;

            const label = document.createElement('div');
            label.textContent = (node.title || node.type).toUpperCase();
            label.style.cssText = `color: ${t.accent}; font-size: 10px; font-weight: 800; cursor: pointer; letter-spacing: 0.5px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px; margin-bottom: 4px;`;
            label.onclick = () => { app.canvas.centerOnNode(node); app.canvas.selectNode(node); };
            wrapper.appendChild(label);

            if (node.widgets) {
                // Main text area (base_prompt)
                const mainPrompt = node.widgets.find(w => w.name === 'base_prompt');
                if (mainPrompt) {
                    const pc = document.createElement('div');
                    pc.style.cssText = 'display: flex; flex-direction: column; gap: 2px;';
                    const textarea = document.createElement('textarea');
                    textarea.value = mainPrompt.value || "";
                    textarea.placeholder = "Enter your base cinematic prompt...";
                    textarea.style.cssText = `width: 100%; height: 80px; background: rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; color: ${t.text}; font-size: 12px; padding: 8px; resize: vertical; font-family: ${t.mono}; outline: none; transition: border-color 0.2s;`;
                    textarea.onfocus = () => textarea.style.borderColor = t.accent;
                    textarea.onblur = () => textarea.style.borderColor = 'rgba(255,255,255,0.1)';
                    textarea.oninput = (e) => {
                        mainPrompt.value = e.target.value;
                        if (mainPrompt.callback) mainPrompt.callback(mainPrompt.value);
                        node.setDirtyCanvas(true);
                    };
                    pc.appendChild(textarea);
                    wrapper.appendChild(pc);
                }

                // Grid for dropdowns and numbers
                const grid = document.createElement('div');
                grid.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr; gap: 10px;';

                node.widgets.forEach(w => {
                    if (w.name === 'base_prompt' || w.name === 'prompt_preview' || w.type === 'converted-widget' || w.name === '_temp') return;

                    const wWrap = document.createElement('div');
                    wWrap.style.cssText = 'display: flex; flex-direction: column; gap: 4px;';

                    const wl = document.createElement('div');
                    wl.textContent = w.name.replace(/_/g, ' ').toUpperCase();
                    wl.style.cssText = `color: ${t.textDim}; font-size: 9px; font-weight: 600; opacity: 0.8;`;
                    wWrap.appendChild(wl);

                    let input;
                    if (w.type === 'combo') {
                        input = document.createElement('select');
                        input.style.cssText = `width: 100%; background: #0a0a0f; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; color: ${t.text}; font-size: 11px; padding: 4px; cursor: pointer; outline: none;`;
                        if (w.options && w.options.values) {
                            w.options.values.forEach(v => {
                                const opt = document.createElement('option');
                                opt.value = v; opt.textContent = v;
                                input.appendChild(opt);
                            });
                        }
                        input.value = w.value;
                        input.onchange = (e) => {
                            w.value = e.target.value;
                            if (w.callback) w.callback(w.value);
                            node.setDirtyCanvas(true);
                        };
                    } else if (w.type === 'number' || typeof w.value === 'number') {
                        input = document.createElement('input');
                        input.type = 'number';
                        input.value = w.value;
                        if (w.options) {
                            if (w.options.min !== undefined) input.min = w.options.min;
                            if (w.options.max !== undefined) input.max = w.options.max;
                            if (w.options.step !== undefined) input.step = w.options.step;
                        }
                        input.style.cssText = `width: 100%; background: #0a0a0f; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; color: ${t.text}; font-size: 11px; padding: 4px; outline: none;`;
                        input.oninput = (e) => {
                            let val = parseFloat(e.target.value);
                            if (w.options) {
                                if (w.options.min !== undefined) val = Math.max(w.options.min, val);
                                if (w.options.max !== undefined) val = Math.min(w.options.max, val);
                            }
                            w.value = val;
                            if (w.callback) w.callback(w.value);
                            node.setDirtyCanvas(true);
                        };
                    } else if (w.type === 'toggle' || typeof w.value === 'boolean') {
                        const tr = document.createElement('div');
                        tr.style.cssText = 'display: flex; align-items: center; gap: 8px;';
                        const ck = document.createElement('input');
                        ck.type = 'checkbox'; ck.checked = w.value;
                        ck.onchange = (e) => {
                            w.value = e.target.checked;
                            if (w.callback) w.callback(w.value);
                            node.setDirtyCanvas(true);
                        };
                        tr.appendChild(ck);
                        const cl = document.createElement('span'); cl.textContent = w.value ? 'ON' : 'OFF'; cl.style.fontSize = '10px';
                        ck.addEventListener('change', () => cl.textContent = ck.checked ? 'ON' : 'OFF');
                        tr.appendChild(cl);
                        input = tr;
                    } else {
                        input = document.createElement('textarea');
                        input.value = w.value || "";
                        input.style.cssText = `width: 100%; height: 30px; background: rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; color: ${t.text}; font-size: 11px; padding: 4px; resize: vertical; font-family: ${t.mono}; outline: none;`;
                        input.oninput = (e) => {
                            w.value = e.target.value;
                            if (w.callback) w.callback(w.value);
                            node.setDirtyCanvas(true);
                        };
                    }

                    if (input) wWrap.appendChild(input);
                    grid.appendChild(wWrap);
                });
                wrapper.appendChild(grid);
            }
            container.appendChild(wrapper);
        });

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
            background: rgba(10, 10, 14, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 12px;
            z-index: 100;
            display: flex;
            flex-direction: column;
            gap: 12px;
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            box-shadow: 0 16px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05) inset;
            transition: opacity 0.4s cubic-bezier(0.2, 0.8, 0.2, 1.0), transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1.0);
            width: ${this.hudPanelWidth || 640}px;
            height: ${this.hudPanelHeight || 380}px;
            min-width: 400px;
            min-height: 240px;
            max-width: 90vw;
            max-height: 80vh;
            bottom: 40px;
            left: 50%;
            transform: translateX(-50%);
            overflow: auto;
            resize: both;
            font-family: ${t.font};
            opacity: 1;
            pointer-events: auto;
        `;

        // Observe resize and persist dimensions
        const hudResizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const newWidth = Math.round(entry.contentRect.width);
                const newHeight = Math.round(entry.contentRect.height);
                this.hudPanelWidth = newWidth;
                this.hudPanelHeight = newHeight;
                localStorage.setItem('radiance_hud_width', newWidth);
                localStorage.setItem('radiance_hud_height', newHeight);
            }
        });
        hudResizeObserver.observe(this.controlsPanel);

        // Docked HUD uses full width by default

        // 1. Tabs Header (No longer draggable)
        const tabsHeader = document.createElement('div');
        tabsHeader.style.cssText = `
            display: flex; 
            gap: 4px; 
            padding: 4px 8px 10px 8px; 
            border-bottom: 1px solid rgba(255,255,255,0.08); 
            margin-bottom: 2px;
            user-select: none;
        `;

        const activeTabStyle = `background: rgba(255,255,255,0.1); color: ${t.text}; border-bottom: 2px solid ${t.accent}`;
        const inactiveTabStyle = `background: transparent; color: ${t.textDim}; border-bottom: 2px solid transparent`;
        const baseTabStyle = `flex: 1; text-align: center; padding: 6px 0; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; cursor: pointer; border-radius: 4px; transition: all 0.2s;`;

        let activeTab = 'primaries';
        const tabContentContainer = document.createElement('div');
        this.tabContentContainer = tabContentContainer;

        const tabs = [
            { id: 'prompt', label: 'PROMPT' }, // v2.4: Integrated Prompt machine
            { id: 'primaries', label: 'PRIMARIES' },
            { id: 'curves', label: 'CURVES' },
            { id: 'film', label: 'FILM' },
            { id: 'lens', label: 'LENS' },
            { id: 'qualifiers', label: 'QUALIFIER' },
            { id: 'masks', label: 'MASKS' },
            { id: 'scopes', label: 'SCOPES' },
            { id: 'view', label: 'VIEW' }
        ];
        this._hudTabs = []; // Save references

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
                this._hudTabs.push(btn);
                tabsHeader.appendChild(btn);
            });
        };

        this.controlsPanel.appendChild(tabsHeader);
        this.controlsPanel.appendChild(tabContentContainer);

        // 2. Tab Content Renderer
        const renderContent = () => {
            tabContentContainer.innerHTML = '';
            tabContentContainer.style.cssText = 'min-height: 140px; display: flex; flex-direction: column; justify-content: center;';

            if (activeTab === 'prompt') {
                this.renderPromptTab(tabContentContainer);
            } else if (activeTab === 'primaries') {
                this.renderPrimariesTab(tabContentContainer);
            } else if (activeTab === 'curves') {
                this.renderCurvesTab(tabContentContainer);
            } else if (activeTab === 'film') {
                this.renderEffectsTab(tabContentContainer);
            } else if (activeTab === 'lens') {
                this.renderLensTab(tabContentContainer);
            } else if (activeTab === 'qualifiers') {
                this.renderQualifiersTab(tabContentContainer);
            } else if (activeTab === 'masks') {
                this.renderMasksTab(tabContentContainer);
            } else if (activeTab === 'scopes') {
                this.renderScopesTab(tabContentContainer);
            } else if (activeTab === 'view') {
                this.renderViewTab(tabContentContainer);
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

                // --- Printer Lights (1-4) ---
                if (!e.ctrlKey && !e.metaKey && !e.altKey) {
                    const step = e.shiftKey ? -0.01 : 0.01;
                    let changed = false;

                    if (e.key === '1') { // Red
                        this.offset[0] += step; changed = true;
                    } else if (e.key === '2') { // Green
                        this.offset[1] += step; changed = true;
                    } else if (e.key === '3') { // Blue
                        this.offset[2] += step; changed = true;
                    } else if (e.key === '4') { // Master
                        this.offset[0] += step; this.offset[1] += step; this.offset[2] += step;
                        changed = true;
                    }

                    if (changed) {
                        e.preventDefault();
                        this._pushUndoDebounced();
                        if (this.renderer) this.renderer.setOffset(this.offset[0], this.offset[1], this.offset[2]);
                        this.render();
                        if (this._lastRenderContent) this._lastRenderContent(); // Refresh knobs
                    }
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
            bottom: 300px;
            left: 50%;
            transform: translateX(-50%);
            display: ${this.totalFrames > 1 ? 'flex' : 'none'};
            align-items: center;
            justify-content: center;
            gap: 12px;
            background: rgba(10, 10, 12, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 6px 16px;
            backdrop-filter: blur(8px);
            z-index: 101;
            transition: opacity 0.4s, transform 0.4s;
            opacity: 1;
            pointer-events: auto;
        `;

        // The controlsPanel and transportPanel will be appended to canvasWrapper to float over image

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

        // Float these over the canvas
        this.canvasWrapper.appendChild(this.transportPanel);
        this.canvasWrapper.appendChild(this.controlsPanel);

        // Relative ordering for info bar
        if (this.bottomInfoBar) {
            this.container.appendChild(this.bottomInfoBar);
        }
    }

    renderPrimariesTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; gap: 12px; padding: 10px 4px;';

        const createMini = (lbl, min, max, val, step, cb) => {
            const k = this.createKnob(lbl, min, max, val, step, cb);
            k.style.transform = 'scale(0.9)';
            return k;
        };

        // ═════════════════════════════════════════════════════════════════════
        // 1. TOP BAR: Exp | Temp | Tint | Contrast | Pivot | Mid/Detail
        // ═════════════════════════════════════════════════════════════════════
        const topBar = document.createElement('div');
        topBar.style.cssText = 'display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; justify-items: center;';

        // Auto Balance Picker
        const balanceBtn = document.createElement('div');
        balanceBtn.innerHTML = '◎';
        balanceBtn.title = 'Auto White Balance';
        balanceBtn.style.cssText = 'width: 38px; height: 38px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: 0.2s; margin-top: 4px; font-size: 16px;';
        balanceBtn.onmouseenter = () => balanceBtn.style.background = 'rgba(255,255,255,0.12)';
        balanceBtn.onmouseleave = () => balanceBtn.style.background = 'rgba(255,255,255,0.05)';
        balanceBtn.onclick = () => this.activateBalancePicker();
        topBar.appendChild(balanceBtn);

        // Exposure
        topBar.appendChild(createMini('EXP', -10.0, 10.0, this.exposure || 0.0, 0.1, v => {
            this.exposure = v;
            if (this.renderer) this.renderer.setExposure(v);
            this.render();
        }));

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

        const btnStyle = `background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); color: #aaa; padding: 4px 8px; border - radius: 4px; font - size: 10px; cursor: pointer; transition: background 0.15s; `;

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
        addPreset('Kodachrome 64', 0.08, 1.1, 0.1);
        addPreset('Ektachrome 100', 0.10, 1.2, 0.15);
        addPreset('Technicolor', 0.22, 2.2, 0.35);
        addPreset('Fuji Eterna', 0.10, 1.2, 0.08);
        addPreset('CineStill 800T', 0.18, 1.8, 0.2);
        addPreset('Tri-X 400', 0.25, 2.0, 0.0);
        addPreset('Classic Noir', 0.32, 2.8, 0.0);
        addPreset('Push (+2 Stop)', 0.45, 3.2, 0.25);
        addPreset('Expired 35mm', 0.38, 2.6, 0.45);
        addPreset('Vintage 1920s', 0.65, 4.0, 0.1);
        addPreset('Polaroid 600', 0.28, 3.0, 0.2);
        addPreset('Digital Noise', 0.08, 1.0, 0.0);

        container.appendChild(presets);
    }

    renderLensTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; gap: 10px; padding: 10px; max-height: 340px; overflow-y: auto;';

        // 1. Focus & DoF Top Row
        const topRow = document.createElement('div');
        topRow.style.cssText = 'display: flex; gap: 10px; align-items: center; margin-bottom: 5px; justify-content: space-between;';

        const dofLeft = document.createElement('div');
        dofLeft.style.cssText = 'display: flex; align-items: center; gap: 8px;';

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
        dofLeft.appendChild(dofCheck);

        // Pick Focus Tool
        const pickBtn = document.createElement('button');
        pickBtn.textContent = '◎ Pick Focus';
        pickBtn.style.cssText = 'background: #333; color: #ccc; border: 1px solid #555; padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 11px;';
        pickBtn.onclick = () => this.activateFocusPicker(() => {
            // Update the HUD slider if it exists, though knobs are reactive
            this.renderLensTab(this.tabContentContainer);
        });
        dofLeft.appendChild(pickBtn);

        topRow.appendChild(dofLeft);
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

        const presetBtnStyle = `background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); color: #aaa; padding: 4px 10px; border - radius: 4px; font - size: 10px; cursor: pointer; transition: background 0.15s; `;
        const presetBtnActiveStyle = `background: rgba(100, 140, 255, 0.2); border: 1px solid rgba(100, 140, 255, 0.4); color: #9cf; `;

        const bokehPresets = [
            { label: '● Circle', blades: 0, angle: 0 },
            { label: '⬠ Pent', blades: 5, angle: 0 },
            { label: '⬡ Hex', blades: 6, angle: 0 },
            { label: '⬡ Oct', blades: 8, angle: 0 },
        ];

        const signaturePresets = [
            { label: 'Zeiss MP', blades: 0, angle: 0, anamorphic: 1.0, distort: 0.0, fringe: 0.0, halation: 0.05, diffusion: 0.0 },
            { label: 'Cooke S4', blades: 8, angle: 22, anamorphic: 1.0, distort: 0.04, fringe: 0.35, halation: 0.15, diffusion: 0.2 },
            { label: 'Anamorphic', blades: 0, angle: 0, anamorphic: 2.0, distort: 0.18, fringe: 0.75, halation: 0.1, diffusion: 0.15 },
            { label: 'Petzval', blades: 0, angle: 0, anamorphic: 1.0, distort: -0.1, fringe: 0.0, opticalVig: 0.85, highlight: 1.5 },
            { label: 'Dreamy', blades: 0, angle: 0, anamorphic: 1.0, distort: 0.0, fringe: 0.2, halation: 0.4, diffusion: 0.6 }
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

        // ─── Lens Signature Row ──────────────────────────
        const sigLabel = document.createElement('div');
        sigLabel.style.cssText = 'color: #888; font-size: 10px; margin-bottom: 6px; text-transform: uppercase; margin-top: 8px;';
        sigLabel.textContent = 'Lens Signatures';
        shapeGroup.appendChild(sigLabel);

        const sigRow = document.createElement('div');
        sigRow.style.cssText = 'display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap;';

        signaturePresets.forEach(p => {
            const btn = document.createElement('div');
            btn.textContent = p.label;
            btn.style.cssText = presetBtnStyle;
            btn.onmouseenter = () => btn.style.background = 'rgba(255,255,255,0.12)';
            btn.onmouseleave = () => btn.style.background = 'rgba(255,255,255,0.05)';
            btn.onclick = () => {
                this.apertureBlades = p.blades;
                this.apertureRotation = p.angle;
                this.apertureAnamorphic = p.anamorphic || 1.0;
                this.lensDistortion = p.distort || 0.0;
                this.lensFringe = p.fringe || 0.0;
                this.halation = p.halation || 0.0;
                this.diffusion = p.diffusion || 0.0;
                this.bokehOpticalVig = p.opticalVig || 0.0;
                this.bokehHighlightBias = p.highlight || 0.0;

                if (this.renderer) {
                    this.renderer.setApertureShape(p.blades, p.angle, this.apertureAnamorphic);
                    this.renderer.setLensDistortion(this.lensDistortion, this.lensFringe);
                    this.renderer.setHalation(this.halation);
                    this.renderer.setDiffusion(this.diffusion);
                    this.renderer.setBokehPhysics(this.bokehHighlightBias, this.bokehSoapBubble || 0.0, this.bokehOpticalVig);
                }
                this.render();
                this.renderLensTab(this.tabContentContainer);
            };
            sigRow.appendChild(btn);
        });
        shapeGroup.appendChild(sigRow);

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

        const diffPresets = [
            { label: 'Pro-Mist 1/4', bloom: 0.15, halation: 0.1, diffusion: 0.2 },
            { label: 'Pro-Mist 1/2', bloom: 0.25, halation: 0.15, diffusion: 0.35 },
            { label: 'Glimmerglass', bloom: 0.4, halation: 0.05, diffusion: 0.15 },
            { label: 'H.Black Magic', bloom: 0.2, halation: 0.3, diffusion: 0.4 },
        ];

        const dfRow = document.createElement('div');
        dfRow.style.cssText = 'display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap;';
        diffPresets.forEach(p => {
            const btn = document.createElement('button');
            btn.textContent = p.label;
            btn.style.cssText = presetBtnStyle;
            btn.onmouseenter = () => btn.style.background = 'rgba(255,255,255,0.12)';
            btn.onmouseleave = () => btn.style.background = 'rgba(255,255,255,0.05)';
            btn.onclick = () => {
                this.bloom = p.bloom;
                this.halation = p.halation;
                this.diffusion = p.diffusion;
                if (this.renderer) {
                    this.renderer.setBloom(p.bloom);
                    this.renderer.setHalation(p.halation);
                    this.renderer.setDiffusion(p.diffusion);
                }
                this.render();
                this.renderLensTab(this.tabContentContainer);
            };
            dfRow.appendChild(btn);
        });
        fxGroup.appendChild(dfRow);

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

        // 6. Bokeh Physics
        const bokehGroup = document.createElement('div');
        bokehGroup.style.marginTop = '10px';
        bokehGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 4px; text-transform: uppercase;">Bokeh Physics</div>';

        const bokehGrid = document.createElement('div');
        bokehGrid.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;';

        bokehGrid.appendChild(this.createKnob('HIGHLIGHT', 0.0, 5.0, this.bokehHighlightBias || 0.0, 0.1, v => {
            this.bokehHighlightBias = v;
            if (this.renderer) this.renderer.setBokehPhysics(this.bokehHighlightBias, this.bokehSoapBubble || 0.0, this.bokehOpticalVig || 0.0);
            this.render();
        }));

        bokehGrid.appendChild(this.createKnob('SOAP BBL', 0.0, 2.0, this.bokehSoapBubble || 0.0, 0.05, v => {
            this.bokehSoapBubble = v;
            if (this.renderer) this.renderer.setBokehPhysics(this.bokehHighlightBias || 0.0, this.bokehSoapBubble, this.bokehOpticalVig || 0.0);
            this.render();
        }));

        bokehGrid.appendChild(this.createKnob('OPTIC VIG', 0.0, 1.0, this.bokehOpticalVig || 0.0, 0.05, v => {
            this.bokehOpticalVig = v;
            if (this.renderer) this.renderer.setBokehPhysics(this.bokehHighlightBias || 0.0, this.bokehSoapBubble || 0.0, this.bokehOpticalVig);
            this.render();
        }));

        bokehGroup.appendChild(bokehGrid);
        container.appendChild(bokehGroup);
    }

    renderCurvesTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; align-items: stretch; gap: 8px; padding: 10px; height: 100%; overflow: hidden;';

        // 1. Create Editor Container
        const editorContainer = document.createElement('div');
        editorContainer.style.cssText = 'position: relative; flex: 1; min-height: 200px; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;';
        container.appendChild(editorContainer);

        // 2. Initialize Curve Editor
        if (!this.curveEditor) {
            this.curveEditor = new RadianceCurveEditor(280, 280, this.theme, (data) => {
                if (this.renderer) {
                    this.renderer.updateCurveLut(data);
                    this.renderer.setCurveMix(this.curveMix !== undefined ? this.curveMix : 1.0);
                    this.render();
                }
            });
            if (this.image) this.curveEditor.updateHistogram(this.image);
        }

        editorContainer.appendChild(this.curveEditor.canvas);

        // Observe resize
        const curveResizeObs = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    this.curveEditor.resize(width, height);
                }
            }
        });
        curveResizeObs.observe(editorContainer);

        // 3. Channel Selectors (top-left overlay)
        const channels = document.createElement('div');
        channels.style.cssText = 'position: absolute; top: 10px; left: 36px; display: flex; gap: 4px; align-items: center;';

        ['RGB', 'R', 'G', 'B'].forEach(ch => {
            const btn = document.createElement('div');
            btn.textContent = ch;
            const isActive = this.curveEditor.activeChannel === ch;
            const color = ch === 'R' ? '#ff4d4d' : ch === 'G' ? '#4dff4d' : ch === 'B' ? '#4d4dff' : '#ffffff';

            btn.style.cssText = `
                width: 32px; height: 20px;
                background: ${isActive ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.6)'};
                color: ${isActive ? '#fff' : color};
                border: 1px solid ${isActive ? color : 'rgba(255,255,255,0.1)'};
                border-radius: 3px;
                display: flex; align-items: center; justify-content: center;
                font-size: 10px; font-weight: bold; cursor: pointer;
                transition: all 0.15s;
                text-shadow: ${isActive ? `0 0 5px ${color}` : 'none'};
            `;
            btn.onmouseenter = () => { if (!isActive) btn.style.background = 'rgba(255,255,255,0.1)'; };
            btn.onmouseleave = () => { if (!isActive) btn.style.background = 'rgba(0,0,0,0.6)'; };
            btn.onclick = () => {
                this.curveEditor.setActiveChannel(ch);
                this._lastRenderContent();
            };
            channels.appendChild(btn);
        });

        // Copy to All Button
        const copyBtn = document.createElement('div');
        copyBtn.textContent = '📋';
        copyBtn.title = 'Copy RGB to All Channels';
        copyBtn.style.cssText = 'width: 28px; height: 20px; background: rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 3px; display: flex; align-items: center; justify-content: center; font-size: 10px; cursor: pointer; transition: all 0.15s; margin-left: 4px;';
        copyBtn.onmouseenter = () => copyBtn.style.background = 'rgba(255,255,255,0.1)';
        copyBtn.onmouseleave = () => copyBtn.style.background = 'rgba(0,0,0,0.6)';
        copyBtn.onclick = () => {
            this.curveEditor.copyRGBToAll();
            this._lastRenderContent();
        };
        channels.appendChild(copyBtn);

        editorContainer.appendChild(channels);

        // 4. Reset & Presets Group (top-right overlay)
        const topButtons = document.createElement('div');
        topButtons.style.cssText = 'position: absolute; top: 10px; right: 10px; display: flex; gap: 6px; align-items: center;';

        // Presets Dropdown
        const presetsSelect = document.createElement('select');
        presetsSelect.style.cssText = 'background: rgba(0,0,0,0.85); color: #ccc; border: 1px solid rgba(255,255,255,0.15); border-radius: 3px; font-size: 10px; padding: 2px 4px; outline: none; cursor: pointer; height: 20px;';

        ['Presets...', 'Punchy', 'High Contrast', 'Flat / Log', 'Shadow Lift', 'S-Curve', 'Film Print', 'Bleach Bypass', 'Cross Process'].forEach(p => {
            const opt = document.createElement('option');
            opt.value = p === 'Presets...' ? '' : p;
            opt.textContent = p;
            presetsSelect.appendChild(opt);
        });
        presetsSelect.onchange = (e) => {
            if (e.target.value) {
                this.curveEditor.applyIndustryPreset(e.target.value);
                e.target.value = '';
                this._lastRenderContent();
            }
        };
        topButtons.appendChild(presetsSelect);

        const resetGroup = document.createElement('div');
        resetGroup.style.cssText = 'display: flex; gap: 4px;';

        const resetChBtn = document.createElement('div');
        resetChBtn.textContent = '↺';
        resetChBtn.title = 'Reset Active Channel';
        resetChBtn.style.cssText = 'width: 24px; height: 20px; background: rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 3px; display: flex; align-items: center; justify-content: center; color: #888; cursor: pointer; font-size: 14px; transition: all 0.15s;';
        resetChBtn.onmouseenter = () => { resetChBtn.style.color = '#fff'; resetChBtn.style.background = 'rgba(255,255,255,0.1)'; };
        resetChBtn.onmouseleave = () => { resetChBtn.style.color = '#888'; resetChBtn.style.background = 'rgba(0,0,0,0.6)'; };
        resetChBtn.onclick = () => { this.curveEditor.resetActiveChannel(); this._lastRenderContent(); };
        resetGroup.appendChild(resetChBtn);

        const resetAllBtn = document.createElement('div');
        resetAllBtn.textContent = '⟲';
        resetAllBtn.title = 'Reset All Channels';
        resetAllBtn.style.cssText = 'width: 24px; height: 20px; background: rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 3px; display: flex; align-items: center; justify-content: center; color: #888; cursor: pointer; font-size: 14px; transition: all 0.15s;';
        resetAllBtn.onmouseenter = () => { resetAllBtn.style.color = '#ff4d4d'; resetAllBtn.style.background = 'rgba(255,0,0,0.1)'; };
        resetAllBtn.onmouseleave = () => { resetAllBtn.style.color = '#888'; resetAllBtn.style.background = 'rgba(0,0,0,0.6)'; };
        resetAllBtn.onclick = () => {
            this.curveEditor.resetAll();
            this.curveMix = 1.0;
            if (this.renderer) this.renderer.setCurveMix(1.0);
            this._lastRenderContent();
        };
        resetGroup.appendChild(resetAllBtn);

        topButtons.appendChild(resetGroup);
        editorContainer.appendChild(topButtons);

        // 5. Mix Slider
        const mixRow = document.createElement('div');
        mixRow.style.cssText = 'display: flex; align-items: center; gap: 10px; padding: 4px 5px; background: rgba(0,0,0,0.25); border-radius: 4px;';

        const mixLabel = document.createElement('div');
        mixLabel.style.cssText = `color: ${this.theme.textDim}; font-size: 10px; font-weight: 600; text-transform: uppercase; min-width: 35px;`;
        mixLabel.textContent = 'MIX';
        mixRow.appendChild(mixLabel);

        const mixSlider = document.createElement('input');
        mixSlider.type = 'range';
        mixSlider.min = '0'; mixSlider.max = '100'; mixSlider.step = '1';
        mixSlider.value = String(Math.round((this.curveMix !== undefined ? this.curveMix : 1.0) * 100));
        mixSlider.style.cssText = 'flex: 1; accent-color: #00a8ff; height: 4px; cursor: pointer;';
        mixSlider.oninput = (e) => {
            this.curveMix = parseInt(e.target.value) / 100;
            if (this.renderer) this.renderer.setCurveMix(this.curveMix);
            mixValue.textContent = e.target.value + '%';
            this.render();
        };
        mixRow.appendChild(mixSlider);

        const mixValue = document.createElement('div');
        mixValue.style.cssText = 'color: #aaa; font-size: 10px; min-width: 32px; text-align: right; font-family: ' + this.theme.mono + ';';
        mixValue.textContent = Math.round((this.curveMix !== undefined ? this.curveMix : 1.0) * 100) + '%';
        mixRow.appendChild(mixValue);

        container.appendChild(mixRow);

        // 6. Levels Controls
        const levelsRow = document.createElement('div');
        levelsRow.style.cssText = 'display: flex; flex-direction: column; gap: 4px; padding: 10px; background: rgba(0,0,0,0.25); border-radius: 4px; margin-top: 4px;';

        const createLevelSlider = (label, min, max, val, setter) => {
            const row = document.createElement('div');
            row.style.cssText = 'display: flex; align-items: center; gap: 10px;';
            const lbl = document.createElement('div');
            lbl.style.cssText = `color: ${this.theme.textDim}; font-size: 10px; font-weight: 600; text-transform: uppercase; min-width: 60px;`;
            lbl.textContent = label;
            row.appendChild(lbl);

            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = min; slider.max = max; slider.step = '1';
            slider.value = val;
            slider.style.cssText = 'flex: 1; accent-color: #00a8ff; height: 3px; cursor: pointer;';

            const valLabel = document.createElement('div');
            valLabel.style.cssText = 'color: #aaa; font-size: 10px; min-width: 25px; text-align: right;';
            valLabel.textContent = val;

            slider.oninput = (e) => {
                const v = parseInt(e.target.value);
                valLabel.textContent = v;
                setter(v);
            };
            row.appendChild(slider);
            row.appendChild(valLabel);
            return row;
        };

        const inBlackRow = createLevelSlider('IN BLACK', 0, 100, this.curveEditor.levels.inBlack, (v) => {
            this.curveEditor.setLevels(v, this.curveEditor.levels.inWhite);
        });
        const inWhiteRow = createLevelSlider('IN WHITE', 150, 255, this.curveEditor.levels.inWhite, (v) => {
            this.curveEditor.setLevels(this.curveEditor.levels.inBlack, v);
        });

        levelsRow.appendChild(inBlackRow);
        levelsRow.appendChild(inWhiteRow);
        container.appendChild(levelsRow);
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
            title.style.cssText = `color: ${labelColor}; font - size: 10px; width: 30px; font - weight: bold; `;
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

    renderMasksTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; gap: 10px; padding: 10px; max-height: 280px; overflow-y: auto;';

        const update = () => {
            if (this.renderer) {
                this.renderer.setMask(this.maskState);
                this.render();
            }
        };

        // 1. Top Controls (Type, Invert, Show Overlay)
        const topRow = document.createElement('div');
        topRow.style.cssText = 'display: flex; gap: 10px; align-items: center; justify-content: space-between; margin-bottom: 5px;';

        // Mask Type
        const typeSelect = document.createElement('select');
        typeSelect.style.cssText = 'background: #333; color: #ccc; border: 1px solid #555; font-size: 11px; padding: 2px; border-radius: 4px;';
        ['None', 'Circle', 'Box'].forEach((label, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = label;
            if (this.maskState.type === i) opt.selected = true;
            typeSelect.appendChild(opt);
        });
        typeSelect.onchange = (e) => {
            this.maskState.type = parseInt(e.target.value);
            update();
        };
        topRow.appendChild(typeSelect);

        // Invert
        const invertCheck = document.createElement('div');
        invertCheck.innerHTML = `
            <input type="checkbox" id="mask-invert" ${this.maskState.invert ? 'checked' : ''}>
            <label for="mask-invert" style="color: #ccc; font-size: 11px; margin-left: 4px;">Invert</label>
        `;
        invertCheck.querySelector('input').onchange = (e) => {
            this.maskState.invert = e.target.checked;
            update();
        };
        topRow.appendChild(invertCheck);

        // Show Overlay
        const overlayCheck = document.createElement('div');
        overlayCheck.innerHTML = `
            <input type="checkbox" id="mask-overlay" ${this.maskState.showOverlay ? 'checked' : ''}>
            <label for="mask-overlay" style="color: #ccc; font-size: 11px; margin-left: 4px;">Overlay</label>
        `;
        overlayCheck.querySelector('input').onchange = (e) => {
            this.maskState.showOverlay = e.target.checked;
            update();
        };
        topRow.appendChild(overlayCheck);

        container.appendChild(topRow);

        // 2. Transform Controls
        const createRow = (label, controls) => {
            const row = document.createElement('div');
            row.style.cssText = 'display: flex; flex-direction: column; gap: 4px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;';
            const title = document.createElement('div');
            title.textContent = label;
            title.style.cssText = 'color: #888; font-size: 9px; font-weight: bold; letter-spacing: 0.5px;';
            row.appendChild(title);
            const ctrlGroup = document.createElement('div');
            ctrlGroup.style.cssText = 'display: flex; gap: 10px; align-items: center;';
            controls.forEach(c => ctrlGroup.appendChild(c));
            row.appendChild(ctrlGroup);
            return row;
        };

        // Center
        container.appendChild(createRow('CENTER', [
            this.createKnob('X', 0, 1, this.maskState.center[0], 0.01, (v) => { this.maskState.center[0] = v; update(); }),
            this.createKnob('Y', 0, 1, this.maskState.center[1], 0.01, (v) => { this.maskState.center[1] = v; update(); })
        ]));

        // Scale
        container.appendChild(createRow('SCALE', [
            this.createKnob('X', 0.01, 2, this.maskState.scale[0], 0.01, (v) => { this.maskState.scale[0] = v; update(); }),
            this.createKnob('Y', 0.01, 2, this.maskState.scale[1], 0.01, (v) => { this.maskState.scale[1] = v; update(); })
        ]));

        // Rotation & Feather
        const rotFeatherGroup = document.createElement('div');
        rotFeatherGroup.style.cssText = 'display: flex; gap: 10px; width: 100%;';

        const rotRow = createRow('ROTATION', [
            this.createKnob('Rad', -Math.PI, Math.PI, this.maskState.rotation, 0.01, (v) => { this.maskState.rotation = v; update(); })
        ]);
        rotRow.style.flex = '1';
        rotFeatherGroup.appendChild(rotRow);

        const featherRow = createRow('FEATHER', [
            this.createKnob('Soft', 0, 1, this.maskState.feather, 0.01, (v) => { this.maskState.feather = v; update(); })
        ]);
        featherRow.style.flex = '1';
        rotFeatherGroup.appendChild(featherRow);

        container.appendChild(rotFeatherGroup);
    }

    activateBalancePicker() {
        if (this.isPickingBalance) return;
        this.isPickingBalance = true;

        const overlay = document.createElement('div');
        overlay.textContent = 'Click to Neutralize (Gray Balance)';
        overlay.style.cssText = 'position: absolute; top: 10px; left: 50%; transform: translateX(-50%); background: rgba(100,200,100,0.85); color: #fff; padding: 6px 12px; border-radius: 4px; pointer-events: none; z-index: 200; font-size: 11px; font-weight: bold; box-shadow: 0 4px 12px rgba(0,0,0,0.5);';
        this.container.appendChild(overlay);

        this.container.style.cursor = 'crosshair';

        const clickHandler = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const imgX = (x - this.panX) / this.zoom;
            const imgY = (y - this.panY) / this.zoom;

            if (imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
                let r = 0, g = 0, b = 0;

                if (this.hdrData) {
                    const ix = Math.floor(imgX);
                    const iy = Math.floor(imgY);
                    const idx = (iy * this.imageWidth + ix) * this.hdrData.channels;
                    const d = this.hdrData.data;
                    r = d[idx]; g = d[idx + 1]; b = d[idx + 2];
                } else if (this.imageData) {
                    const ix = Math.floor(imgX);
                    const iy = Math.floor(imgY);
                    const idx = (iy * this.imageWidth + ix) * 4;
                    r = this.imageData[idx] / 255.0;
                    g = this.imageData[idx + 1] / 255.0;
                    b = this.imageData[idx + 2] / 255.0;
                }

                // Balance Logic: Neutralize to average luma
                const avg = (r + g + b) / 3.0;
                if (avg > 0) {
                    const dr = avg - r;
                    const dg = avg - g;
                    const db = avg - b;

                    this.offset = [
                        (this.offset[0] || 0) + dr,
                        (this.offset[1] || 0) + dg,
                        (this.offset[2] || 0) + db
                    ];

                    if (this.renderer) {
                        this.renderer.setOffset(this.offset[0], this.offset[1], this.offset[2]);
                        this.render();
                    }
                    this.renderPrimariesTab(this.tabContentContainer);
                }
            }

            this.container.style.cursor = 'default';
            overlay.remove();
            this.container.removeEventListener('click', clickHandler);
            this.isPickingBalance = false;
        };

        this.container.addEventListener('click', clickHandler);
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

    activateFocusPicker(callback) {
        if (this.isPickingFocus) return;
        this.isPickingFocus = true;

        const overlay = document.createElement('div');
        overlay.textContent = 'Click to Set Focus';
        overlay.style.cssText = 'position: absolute; top: 10px; left: 50%; transform: translateX(-50%); background: rgba(106,138,255,0.85); color: #fff; padding: 6px 12px; border-radius: 4px; pointer-events: none; z-index: 200; font-size: 11px; font-weight: bold; box-shadow: 0 4px 12px rgba(0,0,0,0.5);';
        this.container.appendChild(overlay);

        this.container.style.cursor = 'crosshair';

        const clickHandler = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const imgX = (x - this.panX) / this.zoom;
            const imgY = (y - this.panY) / this.zoom;

            if (imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
                if (this.zdepthImage) {
                    // Sample from depth image
                    const pickCanvas = document.createElement('canvas');
                    pickCanvas.width = this.zdepthImage.width;
                    pickCanvas.height = this.zdepthImage.height;
                    const pctx = pickCanvas.getContext('2d');
                    pctx.drawImage(this.zdepthImage, 0, 0);

                    const ix = Math.floor(imgX * (this.zdepthImage.width / this.imageWidth));
                    const iy = Math.floor(imgY * (this.zdepthImage.height / this.imageHeight));

                    const pix = pctx.getImageData(ix, iy, 1, 1).data;
                    // Depth is usually stored in R channel (grayscale)
                    // In ComfyUI/Radiance, normalize it to 0..1
                    this.focusDistance = pix[0] / 255.0;

                    if (this.renderer) {
                        this.renderer.setFocusDistance(this.focusDistance);
                        this.render();
                    }
                    if (callback) callback(this.focusDistance);
                } else {
                    console.warn("[Radiance] No depth map available to pick focus from.");
                }
            }

            this.container.style.cursor = 'default';
            overlay.remove();
            this.container.removeEventListener('click', clickHandler);
            this.isPickingFocus = false;
        };

        this.container.addEventListener('click', clickHandler);
    }

    renderViewTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; gap: 12px; padding: 12px; max-height: 300px; overflow-y: auto;';

        // 1. Comparison Wipe
        const wipeGroup = document.createElement('div');
        wipeGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 8px; text-transform: uppercase; font-weight: bold;">A/B Comparison (Split Screen)</div>';

        const wipeControls = document.createElement('div');
        wipeControls.style.cssText = 'display: flex; align-items: center; gap: 10px; background: rgba(255,255,255,0.03); padding: 8px; border-radius: 6px;';

        const wipeCheck = document.createElement('input');
        wipeCheck.type = 'checkbox';
        wipeCheck.id = 'wipe-enable';
        wipeCheck.checked = this.wipeEnabled || false;
        wipeCheck.onchange = (e) => {
            this.wipeEnabled = e.target.checked;
            if (this.renderer) this.renderer.setWipe(this.wipe || 0.5, this.wipeEnabled);
            this.render();
        };
        wipeControls.appendChild(wipeCheck);

        const wipeLbl = document.createElement('label');
        wipeLbl.htmlFor = 'wipe-enable';
        wipeLbl.textContent = 'Enable Wipe';
        wipeLbl.style.cssText = 'color: #ccc; font-size: 11px; cursor: pointer; flex: 1;';
        wipeControls.appendChild(wipeLbl);

        const wipeSlider = document.createElement('input');
        wipeSlider.type = 'range'; wipeSlider.min = '0'; wipeSlider.max = '100'; wipeSlider.step = '1';
        wipeSlider.value = String(Math.round((this.wipe || 0.5) * 100));
        wipeSlider.style.cssText = 'width: 100px; accent-color: #6a8aff; height: 4px; cursor: pointer;';
        wipeSlider.oninput = (e) => {
            this.wipe = parseInt(e.target.value) / 100;
            if (this.renderer) this.renderer.setWipe(this.wipe, this.wipeEnabled);
            this.render();
        };
        wipeControls.appendChild(wipeSlider);

        wipeGroup.appendChild(wipeControls);
        container.appendChild(wipeGroup);

        // 2. Reference Image
        const refGroup = document.createElement('div');
        refGroup.style.marginTop = '4px';
        refGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 8px; text-transform: uppercase; font-weight: bold;">Reference Gallery</div>';

        const refRow = document.createElement('div');
        refRow.style.cssText = 'display: flex; gap: 8px; align-items: center;';

        const grabBtn = document.createElement('button');
        grabBtn.textContent = '📸 GRAB STILL';
        grabBtn.style.cssText = 'background: #2a2a30; color: #fff; border: 1px solid #444; padding: 6px 12px; border-radius: 4px; font-size: 10px; cursor: pointer; font-weight: bold; flex: 1;';
        grabBtn.onclick = () => this.grabStill();
        refRow.appendChild(grabBtn);

        const refCheckGroup = document.createElement('div');
        refCheckGroup.style.cssText = 'display: flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.03); padding: 5px 10px; border-radius: 4px;';

        const refCheck = document.createElement('input');
        refCheck.type = 'checkbox';
        refCheck.id = 'wipe-ref';
        refCheck.checked = this.wipeRefEnabled || false;
        refCheck.onchange = (e) => {
            this.wipeRefEnabled = e.target.checked;
            if (this.renderer) this.renderer.setWipeRef(this.wipeRefEnabled);
            this.render();
        };
        refCheckGroup.appendChild(refCheck);

        const refLbl = document.createElement('label');
        refLbl.htmlFor = 'wipe-ref';
        refLbl.textContent = 'Use Ref';
        refLbl.style.cssText = 'color: #888; font-size: 10px; cursor: pointer;';
        refCheckGroup.appendChild(refLbl);
        refRow.appendChild(refCheckGroup);

        refGroup.appendChild(refRow);
        container.appendChild(refGroup);

        // 3. Overlays / Grids
        const gridGroup = document.createElement('div');
        gridGroup.style.marginTop = '4px';
        gridGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 8px; text-transform: uppercase; font-weight: bold;">Composition Overlays</div>';

        const gridSelect = document.createElement('select');
        gridSelect.style.cssText = 'width: 100%; background: #1a1a20; color: #ccc; border: 1px solid #333; padding: 6px; border-radius: 4px; font-size: 11px; cursor: pointer;';

        const gridOptions = [
            { id: 0, label: 'None' },
            { id: 1, label: 'Rule of Thirds' },
            { id: 2, label: '2.39:1 Cinematic Mask' },
            { id: 3, label: 'Center Cross' }
        ];

        gridOptions.forEach(opt => {
            const o = document.createElement('option');
            o.value = opt.id;
            o.textContent = opt.label;
            if (opt.id === (this.gridMode || 0)) o.selected = true;
            gridSelect.appendChild(o);
        });

        gridSelect.onchange = (e) => {
            this.gridMode = parseInt(e.target.value);
            if (this.renderer) this.renderer.setGridMode(this.gridMode);
            this.render();
        };
        gridGroup.appendChild(gridSelect);
        container.appendChild(gridGroup);

        // 4. Grade Presets & Export
        const exportGroup = document.createElement('div');
        exportGroup.style.marginTop = '4px';
        exportGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 8px; text-transform: uppercase; font-weight: bold;">Presets & Export</div>';

        const exportRow = document.createElement('div');
        exportRow.style.cssText = 'display: flex; gap: 8px;';

        const saveBtn = document.createElement('button');
        saveBtn.textContent = '💾 SAVE PRESET';
        saveBtn.style.cssText = 'background: #1a1a20; color: #fff; border: 1px solid #333; padding: 6px; border-radius: 4px; font-size: 10px; cursor: pointer; flex: 1;';
        saveBtn.onclick = () => this.saveGrade();
        exportRow.appendChild(saveBtn);

        const cubeBtn = document.createElement('button');
        cubeBtn.textContent = '📤 EXPORT .CUBE';
        cubeBtn.style.cssText = 'background: #2a2a30; color: #ffca28; border: 1px solid #ffca2833; padding: 6px; border-radius: 4px; font-size: 10px; cursor: pointer; flex: 1; font-weight: bold;';
        cubeBtn.onclick = () => this.exportToCube();
        exportRow.appendChild(cubeBtn);

        exportGroup.appendChild(exportRow);

        // Preset List
        const presets = JSON.parse(localStorage.getItem('radiance_presets') || '{}');
        const presetNames = Object.keys(presets);
        if (presetNames.length > 0) {
            const list = document.createElement('div');
            list.style.cssText = 'display: flex; flex-direction: column; gap: 4px; margin-top: 8px; max-height: 100px; overflow-y: auto; background: rgba(0,0,0,0.2); border-radius: 4px; padding: 4px;';
            presetNames.forEach(name => {
                const item = document.createElement('div');
                item.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.03); border-radius: 3px; font-size: 10px; color: #ccc;';

                const nameBtn = document.createElement('div');
                nameBtn.textContent = name;
                nameBtn.style.cssText = 'cursor: pointer; flex: 1;';
                nameBtn.onclick = () => this.loadGrade(name);
                item.appendChild(nameBtn);

                const delBtn = document.createElement('div');
                delBtn.textContent = '✕';
                delBtn.style.cssText = 'cursor: pointer; color: #666; padding: 0 4px;';
                delBtn.onmouseover = () => delBtn.style.color = '#ff6b6b';
                delBtn.onmouseout = () => delBtn.style.color = '#666';
                delBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (confirm(`Delete preset "${name}" ? `)) this.deleteGrade(name);
                };
                item.appendChild(delBtn);

                list.appendChild(item);
            });
            exportGroup.appendChild(list);
        }

        container.appendChild(exportGroup);
    }

    async grabStill() {
        if (!this.renderer || !this.canvas) return;

        // Capture the current rendered output
        const still = new Image();
        still.src = this.canvas.toDataURL('image/png');

        still.onload = () => {
            // Upload to renderer as reference texture
            if (this.renderer) {
                this.renderer.updateReferenceStill(still);
                console.log("[Radiance] Reference still captured.");
                // Flash the screen briefly to indicate capture
                this.canvas.style.filter = 'brightness(2)';
                setTimeout(() => { this.canvas.style.filter = ''; }, 100);
            }
        };
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
                flex: 1; text - align: center; padding: 3px 0;
        background: ${isActive ? 'rgba(106,138,255,0.2)' : 'rgba(255,255,255,0.04)'};
        color: ${isActive ? '#8aafff' : '#777'};
        border: 1px solid ${isActive ? 'rgba(106,138,255,0.4)' : 'rgba(255,255,255,0.08)'};
        border - radius: 3px; font - size: 9px; cursor: pointer;
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
        canvas.style.cssText = `background: #050508; border: 1px solid rgba(255, 255, 255, 0.1); border - radius: 4px; width: 280px; height: ${cH} px; `;
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
                ctx.fillStyle = `rgb(${bright}, ${Math.floor(bright * 1.4)}, ${bright})`;
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

        // Skin Tone Indicator (I-Line)
        ctx.strokeStyle = 'rgba(255, 140, 100, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        const iLineAng = (123 - 90) * Math.PI / 180;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(iLineAng) * rad * 0.9, cy + Math.sin(iLineAng) * rad * 0.9);
        ctx.stroke();
        ctx.setLineDash([]);

        // Plot pixels
        ctx.globalAlpha = 0.04;
        const step = Math.max(1, Math.floor(data.length / 4 / 25000));
        for (let i = 0; i < data.length; i += 4 * step) {
            const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
            const y = r * 0.2126 + g * 0.7152 + b * 0.0722;
            const u = (b - y) * 0.492;
            const v = (r - y) * 0.877;
            ctx.fillStyle = `rgb(${data[i]}, ${data[i + 1]}, ${data[i + 2]})`;
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
            ctx.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
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
        this._lastUndoPush = performance.now();
    }

    // Debounced undo push for rapid-fire inputs (scroll wheel, printer lights)
    // Only captures if last push was >500ms ago — prevents flooding the stack
    _pushUndoDebounced() {
        const now = performance.now();
        if (!this._lastUndoPush || (now - this._lastUndoPush) > 500) {
            this._pushUndo();
        }
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
    //                         GRADE EXPORT & PRESETS
    // ═══════════════════════════════════════════════════════════════════════════

    saveGrade(name) {
        if (!name) name = prompt("Enter preset name:", "New Grade");
        if (!name) return;

        const state = this._captureGradingState();
        const presets = JSON.parse(localStorage.getItem('radiance_presets') || '{}');
        presets[name] = state;
        localStorage.setItem('radiance_presets', JSON.stringify(presets));
        console.log(`[Radiance] Preset "${name}" saved.`);
        if (this._lastRenderContent) this._lastRenderContent(); // Refresh UI
    }

    loadGrade(name) {
        const presets = JSON.parse(localStorage.getItem('radiance_presets') || '{}');
        const state = presets[name];
        if (state) {
            this._pushUndo();
            this._restoreGradingState(state);
            console.log(`[Radiance] Preset "${name}" applied.`);
        }
    }

    deleteGrade(name) {
        const presets = JSON.parse(localStorage.getItem('radiance_presets') || '{}');
        delete presets[name];
        localStorage.setItem('radiance_presets', JSON.stringify(presets));
        if (this._lastRenderContent) this._lastRenderContent();
    }

    exportToCube() {
        console.log("[Radiance] Generating 3D LUT (.cube)...");
        const size = 33;
        let cube = `TITLE "Radiance Export"\nLUT_3D_SIZE ${size} \nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n\n`;

        // Helper to apply math (matching radiance_webgl.js)
        const applyMath = (c) => {
            let r = c[0], g = c[1], b = c[2];

            // 1. Offset
            r += this.offset[0]; g += this.offset[1]; b += this.offset[2];

            // 2. Lift (pivoted at white)
            const luma = r * 0.2126 + g * 0.7152 + b * 0.0722;
            const pivot = Math.max(0.0, Math.min(1.0, 1.0 - luma));
            r += (this.lift[0] || 0) * pivot;
            g += (this.lift[1] || 0) * pivot;
            b += (this.lift[2] || 0) * pivot;

            // 3. Gain
            r *= (this.gain[0] || 1); g *= (this.gain[1] || 1); b *= (this.gain[2] || 1);

            // 4. Gamma
            r = Math.pow(Math.max(0.0, r), 1.0 / (this.gamma[0] || 1));
            g = Math.pow(Math.max(0.0, g), 1.0 / (this.gamma[1] || 1));
            b = Math.pow(Math.max(0.0, b), 1.0 / (this.gamma[2] || 1));

            // 5. Contrast & Pivot
            const con = this.contrast || 1.0;
            const piv = this.pivot || 0.5;
            r = (r - piv) * con + piv;
            g = (g - piv) * con + piv;
            b = (b - piv) * con + piv;

            // 6. Saturation
            const luma2 = r * 0.2126 + g * 0.7152 + b * 0.0722;
            const sat = this.saturation || 1.0;
            r = luma2 + (r - luma2) * sat;
            g = luma2 + (g - luma2) * sat;
            b = luma2 + (b - luma2) * sat;

            return [r, g, b];
        };

        for (let b = 0; b < size; b++) {
            for (let g = 0; g < size; g++) {
                for (let r = 0; r < size; r++) {
                    const result = applyMath([r / (size - 1), g / (size - 1), b / (size - 1)]);
                    cube += `${result[0].toFixed(6)} ${result[1].toFixed(6)} ${result[2].toFixed(6)} \n`;
                }
            }
        }

        const blob = new Blob([cube], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = "radiance_grade.cube";
        a.click();
        URL.revokeObjectURL(url);
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

        // 2. Knob Wrapper (for centering value)
        const knobWrapper = document.createElement('div');
        knobWrapper.style.cssText = 'position: relative; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center;';

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
            pointer-events: none;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
            z-index: 10;
        `;

        knobWrapper.appendChild(svg);
        knobWrapper.appendChild(valDisplay);

        container.appendChild(lbl);
        container.appendChild(knobWrapper);

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
                background: rgba(0, 0, 0, 0.9); border: 1px solid ${this.theme.accent};
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
        // Autodesk Flame Style: Wheel controls Chroma (Hue/Sat), Ring controls Master (Luma).

        const container = document.createElement('div');
        container.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 120px; flex: 1; position: relative;';

        // Label
        const lbl = document.createElement('div');
        lbl.textContent = label;
        lbl.style.cssText = `color: ${this.theme.textDim}; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);`;
        container.appendChild(lbl);

        // SVG Constants
        const size = 110;
        const center = size / 2;
        const wheelRadius = 38;
        const ringOuter = 52;
        const ringInner = 46;
        const puckRadius = 5;

        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("width", size);
        svg.setAttribute("height", size);
        svg.style.cssText = "cursor: default; touch-action: none; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.4));";

        // 1. Master Ring Background
        const ringBg = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        ringBg.setAttribute("cx", center); ringBg.setAttribute("cy", center);
        ringBg.setAttribute("r", (ringOuter + ringInner) / 2);
        ringBg.setAttribute("fill", "none");
        ringBg.setAttribute("stroke", "rgba(0,0,0,0.4)");
        ringBg.setAttribute("stroke-width", ringOuter - ringInner);
        svg.appendChild(ringBg);

        // 2. Master Ring Progress
        const ringProgress = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        const ringCirc = 2 * Math.PI * ((ringOuter + ringInner) / 2);
        ringProgress.setAttribute("cx", center); ringProgress.setAttribute("cy", center);
        ringProgress.setAttribute("r", (ringOuter + ringInner) / 2);
        ringProgress.setAttribute("fill", "none");
        ringProgress.setAttribute("stroke", this.theme.accent);
        ringProgress.setAttribute("stroke-width", ringOuter - ringInner - 2);
        ringProgress.setAttribute("stroke-dasharray", ringCirc);
        ringProgress.setAttribute("stroke-dashoffset", ringCirc); // Start empty
        ringProgress.setAttribute("stroke-linecap", "round");
        ringProgress.style.transformOrigin = "center";
        ringProgress.style.transform = "rotate(-90deg)"; // Start from top
        ringProgress.style.transition = "stroke-dashoffset 0.05s linear, stroke 0.2s";
        svg.appendChild(ringProgress);

        // 3. Inner Wheel Background
        const wheelBg = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        wheelBg.setAttribute("cx", center); wheelBg.setAttribute("cy", center);
        wheelBg.setAttribute("r", wheelRadius);
        wheelBg.setAttribute("fill", "#111");
        wheelBg.setAttribute("stroke", "rgba(255,255,255,0.05)");
        svg.appendChild(wheelBg);

        // Crosshair
        const crossStyle = "stroke: rgba(255,255,255,0.05); stroke-width: 1;";
        const hLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        hLine.setAttribute("x1", center - wheelRadius); hLine.setAttribute("x2", center + wheelRadius);
        hLine.setAttribute("y1", center); hLine.setAttribute("y2", center); hLine.setAttribute("style", crossStyle);
        svg.appendChild(hLine);
        const vLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        vLine.setAttribute("x1", center); vLine.setAttribute("x2", center);
        vLine.setAttribute("y1", center - wheelRadius); vLine.setAttribute("y2", center + wheelRadius);
        vLine.setAttribute("style", crossStyle);
        svg.appendChild(vLine);

        // 4. Puck (Hue/Sat handle)
        const puck = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        puck.setAttribute("r", puckRadius);
        puck.setAttribute("fill", "#fff");
        puck.setAttribute("stroke", "#000");
        puck.setAttribute("stroke-width", "2");
        puck.style.cursor = "crosshair";
        puck.style.filter = "drop-shadow(0 2px 4px rgba(0,0,0,0.5))";
        svg.appendChild(puck);

        container.appendChild(svg);

        // Data State
        let wheelR = 0, wheelG = 0, wheelB = 0; // Chroma offsets
        let masterVal = (defaults[0] + defaults[1] + defaults[2]) / 3.0; // Master offset

        const updateVisuals = () => {
            // Puck Position
            const angleR = 0, angleG = 2 * Math.PI / 3, angleB = 4 * Math.PI / 3;
            let dx = wheelR * Math.cos(angleR) + wheelG * Math.cos(angleG) + wheelB * Math.cos(angleB);
            let dy = wheelR * Math.sin(angleR) + wheelG * Math.sin(angleG) + wheelB * Math.sin(angleB);

            const pxSens = wheelRadius / 0.5;
            let px = dx * pxSens;
            let py = dy * pxSens;

            const dist = Math.sqrt(px * px + py * py);
            if (dist > wheelRadius) {
                px = (px / dist) * wheelRadius;
                py = (py / dist) * wheelRadius;
            }

            puck.setAttribute("cx", center + px);
            puck.setAttribute("cy", center + py);

            // Ring Position
            const range = max - min;
            const t = Math.max(0, Math.min(1, (masterVal - min) / range));
            const offset = ringCirc * (1 - t);
            ringProgress.setAttribute("stroke-dashoffset", offset);
            ringProgress.setAttribute("stroke", Math.abs(masterVal - (defaults[0] + defaults[1] + defaults[2]) / 3) < 0.001 ? "rgba(255,255,255,0.2)" : this.theme.accent);
        };

        // Initialize from defaults
        const initialAvg = (defaults[0] + defaults[1] + defaults[2]) / 3.0;
        wheelR = defaults[0] - initialAvg;
        wheelG = defaults[1] - initialAvg;
        wheelB = defaults[2] - initialAvg;
        masterVal = initialAvg;
        updateVisuals();

        // Interaction
        svg.onpointerdown = (e) => {
            e.preventDefault();
            this._pushUndo();
            svg.setPointerCapture(e.pointerId);

            const rect = svg.getBoundingClientRect();
            const startX = e.clientX, startY = e.clientY;
            const distToCenter = Math.sqrt(Math.pow(startX - (rect.left + center), 2) + Math.pow(startY - (rect.top + center), 2));

            const isRingDrag = distToCenter > wheelRadius;
            const initialMaster = masterVal;

            const onMove = (em) => {
                if (isRingDrag) {
                    const deltaY = startY - em.clientY;
                    let sens = (max - min) * 0.005;
                    if (em.ctrlKey) sens *= 0.1;
                    if (em.shiftKey) sens *= 3.0;
                    masterVal = Math.max(min, Math.min(max, initialMaster + deltaY * sens));
                } else {
                    let dx = em.clientX - (rect.left + center);
                    let dy = em.clientY - (rect.top + center);
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist > wheelRadius) {
                        dx = (dx / dist) * wheelRadius;
                        dy = (dy / dist) * wheelRadius;
                    }
                    const mag = dist / wheelRadius;
                    const ang = Math.atan2(dy, dx);
                    const strength = mag * 0.5;
                    wheelR = Math.cos(ang) * strength;
                    wheelG = Math.cos(ang - 2 * Math.PI / 3) * strength;
                    wheelB = Math.cos(ang - 4 * Math.PI / 3) * strength;
                }
                updateVisuals();
                callback(wheelR + masterVal, wheelG + masterVal, wheelB + masterVal);
            };

            const onUp = () => {
                svg.removeEventListener('pointermove', onMove);
                svg.removeEventListener('pointerup', onUp);
                svg.releasePointerCapture(e.pointerId);
            };

            svg.addEventListener('pointermove', onMove);
            svg.addEventListener('pointerup', onUp);
        };

        svg.ondblclick = (e) => {
            const rect = svg.getBoundingClientRect();
            const distToCenter = Math.sqrt(Math.pow(e.clientX - (rect.left + center), 2) + Math.pow(e.clientY - (rect.top + center), 2));
            this._pushUndo();
            if (distToCenter > wheelRadius) {
                masterVal = (defaults[0] + defaults[1] + defaults[2]) / 3.0;
            } else {
                wheelR = 0; wheelG = 0; wheelB = 0;
            }
            updateVisuals();
            callback(wheelR + masterVal, wheelG + masterVal, wheelB + masterVal);
        };

        svg.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this._pushUndo();
            wheelR = 0; wheelG = 0; wheelB = 0;
            masterVal = (defaults[0] + defaults[1] + defaults[2]) / 3.0;
            updateVisuals();
            callback(wheelR + masterVal, wheelG + masterVal, wheelB + masterVal);
        });

        container.updateValue = (r, g, b) => {
            const m = (r + g + b) / 3.0;
            wheelR = r - m; wheelG = g - m; wheelB = b - m;
            masterVal = m;
            updateVisuals();
        };

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
        font - size: 11px;
        font - family: ${this.theme.font};
        min - width: 50px;
        font - weight: 500;
        `;

        const value = document.createElement('div');
        value.textContent = initial.toFixed(2);
        value.style.cssText = `
        font - family: ${this.theme.mono};
        font - size: 11px;
        color: ${this.theme.accent};
        min - width: 40px;
        text - align: right;
        font - variant - numeric: tabular - nums;
        `;

        metaRow.appendChild(lbl);
        metaRow.appendChild(value);

        const sliderContainer = document.createElement('div');
        sliderContainer.style.cssText = 'position: relative; height: 4px; background: #333; border-radius: 2px; margin-top: 2px;';

        const sliderFill = document.createElement('div');
        sliderFill.style.cssText = `
        position: absolute; left: 0; top: 0; height: 100 %; background: #445;
        width: 50 %; pointer - events: none; border - radius: 2px;
        `;

        const sliderInput = document.createElement('input');
        sliderInput.type = 'range';
        sliderInput.min = min; sliderInput.max = max; sliderInput.step = step; sliderInput.value = initial;
        sliderInput.style.cssText = `
        position: absolute; left: 0; top: -6px; width: 100 %; height: 16px; opacity: 0; cursor: ew - resize; margin: 0;
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
        width: 100 %;
        height: 4px;
        background: rgba(0, 0, 0, 0.5);
        z - index: 50;
        pointer - events: none;
        opacity: 0;
        transition: opacity 0.3s ease;
        `;

        this.progressBar = document.createElement('div');
        this.progressBar.style.cssText = `
        width: 0 %;
        height: 100 %;
        background: linear - gradient(90deg, ${t.accent}, #4f4);
        transition: width 0.1s linear;
        box - shadow: 0 0 10px ${t.accent};
        `;

        this.progressText = document.createElement('div');
        this.progressText.style.cssText = `
        position: absolute;
        bottom: 6px;
        right: 10px;
        font - size: 10px;
        font - family: monospace;
        color: rgba(255, 255, 255, 0.8);
        text - shadow: 0 1px 2px black;
        pointer - events: none;
        opacity: 0;
        transition: opacity 0.3s ease;
        background: rgba(0, 0, 0, 0.6);
        padding: 2px 6px;
        border - radius: 4px;
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
            this.progressBar.style.width = `${pct}% `;

            // ETA Calculation
            const now = Date.now();
            if (value > 0) {
                const elapsed = (now - this.progressStart) / 1000;
                const timePerStep = elapsed / value;
                const remaining = (max - value) * timePerStep;

                // Simple formatting
                const eta = remaining < 60 ? `${remaining.toFixed(1)} s` : `${Math.floor(remaining / 60)}m ${Math.floor(remaining % 60)} s`;
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

    // Synchronous zlib inflate — not supported. Use _zlibInflateAsync() instead.
    _zlibInflate(compressed) {
        throw new Error('[Radiance] Synchronous zlib inflate is not supported. Use _zlibInflateAsync() instead.');
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
            // FIX: Store the raw message data for robust metadata retrieval (foolproof EXR link)
            viewer.lastResult = message.radiance_images;
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
    /**
     * Industry-standard curve editor (DaVinci Resolve / Nuke / Flame paradigm).
     *
     * Interpolation: Monotonic cubic Hermite (Fritsch–Carlson, 1980).
     *   - Guaranteed no overshoot between control points
     *   - Smooth C¹ continuity
     *   - Same algorithm used by DaVinci Resolve and Nuke curve tools
     *
     * Points are simple {x, y} — tangents are auto-computed from neighbors.
     * No manual tangent handles; auto-smooth only (Resolve behaviour).
     *
     * LUT output is Float32Array(256×4 = 1024) RGBA for full HDR precision.
     * Master RGB curve is evaluated first, then per-channel R/G/B on top.
     */
    constructor(width, height, theme, onChange) {
        this.width = width;
        this.height = height;
        this.theme = theme || { mono: 'monospace', textDim: '#888' };
        this.onChange = onChange;
        this.padding = { left: 35, bottom: 25, top: 10, right: 10 };

        this.canvas = document.createElement('canvas');
        this.canvas.width = width;
        this.canvas.height = height;
        this.canvas.style.cursor = 'crosshair';
        this.ctx = this.canvas.getContext('2d');

        this.histograms = { R: null, G: null, B: null, L: null };

        // Simple {x, y} points — endpoints pinned at x=0 and x=1
        this.curves = {
            'RGB': [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            'R': [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            'G': [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            'B': [{ x: 0, y: 0 }, { x: 1, y: 1 }]
        };

        this.activeChannel = 'RGB';
        this.hoverPoint = null;
        this.draggingPoint = null;
        this.mousePos = { x: -1, y: -1 };

        this.channelColors = {
            'RGB': '#ffffff',
            'R': '#ff4d4d',
            'G': '#4dff4d',
            'B': '#4d88ff'
        };

        this.levels = { inBlack: 0, inWhite: 255 };

        this.setupEvents();
        this.draw();
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
        this.canvas.width = width;
        this.canvas.height = height;
        this.draw();
    }

    // ─── Coordinate helpers ────────────────────────────────────
    get plotX() { return this.padding.left; }
    get plotY() { return this.padding.top; }
    get plotW() { return this.width - this.padding.left - this.padding.right; }
    get plotH() { return this.height - this.padding.top - this.padding.bottom; }

    normToCanvas(nx, ny) {
        return {
            cx: this.plotX + nx * this.plotW,
            cy: this.plotY + (1 - ny) * this.plotH
        };
    }

    canvasToNorm(cx, cy) {
        return {
            x: (cx - this.plotX) / this.plotW,
            y: 1.0 - (cy - this.plotY) / this.plotH
        };
    }

    // ─── Histogram ─────────────────────────────────────────────
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
            const luma = Math.min(255, Math.floor(
                data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722
            ));
            L[luma]++;
        }

        // Normalize — ignore extremes (0 and 255 bins often spike)
        let max = 0;
        for (let i = 1; i < buckets - 1; i++) {
            max = Math.max(max, L[i], R[i], G[i], B[i]);
        }
        if (max === 0) max = 1;

        const norm = (arr) => {
            const r = new Float32Array(buckets);
            for (let i = 0; i < buckets; i++) r[i] = Math.min(1.0, arr[i] / max);
            return r;
        };
        this.histograms = { R: norm(R), G: norm(G), B: norm(B), L: norm(L) };
        this.draw();
    }

    // ─── Channel control ───────────────────────────────────────
    setActiveChannel(ch) {
        this.activeChannel = ch;
        this.draw();
        if (this._onChannelSwitch) this._onChannelSwitch(ch);
    }

    resetActiveChannel() {
        this.curves[this.activeChannel] = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
        this.notifyChange();
        this.draw();
    }

    resetAll() {
        for (const ch of ['RGB', 'R', 'G', 'B']) {
            this.curves[ch] = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
        }
        this.levels = { inBlack: 0, inWhite: 255 };
        this.notifyChange();
        this.draw();
    }

    // Legacy compat
    resetAllChannels() { this.resetAll(); }

    applyPreset(preset) {
        for (const ch of ['RGB', 'R', 'G', 'B']) {
            if (preset[ch]) {
                this.curves[ch] = preset[ch].map(p => ({ x: p.x, y: p.y }));
            }
        }
        this.notifyChange();
        this.draw();
    }

    applyIndustryPreset(name) {
        // Industry-standard curve presets (DaVinci Resolve reference values)
        const presets = {
            'Punchy': [
                { x: 0, y: 0 }, { x: 0.15, y: 0.08 },
                { x: 0.50, y: 0.52 }, { x: 0.85, y: 0.95 }, { x: 1, y: 1 }
            ],
            'High Contrast': [
                { x: 0, y: 0 }, { x: 0.20, y: 0.05 },
                { x: 0.50, y: 0.50 }, { x: 0.80, y: 0.95 }, { x: 1, y: 1 }
            ],
            'Flat / Log': [
                { x: 0, y: 0.12 }, { x: 0.25, y: 0.30 },
                { x: 0.50, y: 0.50 }, { x: 0.75, y: 0.70 }, { x: 1, y: 0.88 }
            ],
            'Shadow Lift': [
                { x: 0, y: 0.08 }, { x: 0.15, y: 0.20 },
                { x: 0.50, y: 0.52 }, { x: 1, y: 1 }
            ],
            'S-Curve': [
                { x: 0, y: 0 }, { x: 0.25, y: 0.17 },
                { x: 0.50, y: 0.50 }, { x: 0.75, y: 0.83 }, { x: 1, y: 1 }
            ],
            'Film Print': [
                { x: 0, y: 0.02 }, { x: 0.10, y: 0.06 },
                { x: 0.30, y: 0.28 }, { x: 0.60, y: 0.65 },
                { x: 0.85, y: 0.90 }, { x: 1, y: 0.96 }
            ],
            'Bleach Bypass': [
                { x: 0, y: 0 }, { x: 0.15, y: 0.03 },
                { x: 0.40, y: 0.45 }, { x: 0.70, y: 0.82 }, { x: 1, y: 1 }
            ],
            'Cross Process': {
                'R': [{ x: 0, y: 0.05 }, { x: 0.3, y: 0.2 }, { x: 0.7, y: 0.85 }, { x: 1, y: 0.95 }],
                'G': [{ x: 0, y: 0 }, { x: 0.35, y: 0.4 }, { x: 0.65, y: 0.6 }, { x: 1, y: 1 }],
                'B': [{ x: 0, y: 0.1 }, { x: 0.3, y: 0.35 }, { x: 0.7, y: 0.55 }, { x: 1, y: 0.85 }]
            }
        };

        const preset = presets[name];
        if (!preset) return;

        if (Array.isArray(preset)) {
            // Apply to active channel
            this.curves[this.activeChannel] = preset.map(p => ({ x: p.x, y: p.y }));
        } else {
            // Multi-channel preset (like Cross Process)
            for (const ch of ['RGB', 'R', 'G', 'B']) {
                if (preset[ch]) {
                    this.curves[ch] = preset[ch].map(p => ({ x: p.x, y: p.y }));
                }
            }
        }
        this.notifyChange();
        this.draw();
    }

    copyRGBToAll() {
        const src = this.curves['RGB'].map(p => ({ x: p.x, y: p.y }));
        this.curves['R'] = src.map(p => ({ ...p }));
        this.curves['G'] = src.map(p => ({ ...p }));
        this.curves['B'] = src.map(p => ({ ...p }));
        this.notifyChange();
        this.draw();
    }

    // ─── Events ────────────────────────────────────────────────
    setupEvents() {
        const cvs = this.canvas;
        const GRAB_RADIUS_PX = 8; // Hit radius in pixels

        const hitTest = (e) => {
            const rect = cvs.getBoundingClientRect();
            const px = e.clientX - rect.left;
            const py = e.clientY - rect.top;
            const norm = this.canvasToNorm(px, py);

            const pts = this.curves[this.activeChannel];
            let best = null, bestDist = GRAB_RADIUS_PX * GRAB_RADIUS_PX;
            for (const p of pts) {
                const c = this.normToCanvas(p.x, p.y);
                const d = (px - c.cx) ** 2 + (py - c.cy) ** 2;
                if (d < bestDist) { bestDist = d; best = p; }
            }
            return { norm, best, px, py };
        };

        cvs.onmousedown = (e) => {
            if (e.button !== 0) return; // Left click only
            const { norm, best } = hitTest(e);

            if (best) {
                this.draggingPoint = best;
                cvs.style.cursor = 'grabbing';
            } else if (norm.x > 0.005 && norm.x < 0.995) {
                // Add new point (not at endpoints)
                const pts = this.curves[this.activeChannel];
                const newPt = {
                    x: Math.max(0.005, Math.min(0.995, norm.x)),
                    y: Math.max(0, Math.min(1, norm.y))
                };
                pts.push(newPt);
                pts.sort((a, b) => a.x - b.x);
                this.draggingPoint = newPt;
                cvs.style.cursor = 'grabbing';
                this.notifyChange();
            }
            this.draw();
        };

        cvs.onmousemove = (e) => {
            const rect = cvs.getBoundingClientRect();
            const px = e.clientX - rect.left;
            const py = e.clientY - rect.top;
            const norm = this.canvasToNorm(px, py);
            this.mousePos = norm;

            if (this.draggingPoint) {
                const pts = this.curves[this.activeChannel];
                const idx = pts.indexOf(this.draggingPoint);
                const p = this.draggingPoint;

                // Endpoints: lock X, allow Y
                if (idx === 0) {
                    p.y = Math.max(0, Math.min(1, norm.y));
                } else if (idx === pts.length - 1) {
                    p.y = Math.max(0, Math.min(1, norm.y));
                } else {
                    // Interior point: constrain X between neighbors
                    const minX = pts[idx - 1].x + 0.005;
                    const maxX = pts[idx + 1].x - 0.005;
                    p.x = Math.max(minX, Math.min(maxX, norm.x));
                    p.y = Math.max(0, Math.min(1, norm.y));
                }
                this.notifyChange();
            } else {
                const { best } = hitTest(e);
                const newHover = best || null;
                if (newHover !== this.hoverPoint) {
                    this.hoverPoint = newHover;
                    cvs.style.cursor = newHover ? 'grab' : 'crosshair';
                }
            }
            this.draw();
        };

        const onMouseUp = () => {
            if (this.draggingPoint) {
                this.draggingPoint = null;
                cvs.style.cursor = 'crosshair';
                this.draw();
            }
        };
        window.addEventListener('mouseup', onMouseUp);

        // Double-click: remove interior point
        cvs.ondblclick = (e) => {
            const { best } = hitTest(e);
            if (!best) return;
            const pts = this.curves[this.activeChannel];
            const idx = pts.indexOf(best);
            // Cannot delete first or last (endpoints)
            if (idx > 0 && idx < pts.length - 1) {
                pts.splice(idx, 1);
                this.hoverPoint = null;
                this.notifyChange();
                this.draw();
            }
        };

        // Right-click: also remove
        cvs.oncontextmenu = (e) => {
            e.preventDefault();
            const { best } = hitTest(e);
            if (!best) return;
            const pts = this.curves[this.activeChannel];
            const idx = pts.indexOf(best);
            if (idx > 0 && idx < pts.length - 1) {
                pts.splice(idx, 1);
                this.hoverPoint = null;
                this.notifyChange();
                this.draw();
            }
        };

        // Mouse leave: hide crosshair
        cvs.onmouseleave = () => {
            this.mousePos = { x: -1, y: -1 };
            if (!this.draggingPoint) this.draw();
        };
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return this.canvasToNorm(e.clientX - rect.left, e.clientY - rect.top);
    }

    findPoint(pos) {
        // Legacy compat
        const pts = this.curves[this.activeChannel];
        const threshX = 0.04, threshY = 0.04;
        return pts.find(p =>
            Math.abs(p.x - pos.x) < threshX && Math.abs(p.y - pos.y) < threshY
        );
    }

    // ─── Monotonic Cubic Hermite (Fritsch–Carlson) ─────────────
    /**
     * Industry-standard spline interpolation.
     * Guarantees no overshoot between control points — essential for
     * color grading where overshoot means clipping or color inversions.
     *
     * Algorithm: Fritsch & Carlson (1980), "Monotone Piecewise Cubic Interpolation"
     * Used by: DaVinci Resolve, Nuke, Flame, Baselight
     *
     * @param {Array} points - Sorted array of {x, y} control points
     * @returns {Float32Array} 256-entry LUT
     */
    evaluateCurve(points) {
        const n = points.length;
        if (n === 0) return new Float32Array(256).fill(0);
        if (n === 1) return new Float32Array(256).fill(
            Math.max(0, Math.min(1, points[0].y))
        );

        const xs = points.map(p => p.x);
        const ys = points.map(p => p.y);
        const lut = new Float32Array(256);

        // Step 1: Compute secant slopes (Δk)
        const delta = new Float64Array(n - 1);
        const h = new Float64Array(n - 1);
        for (let i = 0; i < n - 1; i++) {
            h[i] = xs[i + 1] - xs[i];
            delta[i] = (h[i] > 1e-10) ? (ys[i + 1] - ys[i]) / h[i] : 0;
        }

        // Step 2: Compute initial tangents (Catmull-Rom style)
        const m = new Float64Array(n);
        if (n === 2) {
            m[0] = delta[0];
            m[1] = delta[0];
        } else {
            // Endpoint tangents: one-sided difference
            m[0] = delta[0];
            m[n - 1] = delta[n - 2];

            // Interior tangents: weighted harmonic mean (Fritsch-Carlson)
            for (let i = 1; i < n - 1; i++) {
                if (delta[i - 1] * delta[i] <= 0) {
                    // Sign change → flat tangent (prevents overshoot)
                    m[i] = 0;
                } else {
                    // Weighted harmonic mean — adapts to non-uniform spacing
                    const w1 = 2 * h[i] + h[i - 1];
                    const w2 = h[i] + 2 * h[i - 1];
                    m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
                }
            }
        }

        // Step 3: Fritsch-Carlson monotonicity preservation
        for (let i = 0; i < n - 1; i++) {
            if (Math.abs(delta[i]) < 1e-10) {
                // Flat segment: force zero tangents at both ends
                m[i] = 0;
                m[i + 1] = 0;
            } else {
                const alpha = m[i] / delta[i];
                const beta = m[i + 1] / delta[i];

                // Fritsch-Carlson criterion: α² + β² ≤ 9
                // If violated, scale tangents to satisfy
                const tau = alpha * alpha + beta * beta;
                if (tau > 9) {
                    const s = 3.0 / Math.sqrt(tau);
                    m[i] = s * alpha * delta[i];
                    m[i + 1] = s * beta * delta[i];
                }
            }
        }

        // Step 4: Evaluate cubic Hermite at each LUT index
        for (let i = 0; i < 256; i++) {
            const t = i / 255;

            // Clamp to endpoint values outside range
            if (t <= xs[0]) { lut[i] = ys[0]; continue; }
            if (t >= xs[n - 1]) { lut[i] = ys[n - 1]; continue; }

            // Find segment (binary search for efficiency)
            let lo = 0, hi = n - 2;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (xs[mid + 1] < t) lo = mid + 1; else hi = mid;
            }
            const k = lo;

            // Hermite basis evaluation
            const hk = h[k];
            if (hk < 1e-10) { lut[i] = ys[k]; continue; }

            const s = (t - xs[k]) / hk;
            const s2 = s * s;
            const s3 = s2 * s;

            // Hermite basis functions:
            // h00 = 2s³ - 3s² + 1,  h10 = s³ - 2s² + s
            // h01 = -2s³ + 3s²,     h11 = s³ - s²
            const h00 = 2 * s3 - 3 * s2 + 1;
            const h10 = s3 - 2 * s2 + s;
            const h01 = -2 * s3 + 3 * s2;
            const h11 = s3 - s2;

            lut[i] = h00 * ys[k] + h10 * hk * m[k] +
                h01 * ys[k + 1] + h11 * hk * m[k + 1];
        }

        // Clamp output to [0, 1]
        for (let i = 0; i < 256; i++) {
            lut[i] = Math.max(0, Math.min(1, lut[i]));
        }

        return lut;
    }

    // Legacy compat aliases
    solveBezierSpline(points) { return this.evaluateCurve(points); }
    solveMonotonicSpline(points) { return this.evaluateCurve(points); }

    // ─── Drawing ───────────────────────────────────────────────
    draw() {
        const ctx = this.ctx;
        const w = this.width, h = this.height;
        const pX = this.plotX, pY = this.plotY, pW = this.plotW, pH = this.plotH;
        const dpr = window.devicePixelRatio || 1;

        // 1. Background
        ctx.fillStyle = '#0d0d12';
        ctx.fillRect(0, 0, w, h);

        // Plot area
        ctx.fillStyle = '#111118';
        ctx.fillRect(pX, pY, pW, pH);

        ctx.save();
        ctx.beginPath();
        ctx.rect(pX, pY, pW, pH);
        ctx.clip();

        // 2. Grid — 10-stop minor, 4-stop major (Resolve standard)
        ctx.lineWidth = 1;

        // Minor grid (10 divisions)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        ctx.beginPath();
        for (let i = 1; i < 10; i++) {
            const n = i * 0.1;
            const { cx: gx } = this.normToCanvas(n, 0);
            const { cy: gy } = this.normToCanvas(0, n);
            ctx.moveTo(gx, pY); ctx.lineTo(gx, pY + pH);
            ctx.moveTo(pX, gy); ctx.lineTo(pX + pW, gy);
        }
        ctx.stroke();

        // Major grid (4 divisions = quarter-stop)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
        ctx.beginPath();
        for (let i = 1; i < 4; i++) {
            const n = i * 0.25;
            const { cx: gx } = this.normToCanvas(n, 0);
            const { cy: gy } = this.normToCanvas(0, n);
            ctx.moveTo(gx, pY); ctx.lineTo(gx, pY + pH);
            ctx.moveTo(pX, gy); ctx.lineTo(pX + pW, gy);
        }
        ctx.stroke();

        // Center cross (midpoint reference)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.10)';
        ctx.beginPath();
        const mid = this.normToCanvas(0.5, 0.5);
        ctx.moveTo(mid.cx - 6, mid.cy); ctx.lineTo(mid.cx + 6, mid.cy);
        ctx.moveTo(mid.cx, mid.cy - 6); ctx.lineTo(mid.cx, mid.cy + 6);
        ctx.stroke();

        // Identity diagonal
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
        ctx.setLineDash([3, 5]);
        ctx.beginPath();
        const d0 = this.normToCanvas(0, 0), d1 = this.normToCanvas(1, 1);
        ctx.moveTo(d0.cx, d0.cy); ctx.lineTo(d1.cx, d1.cy);
        ctx.stroke();
        ctx.setLineDash([]);

        // 3. Histogram (soft fill, Resolve aesthetic)
        if (this.histograms.L) {
            const drawHist = (hist, color) => {
                if (!hist) return;
                ctx.globalAlpha = 0.25;
                ctx.fillStyle = color;
                ctx.beginPath();
                const b0 = this.normToCanvas(0, 0);
                ctx.moveTo(b0.cx, b0.cy);
                for (let i = 0; i < 256; i++) {
                    // Perceptual compression — sqrt makes quiet areas visible
                    const val = Math.pow(hist[i], 0.5) * 0.85;
                    const { cx, cy } = this.normToCanvas(i / 255, val);
                    ctx.lineTo(cx, cy);
                }
                const bEnd = this.normToCanvas(1, 0);
                ctx.lineTo(bEnd.cx, bEnd.cy);
                ctx.closePath();
                ctx.fill();
                ctx.globalAlpha = 1.0;
            };

            if (this.activeChannel === 'RGB') {
                drawHist(this.histograms.L, 'rgba(140, 150, 190, 0.5)');
            } else if (this.activeChannel === 'R') {
                drawHist(this.histograms.R, 'rgba(255, 80, 80, 0.4)');
            } else if (this.activeChannel === 'G') {
                drawHist(this.histograms.G, 'rgba(80, 255, 80, 0.4)');
            } else {
                drawHist(this.histograms.B, 'rgba(80, 120, 255, 0.4)');
            }
        }

        // 4. Ghost curves (inactive channels at low opacity)
        const ghostChannels = this.activeChannel === 'RGB' ? ['R', 'G', 'B'] : ['RGB'];
        ghostChannels.forEach(ch => {
            const pts = this.curves[ch];
            if (pts.length < 2) return;
            const lut = this.evaluateCurve(pts);
            ctx.globalAlpha = 0.15;
            ctx.strokeStyle = this.channelColors[ch];
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const { cx, cy } = this.normToCanvas(i / 255, lut[i]);
                if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
            }
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        });

        // 5. Active curve
        const pts = this.curves[this.activeChannel];
        const lut = this.evaluateCurve(pts);
        const color = this.channelColors[this.activeChannel];

        // Glow layer (bloom effect)
        ctx.shadowBlur = 6;
        ctx.shadowColor = color;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.4;
        ctx.beginPath();
        for (let i = 0; i < 256; i++) {
            const { cx, cy } = this.normToCanvas(i / 255, lut[i]);
            if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Main curve line
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8;
        ctx.globalAlpha = 0.95;
        ctx.beginPath();
        for (let i = 0; i < 256; i++) {
            const { cx, cy } = this.normToCanvas(i / 255, lut[i]);
            if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        ctx.restore(); // Unclip

        // 6. Clipping indicators (top/bottom edge bars)
        if (lut[255] >= 0.99) {
            ctx.fillStyle = 'rgba(255, 60, 60, 0.3)';
            ctx.fillRect(pX, pY, pW, 2);
        }
        if (lut[0] > 0.01) {
            ctx.fillStyle = 'rgba(100, 140, 255, 0.3)';
            ctx.fillRect(pX, pY + pH - 2, pW, 2);
        }

        // 7. Control points
        ctx.save();
        pts.forEach((p, idx) => {
            const { cx, cy } = this.normToCanvas(p.x, p.y);
            const isHover = (p === this.hoverPoint);
            const isDrag = (p === this.draggingPoint);
            const isEndpoint = (idx === 0 || idx === pts.length - 1);
            const r = (isHover || isDrag) ? 5.5 : isEndpoint ? 3.5 : 4;

            // Glow
            if (isHover || isDrag) {
                ctx.shadowBlur = 8;
                ctx.shadowColor = color;
            }

            // Endpoint: square. Interior: circle (DaVinci Resolve convention)
            ctx.fillStyle = (isHover || isDrag) ? '#ffffff' : color;
            if (isEndpoint) {
                ctx.fillRect(cx - r, cy - r, r * 2, r * 2);
                ctx.strokeStyle = '#1a1a1f';
                ctx.lineWidth = 1;
                ctx.strokeRect(cx - r, cy - r, r * 2, r * 2);
            } else {
                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#1a1a1f';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
            ctx.shadowBlur = 0;
        });
        ctx.restore();

        // 8. Interactive crosshair + readout
        if (this.mousePos.x >= 0 && this.mousePos.x <= 1 &&
            this.mousePos.y >= 0 && this.mousePos.y <= 1) {
            const mPos = this.normToCanvas(this.mousePos.x, this.mousePos.y);

            ctx.save();
            ctx.beginPath();
            ctx.rect(pX, pY, pW, pH);
            ctx.clip();

            ctx.strokeStyle = 'rgba(0, 168, 255, 0.2)';
            ctx.setLineDash([3, 4]);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(mPos.cx, pY); ctx.lineTo(mPos.cx, pY + pH);
            ctx.moveTo(pX, mPos.cy); ctx.lineTo(pX + pW, mPos.cy);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }

        // 9. Dragging / hover readout
        if (this.draggingPoint || this.hoverPoint) {
            const target = this.draggingPoint || this.hoverPoint;
            const inVal = Math.round(target.x * 255);
            const outVal = Math.round(target.y * 255);
            const { cx, cy } = this.normToCanvas(target.x, target.y);

            // Background pill
            const text = `${inVal} → ${outVal}`;
            ctx.font = `bold 10px ${this.theme.mono}`;
            const tw = ctx.measureText(text).width + 12;
            const tx = Math.min(cx + 12, pX + pW - tw - 4);
            const ty = Math.max(cy - 8, pY + 4);

            ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
            ctx.beginPath();
            // Rounded rect (compat)
            const rx = tx, ry = ty - 12, rw = tw, rh = 18, rr = 3;
            ctx.moveTo(rx + rr, ry);
            ctx.arcTo(rx + rw, ry, rx + rw, ry + rh, rr);
            ctx.arcTo(rx + rw, ry + rh, rx, ry + rh, rr);
            ctx.arcTo(rx, ry + rh, rx, ry, rr);
            ctx.arcTo(rx, ry, rx + rw, ry, rr);
            ctx.closePath();
            ctx.fill();

            ctx.fillStyle = '#ddd';
            ctx.textAlign = 'left';
            ctx.fillText(text, tx + 6, ty + 1);
        }

        // 10. Axis labels
        ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
        ctx.font = `9px ${this.theme.mono}`;
        ctx.textAlign = 'center';
        [0, 64, 128, 192, 255].forEach(v => {
            const { cx } = this.normToCanvas(v / 255, 0);
            ctx.fillText(v.toString(), cx, pY + pH + 13);
        });
        ctx.textAlign = 'right';
        [0, 64, 128, 192, 255].forEach(v => {
            const { cy } = this.normToCanvas(0, v / 255);
            ctx.fillText(v.toString(), pX - 5, cy + 3);
        });
    }

    // ─── LUT Generation → GPU ──────────────────────────────────
    notifyChange() {
        if (!this.onChange) return;

        // Evaluate all four curves
        const master = this.evaluateCurve(this.curves['RGB']);
        const rCurve = this.evaluateCurve(this.curves['R']);
        const gCurve = this.evaluateCurve(this.curves['G']);
        const bCurve = this.evaluateCurve(this.curves['B']);

        // Levels remapping
        const inBlack = this.levels.inBlack / 255;
        const inWhite = this.levels.inWhite / 255;
        const inRange = Math.max(0.001, inWhite - inBlack);

        // Float32 LUT: 256 × RGBA = 1024 floats
        const lut = new Float32Array(256 * 4);

        // Sub-sample lookup with linear interpolation for precision
        const lookup = (curve, u) => {
            const f = u * 255;
            const k = Math.floor(f);
            const frac = f - k;
            if (k >= 255) return curve[255];
            if (k < 0) return curve[0];
            return curve[k] * (1 - frac) + curve[k + 1] * frac;
        };

        for (let i = 0; i < 256; i++) {
            // Apply input levels
            const leveled = Math.max(0, Math.min(1, (i / 255 - inBlack) / inRange));

            // Master curve (RGB)
            const mVal = lookup(master, leveled);

            // Per-channel curves applied to master output
            const r = lookup(rCurve, mVal);
            const g = lookup(gCurve, mVal);
            const b = lookup(bCurve, mVal);

            lut[i * 4 + 0] = Math.max(0, Math.min(1, r));
            lut[i * 4 + 1] = Math.max(0, Math.min(1, g));
            lut[i * 4 + 2] = Math.max(0, Math.min(1, b));
            lut[i * 4 + 3] = 1.0;
        }

        this.onChange(lut);
    }

    setLevels(inBlack, inWhite) {
        this.levels.inBlack = inBlack;
        this.levels.inWhite = inWhite;
        this.notifyChange();
        this.draw();
    }
}

