/* ==========================================================================
   RADIANCE TECHNICAL DOCUMENTATION SYSTEM (v3.5)
   Comprehensive Manuals, Interactive Node DB, Keyboard Visualizer & Sliders
   ========================================================================== */

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. CHAPTERS & DYNAMIC CONTENT
// ═══════════════════════════════════════════════════════════════════════════════

const CHAPTERS_DATABASE = {
    // --- CATEGORY 1: GETTING STARTED ---
    "welcome": {
        title: "Welcome & Philosophy",
        icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>`,
        content: `
            <div class="welcome-hero">
                <h2>◎ Radiance: Professional VFX & Color Suite</h2>
                <p>A professional, 32-bit floating-point color science suite for ComfyUI. Built from the ground up for film colorists and visual effects artists who require absolute mathematical precision in AI-assisted workflows.</p>
                <div class="hero-stats">
                    <div class="stat-card"><span class="stat-num">32-Bit</span><span class="stat-lbl">Float Pipeline</span></div>
                    <div class="stat-card"><span class="stat-num">OCIO v2.2</span><span class="stat-lbl">Color Science</span></div>
                    <div class="stat-card"><span class="stat-num">121</span><span class="stat-lbl">VFX Nodes</span></div>
                </div>
            </div>

            <h3>Core Philosophy: Display-Referred vs. Scene-Linear</h3>
            <p>Traditional image generators operate inside <strong>display-referred spaces</strong> (like sRGB or Rec.709). These spaces are non-linear, cramped, and clip values strictly at <code>1.0</code> (white). In contrast, film-making pipelines operate in <strong>scene-linear spaces</strong>, where color values represent physical light energy (radiance) and can range from <code>0.0</code> to well over <code>100.0</code> (representing highlights, specular glints, and direct sun exposure).</p>
            
            <div class="alert-box important">
                <span class="alert-title">Why Radiance is Crucial</span>
                <span class="alert-content">When you feed scene-linear values directly into standard AI neural networks or VAE encoders, highlights get crushed, detail is lost, and color values are clipped. Radiance solves this by providing a unified mathematical gateway (via log curves, Reinhard compression, and OpenColorIO displays) that protects high-luminance data, letting you composite, grade, and upscale scene-linear footage safely.</span>
            </div>

            <h3>Industry Standards Integration</h3>
            <p>Radiance integrates directly with standard VFX systems:</p>
            <ul>
                <li><strong>OpenColorIO (OCIO) v2.2:</strong> Resolves camera profiles (LogC3/C4, S-Log3, REDLogFilm) and displays Display Transform views.</li>
                <li><strong>ACES 2.0 (Academy Color Encoding System):</strong> Incorporates analytical RRT (Reference Rendering Transform) and ODT (Output Device Transform) calculations.</li>
                <li><strong>DaVinci Resolve / Nuke Sync:</strong> Directly transfers images, EXR sequences, and ASC CDL grade sheets into host compositing platforms via TCP connections.</li>
            </ul>
        `
    },
    
    "installation": {
        title: "Installation & Troubleshoot",
        icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect><rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect><line x1="6" y1="6" x2="6" y2="6"></line><line x1="6" y1="18" x2="6" y2="18"></line></svg>`,
        content: `
            <h2>Installation Guide & Platform Requirements</h2>
            <p>Ensure your system satisfies the deep-compilation requirements needed to write and parse 32-bit floating point OpenEXR files.</p>

            <h3>Method 1: Installation via ComfyUI Manager</h3>
            <ol>
                <li>Launch your active ComfyUI interface.</li>
                <li>Open the <strong>ComfyUI Manager</strong> console.</li>
                <li>Search for <strong>"Radiance"</strong> in the package listings.</li>
                <li>Click <strong>Install</strong> and restart your server completely.</li>
            </ol>

            <h3>Method 2: Manual Installation (Git & CLI)</h3>
            <p>Open your terminal inside your custom nodes folder and run the commands:</p>
            <pre><button class="copy-btn-anchor" data-copy="cd ComfyUI/custom_nodes && git clone https://github.com/fxtdstudios/radiance.git && cd radiance && pip install -r requirements_windows.txt">Copy Commands</button><code>cd ComfyUI/custom_nodes
git clone https://github.com/fxtdstudios/radiance.git
cd radiance
# Install platform requirements (Windows)
pip install -r requirements_windows.txt</code></pre>

            <div class="alert-box warning">
                <span class="alert-title">Platform Notice for Linux and macOS Users</span>
                <span class="alert-content">Before installing python packages via requirements files, you MUST install <code>libopenexr-dev</code> on your operating system so that the C compiler can successfully compile OpenEXR bindings:
                <br><strong>Ubuntu/Debian:</strong> <code>sudo apt-get install libopenexr-dev</code>
                <br><strong>macOS (Homebrew):</strong> <code>brew install openexr</code></span>
            </div>

            <h2>Common Startup Troubleshooting</h2>
            
            <h3>1. OpenEXR Import Failures</h3>
            <p>If you encounter <code>ModuleNotFoundError: No module named 'OpenEXR'</code>, it indicates the binary pip installation failed to compile. Reinstall with pre-compiled wheels:</p>
            <pre><code>pip install --only-binary :all: OpenEXR</code></pre>

            <h3>2. Missing OpenCV (cv2) DLL Load Error (Windows Server)</h3>
            <p>On Windows Server platforms lacking media features, OpenCV may fail to load with a DLL error. Resolve this by installing the headless variation:</p>
            <pre><code>pip uninstall opencv-python
pip install opencv-python-headless</code></pre>

            <h3>3. Missing OCIO Configs</h3>
            <p>If OCIO transforms throw errors, make sure you have loaded a valid <code>config.ocio</code> path. You can download the standard ACES package and set your system environment variable: <code>OCIO=/path/to/config.ocio</code>.</p>
        `
    },

    "first-workflow": {
        title: "Your First Workflow",
        icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>`,
        content: `
            <h2>Getting Started: Your First HDR Color Pipeline</h2>
            <p>Follow these step-by-step instructions to set up a mathematically protected HDR pipeline that generates images with high highlight retention.</p>

            <ul class="step-list">
                <li class="step-item">
                    <div class="step-num">1</div>
                    <div class="step-content">
                        <h4>Establish the Project Manager Workspace</h4>
                        <p>Right-click in your ComfyUI workspace and add <strong>◎ Pipeline > Radiance Workspace</strong>. Point the Project Path widget to your local working directory. This node tracks your frame increments, saves .rad container history, and configures versioning.</p>
                    </div>
                </li>
                <li class="step-item">
                    <div class="step-num">2</div>
                    <div class="step-content">
                        <h4>Load Scene-Linear Footage</h4>
                        <p>Connect your source EXR sequence or high dynamic range photograph. This data sits in scene-linear space (values exceed 1.0). Run it into <strong>◎ Radiance HDR Auto Log Select</strong> to automatically determine the knee-stops required for the highlights.</p>
                    </div>
                </li>
                <li class="step-item">
                    <div class="step-num">3</div>
                    <div class="step-content">
                        <h4>Compress Highlights for the VAE (Encoder)</h4>
                        <p>VAE models strictly expect values in the display range [0, 1]. Place the <strong>◎ Radiance HDR Encoder</strong> after your image loader. This compresses FP32 values into display-referred ranges using soft-knee Reinhard math, protecting details from clipping.</p>
                    </div>
                </li>
                <li class="step-item">
                    <div class="step-num">4</div>
                    <div class="step-content">
                        <h4>Sampler Denoising (Inference)</h4>
                        <p>Feed the compressed output into a standard VAE Encode node and sampler. Since the VAE sees standard SDR ranges, it processes highlights cleanly without artifacts.</p>
                    </div>
                </li>
                <li class="step-item">
                    <div class="step-num">5</div>
                    <div class="step-content">
                        <h4>Reconstruct Scene-Linear Ranges (Decoder)</h4>
                        <p>After your KSampler completes and decodes back to an image, place the <strong>◎ Radiance HDR Decoder</strong>. Wire the same compression ratio from step 3. The decoder inverts the soft-knee math, recovering full 32-bit linear values above 1.0.</p>
                    </div>
                </li>
                <li class="step-item">
                    <div class="step-num">6</div>
                    <div class="step-content">
                        <h4>Bake Grade & Monitor</h4>
                        <p>Grade the recovered plate with <strong>◎ Radiance Grade</strong> (ASC CDL) or monitor dynamic ranges via the **False Color** or **Waveform Scopes**. Export the output as an EXR sequence or send it directly to Nuke/Resolve.</p>
                    </div>
                </li>
            </ul>

            <div class="alert-box tip">
                <span class="alert-title">Pro Tip: Use the Model Preset Loader</span>
                <span class="alert-content">Different models require different compression settings. Add a <strong>◎ Radiance HDR Model Preset Loader</strong> and select your model (e.g., 'flux' or 'wan') to automatically route recommended compression and normalizations throughout your entire canvas.</span>
            </div>
        `
    },

    "video-tutorials": {
        title: "Video Masterclasses",
        icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>`,
        content: `
            <h2>VFX Masterclass: Dynamic Video Tutorials</h2>
            <p>Accelerate your high-dynamic-range workflow. Watch these production-grade walkthroughs detailing setups from basic workspaces to automated host-editor syncs.</p>

            <div class="tutorials-grid">
                <!-- Video Card 1 -->
                <div class="tutorial-video-card">
                    <div class="tutorial-thumbnail-box">
                        <div class="play-button-overlay">▶</div>
                        <span class="duration-tag">04:12</span>
                        <span class="difficulty-tag beginner">Beginner</span>
                    </div>
                    <div class="tutorial-content-box">
                        <h4>1. Workspace Setup & Versioning</h4>
                        <p>Learn how to establish local project directories, build secure version-control sequences, and manage automated workflow backups using .rad containers.</p>
                        <ul class="tutorial-learn-list">
                            <li>Setting directory environments</li>
                            <li>Automated revision sequences</li>
                            <li>Commit metadata configurations</li>
                        </ul>
                    </div>
                </div>

                <!-- Video Card 2 -->
                <div class="tutorial-video-card">
                    <div class="tutorial-thumbnail-box">
                        <div class="play-button-overlay">▶</div>
                        <span class="duration-tag">08:45</span>
                        <span class="difficulty-tag intermediate">Intermediate</span>
                    </div>
                    <div class="tutorial-content-box">
                        <h4>2. Building Your First ACES 2.0 Pipeline</h4>
                        <p>Deep dive into 32-bit float color science. Master highlight dynamic range compression using soft-knee Reinhard encoders and log curve decodes.</p>
                        <ul class="tutorial-learn-list">
                            <li>Reinhard soft-knee mathematics</li>
                            <li>Per-channel normalizations</li>
                            <li>Display-referred color translations</li>
                        </ul>
                    </div>
                </div>

                <!-- Video Card 3 -->
                <div class="tutorial-video-card">
                    <div class="tutorial-thumbnail-box">
                        <div class="play-button-overlay">▶</div>
                        <span class="duration-tag">11:30</span>
                        <span class="difficulty-tag intermediate">Intermediate</span>
                    </div>
                    <div class="tutorial-content-box">
                        <h4>3. Motion-Aware Temporal Denoising</h4>
                        <p>Eliminate high-frequency neural network flickering in AI video sequences. Tweak motion thresholds and cache high-fidelity VAE latent channels.</p>
                        <ul class="tutorial-learn-list">
                            <li>EMA temporal filters</li>
                            <li>Establishing motion thresholds</li>
                            <li>Preventing temporal bleeding</li>
                        </ul>
                    </div>
                </div>

                <!-- Video Card 4 -->
                <div class="tutorial-video-card">
                    <div class="tutorial-thumbnail-box">
                        <div class="play-button-overlay">▶</div>
                        <span class="duration-tag">14:15</span>
                        <span class="difficulty-tag advanced">Advanced</span>
                    </div>
                    <div class="tutorial-content-box">
                        <h4>4. Host Integrations: Nuke & Resolve</h4>
                        <p>Establish high-speed TCP socket connections to push raw EXR frame sequences, CDL grades, and dynamic nodes directly into active edit suites.</p>
                        <ul class="tutorial-learn-list">
                            <li>TCP socket port bindings</li>
                            <li>ASC CDL XML exportation</li>
                            <li>Automatic Nuke Read-node injectors</li>
                        </ul>
                    </div>
                </div>
            </div>
        `
    },

    // --- CATEGORY 2: REFERENCE & INTERACTIVE MAPS ---
    "noderef": {
        title: "Node Reference Catalog",
        icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>`,
        content: `
            <h2>◎ Dynamic Node Catalog & Mappings</h2>
            <p>Explore inputs, outputs, and mathematical formulas for the Radiance node database. Use the global search bar in the header to filter this reference dynamically.</p>
            <div id="nodes-rendered-slot"></div>
        `
    },

    "shortcuts": {
        title: "Keyboard & Viewer HUD",
        icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2" ry="2"></rect><line x1="6" y1="8" x2="6" y2="8"></line><line x1="10" y1="8" x2="10" y2="8"></line><line x1="14" y1="8" x2="14" y2="8"></line><line x1="18" y1="8" x2="18" y2="8"></line><line x1="6" y1="12" x2="6" y2="12"></line><line x1="10" y1="12" x2="10" y2="12"></line><line x1="14" y1="12" x2="14" y2="12"></line><line x1="18" y1="12" x2="18" y2="12"></line><line x1="7" y1="16" x2="17" y2="16"></line></svg>`,
        content: `
            <h2>◎ Interactive Keyboard Shortcuts</h2>
            <p>Radiance Viewer utilizes professional, keyboard-driven navigation schemas. Hover over any glowing keycap on the interactive keyboard below to inspect its precise functionality.</p>
            <div id="keyboard-rendered-slot"></div>
        `
    },

    // --- CATEGORY 3: ADVANCED CLI & LORA ---
    "cli-training": {
        title: "CLI & LoRA Training",
        icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg>`,
        content: `
            <h2>Advanced LoRA Training: 32-bit Highlight Retention</h2>
            <p>The Radiance training pipeline lets you train custom Low-Rank Adaptations (LoRAs) specifically targeted at retaining specular highlights and high dynamic range details.</p>

            <h3>Step 1: Cache Dataset Latents</h3>
            <p>To avoid running VAE encodings repeatedly during training, compile your dataset of EXR files into a high-performance latent cache directory.</p>
            
            <div class="code-header">
                <span>Bash / CMD Terminal</span>
                <button class="copy-btn" data-copy="python -m radiance.dataset_hdr_lora --exr_dirs /data/hdr_footage --cache_dir /data/latent_cache --vae_path /models/ltxv_vae.safetensors --model_name ltx-video --size 512 --n_frames 1 --compression_ratio 0.5">Copy Code</button>
            </div>
            <pre><code>python -m radiance.dataset_hdr_lora \\
    --exr_dirs /data/hdr_footage \\
    --cache_dir /data/latent_cache \\
    --vae_path /models/ltxv_vae.safetensors \\
    --model_name ltx-video \\
    --size 512 \\
    --n_frames 1 \\
    --compression_ratio 0.5</code></pre>

            <h3>Step 2: Train the LoRA</h3>
            <p>Train the weights using highlight-weighted loss coefficients. This applies extra penalization to pixels exceeding 0.85 compressed luminance to force detail retention.</p>

            <div class="code-header">
                <span>Bash / CMD Terminal</span>
                <button class="copy-btn" data-copy="python -m radiance.train_hdr_lora --cache_dir /data/latent_cache --model_path /models/ltxv.safetensors --output_dir /checkpoints/hdr_lora_ltxv --model_name ltx-video --rank 16 --alpha 16.0 --steps 5000 --batch_size 2 --lr 1e-4 --save_every 500">Copy Code</button>
            </div>
            <pre><code>python -m radiance.train_hdr_lora \\
    --cache_dir /data/latent_cache \\
    --model_path /models/ltxv.safetensors \\
    --output_dir /checkpoints/hdr_lora_ltxv \\
    --model_name ltx-video \\
    --rank 16 \\
    --alpha 16.0 \\
    --steps 5000 \\
    --batch_size 2 \\
    --lr 1e-4 --save_every 500</code></pre>

            <h3>Training Parameters Reference</h3>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Recommended</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><code>--rank</code></td>
                        <td>16 or 32</td>
                        <td>Dimension of low-rank matrices. Higher rank captures more grain detail but increases file size.</td>
                    </tr>
                    <tr>
                        <td><code>--alpha</code></td>
                        <td>16.0</td>
                        <td>Scaling multiplier for LoRA weight updates. Should match the rank value.</td>
                    </tr>
                    <tr>
                        <td><code>--lr</code></td>
                        <td>1e-4</td>
                        <td>Learning rate. Flow-matching networks require slightly higher learning rates than DDPM.</td>
                    </tr>
                    <tr>
                        <td><code>--compression_ratio</code></td>
                        <td>0.50</td>
                        <td>Soft-knee Reinhard compression ratio used to normalize EXR highlights.</td>
                    </tr>
                </tbody>
            </table>
        `
    },

    // --- CATEGORY 4: DEVELOPER GUIDE ---
    "developer-api": {
        title: "Developer & API Specs",
        icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>`,
        content: `
            <h2>Developer Reference: Sockets, Schemas & HTTP Routers</h2>
            <p>Technical specifications for pipeline engineers looking to integrate Radiance into custom studio pipelines or automated render queues.</p>

            <h3>1. The Secure .rad (v2) Binary Container Layout</h3>
            <p>Radiance stores pipeline templates inside a binary wrapper with verification checksums to avoid corruption. The layout conforms to this exact structure:</p>
            
            <table>
                <thead>
                    <tr>
                        <th>Section</th>
                        <th>Offset</th>
                        <th>Type</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Magic Bytes</strong></td>
                        <td>0 - 3</td>
                        <td>4s</td>
                        <td><code>RAD!</code> (Magic validation code)</td>
                    </tr>
                    <tr>
                        <td><strong>Version</strong></td>
                        <td>4 - 5</td>
                        <td>unsigned short</td>
                        <td>Format version (currently <code>2</code>)</td>
                    </tr>
                    <tr>
                        <td><strong>Metadata Size</strong></td>
                        <td>6 - 9</td>
                        <td>unsigned int</td>
                        <td>Byte size of metadata JSON header</td>
                    </tr>
                    <tr>
                        <td><strong>Graph Size</strong></td>
                        <td>10 - 13</td>
                        <td>unsigned int</td>
                        <td>Byte size of compressed graph</td>
                    </tr>
                    <tr>
                        <td><strong>Payload</strong></td>
                        <td>14+</td>
                        <td>JSON + zlib</td>
                        <td>Metadata string followed by zlib-compressed workflow</td>
                    </tr>
                    <tr>
                        <td><strong>SHA-256 Checksum</strong></td>
                        <td>EOF - 32</td>
                        <td>32 bytes</td>
                        <td>Integrity signature computed over all previous bytes</td>
                    </tr>
                </tbody>
            </table>

            <h3>2. Studio TCP Socket Protocol</h3>
            <p>The Nuke and DaVinci export pathways communicate with local host applications using TCP network sockets on ports <code>1986</code> (Nuke) and <code>1987</code> (DaVinci). Send JSON packets formatted as follows:</p>
            <pre><code>{
  "action": "import_render",
  "filepath": "/absolute/path/to/render.exr",
  "frame_start": 1001,
  "cdl_metadata": {
    "slope": [1.0, 1.0, 1.0],
    "offset": [0.0, 0.0, 0.0],
    "power": [1.0, 1.0, 1.0],
    "saturation": 1.0
  }
}</code></pre>

            <h3>3. ComfyUI HTTP Endpoints</h3>
            <p>Radiance registers several routes on the internal ComfyUI web server:</p>
            <ul>
                <li><code>POST /radiance/workflows/pack</code> — Packs standard ComfyUI JSON graphs into secure <code>.rad</code> binaries.</li>
                <li><code>POST /radiance/workflows/unpack</code> — Extracts the zlib graph and JSON metadata from a binary <code>.rad</code> plate.</li>
                <li><code>GET /radiance/ocio/displays</code> — Scans loaded configs and lists ACES display configurations.</li>
            </ul>
        `
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. INTERACTIVE REFERENCE MANUAL ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

const NODE_DATABASE = [
  {
    "id": "RadianceCDLTransform",
    "name": "◎ Radiance CDL Transform",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "slope_r",
        "type": "FLOAT"
      },
      {
        "name": "slope_g",
        "type": "FLOAT"
      },
      {
        "name": "slope_b",
        "type": "FLOAT"
      },
      {
        "name": "offset_r",
        "type": "FLOAT"
      },
      {
        "name": "offset_g",
        "type": "FLOAT"
      },
      {
        "name": "offset_b",
        "type": "FLOAT"
      },
      {
        "name": "power_r",
        "type": "FLOAT"
      },
      {
        "name": "power_g",
        "type": "FLOAT"
      },
      {
        "name": "power_b",
        "type": "FLOAT"
      },
      {
        "name": "saturation",
        "type": "FLOAT"
      },
      {
        "name": "cdl_data",
        "type": "STRING (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "cdl_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance CDL Transform in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceCDLImport",
    "name": "◎ Radiance CDL Import",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "file_path",
        "type": "STRING"
      }
    ],
    "outputs": [
      {
        "name": "cdl_data",
        "type": "STRING"
      },
      {
        "name": "slope_r",
        "type": "FLOAT"
      },
      {
        "name": "slope_g",
        "type": "FLOAT"
      },
      {
        "name": "slope_b",
        "type": "FLOAT"
      },
      {
        "name": "offset_r",
        "type": "FLOAT"
      },
      {
        "name": "offset_g",
        "type": "FLOAT"
      },
      {
        "name": "offset_b",
        "type": "FLOAT"
      },
      {
        "name": "power_r",
        "type": "FLOAT"
      },
      {
        "name": "power_g",
        "type": "FLOAT"
      },
      {
        "name": "power_b",
        "type": "FLOAT"
      },
      {
        "name": "saturation",
        "type": "FLOAT"
      }
    ],
    "tip": "Use ◎ Radiance CDL Import in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceCDLExport",
    "name": "◎ Radiance CDL Export",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "file_path",
        "type": "STRING"
      },
      {
        "name": "slope_r",
        "type": "FLOAT"
      },
      {
        "name": "slope_g",
        "type": "FLOAT"
      },
      {
        "name": "slope_b",
        "type": "FLOAT"
      },
      {
        "name": "offset_r",
        "type": "FLOAT"
      },
      {
        "name": "offset_g",
        "type": "FLOAT"
      },
      {
        "name": "offset_b",
        "type": "FLOAT"
      },
      {
        "name": "power_r",
        "type": "FLOAT"
      },
      {
        "name": "power_g",
        "type": "FLOAT"
      },
      {
        "name": "power_b",
        "type": "FLOAT"
      },
      {
        "name": "saturation",
        "type": "FLOAT"
      },
      {
        "name": "cdl_data",
        "type": "STRING (optional)"
      }
    ],
    "outputs": [
      {
        "name": "file_path",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance CDL Export in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceWhiteBalance",
    "name": "◎ Radiance White Balance",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "mode",
        "type": "COMBO (3 options)"
      },
      {
        "name": "preset",
        "type": "COMBO (6 options)"
      },
      {
        "name": "temperature",
        "type": "FLOAT"
      },
      {
        "name": "tint",
        "type": "FLOAT"
      },
      {
        "name": "src_illuminant",
        "type": "COMBO (9 options)"
      },
      {
        "name": "dst_illuminant",
        "type": "COMBO (9 options)"
      },
      {
        "name": "gain_r",
        "type": "FLOAT"
      },
      {
        "name": "gain_g",
        "type": "FLOAT"
      },
      {
        "name": "gain_b",
        "type": "FLOAT"
      },
      {
        "name": "strength",
        "type": "FLOAT"
      },
      {
        "name": "grade_info_in",
        "type": "STRING (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "grade_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance White Balance in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceColorSpaceConvert",
    "name": "◎ Radiance Colorspace Convert",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "src_space",
        "type": "COMBO (16 options)"
      },
      {
        "name": "dst_space",
        "type": "COMBO (16 options)"
      },
      {
        "name": "direction",
        "type": "COMBO (2 options)"
      },
      {
        "name": "strength",
        "type": "FLOAT"
      },
      {
        "name": "grade_info_in",
        "type": "STRING (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "grade_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance Colorspace Convert in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceACESTransform",
    "name": "◎ Radiance ACES Transform",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "odt",
        "type": "COMBO (4 options)"
      },
      {
        "name": "exposure_offset",
        "type": "FLOAT"
      },
      {
        "name": "peak_nits",
        "type": "FLOAT"
      },
      {
        "name": "saturation",
        "type": "FLOAT"
      },
      {
        "name": "grade_info_in",
        "type": "STRING (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "aces_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance ACES Transform in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceBitDepthDegrade",
    "name": "◎ Radiance Bit Depth Degrade",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "bit_depth",
        "type": "INT"
      },
      {
        "name": "dither_mode",
        "type": "COMBO (3 options)"
      },
      {
        "name": "delta_gain",
        "type": "FLOAT (optional)"
      },
      {
        "name": "banding_threshold",
        "type": "FLOAT (optional)"
      },
      {
        "name": "restore_from_quantized",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "quantized",
        "type": "IMAGE"
      },
      {
        "name": "delta_amplified",
        "type": "IMAGE"
      },
      {
        "name": "banding_mask",
        "type": "IMAGE"
      },
      {
        "name": "metrics",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance Bit Depth Degrade in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceHueCurves",
    "name": "◎ Radiance Hue Curves",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "mode",
        "type": "COMBO (3 options)"
      },
      {
        "name": "control_points",
        "type": "STRING"
      },
      {
        "name": "strength",
        "type": "FLOAT"
      },
      {
        "name": "grade_info",
        "type": "STRING (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Radiance Hue Curves in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceCurves",
    "name": "◎ Radiance Curves",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "master",
        "type": "STRING"
      },
      {
        "name": "red",
        "type": "STRING"
      },
      {
        "name": "green",
        "type": "STRING"
      },
      {
        "name": "blue",
        "type": "STRING"
      },
      {
        "name": "strength",
        "type": "FLOAT"
      },
      {
        "name": "grade_info_in",
        "type": "STRING (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "grade_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance Curves in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceGrade",
    "name": "◎ Radiance Grade",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "preset",
        "type": "COMBO (13 options)"
      },
      {
        "name": "preset_strength",
        "type": "FLOAT"
      },
      {
        "name": "reference_image",
        "type": "IMAGE (optional)"
      },
      {
        "name": "match_strength",
        "type": "FLOAT (optional)"
      },
      {
        "name": "preset_file",
        "type": "STRING (optional)"
      },
      {
        "name": "lift_r",
        "type": "FLOAT (optional)"
      },
      {
        "name": "lift_g",
        "type": "FLOAT (optional)"
      },
      {
        "name": "lift_b",
        "type": "FLOAT (optional)"
      },
      {
        "name": "gamma_r",
        "type": "FLOAT (optional)"
      },
      {
        "name": "gamma_g",
        "type": "FLOAT (optional)"
      },
      {
        "name": "gamma_b",
        "type": "FLOAT (optional)"
      },
      {
        "name": "gain_r",
        "type": "FLOAT (optional)"
      },
      {
        "name": "gain_g",
        "type": "FLOAT (optional)"
      },
      {
        "name": "gain_b",
        "type": "FLOAT (optional)"
      },
      {
        "name": "offset_r",
        "type": "FLOAT (optional)"
      },
      {
        "name": "offset_g",
        "type": "FLOAT (optional)"
      },
      {
        "name": "offset_b",
        "type": "FLOAT (optional)"
      },
      {
        "name": "contrast",
        "type": "FLOAT (optional)"
      },
      {
        "name": "pivot",
        "type": "FLOAT (optional)"
      },
      {
        "name": "saturation",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "grade_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance Grade in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceApplyGradeInfo",
    "name": "◎ Radiance Apply Grade Info",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "grade_info",
        "type": "STRING"
      },
      {
        "name": "strength",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "grade_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance Apply Grade Info in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceGradeMatch",
    "name": "◎ Radiance Grade Match",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "source",
        "type": "IMAGE"
      },
      {
        "name": "reference",
        "type": "IMAGE"
      },
      {
        "name": "strength",
        "type": "FLOAT"
      }
    ],
    "outputs": [
      {
        "name": "matched_image",
        "type": "IMAGE"
      },
      {
        "name": "grade_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance Grade Match in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceOCIOContext",
    "name": "◎ Radiance OCIO Context",
    "zone": "FXTD STUDIOS/Radiance/◎ Color",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "config_path",
        "type": "STRING"
      },
      {
        "name": "working_space",
        "type": "STRING"
      }
    ],
    "outputs": [
      {
        "name": "ocio_context",
        "type": "RADIANCE_OCIO"
      }
    ],
    "tip": "Use ◎ Radiance OCIO Context in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceQC",
    "name": "◎ Radiance QC",
    "zone": "FXTD STUDIOS/Radiance/◎ QC & Debug",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "mode",
        "type": "COMBO (2 options)"
      },
      {
        "name": "image",
        "type": "IMAGE (optional)"
      },
      {
        "name": "black_threshold",
        "type": "FLOAT (optional)"
      },
      {
        "name": "white_threshold",
        "type": "FLOAT (optional)"
      },
      {
        "name": "overlay_opacity",
        "type": "FLOAT (optional)"
      },
      {
        "name": "banding_threshold",
        "type": "FLOAT (optional)"
      },
      {
        "name": "enable_focus_check",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "enable_artifacts_check",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "enable_noise_check",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "fail_on_errors",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "qc_report_json",
        "type": "STRING (optional)"
      },
      {
        "name": "output_path",
        "type": "STRING (optional)"
      },
      {
        "name": "filename_prefix",
        "type": "STRING (optional)"
      },
      {
        "name": "export_format",
        "type": "COMBO (4 options) (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "text_report",
        "type": "STRING"
      },
      {
        "name": "json_report",
        "type": "STRING"
      },
      {
        "name": "status",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance QC in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadiancePolicyGuard",
    "name": "◎ Policy Guard",
    "zone": "FXTD STUDIOS/Radiance/◎ QC & Debug",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "mode",
        "type": "COMBO (2 options)"
      },
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "preset",
        "type": "COMBO (5 options) (optional)"
      },
      {
        "name": "policy_file",
        "type": "STRING (optional)"
      },
      {
        "name": "custom_max_peak_nits",
        "type": "FLOAT (optional)"
      },
      {
        "name": "custom_max_clipping",
        "type": "FLOAT (optional)"
      },
      {
        "name": "custom_max_black_crush",
        "type": "FLOAT (optional)"
      },
      {
        "name": "custom_max_saturation",
        "type": "FLOAT (optional)"
      },
      {
        "name": "policy",
        "type": "STRING (optional)"
      },
      {
        "name": "max_clipping",
        "type": "FLOAT (optional)"
      },
      {
        "name": "max_black_crush",
        "type": "FLOAT (optional)"
      },
      {
        "name": "max_saturation",
        "type": "FLOAT (optional)"
      },
      {
        "name": "max_peak_nits",
        "type": "FLOAT (optional)"
      },
      {
        "name": "require_metadata",
        "type": "STRING (optional)"
      },
      {
        "name": "metadata_present",
        "type": "STRING (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "passed",
        "type": "BOOLEAN"
      },
      {
        "name": "data1",
        "type": "STRING"
      },
      {
        "name": "data2",
        "type": "STRING"
      },
      {
        "name": "score",
        "type": "INT"
      }
    ],
    "tip": "Use ◎ Policy Guard in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceACES2Tonescale",
    "name": "◎ ACES 2.0 Tonescale",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "Apply the Daniele Evo forward tonescale (ACES 2.0 official curve).",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "peak_nits",
        "type": "FLOAT"
      },
      {
        "name": "mode",
        "type": "COMBO (2 options)"
      },
      {
        "name": "contrast_g",
        "type": "FLOAT (optional)"
      },
      {
        "name": "grey_target",
        "type": "FLOAT (optional)"
      },
      {
        "name": "toe_scene",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "curve_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ ACES 2.0 Tonescale in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceACES2ReachGamutCompress",
    "name": "◎ ACES 2.0 Gamut Compress",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "ACES 2.0 Reach-Based Gamut Compression.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "strength",
        "type": "FLOAT"
      },
      {
        "name": "limit_cyan",
        "type": "FLOAT (optional)"
      },
      {
        "name": "limit_magenta",
        "type": "FLOAT (optional)"
      },
      {
        "name": "limit_yellow",
        "type": "FLOAT (optional)"
      },
      {
        "name": "threshold_cyan",
        "type": "FLOAT (optional)"
      },
      {
        "name": "threshold_magenta",
        "type": "FLOAT (optional)"
      },
      {
        "name": "threshold_yellow",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "compress_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ ACES 2.0 Gamut Compress in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceACES2OutputTransformFull",
    "name": "◎ ACES 2.0 Output Transform",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "Complete ACES 2.0 Output Transform (reference-accurate).",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "input_colorspace",
        "type": "COMBO (4 options)"
      },
      {
        "name": "output_transform",
        "type": "COMBO (8 options)"
      },
      {
        "name": "peak_luminance",
        "type": "FLOAT (optional)"
      },
      {
        "name": "surround",
        "type": "COMBO (3 options) (optional)"
      },
      {
        "name": "exposure_adjust",
        "type": "FLOAT (optional)"
      },
      {
        "name": "creative_white_scale",
        "type": "FLOAT (optional)"
      },
      {
        "name": "gamut_compress_strength",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "transform_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ ACES 2.0 Output Transform in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceHDRColorPipeline",
    "name": "◎ HDR Color Pipeline",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "◎ Radiance HDR Color Pipeline",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "encoding",
        "type": "COMBO (10 options)"
      },
      {
        "name": "compression_ratio",
        "type": "FLOAT"
      },
      {
        "name": "source_primaries",
        "type": "COMBO (5 options) (optional)"
      },
      {
        "name": "target_primaries",
        "type": "COMBO (5 options) (optional)"
      },
      {
        "name": "chromatic_adaptation",
        "type": "COMBO (5 options) (optional)"
      },
      {
        "name": "pq_peak_nits",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "vae_image",
        "type": "IMAGE"
      },
      {
        "name": "scene_linear",
        "type": "IMAGE"
      },
      {
        "name": "peak_linear",
        "type": "FLOAT"
      },
      {
        "name": "colorspace_json",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ HDR Color Pipeline in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceHDREncode",
    "name": "◎ HDR Encode",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "◎ Radiance HDR Encode",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "format",
        "type": "COMBO (2 options)"
      },
      {
        "name": "peak_nits",
        "type": "COMBO (4 options) (optional)"
      },
      {
        "name": "reference_white_nits",
        "type": "FLOAT (optional)"
      },
      {
        "name": "scene_linear_gain",
        "type": "FLOAT (optional)"
      },
      {
        "name": "apply_bt2020",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "encoded_image",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ HDR Encode in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceHDRMonitor",
    "name": "◎ HDR Monitor",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "◎ Radiance HDR Monitor",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "mode",
        "type": "COMBO (3 options)"
      },
      {
        "name": "operator",
        "type": "COMBO (4 options) (optional)"
      },
      {
        "name": "exposure",
        "type": "FLOAT (optional)"
      },
      {
        "name": "saturation",
        "type": "FLOAT (optional)"
      },
      {
        "name": "gamma",
        "type": "FLOAT (optional)"
      },
      {
        "name": "reinhard_white",
        "type": "FLOAT (optional)"
      },
      {
        "name": "peak_nits",
        "type": "FLOAT (optional)"
      },
      {
        "name": "gamma_correct_sdr",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "preview",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ HDR Monitor in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceHDRAutoLogSelect",
    "name": "◎ HDR Auto Log Select",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "override",
        "type": "COMBO (6 options)"
      },
      {
        "name": "model_hint",
        "type": "STRING (optional)"
      }
    ],
    "outputs": [
      {
        "name": "log_format",
        "type": "STRING"
      },
      {
        "name": "compression_ratio",
        "type": "FLOAT"
      },
      {
        "name": "stops_detected",
        "type": "FLOAT"
      },
      {
        "name": "model_preset_used",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ HDR Auto Log Select in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceHDRDiagnostics",
    "name": "◎ HDR Diagnostics",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "compression_ratio",
        "type": "FLOAT (optional)"
      },
      {
        "name": "model_preset_used",
        "type": "STRING (optional)"
      },
      {
        "name": "stats_json",
        "type": "STRING (optional)"
      },
      {
        "name": "coherence_map",
        "type": "IMAGE (optional)"
      },
      {
        "name": "colorspace",
        "type": "COMBO (4 options) (optional)"
      }
    ],
    "outputs": [
      {
        "name": "report_json",
        "type": "STRING"
      },
      {
        "name": "psnr_estimate",
        "type": "FLOAT"
      },
      {
        "name": "peak_stops",
        "type": "FLOAT"
      },
      {
        "name": "peak_nit",
        "type": "FLOAT"
      },
      {
        "name": "ev_range",
        "type": "FLOAT"
      },
      {
        "name": "clipped_pct",
        "type": "FLOAT"
      },
      {
        "name": "is_hdr",
        "type": "BOOLEAN"
      }
    ],
    "tip": "Use ◎ HDR Diagnostics in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceClipDetector",
    "name": "◎ Clip Detector",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "threshold",
        "type": "FLOAT"
      },
      {
        "name": "channel_mode",
        "type": "COMBO (3 options)"
      },
      {
        "name": "soft_edge",
        "type": "FLOAT"
      },
      {
        "name": "dilate_px",
        "type": "INT"
      }
    ],
    "outputs": [
      {
        "name": "clip_mask",
        "type": "MASK"
      },
      {
        "name": "clip_fraction",
        "type": "FLOAT"
      },
      {
        "name": "visualization",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Clip Detector in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceSDRToHDRPrepare",
    "name": "◎ SDR to HDR Prepare",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "clip_mask",
        "type": "MASK"
      },
      {
        "name": "compression_ratio",
        "type": "FLOAT"
      },
      {
        "name": "inverse_eotf",
        "type": "COMBO (4 options) (optional)"
      },
      {
        "name": "highlight_boost",
        "type": "FLOAT (optional)"
      },
      {
        "name": "boost_gamma",
        "type": "FLOAT (optional)"
      },
      {
        "name": "mask_feather",
        "type": "INT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "mask",
        "type": "MASK"
      },
      {
        "name": "stats_json",
        "type": "STRING"
      },
      {
        "name": "peak_linear",
        "type": "FLOAT"
      }
    ],
    "tip": "Use ◎ SDR to HDR Prepare in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceHDRHighlightComposite",
    "name": "◎ HDR Highlight Composite",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "original_image",
        "type": "IMAGE"
      },
      {
        "name": "hdr_image",
        "type": "IMAGE"
      },
      {
        "name": "clip_mask",
        "type": "MASK"
      },
      {
        "name": "inverse_eotf",
        "type": "COMBO (4 options) (optional)"
      },
      {
        "name": "blend_softness",
        "type": "INT (optional)"
      },
      {
        "name": "highlight_strength",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ HDR Highlight Composite in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceSDRtoHDRExpand",
    "name": "◎ SDR to HDR Expand",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "◎ Radiance SDR to HDR Expand",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "inverse_oetf",
        "type": "COMBO (3 options)"
      },
      {
        "name": "threshold",
        "type": "FLOAT"
      },
      {
        "name": "expansion_gain",
        "type": "FLOAT"
      },
      {
        "name": "expansion_gamma",
        "type": "FLOAT"
      },
      {
        "name": "smoothness",
        "type": "FLOAT"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ SDR to HDR Expand in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceHDRSynthesisEngine",
    "name": "◎ HDR Synthesis Engine",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "◎ Radiance HDR Synthesis Engine",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "energy_target",
        "type": "FLOAT"
      },
      {
        "name": "recovery_iters",
        "type": "INT"
      },
      {
        "name": "chroma_preservation",
        "type": "FLOAT"
      },
      {
        "name": "guidance_mask",
        "type": "MASK (optional)"
      },
      {
        "name": "guidance_nits",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "highlight_mask",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ HDR Synthesis Engine in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceRelightEngine",
    "name": "◎ Relight Engine",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "◎ Radiance Relight Engine",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "normal_map",
        "type": "IMAGE"
      },
      {
        "name": "light_dir_x",
        "type": "FLOAT"
      },
      {
        "name": "light_dir_y",
        "type": "FLOAT"
      },
      {
        "name": "light_dir_z",
        "type": "FLOAT"
      },
      {
        "name": "light_color_r",
        "type": "FLOAT"
      },
      {
        "name": "light_color_g",
        "type": "FLOAT"
      },
      {
        "name": "light_color_b",
        "type": "FLOAT"
      },
      {
        "name": "diffuse_intensity",
        "type": "FLOAT"
      },
      {
        "name": "specular_intensity",
        "type": "FLOAT"
      },
      {
        "name": "specular_roughness",
        "type": "FLOAT"
      },
      {
        "name": "camera",
        "type": "RADIANCE_CAMERA (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "lighting_pass_only",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Relight Engine in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceHDRLatentEncoder",
    "name": "◎ HDR Latent Encoder",
    "zone": "FXTD STUDIOS/Radiance/◎ HDR",
    "desc": "◎ Radiance HDR Latent Encoder",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "vae",
        "type": "VAE"
      },
      {
        "name": "mode",
        "type": "COMBO (2 options)"
      },
      {
        "name": "compression_ratio",
        "type": "FLOAT (optional)"
      },
      {
        "name": "exposure_offset",
        "type": "FLOAT (optional)"
      },
      {
        "name": "energy_normalization",
        "type": "FLOAT (optional)"
      },
      {
        "name": "normalize_channels",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "norm_center",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "latent",
        "type": "LATENT"
      },
      {
        "name": "channel_stats",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ HDR Latent Encoder in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceRead",
    "name": "◎ Radiance Read",
    "zone": "FXTD STUDIOS/Radiance/◎ IO & Delivery",
    "desc": "Universal reader — images, EXR, video, numbered sequences.",
    "inputs": [
      {
        "name": "browse",
        "type": "COMBO (6 options)"
      },
      {
        "name": "media_type",
        "type": "COMBO (4 options) (optional)"
      },
      {
        "name": "path",
        "type": "STRING (optional)"
      },
      {
        "name": "color_space",
        "type": "COMBO (9 options) (optional)"
      },
      {
        "name": "start_frame",
        "type": "INT (optional)"
      },
      {
        "name": "end_frame",
        "type": "INT (optional)"
      },
      {
        "name": "frame_step",
        "type": "INT (optional)"
      },
      {
        "name": "max_video_frames",
        "type": "INT (optional)"
      },
      {
        "name": "proxy_scale",
        "type": "FLOAT (optional)"
      },
      {
        "name": "missing_frames",
        "type": "COMBO (3 options) (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "mask",
        "type": "MASK"
      }
    ],
    "tip": "Use ◎ Radiance Read in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceWrite",
    "name": "◎ Radiance Write",
    "zone": "FXTD STUDIOS/Radiance/◎ IO & Delivery",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "*"
      },
      {
        "name": "output_path",
        "type": "STRING"
      },
      {
        "name": "format",
        "type": "COMBO (20 options)"
      },
      {
        "name": "filename",
        "type": "STRING (optional)"
      },
      {
        "name": "version",
        "type": "INT (optional)"
      },
      {
        "name": "color_space",
        "type": "COMBO (10 options) (optional)"
      },
      {
        "name": "fps",
        "type": "FLOAT (optional)"
      },
      {
        "name": "quality",
        "type": "INT (optional)"
      },
      {
        "name": "exr_compression",
        "type": "COMBO (7 options) (optional)"
      },
      {
        "name": "start_frame",
        "type": "INT (optional)"
      },
      {
        "name": "frame_padding",
        "type": "INT (optional)"
      },
      {
        "name": "audio_source",
        "type": "STRING (optional)"
      },
      {
        "name": "broadcast_safe",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "overwrite",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "proxy_scale",
        "type": "FLOAT (optional)"
      },
      {
        "name": "audio",
        "type": "AUDIO (optional)"
      }
    ],
    "outputs": [],
    "tip": "Use ◎ Radiance Write in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceEXRMultiPart",
    "name": "◎ Radiance EXR Multi-Part",
    "zone": "FXTD STUDIOS/Radiance/◎ IO & Delivery",
    "desc": "Write a named multi-part EXR v2 combining up to 6 AOV layers into one",
    "inputs": [
      {
        "name": "filename_prefix",
        "type": "STRING"
      },
      {
        "name": "beauty",
        "type": "IMAGE"
      },
      {
        "name": "bit_depth",
        "type": "COMBO (2 options)"
      },
      {
        "name": "compression",
        "type": "COMBO (10 options)"
      },
      {
        "name": "depth",
        "type": "IMAGE (optional)"
      },
      {
        "name": "normal",
        "type": "IMAGE (optional)"
      },
      {
        "name": "albedo",
        "type": "IMAGE (optional)"
      },
      {
        "name": "custom_1",
        "type": "IMAGE (optional)"
      },
      {
        "name": "custom_1_name",
        "type": "STRING (optional)"
      },
      {
        "name": "custom_2",
        "type": "IMAGE (optional)"
      },
      {
        "name": "custom_2_name",
        "type": "STRING (optional)"
      },
      {
        "name": "output_path",
        "type": "STRING (optional)"
      },
      {
        "name": "remote_path",
        "type": "STRING (optional)"
      },
      {
        "name": "frame_index",
        "type": "INT (optional)"
      },
      {
        "name": "custom_metadata",
        "type": "STRING (optional)"
      }
    ],
    "outputs": [
      {
        "name": "output_path",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance EXR Multi-Part in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceLoadImageMask",
    "name": "◎ Radiance Load Image Mask",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "Advanced Image Loader + Mask Editor for Radiance.",
    "inputs": [],
    "outputs": [
      {
        "name": "IMAGE",
        "type": "IMAGE"
      },
      {
        "name": "MASK",
        "type": "MASK"
      }
    ],
    "tip": "Use ◎ Radiance Load Image Mask in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceDepthMapGenerator",
    "name": "◎ Depth Map Generator",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "model_size",
        "type": "COMBO (3 options)"
      },
      {
        "name": "normalize",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "invert",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "blur_edges",
        "type": "FLOAT (optional)"
      },
      {
        "name": "use_gpu",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "depth_map",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Depth Map Generator in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceOpticalFlow",
    "name": "◎ Optical Flow",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "◎ Radiance Optical Flow",
    "inputs": [
      {
        "name": "images",
        "type": "IMAGE"
      },
      {
        "name": "preset",
        "type": "COMBO (3 options)"
      },
      {
        "name": "flow_scale",
        "type": "FLOAT"
      },
      {
        "name": "visualize",
        "type": "BOOLEAN"
      }
    ],
    "outputs": [
      {
        "name": "motion_vectors",
        "type": "IMAGE"
      },
      {
        "name": "visualization",
        "type": "IMAGE"
      },
      {
        "name": "stats",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Optical Flow in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceMotionBlur",
    "name": "◎ Motion Blur",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "◎ Radiance Physical Motion Blur",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "motion_vectors",
        "type": "IMAGE"
      },
      {
        "name": "shutter_angle",
        "type": "FLOAT"
      },
      {
        "name": "samples",
        "type": "INT"
      },
      {
        "name": "energy_conservation",
        "type": "BOOLEAN"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Motion Blur in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceLensDistortion",
    "name": "◎ Lens Distortion",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "◎ Radiance Lens Distortion",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "k1",
        "type": "FLOAT"
      },
      {
        "name": "k2",
        "type": "FLOAT"
      },
      {
        "name": "scale",
        "type": "FLOAT"
      },
      {
        "name": "center_x",
        "type": "FLOAT"
      },
      {
        "name": "center_y",
        "type": "FLOAT"
      },
      {
        "name": "padding_mode",
        "type": "COMBO (3 options)"
      },
      {
        "name": "invert",
        "type": "BOOLEAN"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "st_map",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Lens Distortion in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceChromaticAberration",
    "name": "◎ Chromatic Aberration",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "◎ Radiance Chromatic Aberration",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "shift_r",
        "type": "FLOAT"
      },
      {
        "name": "shift_g",
        "type": "FLOAT"
      },
      {
        "name": "shift_b",
        "type": "FLOAT"
      },
      {
        "name": "center_x",
        "type": "FLOAT"
      },
      {
        "name": "center_y",
        "type": "FLOAT"
      },
      {
        "name": "invert",
        "type": "BOOLEAN"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Chromatic Aberration in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceAnamorphicStreaks",
    "name": "◎ Anamorphic Streaks",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "◎ Radiance Anamorphic Streaks",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "threshold",
        "type": "FLOAT"
      },
      {
        "name": "streak_length",
        "type": "INT"
      },
      {
        "name": "streak_color_r",
        "type": "FLOAT"
      },
      {
        "name": "streak_color_g",
        "type": "FLOAT"
      },
      {
        "name": "streak_color_b",
        "type": "FLOAT"
      },
      {
        "name": "intensity",
        "type": "FLOAT"
      },
      {
        "name": "streak_direction",
        "type": "COMBO (4 options)"
      },
      {
        "name": "streak_falloff",
        "type": "FLOAT"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "streak_pass",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Anamorphic Streaks in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceFilmGrain",
    "name": "◎ Film Grain",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "◎ Radiance Film Grain",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "grain_size",
        "type": "FLOAT"
      },
      {
        "name": "grain_strength",
        "type": "FLOAT"
      },
      {
        "name": "grain_size_r_offset",
        "type": "FLOAT"
      },
      {
        "name": "grain_size_g_offset",
        "type": "FLOAT"
      },
      {
        "name": "grain_size_b_offset",
        "type": "FLOAT"
      },
      {
        "name": "hdr_aware",
        "type": "BOOLEAN"
      },
      {
        "name": "seed",
        "type": "INT"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Film Grain in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceVignette",
    "name": "◎ Vignette",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "◎ Radiance Vignette",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "strength",
        "type": "FLOAT"
      },
      {
        "name": "power",
        "type": "FLOAT"
      },
      {
        "name": "center_x",
        "type": "FLOAT"
      },
      {
        "name": "center_y",
        "type": "FLOAT"
      },
      {
        "name": "feather",
        "type": "FLOAT"
      },
      {
        "name": "tint_r",
        "type": "FLOAT (optional)"
      },
      {
        "name": "tint_g",
        "type": "FLOAT (optional)"
      },
      {
        "name": "tint_b",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "vignette_mask",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Vignette in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceMultipassGeometry",
    "name": "◎ Multipass: Surface Geometry",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "beauty",
        "type": "IMAGE"
      },
      {
        "name": "depth_map",
        "type": "IMAGE (optional)"
      },
      {
        "name": "luma_weights",
        "type": "COMBO (3 options) (optional)"
      },
      {
        "name": "auto_depth_model",
        "type": "COMBO (4 options) (optional)"
      },
      {
        "name": "normal_strength",
        "type": "FLOAT (optional)"
      },
      {
        "name": "normal_convention",
        "type": "COMBO (2 options) (optional)"
      },
      {
        "name": "dsine_model_path",
        "type": "STRING (optional)"
      },
      {
        "name": "fov_degrees",
        "type": "FLOAT (optional)"
      },
      {
        "name": "depth_scale",
        "type": "FLOAT (optional)"
      },
      {
        "name": "depth_near_is_white",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "normal_map",
        "type": "IMAGE"
      },
      {
        "name": "world_position",
        "type": "IMAGE"
      },
      {
        "name": "curvature",
        "type": "IMAGE"
      },
      {
        "name": "depth",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Multipass: Surface Geometry in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceMultipassMaterial",
    "name": "◎ Multipass: Material Properties",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "beauty",
        "type": "IMAGE"
      },
      {
        "name": "luma_weights",
        "type": "COMBO (3 options) (optional)"
      },
      {
        "name": "albedo_shading_radius",
        "type": "FLOAT (optional)"
      },
      {
        "name": "albedo_eps",
        "type": "FLOAT (optional)"
      },
      {
        "name": "emission_radius",
        "type": "FLOAT (optional)"
      },
      {
        "name": "emission_boost",
        "type": "FLOAT (optional)"
      },
      {
        "name": "roughness_fine_radius",
        "type": "FLOAT (optional)"
      },
      {
        "name": "roughness_coarse_radius",
        "type": "FLOAT (optional)"
      },
      {
        "name": "transmission_sensitivity",
        "type": "FLOAT (optional)"
      },
      {
        "name": "diffuse_radius",
        "type": "FLOAT (optional)"
      },
      {
        "name": "diffuse_edge_eps",
        "type": "FLOAT (optional)"
      },
      {
        "name": "specular_floor",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "albedo",
        "type": "IMAGE"
      },
      {
        "name": "emission",
        "type": "IMAGE"
      },
      {
        "name": "roughness",
        "type": "IMAGE"
      },
      {
        "name": "transmission",
        "type": "IMAGE"
      },
      {
        "name": "diffuse",
        "type": "IMAGE"
      },
      {
        "name": "specular",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Multipass: Material Properties in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceMultipassLighting",
    "name": "◎ Multipass: Lighting & Masks",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "beauty",
        "type": "IMAGE"
      },
      {
        "name": "depth_map",
        "type": "IMAGE (optional)"
      },
      {
        "name": "normal_map",
        "type": "IMAGE (optional)"
      },
      {
        "name": "luma_weights",
        "type": "COMBO (3 options) (optional)"
      },
      {
        "name": "shadow_threshold",
        "type": "FLOAT (optional)"
      },
      {
        "name": "highlight_threshold",
        "type": "FLOAT (optional)"
      },
      {
        "name": "mask_softness",
        "type": "FLOAT (optional)"
      },
      {
        "name": "ao_radius",
        "type": "FLOAT (optional)"
      },
      {
        "name": "ao_strength",
        "type": "FLOAT (optional)"
      },
      {
        "name": "ao_samples",
        "type": "INT (optional)"
      },
      {
        "name": "depth_near_is_white",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "shadow_mask",
        "type": "IMAGE"
      },
      {
        "name": "midtone_mask",
        "type": "IMAGE"
      },
      {
        "name": "highlight_mask",
        "type": "IMAGE"
      },
      {
        "name": "ao",
        "type": "IMAGE"
      },
      {
        "name": "reflection_mask",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Multipass: Lighting & Masks in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceMultipassRelight",
    "name": "◎ Multipass: Real PBR Relight",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "albedo",
        "type": "IMAGE"
      },
      {
        "name": "normal_map",
        "type": "IMAGE"
      },
      {
        "name": "beauty",
        "type": "IMAGE (optional)"
      },
      {
        "name": "roughness",
        "type": "IMAGE (optional)"
      },
      {
        "name": "metallic",
        "type": "IMAGE (optional)"
      },
      {
        "name": "specular",
        "type": "IMAGE (optional)"
      },
      {
        "name": "ao",
        "type": "IMAGE (optional)"
      },
      {
        "name": "alpha",
        "type": "IMAGE (optional)"
      },
      {
        "name": "shadow_mask",
        "type": "IMAGE (optional)"
      },
      {
        "name": "depth_map",
        "type": "IMAGE (optional)"
      },
      {
        "name": "normal_convention",
        "type": "COMBO (2 options) (optional)"
      },
      {
        "name": "light_type",
        "type": "COMBO (2 options) (optional)"
      },
      {
        "name": "light_x",
        "type": "FLOAT (optional)"
      },
      {
        "name": "light_y",
        "type": "FLOAT (optional)"
      },
      {
        "name": "light_z",
        "type": "FLOAT (optional)"
      },
      {
        "name": "light_r",
        "type": "FLOAT (optional)"
      },
      {
        "name": "light_g",
        "type": "FLOAT (optional)"
      },
      {
        "name": "light_b",
        "type": "FLOAT (optional)"
      },
      {
        "name": "intensity",
        "type": "FLOAT (optional)"
      },
      {
        "name": "ambient",
        "type": "FLOAT (optional)"
      },
      {
        "name": "specular_intensity",
        "type": "FLOAT (optional)"
      },
      {
        "name": "depth_scale",
        "type": "FLOAT (optional)"
      },
      {
        "name": "mix_with_beauty",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "relit",
        "type": "IMAGE"
      },
      {
        "name": "diffuse_light",
        "type": "IMAGE"
      },
      {
        "name": "specular_light",
        "type": "IMAGE"
      },
      {
        "name": "lighting",
        "type": "IMAGE"
      },
      {
        "name": "alpha",
        "type": "IMAGE"
      },
      {
        "name": "relight_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Multipass: Real PBR Relight in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceMultipassComposite",
    "name": "◎ Multipass: VFX Composite",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "foreground",
        "type": "IMAGE"
      },
      {
        "name": "alpha",
        "type": "IMAGE"
      },
      {
        "name": "background",
        "type": "IMAGE (optional)"
      },
      {
        "name": "relit_foreground",
        "type": "IMAGE (optional)"
      },
      {
        "name": "foreground_depth",
        "type": "IMAGE (optional)"
      },
      {
        "name": "background_depth",
        "type": "IMAGE (optional)"
      },
      {
        "name": "shadow_mask",
        "type": "IMAGE (optional)"
      },
      {
        "name": "alpha_invert",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "premultiplied_input",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "depth_near_is_white",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "depth_bias",
        "type": "FLOAT (optional)"
      },
      {
        "name": "shadow_strength",
        "type": "FLOAT (optional)"
      },
      {
        "name": "light_wrap",
        "type": "FLOAT (optional)"
      },
      {
        "name": "light_wrap_radius",
        "type": "INT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "composite",
        "type": "IMAGE"
      },
      {
        "name": "premultiplied_foreground",
        "type": "IMAGE"
      },
      {
        "name": "holdout_mask",
        "type": "IMAGE"
      },
      {
        "name": "depth_matte",
        "type": "IMAGE"
      },
      {
        "name": "comp_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Multipass: VFX Composite in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceMultipassMotionOptics",
    "name": "◎ Multipass: Motion & Optics",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "beauty",
        "type": "IMAGE"
      },
      {
        "name": "prev_frame",
        "type": "IMAGE (optional)"
      },
      {
        "name": "luma_weights",
        "type": "COMBO (3 options) (optional)"
      },
      {
        "name": "lk_window_radius",
        "type": "INT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "motion_vector",
        "type": "IMAGE"
      },
      {
        "name": "edge_map",
        "type": "IMAGE"
      },
      {
        "name": "colorfulness",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Multipass: Motion & Optics in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceMultipassCrypto",
    "name": "◎ Multipass: Cryptomatte ID",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "beauty",
        "type": "IMAGE"
      },
      {
        "name": "luma_weights",
        "type": "COMBO (3 options) (optional)"
      },
      {
        "name": "id_segments",
        "type": "INT (optional)"
      },
      {
        "name": "id_spatial_weight",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "object_id_matte",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Multipass: Cryptomatte ID in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceAudioCut",
    "name": "◎ Audio Cut",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "audio_filepath",
        "type": "STRING"
      },
      {
        "name": "fps",
        "type": "FLOAT"
      },
      {
        "name": "method",
        "type": "COMBO (3 options)"
      },
      {
        "name": "sensitivity",
        "type": "FLOAT"
      },
      {
        "name": "min_interval_frames",
        "type": "INT"
      },
      {
        "name": "backend",
        "type": "COMBO (4 options)"
      },
      {
        "name": "frame_offset",
        "type": "INT (optional)"
      },
      {
        "name": "max_cuts",
        "type": "INT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "cut_frames_json",
        "type": "STRING"
      },
      {
        "name": "cut_times_json",
        "type": "STRING"
      },
      {
        "name": "cut_count",
        "type": "INT"
      },
      {
        "name": "analysis_report",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Audio Cut in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceProjectManager",
    "name": "◎ Project Manager",
    "zone": "FXTD STUDIOS/Radiance/◎ Pipeline",
    "desc": "Pipeline project manager — save, list, load, delete, and inspect .rad workflow containers.",
    "inputs": [
      {
        "name": "filename",
        "type": "STRING (optional)"
      },
      {
        "name": "artist",
        "type": "STRING (optional)"
      },
      {
        "name": "version",
        "type": "INT (optional)"
      }
    ],
    "outputs": [],
    "tip": "Use ◎ Project Manager in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceBlendComposite",
    "name": "◎ Blend Composite",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "Composite two images using industry-standard blend modes.",
    "inputs": [
      {
        "name": "base",
        "type": "IMAGE"
      },
      {
        "name": "blend",
        "type": "IMAGE"
      },
      {
        "name": "mode",
        "type": "COMBO (8 options)"
      },
      {
        "name": "opacity",
        "type": "FLOAT"
      },
      {
        "name": "mask",
        "type": "MASK (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Blend Composite in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceCinemaStudio",
    "name": "◎ Cinema Studio",
    "zone": "FXTD STUDIOS/Radiance/◎ Generate",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "base_prompt",
        "type": "STRING"
      },
      {
        "name": "camera",
        "type": "COMBO (20 options)"
      },
      {
        "name": "lens_series",
        "type": "COMBO (18 options)"
      },
      {
        "name": "focal_length",
        "type": "COMBO (22 options)"
      },
      {
        "name": "aperture",
        "type": "COMBO (12 options)"
      },
      {
        "name": "shutter",
        "type": "COMBO (7 options)"
      },
      {
        "name": "iso",
        "type": "COMBO (9 options)"
      },
      {
        "name": "shot_type",
        "type": "COMBO (10 options) (optional)"
      },
      {
        "name": "camera_movement",
        "type": "COMBO (12 options) (optional)"
      }
    ],
    "outputs": [
      {
        "name": "prompt",
        "type": "STRING"
      },
      {
        "name": "technical_data_str",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Cinema Studio in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceMCP",
    "name": "◎ Radiance MCP",
    "zone": "FXTD STUDIOS/Radiance/◎ Pipeline",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "mode",
        "type": "COMBO (2 options)"
      },
      {
        "name": "source",
        "type": "COMBO (4 options)"
      },
      {
        "name": "target",
        "type": "COMBO (3 options)"
      },
      {
        "name": "output_path",
        "type": "STRING"
      },
      {
        "name": "format",
        "type": "COMBO (4 options)"
      },
      {
        "name": "images",
        "type": "IMAGE (optional)"
      },
      {
        "name": "video_path",
        "type": "STRING (optional)"
      },
      {
        "name": "sequence_path",
        "type": "STRING (optional)"
      },
      {
        "name": "fps",
        "type": "FLOAT (optional)"
      },
      {
        "name": "frame_start",
        "type": "INT (optional)"
      },
      {
        "name": "frame_end",
        "type": "INT (optional)"
      },
      {
        "name": "filename_prefix",
        "type": "STRING (optional)"
      },
      {
        "name": "bridge_port",
        "type": "INT (optional)"
      },
      {
        "name": "bridge_host",
        "type": "STRING (optional)"
      }
    ],
    "outputs": [
      {
        "name": "status",
        "type": "STRING"
      },
      {
        "name": "render_path",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Radiance MCP in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceNukeSend",
    "name": "◎ Send to Nuke",
    "zone": "FXTD STUDIOS/Radiance/◎ Studio",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "nuke_folder",
        "type": "STRING"
      },
      {
        "name": "filename",
        "type": "STRING"
      },
      {
        "name": "frame_start",
        "type": "INT (optional)"
      },
      {
        "name": "push_to_nuke",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "nuke_host",
        "type": "STRING (optional)"
      },
      {
        "name": "nuke_port",
        "type": "INT (optional)"
      },
      {
        "name": "half_float",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "status",
        "type": "STRING"
      },
      {
        "name": "render_path",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Send to Nuke in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceDaVinciSend",
    "name": "◎ Send to DaVinci Resolve",
    "zone": "FXTD STUDIOS/Radiance/◎ Studio",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "resolve_folder",
        "type": "STRING"
      },
      {
        "name": "filename",
        "type": "STRING"
      },
      {
        "name": "bit_depth",
        "type": "COMBO (3 options)"
      },
      {
        "name": "frame_start",
        "type": "INT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "status",
        "type": "STRING"
      },
      {
        "name": "render_path",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Send to DaVinci Resolve in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceWirelessSave",
    "name": "◎ Wireless State Save",
    "zone": "FXTD STUDIOS/Radiance/◎ Wireless",
    "desc": "Saves any variable (Model, CLIP, Latent, Image, etc.) into a named registry slot.",
    "inputs": [
      {
        "name": "key",
        "type": "STRING"
      },
      {
        "name": "value",
        "type": "*"
      }
    ],
    "outputs": [
      {
        "name": "value",
        "type": "*"
      }
    ],
    "tip": "Use ◎ Wireless State Save in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceWirelessLoad",
    "name": "◎ Wireless State Load",
    "zone": "FXTD STUDIOS/Radiance/◎ Wireless",
    "desc": "Loads a variable from the named registry slot by key.",
    "inputs": [
      {
        "name": "key",
        "type": "STRING"
      }
    ],
    "outputs": [
      {
        "name": "value",
        "type": "*"
      }
    ],
    "tip": "Use ◎ Wireless State Load in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadiancePipelineBus",
    "name": "◎ Pipeline Bus (Pack)",
    "zone": "FXTD STUDIOS/Radiance/◎ Pipeline Bus",
    "desc": "Packs key pipeline components into a single multiplexed custom Bus object.",
    "inputs": [
      {
        "name": "bus",
        "type": "RADIANCE_BUS (optional)"
      },
      {
        "name": "model",
        "type": "MODEL (optional)"
      },
      {
        "name": "clip",
        "type": "CLIP (optional)"
      },
      {
        "name": "vae",
        "type": "VAE (optional)"
      },
      {
        "name": "latent",
        "type": "LATENT (optional)"
      },
      {
        "name": "positive",
        "type": "CONDITIONING (optional)"
      },
      {
        "name": "negative",
        "type": "CONDITIONING (optional)"
      },
      {
        "name": "image",
        "type": "IMAGE (optional)"
      },
      {
        "name": "mask",
        "type": "MASK (optional)"
      }
    ],
    "outputs": [
      {
        "name": "bus",
        "type": "RADIANCE_BUS"
      }
    ],
    "tip": "Use ◎ Pipeline Bus (Pack) in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadiancePipelineBusUnpack",
    "name": "◎ Pipeline Bus (Unpack)",
    "zone": "FXTD STUDIOS/Radiance/◎ Pipeline Bus",
    "desc": "Unpacks elements from a pipeline Bus object.",
    "inputs": [
      {
        "name": "bus",
        "type": "RADIANCE_BUS"
      }
    ],
    "outputs": [
      {
        "name": "model",
        "type": "MODEL"
      },
      {
        "name": "clip",
        "type": "CLIP"
      },
      {
        "name": "vae",
        "type": "VAE"
      },
      {
        "name": "latent",
        "type": "LATENT"
      },
      {
        "name": "positive",
        "type": "CONDITIONING"
      },
      {
        "name": "negative",
        "type": "CONDITIONING"
      },
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "mask",
        "type": "MASK"
      }
    ],
    "tip": "Use ◎ Pipeline Bus (Unpack) in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceUpscaleTiler",
    "name": "◎ Upscale Tiler",
    "zone": "FXTD STUDIOS/Radiance/◎ Upscale",
    "desc": "◎ Radiance Upscale Tiler",
    "inputs": [
      {
        "name": "operation",
        "type": "COMBO (2 options)"
      },
      {
        "name": "images",
        "type": "IMAGE (optional)"
      },
      {
        "name": "scale",
        "type": "COMBO (3 options) (optional)"
      },
      {
        "name": "tile_size",
        "type": "INT (optional)"
      },
      {
        "name": "overlap",
        "type": "INT (optional)"
      },
      {
        "name": "blend_mode",
        "type": "COMBO (3 options) (optional)"
      },
      {
        "name": "upscale_model",
        "type": "UPSCALE_MODEL (optional)"
      },
      {
        "name": "model_tier",
        "type": "COMBO (6 options) (optional)"
      },
      {
        "name": "source",
        "type": "IMAGE (optional)"
      },
      {
        "name": "reference",
        "type": "IMAGE (optional)"
      },
      {
        "name": "cf_strength",
        "type": "FLOAT (optional)"
      },
      {
        "name": "n_bins",
        "type": "INT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image_a",
        "type": "IMAGE"
      },
      {
        "name": "image_b",
        "type": "IMAGE"
      },
      {
        "name": "info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Upscale Tiler in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceUpscaleImage",
    "name": "◎ Upscale Image",
    "zone": "FXTD STUDIOS/Radiance/◎ Upscale",
    "desc": "◎ Radiance Upscale Image",
    "inputs": [
      {
        "name": "operation",
        "type": "COMBO (2 options)"
      },
      {
        "name": "images",
        "type": "IMAGE"
      },
      {
        "name": "scale",
        "type": "COMBO (3 options) (optional)"
      },
      {
        "name": "mode",
        "type": "COMBO (3 options) (optional)"
      },
      {
        "name": "tile_size",
        "type": "INT (optional)"
      },
      {
        "name": "overlap",
        "type": "INT (optional)"
      },
      {
        "name": "sharpness_boost",
        "type": "FLOAT (optional)"
      },
      {
        "name": "denoise_pre",
        "type": "FLOAT (optional)"
      },
      {
        "name": "upscale_model",
        "type": "UPSCALE_MODEL (optional)"
      },
      {
        "name": "model_tier",
        "type": "COMBO (6 options) (optional)"
      },
      {
        "name": "diffusion_steps",
        "type": "INT (optional)"
      },
      {
        "name": "diffusion_noise_level",
        "type": "INT (optional)"
      },
      {
        "name": "guidance_scale",
        "type": "FLOAT (optional)"
      },
      {
        "name": "enhancement_prompt",
        "type": "STRING (optional)"
      },
      {
        "name": "prefer_speed",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "sample_frame",
        "type": "INT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "image_a",
        "type": "IMAGE"
      },
      {
        "name": "image_b",
        "type": "IMAGE"
      },
      {
        "name": "info",
        "type": "STRING"
      },
      {
        "name": "data1",
        "type": "STRING"
      },
      {
        "name": "data2",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Upscale Image in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceUpscaleVideo",
    "name": "◎ Upscale Video",
    "zone": "FXTD STUDIOS/Radiance/◎ Upscale",
    "desc": "◎ Radiance Upscale Video",
    "inputs": [
      {
        "name": "frames",
        "type": "IMAGE"
      },
      {
        "name": "scale",
        "type": "COMBO (3 options)"
      },
      {
        "name": "tile_size",
        "type": "INT"
      },
      {
        "name": "overlap_spatial",
        "type": "INT"
      },
      {
        "name": "window_size",
        "type": "INT"
      },
      {
        "name": "overlap_temporal",
        "type": "INT"
      },
      {
        "name": "flow_compensation",
        "type": "BOOLEAN"
      },
      {
        "name": "sharpness_boost",
        "type": "FLOAT"
      },
      {
        "name": "upscale_model",
        "type": "UPSCALE_MODEL (optional)"
      },
      {
        "name": "model_tier",
        "type": "COMBO (6 options) (optional)"
      },
      {
        "name": "enhancement_prompt",
        "type": "STRING (optional)"
      },
      {
        "name": "diffusion_steps",
        "type": "INT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "upscaled",
        "type": "IMAGE"
      },
      {
        "name": "confidence_map",
        "type": "IMAGE"
      },
      {
        "name": "pass_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Upscale Video in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceUpscaleFaceRestore",
    "name": "◎ Upscale Face Restore",
    "zone": "FXTD STUDIOS/Radiance/◎ Upscale",
    "desc": "◎ Radiance Upscale Face Restore",
    "inputs": [
      {
        "name": "images",
        "type": "IMAGE"
      },
      {
        "name": "face_model",
        "type": "COMBO (4 options)"
      },
      {
        "name": "fidelity_weight",
        "type": "FLOAT"
      },
      {
        "name": "blend_radius",
        "type": "INT"
      },
      {
        "name": "face_pad_frac",
        "type": "FLOAT"
      },
      {
        "name": "min_face_px",
        "type": "INT"
      },
      {
        "name": "colour_correct",
        "type": "BOOLEAN"
      },
      {
        "name": "colour_strength",
        "type": "FLOAT"
      },
      {
        "name": "original_images",
        "type": "IMAGE (optional)"
      }
    ],
    "outputs": [
      {
        "name": "restored",
        "type": "IMAGE"
      },
      {
        "name": "face_mask",
        "type": "IMAGE"
      },
      {
        "name": "pass_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Upscale Face Restore in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceVideoModelInfo",
    "name": "◎ Video Model Info",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "Inspect a ComfyUI MODEL object and produce a DiT config JSON describing",
    "inputs": [
      {
        "name": "model",
        "type": "MODEL"
      },
      {
        "name": "model_preset",
        "type": "COMBO (5 options)"
      },
      {
        "name": "override_channels",
        "type": "INT (optional)"
      },
      {
        "name": "override_latent_scale",
        "type": "FLOAT (optional)"
      },
      {
        "name": "print_info",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "model",
        "type": "MODEL"
      },
      {
        "name": "dit_config",
        "type": "STRING"
      },
      {
        "name": "info_report",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Video Model Info in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceVideoLatentNoise",
    "name": "◎ Video Latent Noise",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "Generate correctly-shaped Gaussian noise for a given DiT video model",
    "inputs": [
      {
        "name": "dit_config",
        "type": "STRING"
      },
      {
        "name": "width",
        "type": "INT"
      },
      {
        "name": "height",
        "type": "INT"
      },
      {
        "name": "frames",
        "type": "INT"
      },
      {
        "name": "batch_size",
        "type": "INT"
      },
      {
        "name": "seed",
        "type": "INT"
      },
      {
        "name": "noise_scale",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "noise_latent",
        "type": "LATENT"
      },
      {
        "name": "shape_report",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Video Latent Noise in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceVideoCondMerge",
    "name": "◎ Video Cond Merge",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "Merge up to three conditioning inputs (text, character, HDR) into a",
    "inputs": [
      {
        "name": "text_conditioning",
        "type": "CONDITIONING"
      },
      {
        "name": "merge_mode",
        "type": "COMBO (3 options)"
      },
      {
        "name": "character_conditioning",
        "type": "CONDITIONING (optional)"
      },
      {
        "name": "hdr_conditioning",
        "type": "CONDITIONING (optional)"
      },
      {
        "name": "text_weight",
        "type": "FLOAT (optional)"
      },
      {
        "name": "character_weight",
        "type": "FLOAT (optional)"
      },
      {
        "name": "hdr_weight",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "merged_conditioning",
        "type": "CONDITIONING"
      },
      {
        "name": "merge_report",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Video Cond Merge in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceVideoSampler",
    "name": "◎ Video Sampler",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "model",
        "type": "MODEL"
      },
      {
        "name": "positive",
        "type": "CONDITIONING"
      },
      {
        "name": "negative",
        "type": "CONDITIONING"
      },
      {
        "name": "latent_noise",
        "type": "LATENT"
      },
      {
        "name": "dit_config",
        "type": "STRING"
      },
      {
        "name": "steps",
        "type": "INT"
      },
      {
        "name": "cfg",
        "type": "FLOAT"
      },
      {
        "name": "sampler_name",
        "type": "COMBO (25 options)"
      },
      {
        "name": "scheduler",
        "type": "COMBO (7 options)"
      },
      {
        "name": "seed",
        "type": "INT"
      },
      {
        "name": "cfg_schedule_json",
        "type": "STRING (optional)"
      },
      {
        "name": "denoise",
        "type": "FLOAT (optional)"
      },
      {
        "name": "tiling",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "samples",
        "type": "LATENT"
      },
      {
        "name": "sampler_report",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Video Sampler in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceT2VPipeline",
    "name": "◎ T2V Pipeline",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "model",
        "type": "MODEL"
      },
      {
        "name": "clip",
        "type": "CLIP"
      },
      {
        "name": "vae",
        "type": "VAE"
      },
      {
        "name": "positive_prompt",
        "type": "STRING"
      },
      {
        "name": "negative_prompt",
        "type": "STRING"
      },
      {
        "name": "width",
        "type": "INT"
      },
      {
        "name": "height",
        "type": "INT"
      },
      {
        "name": "frames",
        "type": "INT"
      },
      {
        "name": "seed",
        "type": "INT"
      },
      {
        "name": "dit_config",
        "type": "STRING (optional)"
      },
      {
        "name": "character_conditioning",
        "type": "CONDITIONING (optional)"
      },
      {
        "name": "cfg_schedule_json",
        "type": "STRING (optional)"
      },
      {
        "name": "steps",
        "type": "INT (optional)"
      },
      {
        "name": "cfg",
        "type": "FLOAT (optional)"
      },
      {
        "name": "sampler_name",
        "type": "COMBO (25 options) (optional)"
      },
      {
        "name": "scheduler",
        "type": "COMBO (7 options) (optional)"
      },
      {
        "name": "peak_nits",
        "type": "COMBO (7 options) (optional)"
      },
      {
        "name": "target_gamut",
        "type": "COMBO (5 options) (optional)"
      },
      {
        "name": "hdr_eotf",
        "type": "COMBO (4 options) (optional)"
      },
      {
        "name": "hdr_strength",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "video_latent",
        "type": "LATENT"
      },
      {
        "name": "preview_frames",
        "type": "IMAGE"
      },
      {
        "name": "positive_cond",
        "type": "CONDITIONING"
      },
      {
        "name": "pipeline_report",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ T2V Pipeline in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceI2VPipeline",
    "name": "◎ I2V Pipeline",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "model",
        "type": "MODEL"
      },
      {
        "name": "clip",
        "type": "CLIP"
      },
      {
        "name": "vae",
        "type": "VAE"
      },
      {
        "name": "reference_image",
        "type": "IMAGE"
      },
      {
        "name": "positive_prompt",
        "type": "STRING"
      },
      {
        "name": "negative_prompt",
        "type": "STRING"
      },
      {
        "name": "frames",
        "type": "INT"
      },
      {
        "name": "seed",
        "type": "INT"
      },
      {
        "name": "dit_config",
        "type": "STRING (optional)"
      },
      {
        "name": "character_conditioning",
        "type": "CONDITIONING (optional)"
      },
      {
        "name": "cfg_schedule_json",
        "type": "STRING (optional)"
      },
      {
        "name": "i2v_strategy",
        "type": "COMBO (5 options) (optional)"
      },
      {
        "name": "image_strength",
        "type": "FLOAT (optional)"
      },
      {
        "name": "motion_strength",
        "type": "FLOAT (optional)"
      },
      {
        "name": "steps",
        "type": "INT (optional)"
      },
      {
        "name": "cfg",
        "type": "FLOAT (optional)"
      },
      {
        "name": "sampler_name",
        "type": "COMBO (25 options) (optional)"
      },
      {
        "name": "scheduler",
        "type": "COMBO (7 options) (optional)"
      },
      {
        "name": "peak_nits",
        "type": "COMBO (7 options) (optional)"
      },
      {
        "name": "target_gamut",
        "type": "COMBO (5 options) (optional)"
      },
      {
        "name": "hdr_eotf",
        "type": "COMBO (4 options) (optional)"
      }
    ],
    "outputs": [
      {
        "name": "video_latent",
        "type": "LATENT"
      },
      {
        "name": "preview_frames",
        "type": "IMAGE"
      },
      {
        "name": "pipeline_report",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ I2V Pipeline in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceVideoBatchDecode",
    "name": "◎ Video Batch Decode",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "Decode a DiT video LATENT tensor into an IMAGE frame batch.",
    "inputs": [
      {
        "name": "vae",
        "type": "VAE"
      },
      {
        "name": "latent",
        "type": "LATENT"
      },
      {
        "name": "dit_config",
        "type": "STRING (optional)"
      },
      {
        "name": "tile_decode",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "tile_overlap",
        "type": "INT (optional)"
      },
      {
        "name": "output_linear",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "frames",
        "type": "IMAGE"
      },
      {
        "name": "frame_count",
        "type": "INT"
      },
      {
        "name": "decode_report",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Video Batch Decode in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceVideoExport",
    "name": "◎ Video Export",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "Route a decoded video IMAGE batch to the appropriate output format.",
    "inputs": [
      {
        "name": "frames",
        "type": "IMAGE"
      },
      {
        "name": "mode",
        "type": "COMBO (4 options)"
      },
      {
        "name": "hdr_metadata_json",
        "type": "STRING (optional)"
      },
      {
        "name": "output_folder",
        "type": "STRING (optional)"
      },
      {
        "name": "filename_prefix",
        "type": "STRING (optional)"
      },
      {
        "name": "fps",
        "type": "FLOAT (optional)"
      },
      {
        "name": "frame_offset",
        "type": "INT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "frames",
        "type": "IMAGE"
      },
      {
        "name": "frame_count",
        "type": "INT"
      },
      {
        "name": "export_report",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Video Export in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceVideoHDRConditioner",
    "name": "◎ Video HDR Conditioner",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "positive",
        "type": "CONDITIONING"
      },
      {
        "name": "peak_nits",
        "type": "COMBO (7 options)"
      },
      {
        "name": "target_gamut",
        "type": "COMBO (6 options)"
      },
      {
        "name": "eotf",
        "type": "COMBO (4 options)"
      },
      {
        "name": "camera_move",
        "type": "COMBO (7 options) (optional)"
      },
      {
        "name": "mood",
        "type": "COMBO (8 options) (optional)"
      },
      {
        "name": "extra_hdr_prompt",
        "type": "STRING (optional)"
      },
      {
        "name": "inject_metadata_embedding",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "token_strength",
        "type": "FLOAT (optional)"
      }
    ],
    "outputs": [
      {
        "name": "positive",
        "type": "CONDITIONING"
      },
      {
        "name": "hdr_metadata_json",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Video HDR Conditioner in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceVideoHDRDecode",
    "name": "◎ Video HDR Decode",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "image",
        "type": "IMAGE"
      },
      {
        "name": "hdr_metadata_json",
        "type": "STRING"
      },
      {
        "name": "tonemap",
        "type": "COMBO (3 options)"
      },
      {
        "name": "exposure_compensation_ev",
        "type": "FLOAT (optional)"
      },
      {
        "name": "output_eotf",
        "type": "COMBO (4 options) (optional)"
      },
      {
        "name": "sdr_preview_nits",
        "type": "FLOAT (optional)"
      },
      {
        "name": "gamut_clip",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "hdr_image",
        "type": "IMAGE"
      },
      {
        "name": "sdr_preview",
        "type": "IMAGE"
      },
      {
        "name": "decode_report",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Video HDR Decode in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceVideoFrameRouter",
    "name": "◎ Video Frame Router",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "Extract individual frames from a decoded video IMAGE tensor",
    "inputs": [
      {
        "name": "video_image",
        "type": "IMAGE"
      },
      {
        "name": "frame_index",
        "type": "INT"
      },
      {
        "name": "wrap_index",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "frame_image",
        "type": "IMAGE"
      },
      {
        "name": "frame_index",
        "type": "INT"
      },
      {
        "name": "total_frames",
        "type": "INT"
      },
      {
        "name": "passthrough",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Video Frame Router in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceVideoAssembler",
    "name": "◎ Video Assembler",
    "zone": "FXTD STUDIOS/Radiance/◎ Video",
    "desc": "Collect per-frame IMAGE tensors into a single video batch tensor",
    "inputs": [
      {
        "name": "frame",
        "type": "IMAGE"
      },
      {
        "name": "session_key",
        "type": "STRING"
      },
      {
        "name": "expected_total_frames",
        "type": "INT"
      },
      {
        "name": "flush",
        "type": "BOOLEAN (optional)"
      },
      {
        "name": "reset",
        "type": "BOOLEAN (optional)"
      }
    ],
    "outputs": [
      {
        "name": "video_image",
        "type": "IMAGE"
      },
      {
        "name": "frames_accumulated",
        "type": "INT"
      },
      {
        "name": "is_complete",
        "type": "BOOLEAN"
      }
    ],
    "tip": "Use ◎ Video Assembler in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceSceneCutDetect",
    "name": "◎ Scene Cut Detect",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "Detect hard shot cuts in a video sequence batch.",
    "inputs": [
      {
        "name": "images",
        "type": "IMAGE"
      },
      {
        "name": "threshold",
        "type": "FLOAT"
      },
      {
        "name": "min_shot_frames",
        "type": "INT"
      },
      {
        "name": "method",
        "type": "COMBO (3 options)"
      }
    ],
    "outputs": [
      {
        "name": "cut_data",
        "type": "STRING"
      },
      {
        "name": "shot_count",
        "type": "INT"
      },
      {
        "name": "score_plot",
        "type": "IMAGE"
      }
    ],
    "tip": "Use ◎ Scene Cut Detect in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceSceneCutSplit",
    "name": "◎ Scene Cut Split",
    "zone": "FXTD STUDIOS/Radiance/◎ VFX",
    "desc": "Split an IMAGE batch into per-shot sub-batches using cut_data from",
    "inputs": [
      {
        "name": "images",
        "type": "IMAGE"
      },
      {
        "name": "cut_data",
        "type": "STRING"
      },
      {
        "name": "shot_index",
        "type": "INT"
      }
    ],
    "outputs": [
      {
        "name": "frames",
        "type": "IMAGE"
      },
      {
        "name": "shot_index",
        "type": "INT"
      },
      {
        "name": "start_frame",
        "type": "INT"
      },
      {
        "name": "end_frame",
        "type": "INT"
      },
      {
        "name": "shot_info",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Scene Cut Split in your pipelines to achieve high dynamic range consistency."
  },
  {
    "id": "RadianceTurboCheckpointLoader",
    "name": "◎ Turbo Checkpoint Loader",
    "zone": "FXTD STUDIOS/Radiance/⚙ Training",
    "desc": "VFX processing component.",
    "inputs": [
      {
        "name": "checkpoint_path",
        "type": "STRING"
      },
      {
        "name": "model_type",
        "type": "COMBO (12 options)"
      },
      {
        "name": "model_size",
        "type": "COMBO (2 options)"
      }
    ],
    "outputs": [
      {
        "name": "status",
        "type": "STRING"
      }
    ],
    "tip": "Use ◎ Turbo Checkpoint Loader in your pipelines to achieve high dynamic range consistency."
  }
];

const KEY_SHORTCUTS = {
    "Space": "Toggles timeline playback inside the Radiance Viewer. Controls multi-frame caching and looping.",
    "1": "Forces a 1:1 pixel-perfect zoom layout. Essential for evaluating details and checking focus.",
    "F": "Fits the image back to center. Reset panning offsets and scale matching instantly.",
    "R": "Isolates the Red channel in the monitor. Helps locate color noise or banding in the highlights.",
    "G": "Isolates the Green channel in the monitor. Green typically contains the highest spatial detail.",
    "B": "Isolates the Blue channel. Blue usually exhibits the highest sensor noise in camera plates.",
    "L": "Renders a grayscale representation showing only Luma distribution (perceptual luminance).",
    "Shift+A": "Isolates the Alpha channel transparency mask. Checks edge blending and feathering contours.",
    "C": "Restores full RGB Color representation. Deactivates isolated channel mode.",
    "W": "Toggles the high-speed HDR-aware Waveform Parade scope on the overlay screen.",
    "M": "Switches the Waveform parade mode (cycles between overlaid RGB Luma and side-by-side parade).",
    "V": "Toggles the Vectorscope display. Monitor hue saturation distributions against the skin tone line.",
    "S": "Cycles safe area aspect ratio overlays (Action Safe / Title Safe frame boxes).",
    "A": "Cycles A/B split screen comparisons. Compare active buffer against the pinned reference plate.",
    "Esc": "Exits active fullscreen display, closes panel HUD screens, or resets tool scopes."
};

// ═══════════════════════════════════════════════════════════════════════════════
//                           3. BOOTSTRAP DOCUMENTATION ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

let activeChapter = "welcome";

document.addEventListener("DOMContentLoaded", () => {
    initChapterNavigation();
    initTabRouting();
    renderActiveChapter();
    initColorSimulator();
    initGlobalSearch();
});

// Setup Chapter Sidebar and Chapter Switching
function initChapterNavigation() {
    const chaptersList = document.getElementById("sidebar-chapters");
    if (!chaptersList) return;

    chaptersList.innerHTML = "";

    // Outline high-level folders
    const sidebarStructure = [
        {
            title: "Getting Started",
            id: "getting-started",
            keys: ["welcome", "installation", "first-workflow", "video-tutorials"]
        },
        {
            title: "Interactive Reference",
            id: "interactive-ref",
            keys: ["noderef", "shortcuts"]
        },
        {
            title: "Advanced CLI & Training",
            id: "cli-train",
            keys: ["cli-training"]
        },
        {
            title: "Developer Area",
            id: "dev-area",
            keys: ["developer-api"]
        }
    ];

    // Maintain expanded states in local memory
    if (!window.sidebarFolderStates) {
        window.sidebarFolderStates = {
            "getting-started": true,
            "interactive-ref": true,
            "cli-train": true,
            "dev-area": true
        };
    }

    sidebarStructure.forEach(folder => {
        const folderDiv = document.createElement("div");
        folderDiv.className = `sidebar-folder-group ${window.sidebarFolderStates[folder.id] ? "expanded" : ""}`;
        folderDiv.id = `folder-group-${folder.id}`;

        // Folder Header
        const header = document.createElement("div");
        header.className = "sidebar-folder-header";
        header.innerHTML = `
            <div class="folder-header-left">
                <svg class="chevron-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"></polyline></svg>
                <span class="folder-title-text">${folder.title}</span>
            </div>
            <span class="folder-tab-decor"></span>
        `;

        const ulWrapper = document.createElement("div");
        ulWrapper.className = "sidebar-folder-content";

        const ul = document.createElement("ul");
        ul.className = "sidebar-menu";

        folder.keys.forEach(key => {
            const ch = CHAPTERS_DATABASE[key];
            if (!ch) return;

            const li = document.createElement("li");
            if (key === activeChapter) li.className = "active";
            li.setAttribute("data-chapter", key);

            // Determine custom dynamic badges
            let badgeHTML = "";
            if (key === "video-tutorials") {
                badgeHTML = `<span class="menu-badge badge-new-neon">NEW</span>`;
            } else if (key === "shortcuts") {
                badgeHTML = `<span class="menu-badge badge-interactive">LIVE</span>`;
            } else if (key === "developer-api") {
                badgeHTML = `<span class="menu-badge badge-api">API</span>`;
            } else if (key === "welcome") {
                badgeHTML = `<span class="menu-badge badge-core">CORE</span>`;
            }

            li.innerHTML = `<a href="#chapter-${key}">
                <span class="menu-item-left">${ch.icon || "📘"} ${ch.title}</span>
                ${badgeHTML}
            </a>`;

            li.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                switchChapter(key);
            });

            ul.appendChild(li);
        });

        ulWrapper.appendChild(ul);
        folderDiv.appendChild(header);
        folderDiv.appendChild(ulWrapper);

        // Header click toggles expand/collapse
        header.addEventListener("click", () => {
            const isExpanded = folderDiv.classList.contains("expanded");
            window.sidebarFolderStates[folder.id] = !isExpanded;
            
            if (isExpanded) {
                folderDiv.classList.remove("expanded");
            } else {
                folderDiv.classList.add("expanded");
            }
        });

        chaptersList.appendChild(folderDiv);
    });
}

function switchChapter(key) {
    if (!CHAPTERS_DATABASE[key]) return;
    
    activeChapter = key;
    
    // Update sidebar styles
    const items = document.querySelectorAll("#sidebar-chapters li");
    items.forEach(item => {
        if (item.getAttribute("data-chapter") === key) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });

    renderActiveChapter();
    
    // Scroll content area back to top
    document.getElementById("main-content").scrollTop = 0;
}

// Renders markdown block corresponding to current active chapter
function renderActiveChapter() {
    const contentArea = document.getElementById("docs-content-area");
    if (!contentArea) return;

    const ch = CHAPTERS_DATABASE[activeChapter];
    if (!ch) return;

    contentArea.innerHTML = `
        <h1>${ch.title}</h1>
        ${ch.content}
    `;

    // Hook sub-interactive slots
    if (activeChapter === "noderef") {
        renderNodeReferenceGrid();
    } else if (activeChapter === "shortcuts") {
        renderShortcutsVisualizer();
    }

    // Attach copy button listeners to any rendered preblocks
    setupCodeCopyButtons();
}

// Tab navigation handler (Docs vs Playground)
function initTabRouting() {
    const tabs = document.querySelectorAll(".tab-btn");
    const panes = document.querySelectorAll(".tab-pane");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetPane = `tab-${tab.getAttribute("data-tab")}`;
            
            tabs.forEach(t => t.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            
            const activePane = document.getElementById(targetPane);
            if (activePane) {
                activePane.classList.add("active");
            }
        });
    });

    // Press '/' or 'Ctrl+K' to focus search
    window.addEventListener("keydown", (e) => {
        if (e.key === "/" || (e.ctrlKey && e.key.toLowerCase() === "k")) {
            const searchInput = document.getElementById("global-search");
            if (document.activeElement !== searchInput) {
                e.preventDefault();
                searchInput.focus();
                searchInput.select();
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           4. RENDERING SPECIAL REFERENCE SLOTS
// ═══════════════════════════════════════════════════════════════════════════════

function renderNodeReferenceGrid(filtered = NODE_DATABASE) {
    const slot = document.getElementById("nodes-rendered-slot");
    if (!slot) return;

    slot.innerHTML = "";

    const grid = document.createElement("div");
    grid.className = "nodes-grid";

    filtered.forEach(node => {
        const card = document.createElement("div");
        card.className = "node-card";
        
        let paramsHTML = "";
        if (node.inputs && node.inputs.length > 0) {
            paramsHTML = `
                <div class="params-box">
                    <span class="param-title">Inputs</span>
                    <div class="param-list">
                        ${node.inputs.map(i => `
                            <div class="param-item">
                                <span class="param-name">${i.name}</span>
                                <span class="param-type">${i.type}</span>
                            </div>
                        `).join("")}
                    </div>
                </div>
            `;
        }

        let outputsHTML = "";
        if (node.outputs && node.outputs.length > 0) {
            outputsHTML = `
                <div class="outputs-box">
                    <div class="param-list">
                        ${node.outputs.map(o => `
                            <div class="output-item">
                                <span class="output-name">${o.name}</span>
                                <span class="output-type">${o.type}</span>
                            </div>
                        `).join("")}
                    </div>
                </div>
            `;
        }

        let formulaHTML = "";
        if (node.formula) {
            formulaHTML = `<div class="math-formula"><code>${node.formula}</code></div>`;
        }

        card.innerHTML = `
            <div class="node-meta">
                <span class="node-cat">${node.id}</span>
            </div>
            <h3>${node.name}</h3>
            <p class="node-desc">${node.desc}</p>
            ${paramsHTML}
            ${outputsHTML}
            ${formulaHTML}
            <div class="expert-tip">${node.tip}</div>
        `;
        grid.appendChild(card);
    });

    slot.appendChild(grid);

    if (filtered.length === 0) {
        slot.innerHTML = `
            <div class="welcome-hero" style="text-align: center; padding: 40px; margin-top: 20px;">
                <h2>No matching nodes found</h2>
                <p>Try searching "CDL", "Log", or "EXR" to find dynamic VFX color components.</p>
            </div>
        `;
    }
}

// Renders the keyboard shortcut visually under its tab
function renderShortcutsVisualizer() {
    const slot = document.getElementById("keyboard-rendered-slot");
    if (!slot) return;

    slot.innerHTML = `
        <div class="keyboard-visualizer">
            <div class="keyboard-row">
                <div class="key" data-key="Esc"><span>Esc</span></div>
                <div class="key spacer-small"></div>
                <div class="key" data-key="1"><span>1</span><span class="subtext">1:1 Zoom</span></div>
                <div class="key" data-key="2"><span>2</span></div>
                <div class="key" data-key="3"><span>3</span></div>
                <div class="key" data-key="4"><span>4</span></div>
                <div class="key" data-key="5"><span>5</span></div>
                <div class="key" data-key="6"><span>6</span></div>
                <div class="key" data-key="7"><span>7</span></div>
                <div class="key" data-key="8"><span>8</span></div>
                <div class="key" data-key="9"><span>9</span></div>
                <div class="key" data-key="0"><span>0</span></div>
                <div class="key" data-key="-"><span>-</span></div>
                <div class="key" data-key="="><span>=</span></div>
            </div>
            <div class="keyboard-row">
                <div class="key wide-tab"><span>Tab</span></div>
                <div class="key" data-key="q"><span>Q</span></div>
                <div class="key" data-key="w"><span>W</span><span class="subtext">Waveform</span></div>
                <div class="key" data-key="e"><span>E</span></div>
                <div class="key" data-key="r"><span>R</span><span class="subtext">Red Ch</span></div>
                <div class="key" data-key="t"><span>T</span></div>
                <div class="key" data-key="y"><span>Y</span></div>
                <div class="key" data-key="u"><span>U</span></div>
                <div class="key" data-key="i"><span>I</span></div>
                <div class="key" data-key="o"><span>O</span></div>
                <div class="key" data-key="p"><span>P</span></div>
                <div class="key" data-key="["><span>[</span></div>
                <div class="key" data-key="]"><span>]</span></div>
            </div>
            <div class="keyboard-row">
                <div class="key wide-caps"><span>Caps</span></div>
                <div class="key" data-key="a"><span>A</span><span class="subtext">A/B Mode</span></div>
                <div class="key" data-key="s"><span>S</span><span class="subtext">Safe Area</span></div>
                <div class="key" data-key="d"><span>D</span></div>
                <div class="key" data-key="f"><span>F</span><span class="subtext">Fit View</span></div>
                <div class="key" data-key="g"><span>G</span><span class="subtext">Green Ch</span></div>
                <div class="key" data-key="h"><span>H</span></div>
                <div class="key" data-key="j"><span>J</span></div>
                <div class="key" data-key="k"><span>K</span></div>
                <div class="key" data-key="l"><span>L</span><span class="subtext">Luma Ch</span></div>
                <div class="key" data-key=";"><span>;</span></div>
                <div class="key" data-key="'"><span>'</span></div>
                <div class="key wide-enter"><span>Enter</span></div>
            </div>
            <div class="keyboard-row">
                <div class="key extra-wide"><span>Shift</span></div>
                <div class="key" data-key="z"><span>Z</span></div>
                <div class="key" data-key="x"><span>X</span></div>
                <div class="key" data-key="c"><span>C</span><span class="subtext">RGB Color</span></div>
                <div class="key" data-key="v"><span>V</span><span class="subtext">Vectorscope</span></div>
                <div class="key" data-key="b"><span>B</span><span class="subtext">Blue Ch</span></div>
                <div class="key" data-key="n"><span>N</span></div>
                <div class="key" data-key="m"><span>M</span><span class="subtext">Parade</span></div>
                <div class="key" data-key=","><span>,</span></div>
                <div class="key" data-key="."><span>.</span></div>
                <div class="key" data-key="/"><span>/</span></div>
                <div class="key extra-wide"><span>Shift</span></div>
            </div>
            <div class="keyboard-row">
                <div class="key"><span>Ctrl</span></div>
                <div class="key"><span>Alt</span></div>
                <div class="key spacebar" data-key="Space"><span>Space (Toggle Playback)</span></div>
                <div class="key"><span>Alt</span></div>
                <div class="keyArrow" data-key="Left"><span>←</span><span class="subtext">Prev</span></div>
                <div class="keyArrow" data-key="Right"><span>→</span><span class="subtext">Next</span></div>
            </div>
        </div>

        <div id="key-info-box" class="key-info-overlay">
            <div class="key-info-inner">
                <span class="indicator-kbd">Shortcut Inspector</span>
                <h4 id="info-key-name">Hover over a mapped keycap...</h4>
                <p id="info-key-desc">Interactive key definitions will populate here automatically.</p>
            </div>
        </div>

        <div class="shortcuts-grid">
            <div class="shortcut-card">
                <h3>◎ Navigation & Playback</h3>
                <table class="shortcut-table">
                    <tr><td><kbd>Space</kbd></td><td>Toggle Playback</td></tr>
                    <tr><td><kbd>←</kbd> / <kbd>→</kbd></td><td>Previous / Next Frame</td></tr>
                    <tr><td><kbd>F</kbd></td><td>Fit to View (Recenter)</td></tr>
                    <tr><td><kbd>1</kbd></td><td>1:1 Pixel Zoom</td></tr>
                    <tr><td><kbd>Shift + Drag</kbd></td><td>Pan Canvas Image</td></tr>
                    <tr><td><kbd>Esc</kbd></td><td>Exit Fullscreen / Close Panels</td></tr>
                </table>
            </div>
            <div class="shortcut-card">
                <h3>◎ Channel Monitoring</h3>
                <table class="shortcut-table">
                    <tr><td><kbd>C</kbd></td><td>Full RGB Color View</td></tr>
                    <tr><td><kbd>R</kbd></td><td>Red Color Channel Only</td></tr>
                    <tr><td><kbd>G</kbd></td><td>Green Color Channel Only</td></tr>
                    <tr><td><kbd>B</kbd></td><td>Blue Color Channel Only</td></tr>
                    <tr><td><kbd>L</kbd></td><td>Luminance Channel Only</td></tr>
                    <tr><td><kbd>Shift + A</kbd></td><td>Alpha Channel Monitor</td></tr>
                </table>
            </div>
            <div class="shortcut-card">
                <h3>◎ Scopes & Overlays</h3>
                <table class="shortcut-table">
                    <tr><td><kbd>W</kbd></td><td>Toggle Waveform (HDR-Aware)</td></tr>
                    <tr><td><kbd>M</kbd></td><td>Toggle Waveform Parade Mode</td></tr>
                    <tr><td><kbd>V</kbd></td><td>Toggle Vectorscope (w/ Skin Tone Line)</td></tr>
                    <tr><td><kbd>S</kbd></td><td>Cycle Safe Area Action/Title Grids</td></tr>
                    <tr><td><kbd>Shift + G</kbd></td><td>Cycle Aspect Ratio Grid Guides</td></tr>
                    <tr><td><kbd>A</kbd></td><td>Cycle A/B Split Comparison Modes</td></tr>
                </table>
            </div>
        </div>
    `;

    // Initialize keymap listeners inside the dynamically loaded container
    const keys = slot.querySelectorAll(".key");
    const keyBoxName = slot.querySelector("#info-key-name");
    const keyBoxDesc = slot.querySelector("#info-key-desc");

    keys.forEach(key => {
        const code = key.getAttribute("data-key");
        
        if (code && KEY_SHORTCUTS[code]) {
            key.addEventListener("mouseenter", () => {
                key.classList.add("active-hover");
                keyBoxName.innerText = `Key [ ${code} ] — Viewer Command`;
                keyBoxDesc.innerText = KEY_SHORTCUTS[code];
            });

            key.addEventListener("mouseleave", () => {
                key.classList.remove("active-hover");
            });

            key.addEventListener("click", () => {
                keyBoxName.innerText = `Key [ ${code} ] — Selected`;
                keyBoxDesc.innerText = KEY_SHORTCUTS[code];
            });
        } else if (code) {
            key.addEventListener("mouseenter", () => {
                keyBoxName.innerText = `Key [ ${code} ]`;
                keyBoxDesc.innerText = "This key is currently unassigned in the Radiance Viewer.";
            });
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           5. GLOBAL REAL-TIME SEARCH INDEX
// ═══════════════════════════════════════════════════════════════════════════════

function initGlobalSearch() {
    const searchInput = document.getElementById("global-search");
    if (!searchInput) return;

    searchInput.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase().trim();
        
        if (!query) {
            // Restore active chapter view
            renderActiveChapter();
            return;
        }

        // Toggles tab back to "Docs" if searching from Playground
        const docTab = document.querySelector('[data-tab="docs"]');
        if (docTab) docTab.click();

        // Search through the 121 Node Reference and display matches
        const filteredNodes = NODE_DATABASE.filter(node => {
            return (
                node.name.toLowerCase().includes(query) ||
                node.id.toLowerCase().includes(query) ||
                node.zone.toLowerCase().includes(query) ||
                node.desc.toLowerCase().includes(query) ||
                node.tip.toLowerCase().includes(query) ||
                (node.inputs && node.inputs.some(i => i.name.toLowerCase().includes(query)))
            );
        });

        // Set Node Reference as active chapter during search feedback
        activeChapter = "noderef";
        const items = document.querySelectorAll("#sidebar-chapters li");
        items.forEach(item => {
            if (item.getAttribute("data-chapter") === "noderef") {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });

        renderActiveChapter();
        renderNodeReferenceGrid(filteredNodes);
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           6. REAL-TIME COLOR SCIENCE SIMULATOR
// ═══════════════════════════════════════════════════════════════════════════════

function initColorSimulator() {
    const canvasGrad = document.getElementById("canvas-gradient");
    const canvasWave = document.getElementById("canvas-waveform");
    
    if (!canvasGrad || !canvasWave) return;

    const ctxGrad = canvasGrad.getContext("2d");
    const ctxWave = canvasWave.getContext("2d");

    // Slider Nodes
    const sliders = {
        exposure: document.getElementById("slide-exposure"),
        slope: document.getElementById("slide-slope"),
        offset: document.getElementById("slide-offset"),
        power: document.getElementById("slide-power"),
        saturation: document.getElementById("slide-saturation")
    };

    // Value HUD Labels
    const labels = {
        exposure: document.getElementById("val-exposure"),
        slope: document.getElementById("val-slope"),
        offset: document.getElementById("val-offset"),
        power: document.getElementById("val-power"),
        saturation: document.getElementById("val-saturation")
    };

    const resetBtn = document.getElementById("btn-reset-sim");

    let isDrawing = false;

    function queueRedraw() {
        if (!isDrawing) {
            isDrawing = true;
            requestAnimationFrame(drawSimulatorState);
        }
    }

    function drawSimulatorState() {
        isDrawing = false;
        
        // Grab current sliders values
        const ev = parseFloat(sliders.exposure.value);
        const slope = parseFloat(sliders.slope.value);
        const offset = parseFloat(sliders.offset.value);
        const power = parseFloat(sliders.power.value);
        const sat = parseFloat(sliders.saturation.value);

        // Update Text HUD labels
        labels.exposure.innerText = `${ev >= 0 ? '+' : ''}${ev.toFixed(2)} EV`;
        labels.slope.innerText = slope.toFixed(2);
        labels.offset.innerText = `${offset >= 0 ? '+' : ''}${offset.toFixed(2)}`;
        labels.power.innerText = power.toFixed(2);
        labels.saturation.innerText = sat.toFixed(2);

        const width = canvasGrad.width;
        const height = canvasGrad.height;

        // 1. Draw gradient test pattern on gradient canvas
        const imgData = ctxGrad.createImageData(width, height);
        const data = imgData.data;

        // Precalculate exposure offset scaler
        const expMultiplier = Math.pow(2, ev);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;

                // Create a beautiful linear gradient representing colorful highlights
                let r_lin = (x / width) * 1.25;
                let g_lin = (y / height) * 0.95 + (x / width) * 0.3;
                let b_lin = 1.0 - (x / width) * 1.0;

                // --- PIPELINE 1: EXPOSURE ADJUSTMENT ---
                r_lin *= expMultiplier;
                g_lin *= expMultiplier;
                b_lin *= expMultiplier;

                // --- PIPELINE 2: CDL SLOPE + OFFSET + POWER ---
                let r_cdl = Math.pow(Math.max(0, r_lin * slope + offset), power);
                let g_cdl = Math.pow(Math.max(0, g_lin * slope + offset), power);
                let b_cdl = Math.pow(Math.max(0, b_lin * slope + offset), power);

                // --- PIPELINE 3: GLOBAL LUMINANCE & SATURATION ---
                const luma = 0.2126 * r_cdl + 0.7152 * g_cdl + 0.0722 * b_cdl;
                
                let r_sat = luma + sat * (r_cdl - luma);
                let g_sat = luma + sat * (g_cdl - luma);
                let b_sat = luma + sat * (b_cdl - luma);

                // Safe clamping to display sRGB limits
                data[idx]     = Math.min(255, Math.max(0, r_sat * 255));     // Red
                data[idx + 1] = Math.min(255, Math.max(0, g_sat * 255));     // Green
                data[idx + 2] = Math.min(255, Math.max(0, b_sat * 255));     // Blue
                data[idx + 3] = 255;                                         // Alpha
            }
        }
        ctxGrad.putImageData(imgData, 0, 0);

        // 2. Draw Waveform Scope Overlay
        drawWaveformParade(data, width, height, ctxWave, canvasWave.width, canvasWave.height);
    }

    // Advanced Waveform Scope drawing logic
    function drawWaveformParade(gradientData, srcW, srcH, scopeCtx, scopeW, scopeH) {
        scopeCtx.fillStyle = "#040405";
        scopeCtx.fillRect(0, 0, scopeW, scopeH);

        // Draw horizontal grid lines
        scopeCtx.strokeStyle = "rgba(255, 255, 255, 0.06)";
        scopeCtx.lineWidth = 1;
        
        const lines = [0.1, 0.3, 0.5, 0.7, 0.9];
        lines.forEach(ratio => {
            const h = scopeH - ratio * scopeH;
            scopeCtx.beginPath();
            scopeCtx.moveTo(0, h);
            scopeCtx.lineTo(scopeW, h);
            scopeCtx.stroke();
            
            scopeCtx.fillStyle = "#555";
            scopeCtx.font = "8px monospace";
            scopeCtx.fillText(`${Math.round(ratio * 100)}%`, 6, h - 3);
        });

        const paradeWidth = Math.floor(scopeW / 3);
        const colStep = 2; 
        const pixelData = scopeCtx.createImageData(scopeW, scopeH);
        const out = pixelData.data;

        for (let i = 0; i < out.length; i += 4) {
            out[i] = 4; out[i+1] = 4; out[i+2] = 5; out[i+3] = 255;
        }

        // Analyze columns
        for (let col = 0; col < srcW; col += colStep) {
            const targetXRed = Math.floor((col / srcW) * paradeWidth);
            const targetXGreen = targetXRed + paradeWidth;
            const targetXBlue = targetXGreen + paradeWidth;

            for (let row = 0; row < srcH; row += 2) {
                const idx = (row * srcW + col) * 4;
                const r = gradientData[idx];
                const g = gradientData[idx + 1];
                const b = gradientData[idx + 2];

                const yRed = Math.floor(scopeH - (r / 255) * (scopeH - 4));
                const yGreen = Math.floor(scopeH - (g / 255) * (scopeH - 4));
                const yBlue = Math.floor(scopeH - (b / 255) * (scopeH - 4));

                if (yRed >= 0 && yRed < scopeH && targetXRed < scopeW) {
                    const outIdx = (yRed * scopeW + targetXRed) * 4;
                    out[outIdx] = Math.min(255, out[outIdx] + 90);
                    out[outIdx + 3] = 255;
                }

                if (yGreen >= 0 && yGreen < scopeH && targetXGreen < scopeW) {
                    const outIdx = (yGreen * scopeW + targetXGreen) * 4;
                    out[outIdx + 1] = Math.min(255, out[outIdx + 1] + 90);
                    out[outIdx + 3] = 255;
                }

                if (yBlue >= 0 && yBlue < scopeH && targetXBlue < scopeW) {
                    const outIdx = (yBlue * scopeW + targetXBlue) * 4;
                    out[outIdx + 2] = Math.min(255, out[outIdx + 2] + 90);
                    out[outIdx + 3] = 255;
                }
            }
        }

        scopeCtx.putImageData(pixelData, 0, 0);

        // Overlay Parade channel dividers
        scopeCtx.strokeStyle = "rgba(0, 240, 255, 0.12)";
        scopeCtx.lineWidth = 1;
        scopeCtx.beginPath();
        scopeCtx.moveTo(paradeWidth, 0); scopeCtx.lineTo(paradeWidth, scopeH);
        scopeCtx.moveTo(paradeWidth * 2, 0); scopeCtx.lineTo(paradeWidth * 2, scopeH);
        scopeCtx.stroke();

        scopeCtx.fillStyle = "rgba(255, 255, 255, 0.35)";
        scopeCtx.font = "8px monospace";
        scopeCtx.fillText("RED CHANNEL", paradeWidth / 2 - 25, 14);
        scopeCtx.fillText("GREEN CHANNEL", paradeWidth + paradeWidth / 2 - 30, 14);
        scopeCtx.fillText("BLUE CHANNEL", paradeWidth * 2 + paradeWidth / 2 - 28, 14);
    }

    Object.values(sliders).forEach(slider => {
        slider.addEventListener("input", queueRedraw);
    });

    resetBtn.addEventListener("click", () => {
        sliders.exposure.value = "0.0";
        sliders.slope.value = "1.0";
        sliders.offset.value = "0.0";
        sliders.power.value = "1.0";
        sliders.saturation.value = "1.0";
        queueRedraw();
    });

    queueRedraw();
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           7. RECIPES COPY TO CLIPBOARD
// ═══════════════════════════════════════════════════════════════════════════════

function setupCodeCopyButtons() {
    const buttons = document.querySelectorAll(".copy-btn, .copy-btn-anchor");
    
    buttons.forEach(button => {
        // Clear previous listeners by cloning if necessary (avoid duplicate bindings)
        const newBtn = button.cloneNode(true);
        button.parentNode.replaceChild(newBtn, button);

        newBtn.addEventListener("click", () => {
            const rawCode = newBtn.getAttribute("data-copy");
            
            navigator.clipboard.writeText(rawCode).then(() => {
                const oldText = newBtn.innerText;
                newBtn.innerText = "Copied!";
                newBtn.style.borderColor = "var(--accent-cyan)";
                newBtn.style.color = "var(--accent-cyan)";
                
                setTimeout(() => {
                    newBtn.innerText = oldText;
                    newBtn.style.borderColor = "";
                    newBtn.style.color = "";
                }, 1500);
            }).catch(err => {
                console.error("Clipboard copy failed: ", err);
            });
        });
    });
}
