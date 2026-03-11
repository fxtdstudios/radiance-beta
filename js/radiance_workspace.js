// radiance_workspace.js
// Handles the serialization of the entire ComfyUI app.graph into a secure .rad file

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HEADER = "RAD_WORKSPACE_V1::";

// Inject styles for the Assets Library modal
const style = document.createElement('style');
style.textContent = `
    .radiance-modal-overlay {
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0,0,0,0.7); backdrop-filter: blur(8px); z-index: 10000;
        display: flex; justify-content: center; align-items: center;
        font-family: 'Inter', sans-serif;
    }
    .radiance-modal {
        background: rgba(30, 32, 40, 0.95); color: #ddd; padding: 25px; border-radius: 12px;
        width: 85%; max-width: 750px; max-height: 85%; overflow-y: auto;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.1);
    }
    .radiance-modal-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px;
    }
    .radiance-modal h2 { margin: 0; color: #fff; font-size: 1.5em; letter-spacing: -0.02em; }
    
    .radiance-search-container { margin-bottom: 20px; }
    .radiance-search-input {
        width: 100%; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);
        padding: 10px 15px; border-radius: 6px; color: #fff; font-size: 1em; outline: none;
    }
    .radiance-search-input:focus { border-color: #4db3ff; box-shadow: 0 0 0 2px rgba(77, 179, 255, 0.2); }

    .radiance-workflow-list { list-style: none; padding: 0; margin: 0; }
    .radiance-workflow-item {
        display: flex; justify-content: space-between; align-items: center;
        padding: 15px; border-bottom: 1px solid rgba(255,255,255,0.05); 
        background: rgba(255,255,255,0.02); margin-bottom: 8px; border-radius: 8px;
        transition: transform 0.2s, background 0.2s;
    }
    .radiance-workflow-item:hover { background: rgba(255,255,255,0.05); transform: translateX(4px); }
    .radiance-workflow-info { display: flex; flex-direction: column; flex: 1; margin-right: 15px; }
    .radiance-workflow-name { font-weight: 600; color: #fff; font-size: 1.1em; }
    .radiance-workflow-desc { font-size: 0.85em; color: #888; margin-top: 4px; line-height: 1.4; }
    .radiance-workflow-meta { font-size: 0.75em; color: #555; margin-top: 6px; font-variant-numeric: tabular-nums; }
    
    .radiance-actions { display: flex; gap: 8px; }
    .radiance-btn {
        background: rgba(255,255,255,0.1); color: white; border: none; padding: 8px 14px;
        border-radius: 6px; cursor: pointer; font-size: 0.85em; font-weight: 500;
        transition: background 0.2s, color 0.2s;
    }
    .radiance-btn:hover { background: rgba(255,255,255,0.2); }
    .radiance-btn.primary { background: #005bb7; }
    .radiance-btn.primary:hover { background: #0076eb; }
    .radiance-btn.danger { background: rgba(255, 59, 48, 0.1); color: #ff3b30; }
    .radiance-btn.danger:hover { background: rgba(255, 59, 48, 0.2); }
    
    .radiance-badge {
        background: #444; color: #aaa; padding: 2px 6px; border-radius: 4px;
        font-size: 0.7em; text-transform: uppercase; margin-left: 8px; vertical-align: middle;
    }
    .radiance-badge.system { background: #1a1c23; color: #4db3ff; border: 1px solid #005bb7; }
    
    .radiance-folder-group { margin-bottom: 25px; }
    .radiance-folder-header {
        font-size: 0.9em; font-weight: 700; color: #4db3ff; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 12px; display: flex; align-items: center;
        opacity: 0.8;
    }
    .radiance-folder-header::before {
        content: ""; display: inline-block; width: 4px; height: 14px;
        background: #4db3ff; margin-right: 10px; border-radius: 2px;
    }

    .radiance-modal-close { margin-top: 25px; width: 100%; padding: 12px; font-weight: bold; background: #333; }
`;
document.head.appendChild(style);

app.registerExtension({
    name: "Radiance.Workspace",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RadianceWorkspace") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                // Add Export Button
                this.addWidget("button", "Export .rad", "export", () => {
                    this.exportRadWorkspace();
                });

                // Add Import Button
                this.addWidget("button", "Import .rad", "import", () => {
                    this.importRadWorkspace();
                });

                // Add Assets Library Buttons
                this.addWidget("button", "Save to Library", "save_lib", () => {
                    this.saveToLibrary();
                });
                this.addWidget("button", "Open Library", "open_lib", () => {
                    this.openLibrary();
                });

                // Documentation links
                this.addWidget("button", "Docs", "docs_link", () => {
                    window.open("https://radiance.fxtd.org", "_blank");
                });
                this.addWidget("button", "FXTD Studios", "site_link", () => {
                    window.open("https://www.fxtd.org", "_blank");
                });

                // Style the node
                this.color = "#1a1c23";
                this.bgcolor = "#1a1c23";
                this.size = [300, 200];
            };

            nodeType.prototype.buildRadContent = function () {
                const graphData = app.graph.serialize();
                const jsonString = JSON.stringify(graphData);
                const base64Data = btoa(unescape(encodeURIComponent(jsonString)));
                return HEADER + base64Data;
            };

            nodeType.prototype.exportRadWorkspace = function () {
                try {
                    let defaultName = `workspace_${Date.now()}`;
                    let chosenName = prompt("Enter a name for the workflow export:", defaultName);

                    if (chosenName === null) {
                        return; // User cancelled
                    }
                    if (chosenName.trim() === "") {
                        chosenName = defaultName;
                    }
                    if (!chosenName.endsWith(".rad")) {
                        chosenName += ".rad";
                    }

                    const radContent = this.buildRadContent();

                    const blob = new Blob([radContent], { type: 'application/octet-stream' });
                    const url = URL.createObjectURL(blob);

                    const a = document.createElement('a');
                    a.href = url;
                    a.download = chosenName;
                    document.body.appendChild(a);
                    a.click();

                    setTimeout(() => {
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                    }, 0);

                    console.log("[Radiance] Successfully exported .rad workspace.");
                } catch (e) {
                    console.error("[Radiance] Failed to export workspace:", e);
                    alert("Failed to export Radiance Workspace. Check console for details.");
                }
            };

            nodeType.prototype.saveToLibrary = async function () {
                try {
                    let defaultName = `workspace_${Date.now()}`;
                    let chosenName = prompt("Enter filename (e.g., Experiments/Test01):", defaultName);

                    if (chosenName === null || chosenName.trim() === "") {
                        return;
                    }
                    if (!chosenName.endsWith(".rad")) {
                        chosenName += ".rad";
                    }

                    let description = prompt("Optional: Enter a short description for this workflow:");
                    if (description === null) description = "";

                    const radContent = this.buildRadContent();

                    const response = await api.fetchApi("/radiance/workflows/save", {
                        method: "POST",
                        body: JSON.stringify({ 
                            filename: chosenName, 
                            content: radContent,
                            description: description
                        }),
                        headers: { "Content-Type": "application/json" }
                    });

                    if (response.ok) {
                        alert(`Saved ${chosenName} to Assets Library successfully.`);
                    } else {
                        const err = await response.json();
                        alert(`Failed to save: ${err.error || 'Unknown error'}`);
                    }
                } catch (e) {
                    console.error("[Radiance] Failed to save to library:", e);
                    alert("Failed to save to Assets Library. Check console for details.");
                }
            };

            nodeType.prototype.openLibrary = async function () {
                try {
                    const response = await api.fetchApi("/radiance/workflows/list");
                    if (!response.ok) throw new Error("Failed to fetch library.");
                    const data = await response.json();

                    this.showLibraryModal(data.workflows || []);
                } catch (e) {
                    console.error("[Radiance] Failed to load library:", e);
                    alert("Failed to load Assets Library. Check console for details.");
                }
            };

            nodeType.prototype.showLibraryModal = function (workflows) {
                const overlay = document.createElement('div');
                overlay.className = 'radiance-modal-overlay';

                const modal = document.createElement('div');
                modal.className = 'radiance-modal';

                const header = document.createElement('div');
                header.className = 'radiance-modal-header';
                
                const title = document.createElement('h2');
                title.textContent = "Radiance Assets Library";
                header.appendChild(title);
                modal.appendChild(header);

                // Search Bar
                const searchContainer = document.createElement('div');
                searchContainer.className = 'radiance-search-container';
                const searchInput = document.createElement('input');
                searchInput.placeholder = "Search workflows by name or description...";
                searchInput.className = 'radiance-search-input';
                searchContainer.appendChild(searchInput);
                modal.appendChild(searchContainer);

                const list = document.createElement('ul');
                list.className = 'radiance-workflow-list';
                modal.appendChild(list);

                const renderList = (filter = "") => {
                    list.innerHTML = "";
                    const filtered = workflows.filter(wf => {
                        const searchStr = `${wf.filename} ${wf.metadata?.description || ""}`.toLowerCase();
                        return searchStr.includes(filter.toLowerCase());
                    });

                    if (filtered.length === 0) {
                        const emptyMsg = document.createElement('p');
                        emptyMsg.textContent = "No workflows found.";
                        emptyMsg.style.textAlign = "center";
                        emptyMsg.style.opacity = "0.5";
                        list.appendChild(emptyMsg);
                        return;
                    }

                    // Group by folder
                    const groups = {};
                    filtered.forEach(wf => {
                        const parts = wf.filename.split('/');
                        const folder = parts.length > 1 ? parts[0] : "Root";
                        if (!groups[folder]) groups[folder] = [];
                        groups[folder].push(wf);
                    });

                    // Sort folders (Root first)
                    const sortedFolders = Object.keys(groups).sort((a, b) => {
                        if (a === "Root") return -1;
                        if (b === "Root") return 1;
                        return a.localeCompare(b);
                    });

                    sortedFolders.forEach(folder => {
                        const groupDiv = document.createElement('div');
                        groupDiv.className = 'radiance-folder-group';
                        
                        const header = document.createElement('div');
                        header.className = 'radiance-folder-header';
                        header.textContent = folder === "Root" ? "General Workflows" : folder;
                        groupDiv.appendChild(header);

                        const groupList = document.createElement('ul');
                        groupList.className = 'radiance-workflow-list';

                        groups[folder].forEach(wf => {
                            const li = document.createElement('li');
                            li.className = 'radiance-workflow-item';

                            const info = document.createElement('div');
                            info.className = 'radiance-workflow-info';

                            const nameSpan = document.createElement('span');
                            nameSpan.className = 'radiance-workflow-name';
                            const displayName = wf.filename.split('/').pop().replace('.rad', '');
                            nameSpan.textContent = displayName;

                            if (wf.filename === 'default.rad') {
                                const badge = document.createElement('span');
                                badge.className = 'radiance-badge system';
                                badge.textContent = 'System';
                                nameSpan.appendChild(badge);
                            }

                            info.appendChild(nameSpan);

                            if (wf.metadata?.description) {
                                const descDiv = document.createElement('div');
                                descDiv.className = 'radiance-workflow-desc';
                                descDiv.textContent = wf.metadata.description;
                                info.appendChild(descDiv);
                            }

                            const metaSpan = document.createElement('span');
                            metaSpan.className = 'radiance-workflow-meta';
                            const date = new Date(wf.mtime * 1000).toLocaleString();
                            const sizeKB = Math.round(wf.size / 1024);
                            metaSpan.textContent = `${date}  •  ${sizeKB} KB`;

                            info.appendChild(metaSpan);
                            li.appendChild(info);

                            const actions = document.createElement('div');
                            actions.className = 'radiance-actions';

                            const appendBtn = document.createElement('button');
                            appendBtn.className = 'radiance-btn';
                            appendBtn.textContent = 'Append';
                            appendBtn.onclick = () => {
                                this.loadFromLibrary(wf.filename, true);
                                document.body.removeChild(overlay);
                            };

                            const loadBtn = document.createElement('button');
                            loadBtn.className = 'radiance-btn primary';
                            loadBtn.textContent = 'Load';
                            loadBtn.onclick = () => {
                                if (confirm(`Replace graph with '${displayName}'?`)) {
                                    this.loadFromLibrary(wf.filename, false);
                                    document.body.removeChild(overlay);
                                }
                            };

                            const delBtn = document.createElement('button');
                            delBtn.className = 'radiance-btn danger';
                            delBtn.textContent = 'Delete';
                            delBtn.onclick = async () => {
                                if (confirm(`Permanently delete '${wf.filename}'?`)) {
                                    await this.deleteFromLibrary(wf.filename);
                                    const response = await api.fetchApi("/radiance/workflows/list");
                                    const data = await response.json();
                                    workflows = data.workflows || [];
                                    renderList(searchInput.value);
                                }
                            };

                            actions.appendChild(appendBtn);
                            actions.appendChild(loadBtn);
                            actions.appendChild(delBtn);
                            li.appendChild(actions);

                            groupList.appendChild(li);
                        });
                        
                        groupDiv.appendChild(groupList);
                        list.appendChild(groupDiv);
                    });
                };

                searchInput.oninput = (e) => renderList(e.target.value);
                renderList();

                const closeBtn = document.createElement('button');
                closeBtn.className = 'radiance-btn radiance-modal-close';
                closeBtn.textContent = 'Close';
                closeBtn.onclick = () => document.body.removeChild(overlay);
                modal.appendChild(closeBtn);

                overlay.onclick = (e) => {
                    if (e.target === overlay) document.body.removeChild(overlay);
                };

                overlay.appendChild(modal);
                document.body.appendChild(overlay);
            };

            nodeType.prototype.loadFromLibrary = async function (filename, append = false) {
                try {
                    const response = await api.fetchApi(`/radiance/workflows/get?filename=${encodeURIComponent(filename)}`);
                    if (!response.ok) throw new Error("Failed to fetch workflow content.");
                    const data = await response.json();
                    if (!data.success) throw new Error(data.error);

                    this.parseAndLoadRadContent(data.content, append);
                } catch (e) {
                    console.error("[Radiance] Failed to load workflow:", e);
                    alert("Failed to load workflow from Library.");
                }
            };

            nodeType.prototype.deleteFromLibrary = async function (filename) {
                try {
                    const response = await api.fetchApi("/radiance/workflows/delete", {
                        method: "POST",
                        body: JSON.stringify({ filename }),
                        headers: { "Content-Type": "application/json" }
                    });
                    if (!response.ok) throw new Error("Delete request failed.");
                } catch (e) {
                    console.error("[Radiance] Failed to delete workflow:", e);
                    alert("Failed to delete from Library.");
                }
            };

            nodeType.prototype.parseAndLoadRadContent = function (content, append = false) {
                if (!content.startsWith(HEADER)) {
                    alert("Invalid .rad file: Missing Radiance Header.");
                    return;
                }

                const base64Data = content.replace(HEADER, '');
                const jsonString = decodeURIComponent(escape(atob(base64Data)));
                const graphData = JSON.parse(jsonString);

                app.loadGraphData(graphData, append);
                console.log(`[Radiance] Successfully ${append ? 'appended' : 'loaded'} .rad workspace.`);
            };

            nodeType.prototype.importRadWorkspace = function () {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.rad';

                input.onchange = e => {
                    const file = e.target.files[0];
                    if (!file) return;

                    const reader = new FileReader();
                    reader.onload = (event) => {
                        try {
                            this.parseAndLoadRadContent(event.target.result);
                        } catch (err) {
                            console.error("[Radiance] Failed to parse .rad file:", err);
                            alert("Failed to load .rad Workspace: File may be corrupted.");
                        }
                    };
                    reader.readAsText(file);
                };

                input.click();
            };
        }
    }
});
