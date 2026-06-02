const futureApi = {
    dashboard: "/radiance/projects/dashboard",
    projects: "/radiance/projects",
    recentProjects: "/radiance/projects/recent",
    versions: (projectId) => `/radiance/projects/${projectId}/versions`,
    outputs: (projectId) => `/radiance/projects/${projectId}/outputs`,
    notes: (projectId) => `/radiance/projects/${projectId}/notes`,
    saveVersion: (projectId) => `/radiance/projects/${projectId}/save-version`,
    exportPackage: (projectId) => `/radiance/projects/${projectId}/export-package`,
};

const mockProjectManagerData = {
    user: {
        name: "Ahmed",
        role: "Artist",
    },
    storage: {
        usedLabel: "412 GB",
        totalLabel: "640 GB",
        percent: 64,
    },
    continueWorking: {
        workflow: "SH010_HDR_Grade_v004",
        project: "HDR_Test",
        shot: "SH010",
        status: "Approved",
        lastModified: "2 hours ago",
    },
    projects: [
        { id: "hdr-test", name: "HDR_Test", shots: 12, workflows: 28, updated: "Today", favorite: true },
        { id: "commercial-01", name: "Commercial_01", shots: 18, workflows: 41, updated: "Yesterday", favorite: false },
        { id: "lookdev-project", name: "LookDev_Project", shots: 6, workflows: 13, updated: "3 days ago", favorite: true },
        { id: "personal-studies", name: "Personal_Studies", shots: 9, workflows: 19, updated: "Last week", favorite: false },
    ],
    versions: [
        { version: "v004", status: "Approved", shot: "SH010" },
        { version: "v003", status: "Review", shot: "SH020" },
        { version: "v002", status: "WIP", shot: "SH030" },
        { version: "v001", status: "On Hold", shot: "SH040" },
    ],
    outputs: [
        { name: "SH010_beauty_v004.exr", type: "EXR", size: "164 MB", date: "Today" },
        { name: "SH010_preview_v004.png", type: "PNG", size: "8 MB", date: "Today" },
        { name: "SH020_comp_v003.exr", type: "EXR", size: "142 MB", date: "Yesterday" },
        { name: "SH020_preview_v003.png", type: "PNG", size: "7 MB", date: "Yesterday" },
        { name: "SH030_light_v002.exr", type: "EXR", size: "188 MB", date: "3 days ago" },
    ],
    notes: [
        "Highlight clipping in SH010",
        "Check skin tones in SH020",
        "Reduce noise in shadows",
    ],
    quickActions: [
        "Save Version",
        "Compare Versions",
        "Import Workflow",
        "Export Package",
        "Project Settings",
    ],
    nav: [
        "Dashboard",
        "Projects",
        "Shots",
        "Workflows",
        "Versions",
        "Review",
        "Assets",
        "Settings",
    ],
    activeProjectId: "hdr-test",
    source: "Mock fallback",
};

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#39;",
    })[char]);
}

function iconFor(label) {
    const icons = {
        Dashboard: "D",
        Projects: "[]",
        Shots: "::",
        Workflows: "<>",
        Versions: "=",
        Review: "OK",
        Assets: "#",
        Settings: "*",
    };
    return icons[label] || "-";
}

function Sidebar(data) {
    return `
        <aside class="rpm-sidebar">
            <div class="rpm-brand">
                <img class="rpm-brand-mark" src="/extensions/radiance/r_icon.png" alt="Radiance" />
                <div>
                    <div class="rpm-brand-title">Radiance</div>
                    <div class="rpm-brand-subtitle">Project Manager</div>
                </div>
            </div>
            <nav class="rpm-nav" aria-label="Project Manager">
                ${data.nav.map((item) => `
                    <div class="rpm-nav-item ${item === "Dashboard" ? "is-active" : ""}">
                        <span class="rpm-nav-icon">${iconFor(item)}</span>
                        <span>${escapeHtml(item)}</span>
                    </div>
                `).join("")}
            </nav>
            <div class="rpm-sidebar-spacer"></div>
            ${StorageMiniBar(data.storage)}
        </aside>
    `;
}

function StorageMiniBar(storage) {
    return `
        <section class="rpm-storage" style="--storage-used: ${storage.percent}%">
            <div class="rpm-storage-row">
                <span class="rpm-storage-title">Storage</span>
                <span>${escapeHtml(storage.percent)}%</span>
            </div>
            <div class="rpm-storage-track" aria-hidden="true">
                <div class="rpm-storage-fill"></div>
            </div>
            <div class="rpm-storage-row">
                <span>${escapeHtml(storage.usedLabel)} used</span>
                <span>${escapeHtml(storage.totalLabel)}</span>
            </div>
        </section>
    `;
}

function TopBar(user) {
    return `
        <header class="rpm-topbar">
            <h1 class="rpm-page-title">Dashboard</h1>
            <div class="rpm-top-actions">
                <input class="rpm-search" data-search type="search" placeholder="Search projects, shots, versions">
                <button class="rpm-icon-button" type="button" title="Notifications" data-action="Notifications">!</button>
                <div class="rpm-user">
                    <div class="rpm-user-avatar">A</div>
                    <div>
                        <div class="rpm-user-name">${escapeHtml(user.name)}</div>
                        <div class="rpm-user-role">${escapeHtml(user.role)}</div>
                    </div>
                </div>
            </div>
        </header>
    `;
}

function Card(title, body, kicker = "") {
    return `
        <section class="rpm-card">
            <div class="rpm-card-header">
                <h2 class="rpm-card-title">${escapeHtml(title)}</h2>
                ${kicker ? `<span class="rpm-card-kicker">${escapeHtml(kicker)}</span>` : ""}
            </div>
            <div class="rpm-card-body">${body}</div>
        </section>
    `;
}

function ContinueWorkingCard(item) {
    return Card("Continue Working", `
        <div class="rpm-continue">
            <div class="rpm-thumbnail"></div>
            <div>
                <h3 class="rpm-work-title">${escapeHtml(item.workflow)}</h3>
                <div class="rpm-meta-grid">
                    ${MetaItem("Project", item.project)}
                    ${MetaItem("Shot", item.shot)}
                    ${MetaItem("Status", `<span class="rpm-status-pill">${escapeHtml(item.status)}</span>`, true)}
                    ${MetaItem("Last Modified", item.lastModified)}
                </div>
                <button class="rpm-button rpm-button-primary" type="button" data-action="Open Workflow">Open Workflow</button>
            </div>
        </div>
    `);
}

function MetaItem(label, value, raw = false) {
    return `
        <div>
            <div class="rpm-meta-label">${escapeHtml(label)}</div>
            <div class="rpm-meta-value">${raw ? value : escapeHtml(value)}</div>
        </div>
    `;
}

function RecentProjects(projects, source) {
    return Card("Recent Projects", `
        <div class="rpm-list" data-project-list>
            ${projects.length ? projects.map(ProjectRow).join("") : EmptyState("No projects found. Save a .rad workflow to create one.")}
        </div>
    `, source);
}

function ProjectRow(project) {
    return `
        <div class="rpm-list-row rpm-project-row" data-searchable="${escapeHtml(project.name).toLowerCase()}" data-action="Select Project" data-project-id="${escapeHtml(project.id)}">
            <div class="rpm-primary-text">${escapeHtml(project.name)}</div>
            <div class="rpm-secondary-text">${project.shots} shots</div>
            <div class="rpm-secondary-text">${project.workflows} workflows</div>
            <div class="rpm-secondary-text">${escapeHtml(project.updated)}</div>
            <button class="rpm-favorite" type="button" title="Favorite project" data-action="Toggle Favorite">${project.favorite ? "&#9733;" : "&#9734;"}</button>
        </div>
    `;
}

function RecentVersions(versions) {
    return Card("Recent Versions", `
        <div class="rpm-list">
            ${versions.length ? versions.map((item) => `
                <div class="rpm-list-row rpm-version-row">
                    <div class="rpm-primary-text">${escapeHtml(item.version)}</div>
                    <div class="rpm-secondary-text">${escapeHtml(item.status)}</div>
                    <div class="rpm-secondary-text">${escapeHtml(item.shot)}</div>
                </div>
            `).join("") : EmptyState("No versions tracked for this project yet.")}
        </div>
    `);
}

function RecentOutputs(outputs) {
    return Card("Recent Outputs", `
        <div class="rpm-list">
            ${outputs.length ? outputs.map((item) => `
                <div class="rpm-list-row rpm-output-row">
                    <div class="rpm-primary-text" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</div>
                    <div class="rpm-secondary-text">${escapeHtml(item.type)}</div>
                    <div class="rpm-secondary-text">${escapeHtml(item.size)}</div>
                    <div class="rpm-secondary-text">${escapeHtml(item.date)}</div>
                </div>
            `).join("") : EmptyState("No matching render outputs found in the ComfyUI output folder.")}
        </div>
    `);
}

function ReviewNotes(notes) {
    return Card("Review Notes", `
        <div class="rpm-list">
            ${notes.length ? notes.map((note) => `
                <div class="rpm-list-row rpm-note-row">
                    <div class="rpm-dot"></div>
                    <div class="rpm-primary-text">${escapeHtml(note)}</div>
                </div>
            `).join("") : EmptyState("No review notes for this project.")}
        </div>
    `);
}

function QuickActions(actions) {
    return Card("Quick Actions", `
        <div class="rpm-actions-grid">
            ${actions.map((action) => `
                <button class="rpm-button" type="button" data-action="${escapeHtml(action)}">${escapeHtml(action)}</button>
            `).join("")}
        </div>
    `);
}

function EmptyState(message) {
    return `<div class="rpm-empty">${escapeHtml(message)}</div>`;
}

function ProjectManagerDashboard(data) {
    return `
        ${Sidebar(data)}
        <main class="rpm-main">
            ${TopBar(data.user)}
            <div class="rpm-content">
                <div class="rpm-grid">
                    <div class="rpm-column">
                        ${ContinueWorkingCard(data.continueWorking)}
                        ${RecentProjects(data.projects, data.source || "Live API")}
                        ${RecentOutputs(data.outputs)}
                    </div>
                    <div class="rpm-column">
                        ${RecentVersions(data.versions)}
                        ${ReviewNotes(data.notes)}
                        ${QuickActions(data.quickActions)}
                    </div>
                </div>
            </div>
        </main>
        <div class="rpm-toast" data-toast></div>
    `;
}

function showToast(message) {
    const toast = document.querySelector("[data-toast]");
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("is-visible");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => {
        toast.classList.remove("is-visible");
    }, 1800);
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || response.statusText);
    }
    return response.json();
}

async function loadDashboardData() {
    try {
        const data = await fetchJson(futureApi.dashboard);
        return {
            ...mockProjectManagerData,
            ...data,
            projects: data.projects || [],
            versions: data.versions || [],
            outputs: data.outputs || [],
            notes: data.notes || [],
            quickActions: data.quickActions || mockProjectManagerData.quickActions,
            nav: data.nav || mockProjectManagerData.nav,
            source: data.source || "Live API",
        };
    } catch (error) {
        console.warn("[Radiance Project Manager] Falling back to mock data:", error);
        return mockProjectManagerData;
    }
}

async function loadProjectDetails(state, projectId) {
    const [versions, outputs, notes] = await Promise.all([
        fetchJson(futureApi.versions(projectId)),
        fetchJson(futureApi.outputs(projectId)),
        fetchJson(futureApi.notes(projectId)),
    ]);

    const project = state.data.projects.find((item) => item.id === projectId);
    const latestVersion = versions.versions?.[0];

    return {
        ...state.data,
        activeProjectId: projectId,
        versions: versions.versions || [],
        outputs: outputs.outputs || [],
        notes: notes.notes || [],
        continueWorking: latestVersion ? {
            workflow: latestVersion.workflow || latestVersion.filename || "No workflow saved yet",
            project: project?.name || state.data.continueWorking.project,
            shot: latestVersion.shot || "GENERAL",
            status: latestVersion.status || "WIP",
            lastModified: latestVersion.updated || "",
            filename: latestVersion.filename || "",
        } : {
            workflow: "No workflow saved yet",
            project: project?.name || state.data.continueWorking.project,
            shot: "GENERAL",
            status: "No Versions",
            lastModified: "",
            filename: "",
        },
    };
}

async function openWorkflow(filename) {
    if (!filename) {
        showToast("No workflow is available to open yet.");
        return;
    }

    const data = await fetchJson(`/radiance/workflows/get?filename=${encodeURIComponent(filename)}`);
    const target = window.opener || (window.parent !== window ? window.parent : null);
    if (target) {
        target.postMessage({ type: "radiance_load_workflow", content: data.content }, "*");
        showToast("Workflow sent to the active ComfyUI canvas.");
    } else {
        showToast("Open this dashboard from ComfyUI to load workflows into the canvas.");
    }
}

async function saveVersion(projectId) {
    const target = window.opener || (window.parent !== window ? window.parent : null);
    if (target) {
        target.postMessage({ type: "radiance_save_project_version", projectId }, "*");
        showToast("Save requested from the active ComfyUI canvas.");
        return;
    }

    const result = await fetchJson(futureApi.saveVersion(projectId), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "Dashboard save version", author: "Ahmed" }),
    });
    showToast(result.message || "Version saved.");
}

async function exportPackage(projectId) {
    const response = await fetch(futureApi.exportPackage(projectId), { method: "POST" });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || response.statusText);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${projectId || "radiance"}_package.zip`;
    document.body.appendChild(anchor);
    anchor.click();
    setTimeout(() => {
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
    }, 100);
}

function bindInteractions(root, state) {
    root.querySelector("[data-search]")?.addEventListener("input", (event) => {
        const query = event.target.value.trim().toLowerCase();
        root.querySelectorAll("[data-searchable]").forEach((row) => {
            row.hidden = query.length > 0 && !row.dataset.searchable.includes(query);
        });
    });

    root.addEventListener("click", (event) => {
        const button = event.target.closest("[data-action]");
        if (!button) return;

        const action = button.dataset.action;
        const activeProjectId = state.data.activeProjectId || state.data.projects[0]?.id || "";

        if (action === "Select Project") {
            const projectId = button.dataset.projectId;
            loadProjectDetails(state, projectId)
                .then((nextData) => render(nextData))
                .catch((error) => showToast(error.message));
            return;
        }

        if (action === "Open Workflow") {
            openWorkflow(state.data.continueWorking.filename).catch((error) => showToast(error.message));
            return;
        }

        if (action === "Save Version") {
            if (!activeProjectId) {
                showToast("Save a workflow first to create a project.");
                return;
            }
            saveVersion(activeProjectId)
                .then(() => loadDashboardData())
                .then((nextData) => render(nextData))
                .catch((error) => showToast(error.message));
            return;
        }

        if (action === "Export Package") {
            if (!activeProjectId) {
                showToast("Save a workflow first to create a project.");
                return;
            }
            exportPackage(activeProjectId)
                .then(() => showToast("Package export started."))
                .catch((error) => showToast(error.message));
            return;
        }

        showToast(`${action} is not wired to a backend action yet.`);
    });
}

const mount = document.getElementById("radiance-project-manager");

function render(data) {
    if (!mount) return;
    const state = { data };
    mount.innerHTML = ProjectManagerDashboard(data);
    bindInteractions(mount, state);
}

if (mount) {
    mount.innerHTML = ProjectManagerDashboard({
        ...mockProjectManagerData,
        projects: [],
        versions: [],
        outputs: [],
        notes: [],
        source: "Loading",
    });
    loadDashboardData().then(render);
}

window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) return;
    if (event.data?.type !== "radiance_project_action_result") return;
    if (event.data.success) {
        showToast("Current canvas saved to the project.");
        loadDashboardData().then(render);
    } else {
        showToast(event.data.error || "Project action failed.");
    }
});

window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        if (window.parent && window.parent !== window && window.parent.closeRadianceDashboard) {
            window.parent.closeRadianceDashboard();
        }
    }
});
