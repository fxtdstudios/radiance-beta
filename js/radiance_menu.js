// radiance_menu.js
// ─────────────────────────────────────────────────────────────────────────────
// Floating launcher so the Radiance dashboards are reachable without adding the
// Project Manager node. Renders a fixed ◎ button (bottom-right) that opens a
// small menu — Project Manager / Workflow Library / Assets — via the in-canvas
// overlay (window.showRadianceDashboard, defined by radiance_workspace.js).
// Self-contained: a plain DOM element on document.body, no ComfyUI widget API.
// To revert: delete this file.
// ─────────────────────────────────────────────────────────────────────────────
import { app } from "../../scripts/app.js";

const GOLD = "#c8a96e";

function openDash(file, title) {
    if (window.showRadianceDashboard) window.showRadianceDashboard(`/extensions/radiance/${file}`, title);
    else window.open(`/extensions/radiance/${file}`, "_blank");
}

function build() {
    if (document.getElementById("radiance-launcher-fab")) return;

    const root = document.createElement("div");
    root.id = "radiance-launcher-fab";
    Object.assign(root.style, { position: "fixed", right: "18px", bottom: "18px", zIndex: "9000", fontFamily: "'Inter',system-ui,sans-serif" });

    const menu = document.createElement("div");
    Object.assign(menu.style, {
        position: "absolute", right: "0", bottom: "52px", minWidth: "190px",
        background: "#111114", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px",
        padding: "6px", display: "none", boxShadow: "0 24px 60px -30px rgba(0,0,0,0.9)",
    });
    const item = (icon, label, file, title) => {
        const b = document.createElement("button");
        b.innerHTML = `<span style="color:${GOLD};width:16px;display:inline-block;">${icon}</span>${label}`;
        Object.assign(b.style, {
            display: "flex", alignItems: "center", gap: "9px", width: "100%", textAlign: "left",
            background: "none", border: "0", color: "#e8e8ec", fontFamily: "inherit", fontSize: "12.5px",
            padding: "8px 10px", borderRadius: "7px", cursor: "pointer",
        });
        b.onmouseenter = () => { b.style.background = "rgba(255,255,255,0.05)"; };
        b.onmouseleave = () => { b.style.background = "none"; };
        b.onclick = () => { menu.style.display = "none"; openDash(file, title); };
        return b;
    };
    menu.appendChild(item("◎", "Project Manager", "project_manager_dashboard.html", "Radiance Project Manager"));
    menu.appendChild(item("▤", "Workflow Library", "workspace_dashboard.html", "Radiance Workflow Library"));
    menu.appendChild(item("▦", "Assets", "assets_dashboard.html", "Radiance Assets"));

    const fab = document.createElement("button");
    fab.title = "Radiance dashboards";
    fab.textContent = "◎";
    Object.assign(fab.style, {
        width: "42px", height: "42px", borderRadius: "50%", border: `1.5px solid ${GOLD}`,
        background: "#16161a", color: GOLD, fontSize: "20px", cursor: "pointer", lineHeight: "1",
        boxShadow: "0 8px 24px -8px rgba(0,0,0,0.8)",
    });
    fab.onclick = (e) => {
        e.stopPropagation();
        menu.style.display = menu.style.display === "none" ? "block" : "none";
    };
    document.addEventListener("click", () => { menu.style.display = "none"; });

    root.appendChild(menu);
    root.appendChild(fab);
    document.body.appendChild(root);
}

app.registerExtension({
    name: "Radiance.Launcher.FAB",
    async setup() {
        build();
    },
});

console.log("[Radiance Launcher] floating menu ready");
