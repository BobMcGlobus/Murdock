// Murdock Web UI — no framework, just DOM.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const statusEl = $("#status");
function setStatus(msg, kind = "") {
    statusEl.textContent = msg;
    statusEl.className = "status " + kind;
}

// API base URL — normally empty (direct access). Home Assistant ingress
// sets this to the rewritten prefix so that absolute /api/... paths
// resolve through the ingress reverse-proxy instead of hitting HA itself.
// The value is injected by the server into index.html via
// <script>window.API_BASE="..."</script>.
const API_BASE = (typeof window !== "undefined" && window.API_BASE) || "";

// Resolve an absolute /api/... path against API_BASE. Leaves non-/api
// paths and already-prefixed URLs untouched.
function apiUrl(path) {
    if (!path) return path;
    if (!API_BASE) return path;
    // Don't double-prefix if caller already passed a full ingress URL.
    if (path.startsWith(API_BASE)) return path;
    if (path.startsWith("/api/") || path === "/api") {
        return API_BASE.replace(/\/+$/, "") + path;
    }
    return path;
}

async function api(path, options = {}) {
    const res = await fetch(apiUrl(path), options);
    if (!res.ok) {
        let detail = res.statusText;
        try {
            const body = await res.json();
            if (body.detail) detail = body.detail;
        } catch {}
        throw new Error(detail);
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) return res.json();
    return res;
}

// --- i18n setup --------------------------------------------------------------

// Language selector: wire the <select> in the header.
const langSelector = $("#lang-selector");
if (langSelector) {
    langSelector.value = I18N.preference();
    langSelector.addEventListener("change", () => {
        I18N.setLocale(langSelector.value);
        // Re-render dynamic content that's already on screen.
        const activeTab = $(".tab-btn.active");
        if (activeTab) {
            const tab = activeTab.dataset.tab;
            if (tab === "speakers") loadSpeakers();
            if (tab === "unknown") loadUnknown();
            if (tab === "settings") loadSettings();
            if (tab === "recognition") loadRecognition();
        }
    });
}
// Translate static elements on first load.
I18N.translatePage();

// --- Overview -------------------------------------------------------------

// Ordered by dependency, not importance: a transcription backend is
// useless without a speaker to recognise, and a speaker is useless if
// nothing reaches Home Assistant. Somebody following the list top to
// bottom never hits a step that cannot succeed yet.
const SETUP_ORDER = ["stt", "speakers", "samples", "delivery", "first_recognition"];

function renderSetup(status) {
    const card = $("#setup-card");
    const list = $("#setup-steps");
    if (!card || !list) return;
    // Once everything is done the checklist stops taking up the screen.
    card.hidden = !!status.setup_complete;
    if (status.setup_complete) return;

    const byKey = {};
    (status.setup || []).forEach((s) => { byKey[s.key] = s; });
    list.innerHTML = SETUP_ORDER.map((key) => {
        const step = byKey[key];
        if (!step) return "";
        const mark = step.done ? "✓" : "○";
        const cls = step.done ? "done" : "todo";
        const detail = step.detail
            ? `<span class="meta">${escapeHtml(t("overview.step_" + key + "_detail", { detail: step.detail }))}</span>`
            : "";
        return `<li class="${cls}"><span class="setup-mark">${mark}</span>` +
               `<span>${escapeHtml(t("overview.step_" + key))}</span> ${detail}</li>`;
    }).join("");
}

function renderStatusTiles(status) {
    const box = $("#status-tiles");
    if (!box) return;
    const tiles = [
        { label: t("overview.tile_speakers"),
          value: status.speakers,
          sub: t("overview.tile_samples", { n: status.samples }) },
        { label: t("overview.tile_recognised"),
          value: status.matches_24h,
          sub: t("overview.tile_of_events", { n: status.events_24h }) },
        { label: t("overview.tile_unknown"),
          value: status.unknown_24h,
          sub: t("overview.tile_last_24h") },
        { label: t("overview.tile_delivery"),
          value: status.delivery === "none"
              ? t("overview.delivery_none")
              : status.delivery.toUpperCase(),
          sub: t("overview.tile_stt", { backend: status.stt_backend }) },
    ];
    box.innerHTML = tiles.map((x) =>
        `<div class="status-tile"><div class="status-value">${escapeHtml(String(x.value))}</div>` +
        `<div class="status-label">${escapeHtml(x.label)}</div>` +
        `<div class="meta">${escapeHtml(x.sub)}</div></div>`
    ).join("");

    const note = $("#status-note");
    if (note) {
        note.textContent = status.last_event_at
            ? t("overview.last_event", { when: formatTimestamp(status.last_event_at) })
            : t("overview.no_events_yet");
    }
}

async function loadOverview() {
    try {
        const status = await api("/api/status");
        renderSetup(status);
        renderStatusTiles(status);
    } catch (err) {
        const note = $("#status-note");
        if (note) note.textContent = t("generic.error", { err: err.message });
    }
}

// --- Tabs -----------------------------------------------------------------

$$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        $$(".tab-btn").forEach((b) => b.classList.remove("active"));
        $$(".tab").forEach((t) => t.classList.remove("active"));
        btn.classList.add("active");
        $("#tab-" + btn.dataset.tab).classList.add("active");
        if (btn.dataset.tab === "overview") loadOverview();
        if (btn.dataset.tab === "speakers") { loadSpeakers(); loadCoach(); }
        if (btn.dataset.tab === "verify") loadRoles();
        if (btn.dataset.tab === "unknown") loadUnknown();
        if (btn.dataset.tab === "settings") loadSettings();
        if (btn.dataset.tab === "recognition") loadRecognition();
        // The experimental tab hosts settings cards, so it needs the
        // same population pass as the settings tab.
        if (btn.dataset.tab === "experimental") loadSettings();
    });
});

// --- Role / source vocab --------------------------------------------------

let ROLES = [];
let SOURCES = [];
let rolesPromise = null;

async function loadRoles() {
    if (rolesPromise) return rolesPromise;
    rolesPromise = (async () => {
        try {
            const data = await api("/api/speakers/roles");
            ROLES = data.roles || [];
            SOURCES = data.sources || [];
            for (const selId of ["#enroll-role", "#create-speaker-role"]) {
                const select = $(selId);
                if (!select) continue;
                select.innerHTML = `<option value="">${t("speakers.none")}</option>`;
                for (const r of ROLES) {
                    const opt = document.createElement("option");
                    opt.value = r;
                    opt.textContent = r;
                    select.appendChild(opt);
                }
            }
        } catch (err) {
            console.warn("Failed to load roles", err);
            rolesPromise = null;
        }
    })();
    return rolesPromise;
}

// --- Recorder -------------------------------------------------------------

let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;

async function encodeWavFromBlob(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
    });
    const decoded = await ctx.decodeAudioData(arrayBuffer);
    const data = decoded.getChannelData(0);
    const numSamples = data.length;

    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);
    const writeStr = (off, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + numSamples * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true);
    view.setUint32(28, 16000 * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, numSamples * 2, true);

    let offset = 44;
    for (let i = 0; i < numSamples; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, data[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    ctx.close();
    return new Blob([view], { type: "audio/wav" });
}

const recordBtn = $("#record-btn");
const recordStatus = $("#record-status");
const recordPlayback = $("#record-playback");

// Browser microphone capture needs a secure context (HTTPS or localhost).
// Home Assistant's ingress serves this UI over plain HTTP, so
// navigator.mediaDevices is undefined there. Detect it up front and guide
// the user to file upload or voice-satellite training instead of throwing
// a cryptic "Cannot read properties of undefined" at click time.
function micAvailable() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function initMicAvailability() {
    if (micAvailable()) return;
    if (recordBtn) {
        recordBtn.disabled = true;
        recordBtn.classList.add("disabled");
    }
    if (recordStatus) {
        recordStatus.textContent = t("speakers.mic_unavailable");
        recordStatus.className = "feedback";
    }
}

recordBtn.addEventListener("click", async () => {
    if (!micAvailable()) {
        recordStatus.textContent = t("speakers.mic_unavailable");
        return;
    }
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach((t) => t.stop());
            const raw = new Blob(audioChunks, {
                type: mediaRecorder.mimeType || "audio/webm",
            });
            try {
                recordedBlob = await encodeWavFromBlob(raw);
                recordPlayback.hidden = false;
                recordPlayback.src = URL.createObjectURL(recordedBlob);
                recordStatus.textContent =
                    t("speakers.recorded_kb", { kb: (recordedBlob.size / 1024).toFixed(0) });
                recordBtn.textContent = t("speakers.start_recording");
            } catch (err) {
                recordStatus.textContent = t("speakers.encode_error", { err: err.message });
            }
        };
        mediaRecorder.start();
        recordBtn.textContent = t("speakers.stop_recording");
        recordStatus.textContent = t("speakers.recording");
    } catch (err) {
        recordStatus.textContent = t("speakers.mic_error", { err: err.message });
    }
});

// --- Enrollment -----------------------------------------------------------

$("#enroll-existing").addEventListener("change", (ev) => {
    const form = $("#enroll-form");
    const id = ev.target.value;
    if (!id) {
        form.name.readOnly = false;
        form.ha_user_id.readOnly = false;
        form.role.disabled = false;
        return;
    }
    const speaker = CACHED_SPEAKERS.find((s) => String(s.id) === id);
    if (!speaker) return;
    form.name.value = speaker.name;
    form.name.readOnly = true;
    form.ha_user_id.value = speaker.ha_user_id || "";
    form.ha_user_id.readOnly = true;
    form.role.value = speaker.role || "";
    form.role.disabled = true;
});

$("#enroll-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const name = form.name.value.trim();
    const haUserId = form.ha_user_id.value.trim();
    const role = form.role ? form.role.value : "";
    const feedback = $("#enroll-feedback");
    feedback.className = "feedback";
    feedback.textContent = t("speakers.uploading");

    let audioBlob = recordedBlob;
    let source = "recording";
    let filename = `recording-${Date.now()}.wav`;
    const uploadFile = $("#upload-file").files[0];
    if (uploadFile) {
        audioBlob = uploadFile;
        source = "upload";
        filename = uploadFile.name || filename;
    }
    if (!audioBlob) {
        feedback.className = "feedback err";
        feedback.textContent = t("speakers.no_sample");
        return;
    }

    const body = new FormData();
    body.append("name", name);
    if (haUserId) body.append("ha_user_id", haUserId);
    if (role) body.append("role", role);
    body.append("source", source);
    body.append("filename", filename);
    // Whispered samples train a separate profile and are kept out of the
    // normal voiceprint.
    const styleSel = $("#enroll-style");
    body.append("style", styleSel ? styleSel.value : "normal");
    body.append("audio", audioBlob, filename);

    try {
        const res = await api("/api/speakers/enroll", { method: "POST", body });
        let msg = t("speakers.enrolled_ok", { name: res.speaker_name, n: res.total_samples });
        if (res.vad_speech_ratio != null) {
            msg +=
                ` (${t("speakers.speech_ratio", {
                    pct: (res.vad_speech_ratio * 100).toFixed(0),
                    sec: res.vad_speech_seconds.toFixed(1),
                })})`;
        }
        feedback.className =
            "feedback " + (res.warnings.length ? "warn" : "ok");
        feedback.textContent =
            msg + (res.warnings.length ? " — " + res.warnings.join("; ") : "");
        recordedBlob = null;
        recordPlayback.hidden = true;
        form.reset();
        form.name.readOnly = false;
        form.ha_user_id.readOnly = false;
        form.role.disabled = false;
        $("#enroll-existing").value = "";
        loadSpeakers();
    } catch (err) {
        feedback.className = "feedback err";
        feedback.textContent = t("generic.error", { err: err.message });
    }
});

$("#create-speaker-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const name = form.name.value.trim();
    const role = form.role ? form.role.value : "";
    const feedback = $("#create-speaker-feedback");
    feedback.className = "feedback";
    if (!name) {
        feedback.className = "feedback err";
        feedback.textContent = t("speakers.no_name");
        return;
    }
    const body = { name };
    if (role) body.role = role;
    try {
        const res = await api("/api/speakers", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        feedback.className = "feedback ok";
        feedback.textContent = t("speakers.created_ok", { name: res.name });
        form.reset();
        loadSpeakers();
    } catch (err) {
        feedback.className = "feedback err";
        feedback.textContent = t("generic.error", { err: err.message });
    }
});

// --- Voice map (embedding space) -------------------------------------------

const MAP_PALETTE = [
    "#dc2626", "#2563eb", "#16a34a", "#d97706", "#9333ea",
    "#0891b2", "#db2777", "#65a30d", "#7c3aed", "#ea580c",
];

function renderEmbeddingMap(data) {
    const out = $("#embedding-map-result");
    if (!data.points || data.points.length === 0) {
        out.innerHTML = `<p class="meta">${escapeHtml(t("map.not_enough"))}</p>`;
        return;
    }
    // Scale data coords into the SVG viewBox with padding.
    const W = 640, H = 420, PAD = 30;
    const xs = data.points.map((p) => p.x);
    const ys = data.points.map((p) => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const sx = (v) => PAD + ((v - xMin) / ((xMax - xMin) || 1)) * (W - 2 * PAD);
    const sy = (v) => PAD + ((v - yMin) / ((yMax - yMin) || 1)) * (H - 2 * PAD);

    // Stable speaker → color mapping (sorted by id).
    const speakerIds = [...new Set(
        data.points.filter((p) => p.speaker_id != null).map((p) => p.speaker_id)
    )].sort((a, b) => a - b);
    const colorOf = (sid) => MAP_PALETTE[speakerIds.indexOf(sid) % MAP_PALETTE.length];

    let shapes = "";
    // Draw unknowns first (background), then samples, then centroids on top.
    const byKind = { unknown: [], sample: [], centroid: [] };
    for (const p of data.points) (byKind[p.kind] || byKind.sample).push(p);

    for (const p of byKind.unknown) {
        const x = sx(p.x), y = sy(p.y);
        const tip = `unknown #${p.sample_id}` +
            (p.best_speaker ? ` · ${t("rec.nearest", { name: p.best_speaker })}` : "") +
            (p.distance != null ? ` · d=${p.distance}` : "");
        shapes += `<g class="map-unknown map-clickable" data-map-kind="unknown"
            data-map-id="${p.sample_id}" data-map-nearest="${escapeHtml(p.best_speaker || "")}">
            <title>${escapeHtml(tip)}</title>
            <circle cx="${x}" cy="${y}" r="8" fill="transparent"/>
            <line x1="${x - 4}" y1="${y - 4}" x2="${x + 4}" y2="${y + 4}"/>
            <line x1="${x - 4}" y1="${y + 4}" x2="${x + 4}" y2="${y - 4}"/></g>`;
    }
    for (const p of byKind.sample) {
        const x = sx(p.x), y = sy(p.y);
        const tip = `${p.speaker} #${p.sample_id} (${p.source || "?"})` +
            (p.quality != null ? ` · q=${p.quality.toFixed(2)}` : "") +
            (p.distance != null ? ` · d=${p.distance}` : "");
        shapes += `<circle class="map-clickable" cx="${x}" cy="${y}" r="4"
            fill="${colorOf(p.speaker_id)}" fill-opacity="0.75"
            data-map-kind="sample" data-map-id="${p.sample_id}"
            data-map-speaker="${escapeHtml(p.speaker)}">
            <title>${escapeHtml(tip)}</title></circle>`;
    }
    for (const p of byKind.centroid) {
        const x = sx(p.x), y = sy(p.y);
        shapes += `<circle cx="${x}" cy="${y}" r="9" fill="none"
            stroke="${colorOf(p.speaker_id)}" stroke-width="2.5">
            <title>${escapeHtml(t("map.centroid", { name: p.speaker }))}</title></circle>`;
    }

    const legend = speakerIds.map((sid) => {
        const name = (byKind.centroid.find((c) => c.speaker_id === sid) ||
                      byKind.sample.find((s) => s.speaker_id === sid) || {}).speaker || sid;
        return `<span class="map-legend-item">
            <span class="map-dot" style="background:${colorOf(sid)}"></span>${escapeHtml(name)}</span>`;
    }).join(" ");
    const unknownLegend = byKind.unknown.length
        ? `<span class="map-legend-item"><span class="map-cross">✕</span>${escapeHtml(t("map.unknown_legend", { n: byKind.unknown.length }))}</span>`
        : "";

    out.innerHTML = `
        <svg class="embedding-map" viewBox="0 0 ${W} ${H}" role="img">${shapes}</svg>
        <div class="map-legend">${legend} ${unknownLegend}</div>
        <div id="map-action"></div>
        <p class="meta">${escapeHtml(t("map.stats", {
            n: data.count,
            pc1: Math.round((data.explained?.[0] || 0) * 100),
            pc2: Math.round((data.explained?.[1] || 0) * 100),
            ms: Math.round(data.computed_ms || 0),
        }))}</p>`;

    // Click a point → action panel (play / delete / assign).
    out.querySelector("svg").addEventListener("click", (ev) => {
        const el = ev.target.closest("[data-map-kind]");
        if (!el) return;
        showMapAction(el.dataset);
    });
}

function showMapAction(d) {
    const panel = $("#map-action");
    if (!panel) return;
    const id = d.mapId;
    if (d.mapKind === "sample") {
        panel.innerHTML = `
            <div class="row">
                <strong>${escapeHtml(d.mapSpeaker)}</strong> <code>#${escapeHtml(id)}</code>
                <audio controls src="${apiUrl(`/api/speakers/samples/${id}/audio`)}"></audio>
                <button type="button" class="danger small" data-map-del-sample="${escapeHtml(id)}">${escapeHtml(t("map.delete_sample"))}</button>
            </div>`;
        panel.querySelector("[data-map-del-sample]").addEventListener("click", async (ev) => {
            if (!confirm(t("map.confirm_delete_sample"))) return;
            ev.target.disabled = true;
            try {
                await api(`/api/speakers/samples/${id}`, { method: "DELETE" });
                setStatus(t("map.sample_deleted"), "ok");
                loadSpeakers();
                loadEmbeddingMap();
            } catch (err) {
                setStatus(err.message, "err");
                ev.target.disabled = false;
            }
        });
    } else if (d.mapKind === "unknown") {
        panel.innerHTML = `
            <div class="row">
                <strong>${escapeHtml(t("rec.unknown"))}</strong> <code>#${escapeHtml(id)}</code>
                <audio controls src="${apiUrl(`/api/unknown/${id}/audio`)}"></audio>
                <button type="button" class="secondary small" data-map-assign="${escapeHtml(id)}">${escapeHtml(t("rec.add_to_speaker"))}</button>
                <button type="button" class="danger small" data-map-del-unknown="${escapeHtml(id)}">${escapeHtml(t("map.delete_sample"))}</button>
            </div>`;
        panel.querySelector("[data-map-assign]").addEventListener("click", async (ev) => {
            const name = prompt(t("rec.assign_prompt"), d.mapNearest || "");
            if (!name || !name.trim()) return;
            ev.target.disabled = true;
            try {
                const res = await api(`/api/unknown/${id}/assign`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ speaker_name: name.trim(), create_if_missing: true }),
                });
                setStatus(t("rec.assigned_ok", { name: res.speaker_name, n: res.total_samples }), "ok");
                loadSpeakers();
                loadEmbeddingMap();
            } catch (err) {
                setStatus(err.message, "err");
                ev.target.disabled = false;
            }
        });
        panel.querySelector("[data-map-del-unknown]").addEventListener("click", async (ev) => {
            ev.target.disabled = true;
            try {
                await api(`/api/unknown/${id}`, { method: "DELETE" });
                setStatus(t("map.sample_deleted"), "ok");
                loadEmbeddingMap();
            } catch (err) {
                setStatus(err.message, "err");
                ev.target.disabled = false;
            }
        });
    }
}

async function loadEmbeddingMap() {
    const out = $("#embedding-map-result");
    const btn = $("#embedding-map-btn");
    if (!out || !btn) return;
    btn.disabled = true;
    btn.textContent = t("map.rendering");
    out.innerHTML = `<p class="meta">${escapeHtml(t("map.computing"))}</p>`;
    try {
        const inc = $("#embedding-map-unknown").checked;
        const data = await api(`/api/speakers/embedding-map?include_unknown=${inc}`);
        renderEmbeddingMap(data);
    } catch (err) {
        out.innerHTML = `<p class="feedback err">${escapeHtml(err.message)}</p>`;
    } finally {
        btn.disabled = false;
        btn.textContent = t("map.render");
    }
}

const embeddingMapBtn = $("#embedding-map-btn");
if (embeddingMapBtn) {
    embeddingMapBtn.addEventListener("click", loadEmbeddingMap);
}

// --- Verify ---------------------------------------------------------------

$("#verify-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = $("#verify-file").files[0];
    const out = $("#verify-result");
    if (!file) {
        out.innerHTML = `<p class="feedback err">${escapeHtml(t("verify.pick_file"))}</p>`;
        return;
    }
    out.innerHTML = `<p class="meta">${escapeHtml(t("verify.scoring"))}</p>`;
    const body = new FormData();
    body.append("audio", file, file.name || "verify.wav");
    try {
        const res = await api("/api/speakers/verify", { method: "POST", body });
        const matchClass = res.is_match ? "ok" : "err";
        const matchText = res.is_match
            ? t("verify.match", { name: res.matched_speaker })
            : t("verify.no_match");
        let html = `
            <p class="feedback ${matchClass}">${escapeHtml(matchText)}</p>
            <p class="meta">
                ${escapeHtml(t("verify.best_distance", {
                    d: res.distance.toFixed(3),
                    t: res.threshold.toFixed(3),
                }))}
            </p>
        `;
        const entries = Object.entries(res.all_distances || {}).sort(
            (a, b) => a[1] - b[1]
        );
        if (entries.length) {
            html += `<table class="distances"><thead><tr><th>${escapeHtml(t("verify.col_speaker"))}</th><th>${escapeHtml(t("verify.col_distance"))}</th></tr></thead><tbody>`;
            for (const [name, dist] of entries) {
                const cls = dist <= res.threshold ? "ok" : "";
                html += `<tr class="${cls}"><td>${escapeHtml(name)}</td><td>${dist.toFixed(3)}</td></tr>`;
            }
            html += "</tbody></table>";
        } else {
            html += `<p class="meta">${escapeHtml(t("verify.no_speakers"))}</p>`;
        }
        out.innerHTML = html;
    } catch (err) {
        out.innerHTML = `<p class="feedback err">${escapeHtml(t("generic.error", { err: err.message }))}</p>`;
    }
});

// --- Speaker list ---------------------------------------------------------

let CACHED_SPEAKERS = [];

function populateExistingSpeakerDropdown(speakers) {
    const select = $("#enroll-existing");
    if (!select) return;
    const prev = select.value;
    select.innerHTML = `<option value="">${escapeHtml(t("speakers.new_speaker"))}</option>`;
    for (const s of speakers) {
        const opt = document.createElement("option");
        opt.value = String(s.id);
        opt.textContent = `${s.name} (${s.enrollment_count} ${t("speakers.samples")})`;
        select.appendChild(opt);
    }
    if (prev && speakers.some((s) => String(s.id) === prev)) {
        select.value = prev;
    }
}

// --- Quality helpers -----------------------------------------------------

function qualityClass(score) {
    if (score == null || isNaN(score)) return "unknown";
    if (score >= 0.75) return "good";
    if (score >= 0.50) return "medium";
    return "poor";
}

function qualityLabel(score) {
    if (score == null || isNaN(score)) return t("quality.unknown");
    return `${Math.round(score * 100)}%`;
}

async function fetchSpeakerQuality(speakerId) {
    const badge = document.querySelector(`[data-quality-for="${speakerId}"]`);
    if (!badge) return;
    try {
        const q = await api("/api/speakers/" + speakerId + "/quality");
        const label = qualityLabel(q.training_quality);
        badge.textContent = t("quality.training", { pct: label });
        badge.classList.remove("good", "medium", "poor", "unknown");
        badge.classList.add(qualityClass(q.training_quality));
        const avg = q.avg_sample_score != null
            ? `${Math.round(q.avg_sample_score * 100)}%`
            : "—";
        badge.title = t("quality.training_tooltip", {
            scored: q.scored_count,
            total: q.sample_count,
            avg: avg,
        });
    } catch {
        badge.textContent = "?";
    }
}

async function loadSpeakers() {
    await loadRoles();
    const list = $("#speaker-list");
    list.innerHTML = t("generic.loading");
    try {
        const speakers = await api("/api/speakers");
        CACHED_SPEAKERS = speakers;
        populateExistingSpeakerDropdown(speakers);
        if (speakers.length === 0) {
            list.innerHTML = `<p class="meta">${escapeHtml(t("speakers.no_speakers"))}</p>`;
            return;
        }
        list.innerHTML = "";
        for (const s of speakers) {
            const item = document.createElement("div");
            item.className = "list-item";
            item.dataset.speakerId = s.id;
            item.innerHTML = `
                <div class="row">
                    <h3>${escapeHtml(s.name)}</h3>
                    <span class="badge">${s.enrollment_count} ${escapeHtml(t("speakers.samples"))}</span>
                    <span class="badge quality" data-quality-for="${s.id}" title="${escapeHtml(t("quality.training_label"))}">…</span>
                    ${s.role ? `<span class="badge role">${escapeHtml(s.role)}</span>` : ""}
                    ${s.whisper_samples ? `<span class="badge whisper" title="${escapeHtml(
                        s.has_whisper_profile
                            ? t("speakers.whisper_profile_tooltip")
                            : t("speakers.whisper_partial_tooltip"))}">${escapeHtml(
                        s.has_whisper_profile
                            ? t("speakers.whisper_profile_badge")
                            : t("speakers.whisper_partial_badge", { n: s.whisper_samples })
                    )}</span>` : ""}
                    ${s.ha_user_id ? `<span class="meta">HA: ${escapeHtml(s.ha_user_id)}</span>` : ""}
                </div>
                <div class="row">
                    <button class="secondary" data-view="${s.id}">${escapeHtml(t("speakers.view_samples"))}</button>
                    <button class="secondary" data-edit="${s.id}">${escapeHtml(t("speakers.edit"))}</button>
                    <button class="secondary" data-rescore="${s.id}">${escapeHtml(t("quality.rescore_one"))}</button>
                    <button class="secondary" data-health="${s.id}">${escapeHtml(t("health_panel.btn"))}</button>
                    <button class="danger" data-del="${s.id}">${escapeHtml(t("speakers.delete_speaker"))}</button>
                </div>
                <div class="edit-panel" hidden></div>
                <div class="samples" hidden></div>
                <div class="health-panel" hidden></div>
            `;
            list.appendChild(item);
            // Fire-and-forget per-speaker quality fetch
            fetchSpeakerQuality(s.id);
        }
        list.querySelectorAll("button[data-del]").forEach((btn) =>
            btn.addEventListener("click", async () => {
                if (!confirm(t("speakers.confirm_delete"))) return;
                await api("/api/speakers/" + btn.dataset.del, { method: "DELETE" });
                loadSpeakers();
            })
        );
        list.querySelectorAll("button[data-view]").forEach((btn) =>
            btn.addEventListener("click", () => toggleSamples(btn))
        );
        list.querySelectorAll("button[data-rescore]").forEach((btn) =>
            btn.addEventListener("click", async () => {
                btn.disabled = true;
                btn.textContent = t("quality.rescoring");
                try {
                    const res = await api(
                        "/api/speakers/" + btn.dataset.rescore + "/rescore",
                        { method: "POST" }
                    );
                    setStatus(t("quality.rescored", { n: res.rescored }), "ok");
                    fetchSpeakerQuality(btn.dataset.rescore);
                    // Refresh samples panel if open
                    const item = btn.closest(".list-item");
                    const panel = item.querySelector(".samples");
                    if (panel && !panel.hidden) {
                        const viewBtn = item.querySelector("button[data-view]");
                        toggleSamples(viewBtn);
                        toggleSamples(viewBtn);
                    }
                } catch (err) {
                    setStatus(err.message, "err");
                } finally {
                    btn.disabled = false;
                    btn.textContent = t("quality.rescore_one");
                }
            })
        );
        list.querySelectorAll("button[data-edit]").forEach((btn) =>
            btn.addEventListener("click", () =>
                toggleEdit(btn, speakers.find((x) => x.id == btn.dataset.edit))
            )
        );
        list.querySelectorAll("button[data-health]").forEach((btn) =>
            btn.addEventListener("click", () => toggleHealth(btn))
        );
    } catch (err) {
        list.innerHTML = `<p class="feedback err">${escapeHtml(err.message)}</p>`;
    }
}

async function toggleHealth(btn) {
    const item = btn.closest(".list-item");
    const panel = item.querySelector(".health-panel");
    if (!panel.hidden) {
        panel.hidden = true;
        return;
    }
    panel.hidden = false;
    panel.innerHTML = `<p class="meta">${escapeHtml(t("health_panel.loading"))}</p>`;
    try {
        const h = await api(`/api/speakers/${btn.dataset.health}/health`);
        if (!h.embedded_count) {
            panel.innerHTML = `<p class="meta">${escapeHtml(t("health_panel.no_samples"))}</p>`;
            return;
        }
        let trend = "";
        if (h.quality_trend != null) {
            const up = h.quality_trend >= 0;
            trend = ` · <span class="${up ? "ok" : "warn"}">${escapeHtml(
                t(up ? "health_panel.trend_up" : "health_panel.trend_down",
                  { d: Math.abs(h.quality_trend).toFixed(2) })
            )}</span>`;
        }
        // Flag samples sitting far outside the profile's typical spread.
        const flagAt = Math.max(h.spread_avg * 1.75, 0.12);
        const rows = [...h.samples]
            .sort((a, b) => b.centroid_distance - a.centroid_distance)
            .map((s) => {
                const flagged = s.centroid_distance >= flagAt && h.samples.length > 2;
                return `<div class="row health-row${flagged ? " flagged" : ""}">
                    <code>#${s.id}</code>
                    <span class="badge">${escapeHtml(s.source || "?")}</span>
                    <span class="meta">${escapeHtml(t("health_panel.age", { d: s.age_days.toFixed(1) }))}</span>
                    ${s.quality_score != null ? `<span class="meta">q=${s.quality_score.toFixed(2)}</span>` : ""}
                    <span class="meta">d=${s.centroid_distance.toFixed(3)}</span>
                    ${flagged ? `<span class="badge warn">${escapeHtml(t("health_panel.drifted"))}</span>` : ""}
                </div>`;
            }).join("");
        panel.innerHTML = `
            <p class="meta">${escapeHtml(t("health_panel.summary", {
                n: h.embedded_count,
                avg: h.spread_avg.toFixed(3),
                max: h.spread_max.toFixed(3),
            }))}${trend}</p>
            ${rows}
            <p class="meta">${escapeHtml(t("health_panel.hint"))}</p>`;
    } catch (err) {
        panel.innerHTML = `<p class="feedback err">${escapeHtml(err.message)}</p>`;
    }
}

function toggleEdit(btn, speaker) {
    const item = btn.closest(".list-item");
    const panel = item.querySelector(".edit-panel");
    if (!panel.hidden) {
        panel.hidden = true;
        return;
    }
    panel.hidden = false;
    const roleOptions = [`<option value="">${escapeHtml(t("speakers.none"))}</option>`]
        .concat(
            ROLES.map(
                (r) =>
                    `<option value="${escapeHtml(r)}"${
                        r === speaker.role ? " selected" : ""
                    }>${escapeHtml(r)}</option>`
            )
        )
        .join("");
    panel.innerHTML = `
        <form class="edit-form">
            <label>
                ${escapeHtml(t("speakers.name"))}
                <input type="text" name="name" value="${escapeHtml(speaker.name)}" required>
            </label>
            <label>
                HA user ID
                <input type="text" name="ha_user_id" value="${escapeHtml(speaker.ha_user_id || "")}" placeholder="${escapeHtml(t("speakers.ha_clear_hint"))}">
            </label>
            <label>
                ${escapeHtml(t("speakers.role").split(" (")[0])}
                <select name="role">${roleOptions}</select>
            </label>
            <div class="row">
                <button type="submit">${escapeHtml(t("speakers.save"))}</button>
                <button type="button" class="secondary" data-cancel-edit>${escapeHtml(t("speakers.cancel"))}</button>
            </div>
            <div class="edit-feedback"></div>
        </form>
    `;
    const form = panel.querySelector("form");
    panel.querySelector("[data-cancel-edit]").addEventListener("click", () => {
        panel.hidden = true;
    });
    form.addEventListener("submit", async (ev) => {
        ev.preventDefault();
        const feedback = panel.querySelector(".edit-feedback");
        feedback.className = "edit-feedback feedback";
        feedback.textContent = t("speakers.saving");
        const newName = form.name.value.trim();
        const newHa = form.ha_user_id.value.trim();
        const newRole = form.role.value;
        const body = { name: newName };
        if (newHa === "") {
            body.clear_ha_user_id = true;
        } else if (newHa !== (speaker.ha_user_id || "")) {
            body.ha_user_id = newHa;
        }
        if (newRole === "") {
            body.clear_role = true;
        } else if (newRole !== (speaker.role || "")) {
            body.role = newRole;
        }
        try {
            await api("/api/speakers/" + speaker.id, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            loadSpeakers();
        } catch (err) {
            feedback.className = "edit-feedback feedback err";
            feedback.textContent = t("generic.error", { err: err.message });
        }
    });
}

async function toggleSamples(btn) {
    const item = btn.closest(".list-item");
    const panel = item.querySelector(".samples");
    if (!panel.hidden) {
        panel.hidden = true;
        return;
    }
    panel.hidden = false;
    panel.innerHTML = t("speakers.loading_samples");
    try {
        const samples = await api("/api/speakers/" + btn.dataset.view + "/samples");
        if (samples.length === 0) {
            panel.innerHTML = `<p class="meta">${escapeHtml(t("speakers.no_samples"))}</p>`;
            return;
        }
        panel.innerHTML = "";
        for (const s of samples) {
            const row = document.createElement("div");
            row.className = "sample-row";
            const sourceBadge = s.source
                ? `<span class="badge source-${escapeHtml(s.source)}">${escapeHtml(s.source)}</span>`
                : "";
            const satelliteBadge = s.satellite_id
                ? `<span class="badge satellite" title="${escapeHtml(t("speakers.satellite_tooltip"))}">📡 ${escapeHtml(s.satellite_id)}</span>`
                : "";
            const filename = s.filename
                ? `<span class="filename">${escapeHtml(s.filename)}</span>`
                : `<span class="filename meta">${escapeHtml(t("speakers.no_filename"))}</span>`;
            const qClass = qualityClass(s.quality_score);
            const qLabel = qualityLabel(s.quality_score);
            const qPct = s.quality_score != null ? Math.round(s.quality_score * 100) : 0;
            const qualityHtml = `
                <div class="quality-bar" title="${escapeHtml(t("quality.sample_tooltip"))}">
                    <span class="badge quality ${qClass}">${escapeHtml(t("quality.sample", { pct: qLabel }))}</span>
                    <div class="quality-bar-track"><div class="quality-bar-fill ${qClass}" style="width:${qPct}%"></div></div>
                </div>
            `;
            // Which voiceprint this sample feeds. Whispered ones train a
            // separate profile, so the list has to say which is which.
            const styleBadge = (s.style === "whisper")
                ? `<span class="badge whisper">${escapeHtml(t("speakers.style_whisper_badge"))}</span>`
                : "";
            row.innerHTML = `
                <div class="row">
                    ${filename}
                    ${sourceBadge}
                    ${styleBadge}
                    ${satelliteBadge}
                    <span class="meta">${s.duration_sec.toFixed(1)}s · ${new Date(
                        s.created_at * 1000
                    ).toLocaleString()}</span>
                </div>
                ${qualityHtml}
                <div class="row">
                    <audio controls src="${apiUrl(`/api/speakers/samples/${s.id}/audio`)}"></audio>
                    <button class="danger" data-del-sample="${s.id}">${escapeHtml(t("speakers.delete_sample"))}</button>
                </div>
            `;
            panel.appendChild(row);
        }
        panel.querySelectorAll("button[data-del-sample]").forEach((b) =>
            b.addEventListener("click", async () => {
                await api("/api/speakers/samples/" + b.dataset.delSample, {
                    method: "DELETE",
                });
                toggleSamples(btn);
                toggleSamples(btn);
            })
        );
    } catch (err) {
        panel.innerHTML = `<p class="feedback err">${escapeHtml(err.message)}</p>`;
    }
}

// --- Unknown list ---------------------------------------------------------

async function loadUnknown() {
    const list = $("#unknown-list");
    const includeTagged = $("#include-tagged").checked;
    list.innerHTML = t("generic.loading");
    try {
        // Ensure speaker cache is populated for the datalist autocomplete.
        if (CACHED_SPEAKERS.length === 0) {
            try {
                CACHED_SPEAKERS = await api("/api/speakers");
            } catch (_) {
                /* non-fatal */
            }
        }
        const samples = await api(
            "/api/unknown?include_tagged=" + (includeTagged ? "true" : "false")
        );
        if (samples.length === 0) {
            list.innerHTML = `<p class="meta">${escapeHtml(t("unknown.no_samples"))}</p>`;
            return;
        }
        list.innerHTML = "";
        // Shared datalist for all assign inputs on this page render.
        const dl = document.createElement("datalist");
        dl.id = "unknown-speaker-names";
        dl.innerHTML = CACHED_SPEAKERS.map(
            (sp) => `<option value="${escapeHtml(sp.name)}"></option>`
        ).join("");
        list.appendChild(dl);
        for (const s of samples) {
            const livenessBadge =
                s.liveness_score == null
                    ? ""
                    : s.liveness_score < 0.35
                    ? `<span class="badge tv">${escapeHtml(t("unknown.likely_tv"))}</span>`
                    : `<span class="badge live">${escapeHtml(t("unknown.likely_live"))}</span>`;
            const item = document.createElement("div");
            item.className = "list-item";
            item.innerHTML = `
                <div class="row">
                    <h3>#${s.id}</h3>
                    ${livenessBadge}
                    ${s.tag ? `<span class="badge">tag: ${escapeHtml(s.tag)}</span>` : ""}
                    <span class="meta">${new Date(s.created_at * 1000).toLocaleString()}</span>
                </div>
                <div class="row">
                    <span class="meta">${s.duration_sec.toFixed(1)}s · ${t("unknown.distance")} ${s.best_distance.toFixed(
                3
            )} · ${t("unknown.best")} ${escapeHtml(s.best_speaker || "–")}${
                s.liveness_score != null
                    ? ` · ${t("unknown.liveness")} ${s.liveness_score.toFixed(2)}`
                    : ""
            }</span>
                </div>
                <audio controls src="${apiUrl(`/api/unknown/${s.id}/audio`)}"></audio>
                <div class="row">
                    <input type="text" list="unknown-speaker-names" placeholder="${escapeHtml(t("unknown.assign_ph"))}" data-assign-name="${s.id}">
                    <button data-assign="${s.id}">${escapeHtml(t("unknown.assign_btn"))}</button>
                    <button class="secondary" data-tag-tv="${s.id}">${escapeHtml(t("unknown.tag_tv"))}</button>
                    <button class="danger" data-del-unknown="${s.id}">${escapeHtml(t("unknown.delete"))}</button>
                </div>
            `;
            list.appendChild(item);
        }
        list.querySelectorAll("button[data-del-unknown]").forEach((b) =>
            b.addEventListener("click", async () => {
                await api("/api/unknown/" + b.dataset.delUnknown, { method: "DELETE" });
                loadUnknown();
            })
        );
        list.querySelectorAll("button[data-tag-tv]").forEach((b) =>
            b.addEventListener("click", async () => {
                await api("/api/unknown/" + b.dataset.tagTv + "/tag", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ tag: "tv" }),
                });
                loadUnknown();
            })
        );
        list.querySelectorAll("button[data-assign]").forEach((b) =>
            b.addEventListener("click", async () => {
                const id = b.dataset.assign;
                const input = list.querySelector(`input[data-assign-name="${id}"]`);
                const name = input.value.trim();
                if (!name) {
                    alert(t("unknown.enter_name"));
                    return;
                }
                await api("/api/unknown/" + id + "/assign", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ speaker_name: name, create_if_missing: true }),
                });
                loadUnknown();
            })
        );
    } catch (err) {
        list.innerHTML = `<p class="feedback err">${escapeHtml(err.message)}</p>`;
    }
}

$("#refresh-unknown").addEventListener("click", loadUnknown);
$("#include-tagged").addEventListener("change", loadUnknown);
$("#cleanup-unknown").addEventListener("click", async () => {
    const res = await api("/api/unknown/cleanup", { method: "POST" });
    setStatus(t("unknown.cleaned", { n: res.deleted }), "ok");
    loadUnknown();
});

// --- Voice clusters (bulk-assign) -----------------------------------------

function setClusterFeedback(msg, cls) {
    const el = $("#cluster-feedback");
    if (!el) return;
    el.className = "feedback " + (cls || "");
    el.textContent = msg;
    if (!msg) return;
    setTimeout(() => {
        if (el.textContent === msg) {
            el.textContent = "";
            el.className = "feedback";
        }
    }, 6000);
}

async function loadClusters() {
    const list = $("#cluster-list");
    if (!list) return;
    const thresholdInput = $("#cluster-threshold");
    const threshold = parseFloat(thresholdInput?.value || "0.25");
    if (Number.isNaN(threshold) || threshold < 0 || threshold > 1) {
        setClusterFeedback(t("cluster.invalid_threshold"), "err");
        return;
    }
    // Ensure speakers are cached for the assign datalist.
    if (CACHED_SPEAKERS.length === 0) {
        try {
            CACHED_SPEAKERS = await api("/api/speakers");
        } catch (_) {
            /* non-fatal */
        }
    }
    list.innerHTML = t("generic.loading");
    try {
        const data = await api(
            "/api/unknown/clusters?threshold=" + encodeURIComponent(threshold)
        );
        if (!data.clusters || data.clusters.length === 0) {
            list.innerHTML = `<p class="meta">${escapeHtml(t("cluster.none"))}</p>`;
            return;
        }
        list.innerHTML = "";
        // Shared datalist for all assign inputs.
        const dl = document.createElement("datalist");
        dl.id = "cluster-speaker-names";
        dl.innerHTML = CACHED_SPEAKERS.map(
            (sp) => `<option value="${escapeHtml(sp.name)}"></option>`
        ).join("");
        list.appendChild(dl);

        for (const c of data.clusters) {
            const item = document.createElement("div");
            item.className = "list-item";
            const sats = c.satellites.length
                ? c.satellites
                      .map(
                          (s) =>
                              `<span class="badge satellite">${escapeHtml(s)}</span>`
                      )
                      .join(" ")
                : "";
            const memberRows = c.members
                .map((m) => {
                    const when = new Date(m.created_at * 1000).toLocaleString();
                    const d = m.distance_to_centroid.toFixed(3);
                    const dur = m.duration_sec.toFixed(1);
                    const tagBadge = m.tag
                        ? `<span class="badge">${escapeHtml(m.tag)}</span>`
                        : "";
                    const satBadge = m.satellite_id
                        ? `<span class="badge satellite">${escapeHtml(m.satellite_id)}</span>`
                        : "";
                    return `
                        <div class="sample-row">
                            <div class="row">
                                <span class="filename">#${m.sample_id}</span>
                                <span class="meta">${dur}s · ${escapeHtml(t("cluster.d"))}=${d}</span>
                                ${satBadge}
                                ${tagBadge}
                                <span class="meta">${escapeHtml(when)}</span>
                            </div>
                            <audio controls src="${apiUrl(`/api/unknown/${m.sample_id}/audio`)}"></audio>
                        </div>
                    `;
                })
                .join("");

            item.innerHTML = `
                <div class="row">
                    <h3>${escapeHtml(t("cluster.label", { n: c.cluster_id }))}</h3>
                    <span class="badge">${escapeHtml(t("cluster.size", { n: c.size }))}</span>
                    <span class="meta">${escapeHtml(t("cluster.avg_d"))} ${c.avg_distance.toFixed(3)}</span>
                    ${sats}
                </div>
                ${memberRows}
                <div class="row">
                    <input type="text" list="cluster-speaker-names"
                           data-cluster-name="${c.cluster_id}"
                           placeholder="${escapeHtml(t("cluster.assign_ph"))}">
                    <button data-cluster-assign="${c.cluster_id}">${escapeHtml(t("cluster.assign_btn", { n: c.size }))}</button>
                </div>
            `;
            // Attach member ids as data for the button.
            list.appendChild(item);
            const btn = item.querySelector(`button[data-cluster-assign]`);
            btn.dataset.sampleIds = JSON.stringify(
                c.members.map((m) => m.sample_id)
            );
        }
        list.querySelectorAll("button[data-cluster-assign]").forEach((b) =>
            b.addEventListener("click", async () => {
                const cid = b.dataset.clusterAssign;
                const input = list.querySelector(`input[data-cluster-name="${cid}"]`);
                const name = (input?.value || "").trim();
                if (!name) {
                    setClusterFeedback(t("unknown.enter_name"), "err");
                    return;
                }
                const sampleIds = JSON.parse(b.dataset.sampleIds || "[]");
                b.disabled = true;
                try {
                    const res = await api("/api/unknown/bulk-assign", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            speaker_name: name,
                            create_if_missing: true,
                            sample_ids: sampleIds,
                        }),
                    });
                    setClusterFeedback(
                        t("cluster.assigned", {
                            n: res.assigned,
                            name,
                            skipped: res.skipped,
                        }),
                        res.skipped ? "warn" : "ok"
                    );
                    loadClusters();
                    loadUnknown();
                } catch (err) {
                    setClusterFeedback(err.message, "err");
                } finally {
                    b.disabled = false;
                }
            })
        );
    } catch (err) {
        list.innerHTML = `<p class="feedback err">${escapeHtml(err.message)}</p>`;
    }
}

const clusterRefreshBtn = $("#cluster-refresh");
if (clusterRefreshBtn) {
    clusterRefreshBtn.addEventListener("click", loadClusters);
}

// --- Settings -------------------------------------------------------------

function renderUpstreamHint(s) {
    const hint = $("#upstream-hint");
    if (!hint) return;
    if (s.upstream_uri_source === "override") {
        hint.innerHTML = escapeHtml(t("hint.upstream_override", {
            uri: s.upstream_uri || "",
            default: s.upstream_uri_default || "",
        })).replace(
            escapeHtml(s.upstream_uri || ""),
            `<code>${escapeHtml(s.upstream_uri || "")}</code>`
        ).replace(
            escapeHtml(s.upstream_uri_default || ""),
            `<code>${escapeHtml(s.upstream_uri_default || "")}</code>`
        );
    } else {
        hint.innerHTML = escapeHtml(t("hint.upstream_default", {
            uri: s.upstream_uri || "",
        })).replace(
            escapeHtml(s.upstream_uri || ""),
            `<code>${escapeHtml(s.upstream_uri || "")}</code>`
        );
    }
}

function renderLangHint(s) {
    const hint = $("#lang-hint");
    if (!hint) return;
    const langs = (s.advertised_languages || []).join(", ");
    if (s.advertised_languages_source === "override") {
        hint.textContent = t("hint.lang_override", { langs: langs || "(none)" });
    } else if (langs) {
        hint.textContent = t("hint.lang_auto", { langs });
    } else {
        hint.textContent = t("hint.lang_none");
    }
}

async function loadSettings() {
    try {
        const s = await api("/api/settings");
        const form = $("#settings-form");
        form.verify_threshold.value = s.verify_threshold.toFixed(3);
        if (form.margin_gate) {
            form.margin_gate.value = (s.margin_gate ?? 0).toFixed(2);
        }
        form.skip_leading_seconds.value =
            (s.skip_leading_seconds ?? 1).toFixed(1);
        form.unknown_logging.checked = s.unknown_logging;
        form.require_speaker_match.checked = s.require_speaker_match;
        form.passthrough_when_no_speakers.checked =
            !!s.passthrough_when_no_speakers;
        // Min liveness score (new field — backwards-compatible)
        if (form.min_liveness_score) {
            form.min_liveness_score.value =
                (s.min_liveness_score ?? 0.35).toFixed(2);
        }
        // Auto-enroll (new field — backwards-compatible)
        if (form.auto_enroll) {
            form.auto_enroll.checked = !!s.auto_enroll;
        }
        // Speaker extraction (new fields — backwards-compatible)
        if (form.liveness_media_boost) {
            form.liveness_media_boost.value =
                (s.liveness_media_boost ?? 0.15).toFixed(2);
        }
        if (form.cancel_words) {
            form.cancel_words.value = s.cancel_words ?? "";
        }
        if (form.silence_abort_sec) {
            form.silence_abort_sec.value = (s.silence_abort_sec ?? 3).toFixed(1);
        }
        if (form.enable_extraction) {
            form.enable_extraction.checked = !!s.enable_extraction;
        }
        if (form.extraction_threshold) {
            form.extraction_threshold.value =
                (s.extraction_threshold ?? 0.25).toFixed(2);
        }
        if (form.extraction_min_region_sec) {
            form.extraction_min_region_sec.value =
                (s.extraction_min_region_sec ?? 0.6).toFixed(1);
        }
        // Per-satellite voice profiles (new field — backwards-compatible)
        if (form.enable_satellite_profiles) {
            form.enable_satellite_profiles.checked = !!s.enable_satellite_profiles;
        }
        // Early reject (new fields — backwards-compatible)
        if (form.enable_early_reject) {
            form.enable_early_reject.checked = !!s.enable_early_reject;
        }
        if (form.early_reject_margin) {
            form.early_reject_margin.value =
                (s.early_reject_margin ?? 0.25).toFixed(2);
        }
        // Confidence calibration (new section — backwards-compatible)
        const calForm = $("#calibration-form");
        if (calForm && calForm.enable_calibration) {
            calForm.enable_calibration.checked = !!s.enable_calibration;
            if (calForm.enable_adaptive_thresholds) {
                calForm.enable_adaptive_thresholds.checked =
                    !!s.enable_adaptive_thresholds;
            }
            renderCalibrationStatus(s);
            renderAdaptiveThresholds(s.adaptive_thresholds || {});
        }
        // Speaker context delivery (new section — backwards-compatible)
        const tplForm = $("#transcript-tpl-form");
        if (tplForm && tplForm.speaker_context_mode) {
            tplForm.speaker_context_mode.value = s.speaker_context_mode || "none";
            tplForm.transcript_template_known.value =
                s.transcript_template_known || "";
            tplForm.transcript_template_unknown.value =
                s.transcript_template_unknown || "";
            if (tplForm.transcript_hint_mode) {
                tplForm.transcript_hint_mode.value =
                    s.transcript_hint_mode || "inline";
                updateHintModeHelp(s);
            }
            updateSpeakerContextMode();
        }
        renderUpstreamHint(s);
        if (s.advertised_languages_source === "override") {
            form.advertised_languages.value =
                (s.advertised_languages || []).join(",");
        } else {
            form.advertised_languages.value = "";
        }
        renderLangHint(s);
        $("#settings-info").innerHTML = `
            <p class="meta">
                ${escapeHtml(t("settings.listen"))} <code>${escapeHtml(s.listen_uri)}</code><br>
                ${escapeHtml(t("settings.min_verify", { sec: s.min_verify_seconds.toFixed(1) }))}<br>
                ${escapeHtml(t("settings.ttl", { h: s.unknown_ttl_hours }))} · ${escapeHtml(s.ha_configured ? t("settings.ha_yes") : t("settings.ha_no"))}
            </p>
        `;
        // Populate HA settings form
        const haForm = $("#ha-settings-form");
        if (haForm) {
            haForm.ha_url.value = s.ha_url || "";
            haForm.ha_token.value = "";
            haForm.ha_input_text_entity.value = s.ha_input_text_entity || "input_text.current_speaker";
            haForm.ha_tv_entity.value = s.ha_tv_entity || "";
            haForm.ha_confidence_entity.value = s.ha_confidence_entity || "";
            haForm.ha_distance_entity.value = s.ha_distance_entity || "";
            haForm.ha_nearest_entity.value = s.ha_nearest_entity || "";
            haForm.ha_role_entity.value = s.ha_role_entity || "";
            updateHaTemplatePreview();
            const hint = $("#ha-token-hint");
            if (hint) {
                hint.textContent = s.ha_token_set ? t("ha.token_set") : t("ha.token_empty");
            }
        }
        // Populate MQTT settings form
        const mqttForm = $("#mqtt-settings-form");
        if (mqttForm) {
            mqttForm.mqtt_enabled.checked = !!s.mqtt_enabled;
            mqttForm.mqtt_host.value = s.mqtt_host || "";
            mqttForm.mqtt_port.value = s.mqtt_port || 1883;
            mqttForm.mqtt_username.value = s.mqtt_username || "";
            mqttForm.mqtt_password.value = "";
            mqttForm.mqtt_topic_prefix.value = s.mqtt_topic_prefix || "murdock";
            mqttForm.mqtt_discovery_prefix.value = s.mqtt_discovery_prefix || "homeassistant";
            const pwHint = $("#mqtt-password-hint");
            if (pwHint) {
                pwHint.textContent = s.mqtt_password_set
                    ? t("mqtt.password_set") : t("mqtt.password_empty");
            }
            renderMqttStatus(s);
            updateMqttContextSnippet(s);
            updateMqttSatelliteSnippet(s);
        }
        // Per-satellite thresholds + media restriction matrix
        loadSatelliteThresholds();
        loadMediaRestrictions();
        // STT backend
        const sttForm = $("#stt-form");
        if (sttForm) {
            sttForm.stt_backend.value = s.stt_backend || "upstream";
            sttForm.mistral_api_key.value = "";
            sttForm.mistral_model.value = s.mistral_model || "voxtral-mini-latest";
            const keyHint = $("#stt-key-hint");
            if (keyHint) {
                keyHint.textContent = s.mistral_api_key_set
                    ? t("stt.key_set")
                    : t("stt.key_empty");
            }
            // OpenAI-compatible backend (new fields — backwards-compatible)
            if (sttForm.openai_base_url) {
                sttForm.openai_base_url.value = s.openai_base_url || "";
                sttForm.openai_api_key.value = "";
                sttForm.openai_model.value = s.openai_model || "";
                const oaHint = $("#stt-openai-key-hint");
                if (oaHint) {
                    oaHint.textContent = s.openai_api_key_set
                        ? t("stt.key_set") : t("stt.key_empty");
                }
            }
            if (sttForm.enable_stt_prep) {
                sttForm.enable_stt_prep.checked = !!s.enable_stt_prep;
            }
            if (sttForm.stt_timeout_sec) {
                sttForm.stt_timeout_sec.value = s.stt_timeout_sec ?? 8;
            }
            if (sttForm.stt_language) {
                sttForm.stt_language.value = s.stt_language ?? "";
            }
            if (sttForm.upstream_uri) {
                sttForm.upstream_uri.value =
                    s.upstream_uri_source === "override"
                        ? (s.upstream_uri || "") : "";
            }
            if (sttForm.shadow_rescues_empty) {
                sttForm.shadow_rescues_empty.checked = !!s.shadow_rescues_empty;
            }
            if (sttForm.stt_local_fallback) {
                sttForm.stt_local_fallback.checked = !!s.stt_local_fallback;
            }
            // A/B shadow engine
            if (sttForm.shadow_stt_backend) {
                sttForm.shadow_stt_backend.value = s.shadow_stt_backend || "none";
                sttForm.shadow_upstream_uri.value = s.shadow_upstream_uri || "";
                sttForm.shadow_mistral_model.value = s.shadow_mistral_model || "";
                if (sttForm.shadow_mistral_api_key) {
                    sttForm.shadow_mistral_api_key.value = "";
                }
                sttForm.shadow_openai_base_url.value = s.shadow_openai_base_url || "";
                sttForm.shadow_openai_api_key.value = "";
                sttForm.shadow_openai_model.value = s.shadow_openai_model || "";
                const shHint = $("#shadow-openai-key-hint");
                if (shHint) {
                    shHint.textContent = s.shadow_openai_api_key_set
                        ? t("stt.key_set") : t("stt.key_empty");
                }
                const shMistralHint = $("#shadow-mistral-key-hint");
                if (shMistralHint) {
                    shMistralHint.textContent = s.shadow_mistral_api_key_set
                        ? t("stt.key_set") : t("stt.shadow_mistral_key_empty");
                }
            }
            // Transcript quality tiers
            if (sttForm.enable_stt_vocabulary) {
                sttForm.enable_stt_vocabulary.checked = !!s.enable_stt_vocabulary;
                sttForm.stt_vocabulary.value = s.stt_vocabulary || "";
                sttForm.enable_stt_dictionary.checked = !!s.enable_stt_dictionary;
                sttForm.stt_dictionary.value = s.stt_dictionary || "";
            }
            if (sttForm.enable_canonicalizer) {
                sttForm.enable_canonicalizer.checked = !!s.enable_canonicalizer;
                sttForm.canonicalizer_min_score.value =
                    (s.canonicalizer_min_score ?? 0.82).toFixed(2);
                sttForm.canonicalizer_min_margin.value =
                    (s.canonicalizer_min_margin ?? 0.1).toFixed(2);
            }
            loadVocabularyMirror();
            loadCanonicalizerHits();
            if (sttForm.enable_dual_transcript) {
                sttForm.enable_dual_transcript.checked = !!s.enable_dual_transcript;
            }
            updateSttFieldVisibility();
        }
        const whisperForm = $("#whisper-form");
        if (whisperForm && whisperForm.enable_whisper_detection) {
            whisperForm.enable_whisper_detection.checked =
                !!s.enable_whisper_detection;
            whisperForm.whisper_threshold.value =
                (s.whisper_threshold ?? 0.62).toFixed(2);
        }
        // Populate quality weights form
        const qForm = $("#quality-form");
        if (qForm && s.quality_weights) {
            for (const k of ["speech_ratio", "snr", "liveness", "consistency", "centroid_distance"]) {
                if (qForm[k]) {
                    qForm[k].value = (s.quality_weights[k] ?? 0).toFixed(3);
                }
            }
            const info = $("#quality-weights-info");
            if (info) {
                info.textContent = s.quality_weights_source === "override"
                    ? t("quality.source_override")
                    : t("quality.source_default");
            }
        }
    } catch (err) {
        $("#settings-info").innerHTML = `<p class="feedback err">${escapeHtml(err.message)}</p>`;
    }
}

// --- Per-satellite thresholds ---------------------------------------------

async function loadSatelliteThresholds() {
    const list = $("#sat-threshold-list");
    if (!list) return;
    list.innerHTML = t("generic.loading");
    try {
        const data = await api("/api/settings/satellite-thresholds");
        if (!data.entries || data.entries.length === 0) {
            list.innerHTML = `<p class="meta">${escapeHtml(t("sat_threshold.none"))}</p>`;
            return;
        }
        const defaultTh = data.default_threshold.toFixed(3);
        list.innerHTML = "";
        for (const e of data.entries) {
            const row = document.createElement("div");
            row.className = "list-item";
            const hasOverride = e.threshold != null;
            const thValue = hasOverride ? e.threshold.toFixed(3) : "";
            const lastSeen = e.last_seen
                ? new Date(e.last_seen * 1000).toLocaleString()
                : "–";
            row.innerHTML = `
                <div class="row">
                    <span class="badge satellite" title="${escapeHtml(e.satellite_id)}">${escapeHtml(e.name || e.satellite_id)}</span>
                    <span class="meta">${escapeHtml(t("sat_threshold.events", { n: e.seen_events }))} · ${escapeHtml(t("sat_threshold.last_seen"))} ${escapeHtml(lastSeen)}</span>
                </div>
                <div class="row">
                    <label style="flex-direction:row; align-items:center; gap:0.4rem">
                        <span>${escapeHtml(t("sat_threshold.override"))}</span>
                        <input type="number" step="0.01" min="0" max="2"
                               data-sat-th="${escapeHtml(e.satellite_id)}"
                               value="${thValue}"
                               placeholder="${defaultTh}">
                    </label>
                    <button class="secondary" data-sat-save="${escapeHtml(e.satellite_id)}">${escapeHtml(t("generic.save"))}</button>
                    <button class="danger" data-sat-clear="${escapeHtml(e.satellite_id)}" ${hasOverride ? "" : "disabled"}>${escapeHtml(t("sat_threshold.clear"))}</button>
                </div>
            `;
            list.appendChild(row);
        }
        list.querySelectorAll("button[data-sat-save]").forEach((b) =>
            b.addEventListener("click", async () => {
                const sid = b.dataset.satSave;
                const input = list.querySelector(`input[data-sat-th="${CSS.escape(sid)}"]`);
                const raw = input.value.trim();
                const body = {
                    satellite_id: sid,
                    threshold: raw === "" ? null : parseFloat(raw),
                };
                if (raw !== "" && Number.isNaN(body.threshold)) {
                    setSatFeedback(t("sat_threshold.invalid"), "err");
                    return;
                }
                try {
                    await api("/api/settings/satellite-thresholds", {
                        method: "PATCH",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(body),
                    });
                    setSatFeedback(t("sat_threshold.saved", { sid }), "ok");
                    loadSatelliteThresholds();
                } catch (err) {
                    setSatFeedback(err.message, "err");
                }
            })
        );
        list.querySelectorAll("button[data-sat-clear]").forEach((b) =>
            b.addEventListener("click", async () => {
                const sid = b.dataset.satClear;
                try {
                    await api("/api/settings/satellite-thresholds", {
                        method: "PATCH",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ satellite_id: sid, threshold: null }),
                    });
                    setSatFeedback(t("sat_threshold.cleared", { sid }), "ok");
                    loadSatelliteThresholds();
                } catch (err) {
                    setSatFeedback(err.message, "err");
                }
            })
        );
    } catch (err) {
        list.innerHTML = `<p class="feedback err">${escapeHtml(err.message)}</p>`;
    }
}

function setSatFeedback(msg, cls) {
    const el = $("#sat-threshold-feedback");
    if (!el) return;
    el.className = "feedback " + (cls || "");
    el.textContent = msg;
    setTimeout(() => {
        if (el.textContent === msg) {
            el.textContent = "";
            el.className = "feedback";
        }
    }, 4000);
}

const satRefreshBtn = $("#sat-threshold-refresh");
if (satRefreshBtn) {
    satRefreshBtn.addEventListener("click", loadSatelliteThresholds);
}

// --- Media restriction matrix --------------------------------------------

function setMediaRestrictFeedback(msg, cls) {
    const el = $("#media-restrict-feedback");
    if (!el) return;
    el.className = "feedback " + (cls || "");
    el.textContent = msg;
    setTimeout(() => {
        if (el.textContent === msg) {
            el.textContent = "";
            el.className = "feedback";
        }
    }, 4000);
}

function fillSelect(sel, values, placeholder) {
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = "";
    const ph = document.createElement("option");
    ph.value = "";
    ph.textContent = placeholder;
    sel.appendChild(ph);
    for (const v of values) {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = v;
        sel.appendChild(o);
    }
    if (values.includes(prev)) sel.value = prev;
}

async function loadMediaRestrictions() {
    const list = $("#media-restrict-list");
    if (!list) return;
    list.innerHTML = t("generic.loading");
    try {
        const data = await api("/api/settings/media-restrictions");
        const boost = (data.default_boost ?? 0.05).toFixed(3);
        // Populate the add-row selects.
        fillSelect($("#media-restrict-sat"), data.satellites || [], t("media_restrict.pick_sat"));
        fillSelect(
            $("#media-restrict-source"),
            (data.media || []).map((m) => m.entity_id),
            t("media_restrict.pick_source"),
        );
        // Currently-playing badge map.
        const playing = {};
        for (const m of data.media || []) playing[m.entity_id] = m.playing;

        if (!data.restrictions || data.restrictions.length === 0) {
            list.innerHTML =
                `<p class="meta">${escapeHtml(t("media_restrict.none", { boost }))}</p>`;
            return;
        }
        list.innerHTML = "";
        for (const r of data.restrictions) {
            const row = document.createElement("div");
            row.className = "list-item";
            const playBadge = playing[r.media_entity]
                ? `<span class="badge">${escapeHtml(t("media_restrict.playing"))}</span>`
                : "";
            row.innerHTML = `
                <div class="row">
                    <span class="badge satellite">${escapeHtml(r.satellite_id)}</span>
                    <span class="meta">←</span>
                    <code>${escapeHtml(r.media_entity)}</code>
                    ${playBadge}
                </div>
                <div class="row">
                    <label style="flex-direction:row; align-items:center; gap:0.4rem">
                        <span>${escapeHtml(t("media_restrict.delta"))}</span>
                        <input type="number" step="0.01" min="0" max="2"
                               data-mr-sat="${escapeHtml(r.satellite_id)}"
                               data-mr-src="${escapeHtml(r.media_entity)}"
                               value="${r.delta.toFixed(3)}">
                    </label>
                    <button class="secondary" data-mr-save>${escapeHtml(t("generic.save"))}</button>
                    <button class="danger" data-mr-clear>${escapeHtml(t("sat_threshold.clear"))}</button>
                </div>
            `;
            const save = row.querySelector("[data-mr-save]");
            const clear = row.querySelector("[data-mr-clear]");
            const input = row.querySelector("input");
            save.addEventListener("click", () =>
                patchMediaRestriction(r.satellite_id, r.media_entity, parseFloat(input.value)));
            clear.addEventListener("click", () =>
                patchMediaRestriction(r.satellite_id, r.media_entity, null));
            list.appendChild(row);
        }
    } catch (err) {
        list.innerHTML = `<p class="feedback err">${escapeHtml(err.message)}</p>`;
    }
}

async function patchMediaRestriction(sat, src, delta) {
    if (delta !== null && Number.isNaN(delta)) {
        setMediaRestrictFeedback(t("media_restrict.invalid"), "err");
        return;
    }
    try {
        await api("/api/settings/media-restrictions", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ satellite_id: sat, media_entity: src, delta }),
        });
        setMediaRestrictFeedback(t("media_restrict.saved"), "ok");
        loadMediaRestrictions();
    } catch (err) {
        setMediaRestrictFeedback(err.message, "err");
    }
}

const mediaRestrictForm = $("#media-restrict-form");
if (mediaRestrictForm) {
    mediaRestrictForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const sat = $("#media-restrict-sat").value;
        const src = $("#media-restrict-source").value;
        const raw = $("#media-restrict-delta").value.trim();
        if (!sat || !src) {
            setMediaRestrictFeedback(t("media_restrict.pick_both"), "err");
            return;
        }
        patchMediaRestriction(sat, src, raw === "" ? 0 : parseFloat(raw));
    });
}
const mediaRestrictRefresh = $("#media-restrict-refresh");
if (mediaRestrictRefresh) {
    mediaRestrictRefresh.addEventListener("click", loadMediaRestrictions);
}

// --- STT backend toggle ---------------------------------------------------

function updateSttFieldVisibility() {
    const form = $("#stt-form");
    if (!form) return;
    const backend = form.stt_backend.value;
    const vox = $("#stt-voxtral-fields");
    if (vox) vox.hidden = backend !== "voxtral";
    const oa = $("#stt-openai-fields");
    if (oa) oa.hidden = backend !== "openai";
    // Local fallback only makes sense for buffering cloud backends.
    const fb = $("#stt-fallback-row");
    if (fb) fb.hidden = backend === "upstream";
    // Upload conditioning only applies where audio is uploaded — the
    // upstream path streams while the user is still speaking.
    const prep = $("#stt-prep-row");
    if (prep) prep.hidden = backend === "upstream";
    const to = $("#stt-timeout-row");
    if (to) to.hidden = backend === "upstream";
    const la = $("#stt-language-row");
    if (la) la.hidden = backend === "upstream";
    // The Wyoming URI is the one field the upstream backend needs, and it
    // used to live in a collapsed advanced block on a different tab.
    const up = $("#stt-upstream-fields");
    if (up) up.hidden = backend !== "upstream";
    // Shadow sub-fields per selected shadow engine.
    const shadow = form.shadow_stt_backend ? form.shadow_stt_backend.value : "none";
    const su = $("#shadow-upstream-fields");
    if (su) su.hidden = shadow !== "upstream";
    const sv = $("#shadow-voxtral-fields");
    if (sv) sv.hidden = shadow !== "voxtral";
    const so = $("#shadow-openai-fields");
    if (so) so.hidden = shadow !== "openai";
    // Dual transcript needs a configured shadow engine.
    if (form.enable_dual_transcript) {
        const noShadow = shadow === "none";
        form.enable_dual_transcript.disabled = noShadow;
        if (noShadow) form.enable_dual_transcript.checked = false;
    }
}

const sttForm = $("#stt-form");
if (sttForm) {
    const sttSelect = sttForm.stt_backend;
    sttSelect.addEventListener("change", updateSttFieldVisibility);
    if (sttForm.shadow_stt_backend) {
        sttForm.shadow_stt_backend.addEventListener(
            "change", updateSttFieldVisibility
        );
    }

    sttForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const fb = $("#stt-feedback");
        const body = { stt_backend: sttSelect.value };
        if (sttSelect.value === "voxtral") {
            const keyVal = sttForm.mistral_api_key.value;
            if (keyVal) body.mistral_api_key = keyVal;
            const modelVal = sttForm.mistral_model.value.trim();
            if (modelVal) body.mistral_model = modelVal;
        }
        if (sttSelect.value === "openai" && sttForm.openai_base_url) {
            body.openai_base_url = sttForm.openai_base_url.value.trim();
            body.openai_model = sttForm.openai_model.value.trim();
            const oaKey = sttForm.openai_api_key.value;
            if (oaKey) body.openai_api_key = oaKey;
        }
        if (sttForm.enable_stt_prep) {
            body.enable_stt_prep = sttForm.enable_stt_prep.checked;
        }
        if (sttForm.stt_timeout_sec && sttForm.stt_timeout_sec.value !== "") {
            body.stt_timeout_sec = parseFloat(sttForm.stt_timeout_sec.value);
        }
        if (sttForm.stt_language) {
            body.stt_language = sttForm.stt_language.value.trim();
        }
        if (sttForm.upstream_uri) {
            body.upstream_uri = sttForm.upstream_uri.value.trim();
        }
        if (sttForm.shadow_rescues_empty) {
            body.shadow_rescues_empty = sttForm.shadow_rescues_empty.checked;
        }
        if (sttForm.stt_local_fallback) {
            body.stt_local_fallback = sttForm.stt_local_fallback.checked;
        }
        if (sttForm.shadow_stt_backend) {
            body.shadow_stt_backend = sttForm.shadow_stt_backend.value;
            body.shadow_upstream_uri = sttForm.shadow_upstream_uri.value.trim();
            body.shadow_mistral_model = sttForm.shadow_mistral_model.value.trim();
            body.shadow_openai_base_url = sttForm.shadow_openai_base_url.value.trim();
            body.shadow_openai_model = sttForm.shadow_openai_model.value.trim();
            const shKey = sttForm.shadow_openai_api_key.value;
            if (shKey) body.shadow_openai_api_key = shKey;
            if (sttForm.shadow_mistral_api_key) {
                const shMKey = sttForm.shadow_mistral_api_key.value;
                if (shMKey) body.shadow_mistral_api_key = shMKey;
            }
        }
        if (sttForm.enable_canonicalizer) {
            body.enable_canonicalizer = sttForm.enable_canonicalizer.checked;
            if (sttForm.canonicalizer_min_score.value !== "") {
                body.canonicalizer_min_score =
                    parseFloat(sttForm.canonicalizer_min_score.value);
            }
            if (sttForm.canonicalizer_min_margin.value !== "") {
                body.canonicalizer_min_margin =
                    parseFloat(sttForm.canonicalizer_min_margin.value);
            }
        }
        if (sttForm.enable_stt_vocabulary) {
            body.enable_stt_vocabulary = sttForm.enable_stt_vocabulary.checked;
            body.stt_vocabulary = sttForm.stt_vocabulary.value;
            body.enable_stt_dictionary = sttForm.enable_stt_dictionary.checked;
            body.stt_dictionary = sttForm.stt_dictionary.value;
        }
        if (sttForm.enable_dual_transcript) {
            body.enable_dual_transcript = sttForm.enable_dual_transcript.checked;
        }
        try {
            const saved = await api("/api/settings", {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            // The URI is normalised server-side (a bare host:port grows a
            // tcp:// scheme), so show what was actually stored.
            renderUpstreamHint(saved);
            if (sttForm.upstream_uri) {
                sttForm.upstream_uri.value =
                    saved.upstream_uri_source === "override"
                        ? (saved.upstream_uri || "") : "";
            }
            if (fb) {
                fb.className = "feedback ok";
                fb.textContent = t("stt.saved");
                setTimeout(() => {
                    if (fb.textContent === t("stt.saved")) {
                        fb.textContent = "";
                        fb.className = "feedback";
                    }
                }, 2500);
            }
            sttForm.mistral_api_key.value = "";
            if (sttForm.openai_api_key) sttForm.openai_api_key.value = "";
            if (sttForm.shadow_openai_api_key) sttForm.shadow_openai_api_key.value = "";
            if (sttForm.shadow_mistral_api_key) sttForm.shadow_mistral_api_key.value = "";
            loadSettings();
        } catch (err) {
            if (fb) {
                fb.className = "feedback err";
                fb.textContent = err.message;
            }
        }
    });
}


function parseLanguages(raw) {
    if (!raw) return [];
    return raw
        .split(/[\s,]+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0);
}

$("#settings-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const body = {
        verify_threshold: parseFloat(form.verify_threshold.value),
        skip_leading_seconds: parseFloat(form.skip_leading_seconds.value),
        unknown_logging: form.unknown_logging.checked,
        require_speaker_match: form.require_speaker_match.checked,
        passthrough_when_no_speakers: form.passthrough_when_no_speakers.checked,
        advertised_languages: parseLanguages(form.advertised_languages.value),
    };
    // New fields (backwards-compatible)
    if (form.margin_gate && form.margin_gate.value !== "") {
        body.margin_gate = parseFloat(form.margin_gate.value);
    }
    if (form.min_liveness_score) {
        body.min_liveness_score = parseFloat(form.min_liveness_score.value);
    }
    if (form.auto_enroll) {
        body.auto_enroll = form.auto_enroll.checked;
    }
    if (form.liveness_media_boost && form.liveness_media_boost.value !== "") {
        body.liveness_media_boost = parseFloat(form.liveness_media_boost.value);
    }
    if (form.cancel_words) {
        body.cancel_words = form.cancel_words.value.trim();
    }
    if (form.silence_abort_sec && form.silence_abort_sec.value !== "") {
        body.silence_abort_sec = parseFloat(form.silence_abort_sec.value);
    }
    if (form.enable_extraction) {
        body.enable_extraction = form.enable_extraction.checked;
    }
    if (form.enable_satellite_profiles) {
        body.enable_satellite_profiles = form.enable_satellite_profiles.checked;
    }
    if (form.enable_early_reject) {
        body.enable_early_reject = form.enable_early_reject.checked;
    }
    if (form.early_reject_margin && form.early_reject_margin.value !== "") {
        body.early_reject_margin = parseFloat(form.early_reject_margin.value);
    }
    if (form.extraction_threshold && form.extraction_threshold.value !== "") {
        body.extraction_threshold = parseFloat(form.extraction_threshold.value);
    }
    if (form.extraction_min_region_sec && form.extraction_min_region_sec.value !== "") {
        body.extraction_min_region_sec = parseFloat(form.extraction_min_region_sec.value);
    }
    try {
        const s = await api("/api/settings", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        renderUpstreamHint(s);
        renderLangHint(s);
        setStatus(t("settings.saved"), "ok");
    } catch (err) {
        setStatus(t("generic.error", { err: err.message }), "err");
    }
});

async function pingUpstream(btnSel, outSel, uriSel) {
    const btn = $(btnSel);
    const out = $(outSel);
    // Test what is in the field, falling back to the stored value. A
    // ping that silently uses the saved URI while the user looks at an
    // edited one reads as "my input is being ignored".
    const uriEl = uriSel ? $(uriSel) : null;
    const typed = uriEl ? (uriEl.value || "").trim() : "";
    btn.disabled = true;
    btn.textContent = t("ping.pinging");
    out.innerHTML = "";
    try {
        const res = await api("/api/settings/ping-upstream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(typed ? { uri: typed } : {}),
        });
        if (res.ok) {
            const langArr = res.languages || [];
            const langs = langArr.join(", ") || "(none)";
            out.innerHTML =
                `<span class="feedback ok">${escapeHtml(t("ping.ok"))}</span><br>` +
                `<code>${escapeHtml(res.upstream_uri)}</code> — ` +
                `${res.latency_ms.toFixed(0)}ms<br>` +
                `${escapeHtml(t("ping.upstream_supports", { n: langArr.length }))} ${escapeHtml(langs)}<br>` +
                `<small class="meta">${escapeHtml(t("ping.advertise_note"))}</small>` +
                (typed ? `<br><small class="meta">${escapeHtml(t("ping.unsaved_note"))}</small>` : "");
            setStatus(t("ping.upstream_ok"), "ok");
        } else {
            out.innerHTML =
                `<span class="feedback err">${escapeHtml(t("ping.fail"))}</span><br>` +
                `<code>${escapeHtml(res.upstream_uri)}</code><br>` +
                `${escapeHtml(t("generic.error", { err: res.error || "unknown" }))}`;
            setStatus(t("ping.upstream_unreachable"), "err");
        }
    } catch (err) {
        out.innerHTML =
            `<span class="feedback err">${escapeHtml(t("ping.request_failed"))}</span> ${escapeHtml(err.message)}`;
        setStatus(t("ping.failed", { err: err.message }), "err");
    } finally {
        btn.disabled = false;
        btn.textContent = t("settings.ping");
    }
}

// Both entry points: the one beside the backend selector, and the one
// that has always sat in the recognition group.
$("#ping-upstream-btn").addEventListener("click", () =>
    pingUpstream("#ping-upstream-btn", "#ping-result"));
const pingBtn2 = $("#ping-upstream-btn2");
if (pingBtn2) {
    pingBtn2.addEventListener("click", () =>
        pingUpstream("#ping-upstream-btn2", "#ping-result2",
                     '#stt-form [name="upstream_uri"]'));
}

// --- Threshold recommendation ----------------------------------------------

const thresholdSuggestBtn = $("#threshold-suggest-btn");
if (thresholdSuggestBtn) {
    thresholdSuggestBtn.addEventListener("click", async () => {
        const out = $("#threshold-suggest-result");
        thresholdSuggestBtn.disabled = true;
        out.textContent = t("thsuggest.computing");
        try {
            const r = await api("/api/settings/threshold-recommendation");
            if (r.status === "insufficient_data") {
                out.textContent = t("thsuggest.insufficient", {
                    g: r.genuine_count, i: r.impostor_count,
                });
                return;
            }
            const overlapNote = r.status === "overlap"
                ? ` ${t("thsuggest.overlap")}` : "";
            out.innerHTML = `${escapeHtml(t("thsuggest.result", {
                rec: r.recommended.toFixed(3),
                g95: r.genuine_p95.toFixed(3),
                i05: r.impostor_p05.toFixed(3),
                g: r.genuine_count,
                i: r.impostor_count,
            }))}${escapeHtml(overlapNote)}
                <button type="button" class="secondary small" id="threshold-apply-btn">${escapeHtml(t("thsuggest.apply"))}</button>`;
            $("#threshold-apply-btn").addEventListener("click", async () => {
                try {
                    await api("/api/settings", {
                        method: "PATCH",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ verify_threshold: r.recommended }),
                    });
                    $("#settings-form").verify_threshold.value = r.recommended.toFixed(3);
                    out.textContent = t("thsuggest.applied", { rec: r.recommended.toFixed(3) });
                    setStatus(t("thsuggest.applied", { rec: r.recommended.toFixed(3) }), "ok");
                } catch (err) {
                    setStatus(t("generic.error", { err: err.message }), "err");
                }
            });
        } catch (err) {
            out.textContent = err.message;
        } finally {
            thresholdSuggestBtn.disabled = false;
        }
    });
}

$("#refresh-langs-btn").addEventListener("click", async () => {
    const btn = $("#refresh-langs-btn");
    btn.disabled = true;
    btn.textContent = t("refresh.refreshing");
    try {
        const s = await api("/api/settings/refresh-languages", {
            method: "POST",
        });
        renderLangHint(s);
        setStatus(
            t("refresh.languages", {
                langs: (s.advertised_languages || []).join(", ") || "(none)",
            }),
            "ok"
        );
    } catch (err) {
        setStatus(t("refresh.failed", { err: err.message }), "err");
    } finally {
        btn.disabled = false;
        btn.textContent = t("settings.refresh_langs");
    }
});

$("#restart-btn").addEventListener("click", async () => {
    if (!confirm(t("service.confirm"))) {
        return;
    }
    const feedback = $("#restart-feedback");
    feedback.textContent = t("service.sending");
    try {
        await api("/api/settings/restart", { method: "POST" });
        feedback.textContent = t("service.scheduled");
        setStatus(t("service.restarting"), "warn");
        setTimeout(() => window.location.reload(), 5000);
    } catch (err) {
        feedback.textContent = t("service.triggered");
        setStatus(t("service.restarting"), "warn");
        setTimeout(() => window.location.reload(), 5000);
    }
});

// --- Confidence calibration ----------------------------------------------

function renderCalibrationStatus(s) {
    const el = $("#calibration-status");
    if (!el) return;
    if (!s.enable_calibration) {
        el.textContent = t("calibration.status_disabled");
        el.className = "meta";
        return;
    }
    if (s.calibration_fitted) {
        el.textContent = t("calibration.status_fitted", {
            genuine: s.calibration_n_genuine,
            impostor: s.calibration_n_impostor,
        });
        el.className = "feedback ok";
    } else {
        el.textContent = t("calibration.status_unfitted");
        el.className = "meta";
    }
}

function renderAdaptiveThresholds(map) {
    const el = $("#adaptive-thresholds-list");
    if (!el) return;
    const names = Object.keys(map);
    if (names.length === 0) {
        el.textContent = t("calibration.adaptive_none");
        return;
    }
    el.innerHTML = names
        .sort()
        .map((n) =>
            `<span class="badge">${escapeHtml(n)}: ${map[n].toFixed(3)}</span>`)
        .join(" ");
}

$("#calibration-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const body = { enable_calibration: form.enable_calibration.checked };
    if (form.enable_adaptive_thresholds) {
        body.enable_adaptive_thresholds = form.enable_adaptive_thresholds.checked;
    }
    try {
        const s = await api("/api/settings", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        renderCalibrationStatus(s);
        renderAdaptiveThresholds(s.adaptive_thresholds || {});
        setStatus(t("calibration.saved"), "ok");
    } catch (err) {
        setStatus(t("generic.error", { err: err.message }), "err");
    }
});

function updateSpeakerContextMode() {
    const sel = $("#speaker-context-mode");
    if (!sel) return;
    const mode = sel.value;
    const fields = $("#transcript-tpl-fields");
    if (fields) fields.hidden = mode !== "transcript";
    const help = $("#speaker-context-mode-help");
    if (help) help.textContent = t("transcript_tpl.help_" + mode);
}

// Ambiguity-marker delivery. "auto" resolves server-side against the
// configured sinks, so show what it currently resolves to.
function updateHintModeHelp(settings) {
    const sel = $("#transcript-hint-mode");
    const help = $("#hint-mode-help");
    if (!sel || !help) return;
    let text = t("hint_mode.hint");
    const effective = settings && settings.effective_transcript_hint_mode;
    if (sel.value === "auto" && effective) {
        text += " " + t("hint_mode.auto_now", { mode: effective });
    }
    help.textContent = text;
}

const speakerModeSel = $("#speaker-context-mode");
if (speakerModeSel) {
    speakerModeSel.addEventListener("change", updateSpeakerContextMode);
}

const hintModeSel = $("#transcript-hint-mode");
if (hintModeSel) {
    hintModeSel.addEventListener("change", () => updateHintModeHelp(null));
}

$("#transcript-tpl-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const fb = $("#transcript-tpl-feedback");
    try {
        await api("/api/settings", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                speaker_context_mode: form.speaker_context_mode.value,
                transcript_hint_mode: form.transcript_hint_mode
                    ? form.transcript_hint_mode.value
                    : undefined,
                transcript_template_known: form.transcript_template_known.value,
                transcript_template_unknown: form.transcript_template_unknown.value,
            }),
        });
        fb.className = "feedback ok";
        fb.textContent = t("transcript_tpl.saved");
        setStatus(t("transcript_tpl.saved"), "ok");
    } catch (err) {
        fb.className = "feedback err";
        fb.textContent = err.message;
    }
});

$("#recalibrate-btn").addEventListener("click", async () => {
    const btn = $("#recalibrate-btn");
    const feedback = $("#calibration-feedback");
    btn.disabled = true;
    btn.textContent = t("calibration.recalibrating");
    feedback.innerHTML = "";
    try {
        const res = await api("/api/settings/recalibrate", { method: "POST" });
        if (res.fitted) {
            feedback.innerHTML = `<span class="feedback ok">${escapeHtml(
                t("calibration.fit_ok", { genuine: res.n_genuine, impostor: res.n_impostor })
            )}</span>`;
        } else {
            feedback.innerHTML = `<span class="feedback err">${escapeHtml(t("calibration.fit_insufficient"))}</span>`;
        }
        await loadSettings();
    } catch (err) {
        feedback.innerHTML = `<span class="feedback err">${escapeHtml(t("generic.error", { err: err.message }))}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = t("calibration.recalibrate");
    }
});

// --- MQTT settings --------------------------------------------------------

function renderMqttStatus(s) {
    const el = $("#mqtt-status");
    if (!el) return;
    if (!s.mqtt_enabled) {
        el.textContent = t("mqtt.status_disabled");
        el.className = "meta";
        return;
    }
    if (s.mqtt_connected) {
        el.textContent = t("mqtt.status_connected");
        el.className = "feedback ok";
    } else {
        el.textContent = t("mqtt.status_disconnected");
        el.className = "feedback err";
    }
}

function updateMqttContextSnippet(s) {
    const el = $("#mqtt-context-code");
    if (!el) return;
    const prefix = (s.mqtt_topic_prefix || "murdock").trim() || "murdock";
    // A ready-to-paste HA automation that pushes TV state on a retained
    // context topic. <room> should match the satellite name Murdock sees.
    // One automation for every TV / radio / speaker at once. Each player
    // publishes its own playing-state + area; Murdock tightens the
    // threshold when any player in the active satellite's room is playing.
    const snippet = [
        "alias: Murdock — publish media players",
        "trigger:",
        "  - platform: state",
        "    entity_id:",
        "      - media_player.living_room_tv",
        "      - media_player.kitchen_radio",
        "      # ↑ list every TV / radio / speaker you want to gate on",
        "action:",
        "  - service: mqtt.publish",
        "    data:",
        `      topic: "${prefix}/context/media/{{ trigger.entity_id }}"`,
        "      retain: true",
        "      payload: >-",
        "        {{ {'playing': is_state(trigger.entity_id,'playing'),",
        "            'area': area_name(trigger.entity_id)} | to_json }}",
        "mode: queued",
        "max: 20",
    ].join("\n");
    el.textContent = snippet;
}

function updateMqttSatelliteSnippet(s) {
    const el = $("#mqtt-satellite-code");
    if (!el) return;
    const prefix = (s.mqtt_topic_prefix || "murdock").trim() || "murdock";
    // Publishes the active satellite's room when it starts listening, so
    // Murdock can attribute a recognition to a satellite even though HA's
    // pipeline never passes the device to the STT stage.
    const snippet = [
        "alias: Murdock — publish active satellite",
        "trigger:",
        "  - platform: state",
        "    entity_id:",
        "      - assist_satellite.living_room",
        "      - assist_satellite.kitchen",
        "    to: listening",
        "action:",
        "  - service: mqtt.publish",
        "    data:",
        `      topic: ${prefix}/active_satellite`,
        "      payload: >-",
        "        {{ {'id': trigger.entity_id,",
        "            'area': area_name(trigger.entity_id)} | to_json }}",
        "mode: queued",
        "max: 10",
    ].join("\n");
    el.textContent = snippet;
}

$("#mqtt-settings-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const body = {
        mqtt_enabled: form.mqtt_enabled.checked,
        mqtt_host: form.mqtt_host.value.trim(),
        mqtt_port: parseInt(form.mqtt_port.value, 10) || 1883,
        mqtt_username: form.mqtt_username.value.trim(),
        mqtt_topic_prefix: form.mqtt_topic_prefix.value.trim() || "murdock",
        mqtt_discovery_prefix: form.mqtt_discovery_prefix.value.trim() || "homeassistant",
    };
    // Only send password if the user actually typed something.
    const pwVal = form.mqtt_password.value;
    if (pwVal) {
        body.mqtt_password = pwVal;
    }
    try {
        const s = await api("/api/settings", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        form.mqtt_password.value = "";
        const pwHint = $("#mqtt-password-hint");
        if (pwHint) {
            pwHint.textContent = s.mqtt_password_set
                ? t("mqtt.password_set") : t("mqtt.password_empty");
        }
        renderMqttStatus(s);
        updateMqttContextSnippet(s);
        updateMqttSatelliteSnippet(s);
        setStatus(t("mqtt.saved"), "ok");
    } catch (err) {
        setStatus(t("generic.error", { err: err.message }), "err");
    }
});

$("#mqtt-test-btn").addEventListener("click", async () => {
    const btn = $("#mqtt-test-btn");
    const feedback = $("#mqtt-feedback");
    btn.disabled = true;
    btn.textContent = t("mqtt.testing");
    feedback.innerHTML = "";
    try {
        const res = await api("/api/settings/test-mqtt", { method: "POST" });
        if (res.ok) {
            feedback.innerHTML = `<span class="feedback ok">${escapeHtml(t("mqtt.test_ok"))}</span>`;
            setStatus(t("mqtt.test_ok"), "ok");
        } else {
            feedback.innerHTML = `<span class="feedback err">${escapeHtml(t("mqtt.test_fail", { err: res.error }))}</span>`;
            setStatus(t("mqtt.test_fail", { err: res.error }), "err");
        }
    } catch (err) {
        feedback.innerHTML = `<span class="feedback err">${escapeHtml(t("mqtt.test_fail", { err: err.message }))}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = t("mqtt.test");
    }
});

// Live-update the context snippet as the user edits the topic prefix.
const _mqttPrefixInput = document.querySelector("#mqtt-settings-form input[name='mqtt_topic_prefix']");
if (_mqttPrefixInput) {
    _mqttPrefixInput.addEventListener("input", () => {
        const v = { mqtt_topic_prefix: _mqttPrefixInput.value };
        updateMqttContextSnippet(v);
        updateMqttSatelliteSnippet(v);
    });
}

// --- HA settings ----------------------------------------------------------

$("#ha-settings-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const body = {
        ha_url: form.ha_url.value.trim(),
        ha_input_text_entity: form.ha_input_text_entity.value.trim(),
        ha_tv_entity: form.ha_tv_entity.value.trim(),
        ha_confidence_entity: form.ha_confidence_entity.value.trim(),
        ha_distance_entity: form.ha_distance_entity.value.trim(),
        ha_nearest_entity: form.ha_nearest_entity.value.trim(),
        ha_role_entity: form.ha_role_entity.value.trim(),
    };
    // Only send token if the user actually typed something.
    const tokenVal = form.ha_token.value;
    if (tokenVal) {
        body.ha_token = tokenVal;
    }
    try {
        const s = await api("/api/settings", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        form.ha_token.value = "";
        const hint = $("#ha-token-hint");
        if (hint) {
            hint.textContent = s.ha_token_set ? t("ha.token_set") : t("ha.token_empty");
        }
        updateHaTemplatePreview();
        setStatus(t("ha.saved"), "ok");
    } catch (err) {
        setStatus(t("generic.error", { err: err.message }), "err");
    }
});

$("#ha-test-btn").addEventListener("click", async () => {
    const btn = $("#ha-test-btn");
    const feedback = $("#ha-feedback");
    btn.disabled = true;
    btn.textContent = t("ha.testing");
    feedback.innerHTML = "";
    try {
        const res = await api("/api/settings/test-ha", { method: "POST" });
        if (res.ok) {
            feedback.innerHTML = `<span class="feedback ok">${escapeHtml(t("ha.test_ok"))}</span>`;
            setStatus(t("ha.test_ok"), "ok");
        } else {
            feedback.innerHTML = `<span class="feedback err">${escapeHtml(t("ha.test_fail", { err: res.error }))}</span>`;
            setStatus(t("ha.test_fail", { err: res.error }), "err");
        }
    } catch (err) {
        feedback.innerHTML = `<span class="feedback err">${escapeHtml(t("ha.test_fail", { err: err.message }))}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = t("ha.test");
    }
});

// --- HA template preview ---------------------------------------------------

function buildHaTemplate() {
    const form = $("#ha-settings-form");
    if (!form) return "";
    const speaker = form.ha_input_text_entity.value.trim() || "input_text.current_speaker";
    const conf = form.ha_confidence_entity.value.trim();
    const dist = form.ha_distance_entity.value.trim();
    const nearest = form.ha_nearest_entity.value.trim();
    const role = form.ha_role_entity.value.trim();

    const isDE = I18N.locale() === "de";
    const lines = [];

    lines.push(`{# Murdock – ${isDE ? "in den System-Prompt deines HA-Konversationsagenten einfügen" : "paste into your HA conversation agent system prompt"} #}`);
    lines.push("");

    if (isDE) {
        lines.push(`{% set speaker = states('${speaker}') %}`);
        lines.push(`{% if speaker and speaker not in ['unknown', 'unavailable', ''] %}`);
        lines.push(`Der aktuelle Sprecher ist "{{ speaker }}".`);
    } else {
        lines.push(`{% set speaker = states('${speaker}') %}`);
        lines.push(`{% if speaker and speaker not in ['unknown', 'unavailable', ''] %}`);
        lines.push(`The current speaker is "{{ speaker }}".`);
    }

    if (role) {
        lines.push(`{% set role = states('${role}') %}`);
        lines.push(`{% if role and role not in ['unknown', 'unavailable', ''] %}`);
        lines.push(isDE
            ? `Rolle: {{ role }}.`
            : `Role: {{ role }}.`);
        lines.push(`{% endif %}`);
    }

    if (conf) {
        lines.push(`{% set conf = states('${conf}') | float(0) %}`);
        lines.push(`{% if conf > 0 %}`);
        lines.push(isDE
            ? `Erkennungs-Sicherheit: {{ (conf * 100) | round(1) }}%.`
            : `Recognition confidence: {{ (conf * 100) | round(1) }}%.`);
        lines.push(`{% endif %}`);
    }

    if (isDE) {
        lines.push(`{% else %}`);
        lines.push(`Der Sprecher ist unbekannt.`);
    } else {
        lines.push(`{% else %}`);
        lines.push(`The speaker is unknown.`);
    }

    if (nearest) {
        lines.push(`{% set nearest = states('${nearest}') %}`);
        lines.push(`{% if nearest and nearest not in ['unknown', 'unavailable', ''] %}`);
        lines.push(isDE
            ? `Nächster bekannter Sprecher: {{ nearest }}.`
            : `Closest known speaker: {{ nearest }}.`);
        lines.push(`{% endif %}`);
    }

    if (dist) {
        lines.push(`{% set dist = states('${dist}') | float(-1) %}`);
        lines.push(`{% if dist >= 0 %}`);
        lines.push(isDE
            ? `Distanz: {{ dist | round(3) }} (Schwelle ~0.35, kleiner = ähnlicher).`
            : `Distance: {{ dist | round(3) }} (threshold ~0.35, lower = more similar).`);
        lines.push(`{% endif %}`);
    }

    lines.push(`{% endif %}`);

    return lines.join("\n");
}

function updateHaTemplatePreview() {
    const code = $("#ha-template-code");
    if (!code) return;
    code.textContent = buildHaTemplate();
}

// Update preview when any HA entity field changes.
document.querySelectorAll("#ha-settings-form input[name^='ha_']").forEach((el) => {
    el.addEventListener("input", updateHaTemplatePreview);
});

$("#ha-copy-template").addEventListener("click", async () => {
    const text = buildHaTemplate();
    try {
        await navigator.clipboard.writeText(text);
        const fb = $("#ha-copy-feedback");
        fb.textContent = t("ha.copied");
        fb.className = "feedback ok";
        setTimeout(() => { fb.textContent = ""; fb.className = ""; }, 2000);
    } catch {
        // Fallback for non-HTTPS contexts.
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
        const fb = $("#ha-copy-feedback");
        fb.textContent = t("ha.copied");
        fb.className = "feedback ok";
        setTimeout(() => { fb.textContent = ""; fb.className = ""; }, 2000);
    }
});

// --- Backup & Restore -----------------------------------------------------

$("#backup-export-btn").addEventListener("click", () => {
    window.location.href = apiUrl("/api/backup");
});

$("#backup-pick-btn").addEventListener("click", () => {
    $("#backup-file").click();
});

$("#backup-file").addEventListener("change", () => {
    const file = $("#backup-file").files[0];
    const panel = $("#backup-restore-panel");
    if (file) {
        $("#backup-file-name").textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
        panel.hidden = false;
    } else {
        panel.hidden = true;
    }
});

$("#backup-restore-btn").addEventListener("click", async () => {
    const file = $("#backup-file").files[0];
    if (!file) return;
    const mode = $("#backup-mode").value;
    if (mode === "replace" && !confirm(t("backup.confirm_replace"))) return;

    const btn = $("#backup-restore-btn");
    const feedback = $("#backup-feedback");
    btn.disabled = true;
    btn.textContent = t("backup.restoring");
    feedback.innerHTML = "";

    const form = new FormData();
    form.append("file", file);

    try {
        const res = await api(`/api/backup/restore?mode=${mode}`, {
            method: "POST",
            body: form,
        });
        const parts = [
            t("backup.result_created", { n: res.speakers_created }),
            t("backup.result_skipped", { n: res.speakers_skipped }),
            t("backup.result_samples", { n: res.samples_imported }),
        ];
        if (res.errors && res.errors.length) {
            parts.push(t("backup.result_errors", { n: res.errors.length }));
        }
        feedback.innerHTML = `<span class="feedback ok">${escapeHtml(parts.join(" · "))}</span>`;
        setStatus(t("backup.restored"), "ok");
        // Reset file input
        $("#backup-file").value = "";
        $("#backup-restore-panel").hidden = true;
        // Refresh speaker list if visible
        loadSpeakers();
    } catch (err) {
        feedback.innerHTML = `<span class="feedback err">${escapeHtml(t("generic.error", { err: err.message }))}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = t("backup.restore");
    }
});

// --- Recognition log ------------------------------------------------------

function getOutcomeLabels() {
    return I18N.outcomeLabels();
}

function formatTimestamp(unixSec) {
    const d = new Date(unixSec * 1000);
    const pad = (n) => String(n).padStart(2, "0");
    return (
        `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
        `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
    );
}

// Sub-second values read better in ms, longer ones in seconds.
function formatMs(ms) {
    if (ms === null || ms === undefined) return "";
    return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

function renderRecognitionStats(stats) {
    const container = $("#rec-stats");
    if (!container) return;
    const OUTCOME_LABELS = getOutcomeLabels();
    const hours = (stats.window_seconds / 3600).toFixed(0);
    const parts = [];
    const totalOutcomes = Object.entries(stats.per_outcome || {});
    if (totalOutcomes.length) {
        const total = totalOutcomes.reduce((sum, [, n]) => sum + n, 0);
        parts.push(`<strong>${t("rec.last_hours", { h: hours })}</strong> ${t("rec.events_count", { n: total })}`);
        const breakdown = totalOutcomes
            .sort((a, b) => b[1] - a[1])
            .map(
                ([o, n]) =>
                    `${escapeHtml(
                        (OUTCOME_LABELS[o] || { label: o }).label
                    )}: ${n}`
            )
            .join(" · ");
        parts.push(breakdown);
    } else {
        parts.push(`<strong>${t("rec.last_hours", { h: hours })}</strong> ${t("rec.no_events_window")}`);
    }
    const perSpeaker = Object.entries(stats.per_speaker || {});
    if (perSpeaker.length) {
        const speakers = perSpeaker
            .map(([s, n]) => `${escapeHtml(s)} (${n})`)
            .join(", ");
        parts.push(`${t("rec.recognised")} ${speakers}`);
    }
    container.innerHTML = parts.join("<br>");
}

function renderRecognitionEvent(e) {
    const OUTCOME_LABELS = getOutcomeLabels();
    const meta = OUTCOME_LABELS[e.outcome] || { label: e.outcome, cls: "" };
    const badgeCls = meta.cls ? `badge ${meta.cls}` : "badge";
    const ts = formatTimestamp(e.created_at);
    const isMatch = e.outcome === "match";
    const who = isMatch && e.matched_speaker
        ? escapeHtml(e.matched_speaker)
        : (e.outcome && e.outcome.startsWith("passthrough"))
            ? `<span class="muted">${escapeHtml(t("rec.passthrough"))}</span>`
            : `<span class="muted">${escapeHtml(t("rec.unknown"))}</span>`;
    const nearestHint = !isMatch && e.matched_speaker
        ? `<div class="meta">${escapeHtml(t("rec.nearest", { name: e.matched_speaker }))}</div>`
        : "";
    const dist =
        e.distance !== null && e.distance !== undefined
            ? `d=${e.distance.toFixed(4)}`
            : "";
    const thr =
        e.threshold !== null && e.threshold !== undefined
            ? `th=${e.threshold.toFixed(3)}`
            : "";
    const vms =
        e.verify_ms !== null && e.verify_ms !== undefined
            ? `verify=${e.verify_ms.toFixed(0)}ms`
            : "";
    // Request breakdown for the primary cloud engine. TTFB carries the
    // upload, the queue and the decode; the body is a few hundred bytes,
    // so a body figure that isn't ~0 means the network, not the model.
    const tt = e.transcript_timing || null;
    const sttParts = [];
    if (tt && tt.ttfb_ms !== undefined && tt.ttfb_ms !== null) {
        sttParts.push(`ttfb=${formatMs(tt.ttfb_ms)}`);
    }
    if (tt && tt.body_ms !== undefined && tt.body_ms !== null) {
        sttParts.push(`body=${formatMs(tt.body_ms)}`);
    }
    if (tt && tt.sent_bytes) {
        sttParts.push(`${Math.round(tt.sent_bytes / 1024)}kB`);
    }
    if (tt && tt.trimmed_ms > 0) {
        sttParts.push(`-${formatMs(tt.trimmed_ms)} ${t("rec.trimmed")}`);
    }
    if (tt && tt.answer_ms !== undefined && tt.answer_ms !== null) {
        sttParts.push(`${t("rec.answer")}=${formatMs(tt.answer_ms)}`);
    }
    if (tt && tt.gate_ms !== undefined && tt.gate_ms !== null) {
        sttParts.push(`${t("rec.gate")}=${formatMs(tt.gate_ms)}`);
    }
    if (tt && tt.rescued_by) {
        sttParts.push(`${t("rec.rescued")} ${escapeHtml(tt.rescued_by)}`);
    }
    if (tt && tt.failed) {
        sttParts.push(escapeHtml(tt.failed));
    }
    const sttLine = sttParts.length
        ? `<div class="meta">stt: ${sttParts.join(" · ")}</div>`
        : "";
    const scoreLine = [dist, thr, vms].filter(Boolean).join(" · ");
    // Engine timing badge. In upstream mode the primary streams while the
    // audio is still arriving, so its figure is the remaining wait — the
    // shadow always runs on the finished buffer.
    const msBadge = (ms, extraCls) =>
        ms === null || ms === undefined
            ? ""
            : `<span class="badge timing${extraCls || ""}">${formatMs(ms)}</span>`;
    const hasBoth =
        e.transcript_ms !== null && e.transcript_ms !== undefined &&
        e.shadow_ms !== null && e.shadow_ms !== undefined;
    // Mark the slower of the two when both are known.
    const primarySlower = hasBoth && e.transcript_ms > e.shadow_ms;
    const transcript = e.transcript
        ? `<div class="transcript">${msBadge(e.transcript_ms, primarySlower && hasBoth ? " slower" : "")}&ldquo;${escapeHtml(e.transcript)}&rdquo;</div>`
        : `<div class="transcript muted">${escapeHtml(t("rec.no_transcript"))}</div>`;
    // A/B shadow engine result (filled in asynchronously). Highlight when
    // the two engines disagree — that's the signal the A/B test is for.
    let shadow = "";
    if (e.shadow_transcript != null && e.shadow_engine) {
        const same = (e.transcript || "").trim().toLowerCase()
            === (e.shadow_transcript || "").trim().toLowerCase();
        let delta = "";
        if (hasBoth) {
            const diff = e.shadow_ms - e.transcript_ms;
            const key = diff >= 0 ? "rec.slower_by" : "rec.faster_by";
            delta = `<span class="meta"> ${escapeHtml(
                t(key, { ms: formatMs(Math.abs(diff)) })
            )}</span>`;
        }
        shadow = `<div class="transcript shadow${same ? "" : " differs"}">
            <span class="badge">${escapeHtml(t("rec.shadow"))} ${escapeHtml(e.shadow_engine)}</span>
            ${msBadge(e.shadow_ms, hasBoth && !primarySlower ? " slower" : "")}
            &ldquo;${escapeHtml(e.shadow_transcript)}&rdquo;
            ${delta}
            ${same ? "" : `<span class="meta"> ${escapeHtml(t("rec.shadow_differs"))}</span>`}
        </div>`;
    }
    const sat = e.satellite_id
        ? ` · <code>${escapeHtml(e.satellite_id)}</code>`
        : "";
    const whisperChip = (() => {
        const score = e.whisper_score;
        const has = score !== null && score !== undefined;
        if (!e.whisper && !has) return "";
        const label = e.whisper
            ? t("rec.whisper") + (has ? ` ${score.toFixed(2)}` : "")
            : t("rec.whisper_score", { score: score.toFixed(2) });
        // Detected → solid; measured but under the bar → muted.
        const cls = e.whisper ? "badge whisper" : "badge whisper below";
        return `<span class="${cls}">${escapeHtml(label)}</span>`;
    })();
    // Blocked/unknown entries whose audio was captured can be turned into
    // training data directly from the log.
    const assignBtn = e.unknown_sample_id
        ? `<button type="button" class="secondary small" data-assign-sample="${e.unknown_sample_id}"
             data-nearest="${escapeHtml(e.matched_speaker || "")}">${escapeHtml(t("rec.add_to_speaker"))}</button>`
        : "";
    return `
        <div class="list-item">
            <div class="row">
                <span class="${badgeCls}">${escapeHtml(meta.label)}</span>
                <strong>${who}</strong>
                ${whisperChip}
                <span class="meta">${ts} · ${e.duration_sec.toFixed(2)}s${sat}</span>
            </div>
            ${transcript}
            ${shadow}
            ${nearestHint}
            ${scoreLine ? `<div class="meta">${scoreLine}</div>` : ""}
            ${sttLine}
            ${assignBtn ? `<div class="row">${assignBtn}</div>` : ""}
        </div>
    `;
}

// Delegated handler: "add to speaker" on a captured recognition event.
document.addEventListener("click", async (ev) => {
    const btn = ev.target.closest("[data-assign-sample]");
    if (!btn) return;
    const sampleId = btn.dataset.assignSample;
    const nearest = btn.dataset.nearest || "";
    const name = prompt(t("rec.assign_prompt"), nearest);
    if (!name || !name.trim()) return;
    btn.disabled = true;
    try {
        const res = await api(`/api/unknown/${sampleId}/assign`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ speaker_name: name.trim(), create_if_missing: true }),
        });
        setStatus(t("rec.assigned_ok", { name: res.speaker_name, n: res.total_samples }), "ok");
        loadRecognition();
        loadSpeakers();
    } catch (err) {
        setStatus(t("generic.error", { err: err.message }), "err");
        btn.disabled = false;
    }
});

async function loadRecognition() {
    const list = $("#rec-list");
    if (!list) return;
    list.innerHTML = `<p class="meta">${escapeHtml(t("generic.loading"))}</p>`;
    const outcome = $("#rec-filter-outcome").value;
    const limit = $("#rec-filter-limit").value;
    const qs = new URLSearchParams({ limit });
    if (outcome) qs.set("outcome", outcome);
    try {
        const [events, stats] = await Promise.all([
            api(`/api/recognition?${qs.toString()}`),
            api("/api/recognition/stats?hours=24"),
        ]);
        renderRecognitionStats(stats);
        if (!events.events.length) {
            list.innerHTML =
                `<p class="meta">${escapeHtml(t("rec.no_events"))}</p>`;
            return;
        }
        list.innerHTML = events.events.map(renderRecognitionEvent).join("");
    } catch (err) {
        list.innerHTML = `<p class="feedback err">${escapeHtml(err.message)}</p>`;
    }
}

$("#refresh-rec").addEventListener("click", loadRecognition);
$("#rec-filter-outcome").addEventListener("change", loadRecognition);
$("#rec-filter-limit").addEventListener("change", loadRecognition);
$("#clear-rec").addEventListener("click", async () => {
    if (!confirm(t("rec.confirm_clear"))) return;
    try {
        const res = await api("/api/recognition", { method: "DELETE" });
        setStatus(t("rec.cleared", { n: res.deleted }), "ok");
        loadRecognition();
    } catch (err) {
        setStatus(t("generic.error", { err: err.message }), "err");
    }
});
$("#test-rec").addEventListener("click", async () => {
    try {
        const res = await api("/api/recognition/test", { method: "POST" });
        setStatus(res.message, res.ok ? "ok" : "err");
        loadRecognition();
    } catch (err) {
        setStatus(t("generic.error", { err: err.message }), "err");
    }
});

// --- Quality weight form handlers -----------------------------------------

const qualityForm = $("#quality-form");
if (qualityForm) {
    qualityForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const weights = {};
        for (const k of ["speech_ratio", "snr", "liveness", "consistency", "centroid_distance"]) {
            const v = parseFloat(qualityForm[k].value);
            weights[k] = isNaN(v) ? 0 : v;
        }
        const total = Object.values(weights).reduce((a, b) => a + b, 0);
        if (total <= 0) {
            $("#quality-feedback").className = "meta feedback err";
            $("#quality-feedback").textContent = t("quality.err_zero");
            return;
        }
        try {
            $("#quality-feedback").className = "meta";
            $("#quality-feedback").textContent = t("quality.saving");
            const s = await api("/api/settings", {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ quality_weights: weights }),
            });
            // Repopulate (server normalizes)
            for (const k of ["speech_ratio", "snr", "liveness", "consistency", "centroid_distance"]) {
                qualityForm[k].value = (s.quality_weights[k] ?? 0).toFixed(3);
            }
            const info = $("#quality-weights-info");
            if (info) {
                info.textContent = s.quality_weights_source === "override"
                    ? t("quality.source_override")
                    : t("quality.source_default");
            }
            $("#quality-feedback").className = "meta feedback ok";
            $("#quality-feedback").textContent = t("quality.saved");
        } catch (err) {
            $("#quality-feedback").className = "meta feedback err";
            $("#quality-feedback").textContent = t("generic.error", { err: err.message });
        }
    });
}

const qualityResetBtn = $("#quality-reset-btn");
if (qualityResetBtn) {
    qualityResetBtn.addEventListener("click", async () => {
        if (!confirm(t("quality.confirm_reset"))) return;
        try {
            const s = await api("/api/settings", {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ quality_weights: {} }),
            });
            const form = $("#quality-form");
            for (const k of ["speech_ratio", "snr", "liveness", "consistency", "centroid_distance"]) {
                form[k].value = (s.quality_weights[k] ?? 0).toFixed(3);
            }
            const info = $("#quality-weights-info");
            if (info) info.textContent = t("quality.source_default");
            $("#quality-feedback").className = "meta feedback ok";
            $("#quality-feedback").textContent = t("quality.reset_ok");
        } catch (err) {
            $("#quality-feedback").className = "meta feedback err";
            $("#quality-feedback").textContent = t("generic.error", { err: err.message });
        }
    });
}

const qualityRescoreBtn = $("#quality-rescore-btn");
if (qualityRescoreBtn) {
    qualityRescoreBtn.addEventListener("click", async () => {
        if (!confirm(t("quality.confirm_rescore_all"))) return;
        const fb = $("#quality-feedback");
        try {
            qualityRescoreBtn.disabled = true;
            fb.className = "meta";
            fb.textContent = t("quality.rescoring_all");
            const results = await api("/api/speakers/rescore-all", {
                method: "POST",
            });
            const total = results.reduce((a, r) => a + r.rescored, 0);
            fb.className = "meta feedback ok";
            fb.textContent = t("quality.rescore_all_done", {
                n: total, s: results.length,
            });
        } catch (err) {
            fb.className = "meta feedback err";
            fb.textContent = t("generic.error", { err: err.message });
        } finally {
            qualityRescoreBtn.disabled = false;
        }
    });
}

// --- Utilities ------------------------------------------------------------

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    }[c]));
}

// --- Collapsible settings cards --------------------------------------------
//
// The settings tab has grown long; every card collapses to its heading.
// Default: first card (core settings) open, the rest closed. The choice
// is remembered per card in localStorage. Hidden cards still receive
// their loadSettings() population — display:none doesn't affect that.
function initCollapsibleCards() {
    document.querySelectorAll("#tab-settings > .card").forEach((card, i) => {
        const h2 = card.querySelector("h2");
        if (!h2) return;
        const key = "murdock.collapse." + (h2.dataset.i18n || i);
        card.classList.add("collapsible");
        const stored = localStorage.getItem(key);
        const collapsed = stored === null ? i > 0 : stored === "1";
        card.classList.toggle("collapsed", collapsed);
        h2.addEventListener("click", () => {
            card.classList.toggle("collapsed");
            localStorage.setItem(
                key, card.classList.contains("collapsed") ? "1" : "0"
            );
        });
    });
}

// Initial load
loadOverview();
loadRoles();
loadSpeakers();
initMicAvailability();
initCollapsibleCards();
api("/api/health")
    .then((h) => setStatus(t("health.status", { version: h.version, n: h.speakers })))
    .catch((err) => setStatus(t("health.failed", { err: err.message }), "err"));

// -- Bias-prompt vocabulary editor ---------------------------------
//
// Two kinds of chip: your own terms (always sent) and mirrored ones from
// Home Assistant, which are toggleable. Only the first N mirrored terms
// reach the engine, so choosing *which* N matters more than seeing them.
let VOCAB_STATE = null;

async function loadVocabularyMirror() {
    const box = $("#vocab-mirror");
    if (!box) return;
    try {
        VOCAB_STATE = await api("/api/vocabulary");
    } catch (err) {
        box.hidden = true;
        return;
    }
    renderVocabularyPanel();
}

function renderVocabularyPanel() {
    const data = VOCAB_STATE;
    const box = $("#vocab-mirror");
    if (!box || !data) return;
    const own = data.manual_terms || [];
    const mirrored = data.terms || [];
    if (!mirrored.length && !own.length) {
        box.hidden = true;
        return;
    }
    box.hidden = false;

    const selected = new Set((data.selected || []).map((x) => x.toLowerCase()));
    const isManual = data.selection !== null && data.selection !== undefined;

    const meta = $("#vocab-mirror-meta");
    if (meta) {
        meta.textContent = data.available
            ? t("vocab.mirror_meta", {
                  entities: data.entity_count,
                  terms: data.term_count,
                  cap: data.term_cap,
                  version: data.version || "?",
                  age: data.created_at ? formatTimestamp(data.created_at) : "",
              })
            : t("vocab.mirror_none");
    }

    const list = $("#vocab-terms");
    if (list) {
        const ownChips = own.map(
            (term) =>
                '<span class="chip own" title="' +
                escapeHtml(t("vocab.chip_own")) +
                '">' +
                escapeHtml(term) +
                '<button type="button" class="chip-x" data-vocab-remove="' +
                escapeHtml(term) +
                '">&times;</button></span>'
        );
        const mirroredChips = mirrored.map((term) => {
            const on = selected.has(term.toLowerCase());
            return (
                '<button type="button" class="' +
                (on ? "chip" : "chip capped") +
                '" data-vocab-toggle="' +
                escapeHtml(term) +
                '" title="' +
                escapeHtml(on ? t("vocab.chip_sent") : t("vocab.chip_off")) +
                '">' +
                escapeHtml(term) +
                "</button>"
            );
        });
        list.innerHTML = ownChips.concat(mirroredChips).join("");
    }

    const count = $("#vocab-count");
    if (count) {
        count.textContent = t("vocab.count", {
            own: own.length,
            selected: selected.size,
            cap: data.term_cap,
            mode: isManual ? t("vocab.mode_manual") : t("vocab.mode_auto"),
        });
    }

    // Say so when the active backend throws the prompt away.
    const warn = $("#vocab-backend-warning");
    if (warn) {
        const ignored = data.effective_enabled && !data.backend_supports_prompt;
        warn.hidden = !ignored;
        if (ignored) warn.textContent = t("vocab.backend_ignores");
    }

    const eff = $("#vocab-effective");
    if (eff) {
        eff.textContent = data.effective_enabled
            ? data.effective_prompt || t("vocab.effective_empty")
            : t("vocab.effective_disabled");
    }
}

async function putVocabSelection(payload) {
    try {
        VOCAB_STATE = await api("/api/vocabulary/selection", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        renderVocabularyPanel();
        setStatus(t("vocab.saved"), "ok");
    } catch (err) {
        setStatus(String(err), "error");
    }
}

// Toggling freezes the currently sent set as an explicit selection —
// otherwise the first click would be undone by the automatic "first N"
// rule as soon as the next snapshot arrives.
function toggleMirroredTerm(term) {
    if (!VOCAB_STATE) return;
    const current = new Set(VOCAB_STATE.selected || []);
    if (current.has(term)) current.delete(term);
    else current.add(term);
    putVocabSelection({ terms: Array.from(current) });
}

async function saveOwnTerms(terms) {
    const value = terms.join(", ");
    const form = $("#stt-form");
    if (form && form.stt_vocabulary) form.stt_vocabulary.value = value;
    try {
        await api("/api/settings", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ stt_vocabulary: value }),
        });
        await loadVocabularyMirror();
        setStatus(t("vocab.saved"), "ok");
    } catch (err) {
        setStatus(String(err), "error");
    }
}

const vocabTermsBox = $("#vocab-terms");
if (vocabTermsBox) {
    vocabTermsBox.addEventListener("click", (e) => {
        const remove = e.target.closest("[data-vocab-remove]");
        if (remove) {
            const term = remove.dataset.vocabRemove;
            const own = (VOCAB_STATE ? VOCAB_STATE.manual_terms || [] : [])
                .filter((x) => x !== term);
            saveOwnTerms(own);
            return;
        }
        const toggle = e.target.closest("[data-vocab-toggle]");
        if (toggle) toggleMirroredTerm(toggle.dataset.vocabToggle);
    });
}

function addOwnTermFromInput() {
    const input = $("#vocab-add-input");
    if (!input) return;
    const term = input.value.trim().replace(/,+$/, "");
    if (!term) return;
    const own = VOCAB_STATE ? VOCAB_STATE.manual_terms || [] : [];
    input.value = "";
    if (own.some((x) => x.toLowerCase() === term.toLowerCase())) return;
    saveOwnTerms(own.concat([term]));
}

const vocabAddBtn = $("#vocab-add-btn");
if (vocabAddBtn) vocabAddBtn.addEventListener("click", addOwnTermFromInput);

const vocabAddInput = $("#vocab-add-input");
if (vocabAddInput) {
    vocabAddInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            addOwnTermFromInput();
        }
    });
}

const vocabAutoBtn = $("#vocab-auto-btn");
if (vocabAutoBtn) {
    vocabAutoBtn.addEventListener("click", () => putVocabSelection({ auto: true }));
}

// -- Recurring canonicalizations, promotable to exact rules ---------
async function loadCanonicalizerHits() {
    const box = $("#canon-learned");
    const list = $("#canon-learned-list");
    if (!box || !list) return;
    let data;
    try {
        data = await api("/api/canonicalizer/hits?limit=15");
    } catch (err) {
        box.hidden = true;
        return;
    }
    const hits = data.hits || [];
    box.hidden = hits.length === 0;
    list.innerHTML = hits
        .map(
            (h) =>
                '<div class="list-item"><div class="row"><span><code>' +
                escapeHtml(h.original) +
                "</code> &rarr; <code>" +
                escapeHtml(h.replacement) +
                '</code> <span class="badge">' +
                h.count +
                '&times;</span></span><button type="button" class="secondary small"' +
                ' data-promote-original="' + escapeHtml(h.original) + '"' +
                ' data-promote-replacement="' + escapeHtml(h.replacement) + '">' +
                escapeHtml(t("canon.promote")) +
                "</button></div></div>"
        )
        .join("");
}

const canonList = $("#canon-learned-list");
if (canonList) {
    canonList.addEventListener("click", async (e) => {
        const btn = e.target.closest("[data-promote-original]");
        if (!btn) return;
        try {
            await api("/api/canonicalizer/promote", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    original: btn.dataset.promoteOriginal,
                    replacement: btn.dataset.promoteReplacement,
                }),
            });
            setStatus(t("canon.promoted"), "ok");
            await loadCanonicalizerHits();
            await loadSettings();
        } catch (err) {
            setStatus(String(err), "error");
        }
    });
}

const canonRefreshBtn = $("#canon-refresh-btn");
if (canonRefreshBtn) canonRefreshBtn.addEventListener("click", loadCanonicalizerHits);

const vocabRefreshBtn = $("#vocab-refresh-btn");
if (vocabRefreshBtn) vocabRefreshBtn.addEventListener("click", loadVocabularyMirror);

// ── Settings sub-navigation ────────────────────────────────────────
//
// The settings tab grew to a dozen cards; showing one group at a time
// keeps it scannable. The chosen group is remembered so a save + reload
// doesn't dump you back at the top.
const SETTINGS_GROUP_KEY = "murdock.settingsGroup";

function showSettingsGroup(key) {
    const groups = document.querySelectorAll(".settings-group");
    if (!groups.length) return;
    const known = [...groups].map((g) => g.dataset.settingsGroup);
    const target = known.includes(key) ? key : known[0];
    groups.forEach((g) => {
        g.classList.toggle("active", g.dataset.settingsGroup === target);
    });
    document.querySelectorAll(".settings-nav-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.settingsGroup === target);
    });
    try {
        localStorage.setItem(SETTINGS_GROUP_KEY, target);
    } catch (err) {
        /* private mode — remembering is a nicety, not a requirement */
    }
}

document.querySelectorAll(".settings-nav-btn").forEach((btn) => {
    btn.addEventListener("click", () =>
        showSettingsGroup(btn.dataset.settingsGroup)
    );
});

if (document.querySelector(".settings-group")) {
    let remembered = null;
    try {
        remembered = localStorage.getItem(SETTINGS_GROUP_KEY);
    } catch (err) {
        remembered = null;
    }
    showSettingsGroup(remembered || "recognition");
}

// ── Whisper detection (experimental) ───────────────────────────────
const whisperFormEl = $("#whisper-form");
if (whisperFormEl) {
    whisperFormEl.addEventListener("submit", async (e) => {
        e.preventDefault();
        const form = e.target;
        const fb = $("#whisper-feedback");
        const body = {
            enable_whisper_detection: form.enable_whisper_detection.checked,
        };
        if (form.whisper_threshold.value !== "") {
            body.whisper_threshold = parseFloat(form.whisper_threshold.value);
        }
        try {
            await api("/api/settings", {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            fb.className = "feedback ok";
            fb.textContent = t("whisper.saved");
            setStatus(t("whisper.saved"), "ok");
        } catch (err) {
            fb.className = "feedback error";
            fb.textContent = String(err);
        }
    });
}

// ── Enrollment coach ───────────────────────────────────────────────
async function loadCoach() {
    const list = $("#coach-list");
    if (!list) return;
    let data;
    try {
        data = await api("/api/speakers/coach");
    } catch (err) {
        list.innerHTML = "";
        return;
    }
    const findings = data.findings || [];
    if (!findings.length) {
        list.innerHTML = `<p class="meta">${escapeHtml(
            data.speakers_checked
                ? t("coach.all_good", { n: data.speakers_checked })
                : t("coach.no_speakers")
        )}</p>`;
        return;
    }
    list.innerHTML = findings
        .map((f) => {
            const cls = f.severity === "warn" ? "badge warn" : "badge";
            const label = f.severity === "warn"
                ? t("coach.warn") : t("coach.info");
            return `<div class="list-item"><div class="row">
                <span class="${cls}">${escapeHtml(label)}</span>
                <strong>${escapeHtml(f.speaker)}</strong>
            </div><div class="meta">${escapeHtml(f.message)}</div></div>`;
        })
        .join("");
}

const coachRefreshBtn = $("#coach-refresh-btn");
if (coachRefreshBtn) coachRefreshBtn.addEventListener("click", loadCoach);

// ── Enrollment style (normal vs whispered) ─────────────────────────
function updateEnrollStyleHint() {
    const sel = $("#enroll-style");
    const hint = $("#enroll-style-hint");
    if (!sel || !hint) return;
    hint.textContent = sel.value === "whisper"
        ? t("speakers.whisper_hint", { min: 2 })
        : "";
}

const enrollStyleSel = $("#enroll-style");
if (enrollStyleSel) {
    enrollStyleSel.addEventListener("change", updateEnrollStyleHint);
    updateEnrollStyleHint();
}
