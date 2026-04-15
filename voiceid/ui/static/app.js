// VoiceID Web UI — no framework, just DOM.

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

// --- Tabs -----------------------------------------------------------------

$$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        $$(".tab-btn").forEach((b) => b.classList.remove("active"));
        $$(".tab").forEach((t) => t.classList.remove("active"));
        btn.classList.add("active");
        $("#tab-" + btn.dataset.tab).classList.add("active");
        if (btn.dataset.tab === "speakers") loadSpeakers();
        if (btn.dataset.tab === "verify") loadRoles();
        if (btn.dataset.tab === "unknown") loadUnknown();
        if (btn.dataset.tab === "settings") loadSettings();
        if (btn.dataset.tab === "recognition") loadRecognition();
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
            const select = $("#enroll-role");
            if (select) {
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

recordBtn.addEventListener("click", async () => {
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
                    ${s.ha_user_id ? `<span class="meta">HA: ${escapeHtml(s.ha_user_id)}</span>` : ""}
                </div>
                <div class="row">
                    <button class="secondary" data-view="${s.id}">${escapeHtml(t("speakers.view_samples"))}</button>
                    <button class="secondary" data-edit="${s.id}">${escapeHtml(t("speakers.edit"))}</button>
                    <button class="secondary" data-rescore="${s.id}">${escapeHtml(t("quality.rescore_one"))}</button>
                    <button class="danger" data-del="${s.id}">${escapeHtml(t("speakers.delete_speaker"))}</button>
                </div>
                <div class="edit-panel" hidden></div>
                <div class="samples" hidden></div>
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
    } catch (err) {
        list.innerHTML = `<p class="feedback err">${escapeHtml(err.message)}</p>`;
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
            row.innerHTML = `
                <div class="row">
                    ${filename}
                    ${sourceBadge}
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
        if (s.upstream_uri_source === "override") {
            form.upstream_uri.value = s.upstream_uri || "";
        } else {
            form.upstream_uri.value = "";
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
            if (haForm.ha_emotion_entity) {
                haForm.ha_emotion_entity.value = s.ha_emotion_entity || "";
            }
            updateHaTemplatePreview();
            const hint = $("#ha-token-hint");
            if (hint) {
                hint.textContent = s.ha_token_set ? t("ha.token_set") : t("ha.token_empty");
            }
        }
        // Per-satellite thresholds
        loadSatelliteThresholds();
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
                    <span class="badge satellite">${escapeHtml(e.satellite_id)}</span>
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
        upstream_uri: (form.upstream_uri.value || "").trim(),
        advertised_languages: parseLanguages(form.advertised_languages.value),
    };
    // New fields (backwards-compatible)
    if (form.min_liveness_score) {
        body.min_liveness_score = parseFloat(form.min_liveness_score.value);
    }
    if (form.auto_enroll) {
        body.auto_enroll = form.auto_enroll.checked;
    }
    try {
        const s = await api("/api/settings", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        renderUpstreamHint(s);
        renderLangHint(s);
        if (s.upstream_uri_source === "override") {
            form.upstream_uri.value = s.upstream_uri || "";
        } else {
            form.upstream_uri.value = "";
        }
        setStatus(t("settings.saved"), "ok");
    } catch (err) {
        setStatus(t("generic.error", { err: err.message }), "err");
    }
});

$("#ping-upstream-btn").addEventListener("click", async () => {
    const btn = $("#ping-upstream-btn");
    const out = $("#ping-result");
    btn.disabled = true;
    btn.textContent = t("ping.pinging");
    out.innerHTML = "";
    try {
        const res = await api("/api/settings/ping-upstream", { method: "POST" });
        if (res.ok) {
            const langs = (res.languages || []).join(", ") || "(none)";
            out.innerHTML =
                `<span class="feedback ok">${escapeHtml(t("ping.ok"))}</span><br>` +
                `<code>${escapeHtml(res.upstream_uri)}</code> — ` +
                `${res.latency_ms.toFixed(0)}ms, ${t("ping.languages")} ${escapeHtml(langs)}`;
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
});

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
    const emotion = form.ha_emotion_entity ? form.ha_emotion_entity.value.trim() : "";

    const isDE = I18N.locale() === "de";
    const lines = [];

    lines.push(`{# VoiceID – ${isDE ? "in den System-Prompt deines HA-Konversationsagenten einfügen" : "paste into your HA conversation agent system prompt"} #}`);
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

    if (emotion) {
        lines.push(`{% set emotion = states('${emotion}') %}`);
        lines.push(`{% if emotion and emotion not in ['unknown', 'unavailable', ''] %}`);
        lines.push(isDE
            ? `Stimmungslage: {{ emotion }}.`
            : `Emotional tone: {{ emotion }}.`);
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
    const scoreLine = [dist, thr, vms].filter(Boolean).join(" · ");
    const transcript = e.transcript
        ? `<div class="transcript">&ldquo;${escapeHtml(e.transcript)}&rdquo;</div>`
        : `<div class="transcript muted">${escapeHtml(t("rec.no_transcript"))}</div>`;
    const sat = e.satellite_id
        ? ` · <code>${escapeHtml(e.satellite_id)}</code>`
        : "";
    return `
        <div class="list-item">
            <div class="row">
                <span class="${badgeCls}">${escapeHtml(meta.label)}</span>
                <strong>${who}</strong>
                <span class="meta">${ts} · ${e.duration_sec.toFixed(2)}s${sat}</span>
            </div>
            ${transcript}
            ${nearestHint}
            ${scoreLine ? `<div class="meta">${scoreLine}</div>` : ""}
        </div>
    `;
}

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

// Initial load
loadRoles();
loadSpeakers();
api("/api/health")
    .then((h) => setStatus(t("health.status", { version: h.version, n: h.speakers })))
    .catch((err) => setStatus(t("health.failed", { err: err.message }), "err"));
