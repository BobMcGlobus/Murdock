// VoiceID i18n — lightweight translation layer.
//
// Usage:
//   t("nav.speakers")              → "Speakers" / "Sprecher"
//   t("msg.enrolled", {name, n})   → 'Enrolled "jonas" — 3 sample(s)'
//
// Static HTML elements carry data-i18n="key" (textContent),
// data-i18n-ph="key" (placeholder), or data-i18n-html="key" (innerHTML).
// Call translatePage() after changing locale.

const I18N = (() => {
    const TRANSLATIONS = {
        en: {
            // -- Nav / tabs --
            "nav.speakers": "Speakers",
            "nav.verify": "Verify",
            "nav.recognition": "Recognition log",
            "nav.unknown": "Unknown",
            "nav.settings": "Settings",

            // -- Speakers tab --
            "speakers.enroll_title": "Enroll speaker",
            "speakers.enrolled_title": "Enrolled speakers",
            "speakers.add_to_existing": "Add sample to existing speaker",
            "speakers.new_speaker": "— new speaker —",
            "speakers.name": "Name",
            "speakers.ha_user_id": "HA user ID (optional — only applied to new speakers)",
            "speakers.role": "Role (optional — only applied to new speakers)",
            "speakers.none": "— none —",
            "speakers.start_recording": "Start recording",
            "speakers.stop_recording": "Stop recording",
            "speakers.idle": "Idle",
            "speakers.recording": "Recording…",
            "speakers.or_upload": "…or upload an audio file (WAV, MP3, MP4/M4A, OGG, FLAC, WebM)",
            "speakers.enroll_btn": "Enroll",
            "speakers.uploading": "Uploading…",
            "speakers.no_sample": "Record or upload a sample first.",
            "speakers.enrolled_ok": 'Enrolled "{name}" — {n} sample(s)',
            "speakers.speech_ratio": "speech {pct}%, {sec}s",
            "speakers.recorded_kb": "Recorded {kb} KB WAV",
            "speakers.mic_error": "Mic error: {err}",
            "speakers.encode_error": "Encode error: {err}",
            "speakers.no_speakers": "No speakers enrolled yet.",
            "speakers.loading": "Loading…",
            "speakers.view_samples": "View samples",
            "speakers.edit": "Edit",
            "speakers.delete_speaker": "Delete speaker",
            "speakers.confirm_delete": "Delete this speaker and all samples?",
            "speakers.samples": "samples",
            "speakers.loading_samples": "Loading samples…",
            "speakers.no_samples": "No samples.",
            "speakers.no_filename": "(no filename)",
            "speakers.delete_sample": "Delete",
            "speakers.saving": "Saving…",
            "speakers.save": "Save",
            "speakers.cancel": "Cancel",
            "speakers.ha_clear_hint": "leave empty to clear",
            "speakers.ph_name": "e.g. jonas",
            "speakers.ph_ha_id": "uuid from Home Assistant",

            // -- Verify tab --
            "verify.title": "Verify a sample",
            "verify.description": "Upload a WAV (or record one in the Speakers tab first) to score it against every enrolled speaker. Useful for tuning the verify threshold.",
            "verify.audio_file": "Audio file (WAV, MP3, MP4/M4A, OGG, FLAC, WebM)",
            "verify.btn": "Verify",
            "verify.pick_file": "Pick a WAV file first.",
            "verify.scoring": "Scoring…",
            "verify.match": "MATCH: {name}",
            "verify.no_match": "NO MATCH",
            "verify.best_distance": "Best distance {d} · threshold {t}",
            "verify.col_speaker": "Speaker",
            "verify.col_distance": "Distance",
            "verify.no_speakers": "No speakers enrolled yet.",

            // -- Recognition log tab --
            "rec.title": "Recognition log",
            "rec.description": "Every completed Wyoming session is logged here so you can see which voice was recognised when, what was said, and which gate path the session took.",
            "rec.outcome_filter": "Outcome",
            "rec.all": "— all —",
            "rec.limit": "Limit",
            "rec.refresh": "Refresh",
            "rec.insert_test": "Insert test entry",
            "rec.clear_log": "Clear log",
            "rec.no_events": "No recognition events yet. Say something to your satellite.",
            "rec.confirm_clear": "Clear the entire recognition log?",
            "rec.cleared": "Cleared {n} events",
            "rec.no_transcript": "(no transcript)",
            "rec.nearest": "nearest: {name}",
            "rec.unknown": "unknown",
            "rec.passthrough": "— (passthrough)",
            "rec.last_hours": "Last {h}h:",
            "rec.events_count": "{n} event(s)",
            "rec.no_events_window": "no events yet",
            "rec.recognised": "Recognised:",

            // -- Outcomes --
            "outcome.match": "Match",
            "outcome.unknown-forwarded": "Unknown (forwarded)",
            "outcome.blocked-no-match": "Blocked — no match",
            "outcome.blocked-no-speakers": "Blocked — no speakers",
            "outcome.blocked-embed-failed": "Blocked — embed failed",
            "outcome.blocked-tv-noise": "Blocked — TV/noise",
            "outcome.passthrough-short": "Passthrough — short",
            "outcome.passthrough-no-speakers": "Passthrough — no speakers",
            "outcome.empty": "Empty",

            // -- Unknown tab --
            "unknown.title": "Unknown samples",
            "unknown.include_tagged": "Include tagged",
            "unknown.refresh": "Refresh",
            "unknown.cleanup": "Cleanup expired",
            "unknown.no_samples": "No unknown samples logged.",
            "unknown.likely_tv": "Likely TV",
            "unknown.likely_live": "Likely live",
            "unknown.assign_ph": "Assign to speaker",
            "unknown.assign_btn": "Assign",
            "unknown.tag_tv": "Tag as TV",
            "unknown.delete": "Delete",
            "unknown.enter_name": "Enter a speaker name first.",
            "unknown.cleaned": "Cleaned up {n} expired samples",
            "unknown.distance": "distance",
            "unknown.best": "best:",
            "unknown.liveness": "liveness",

            // -- Settings tab --
            "settings.title": "Runtime settings",
            "settings.upstream_uri": "Upstream STT (Wyoming) URI",
            "settings.ph_upstream": "e.g. 192.168.2.107:10300",
            "settings.threshold": "Verify threshold (cosine distance, 0 = identical)",
            "settings.skip_leading": "Skip leading seconds (trim satellite chime before embedding)",
            "settings.skip_hint": "STT still receives the full audio. Typical value: 1.0 s for ESPHome/Wyoming satellites.",
            "settings.unknown_logging": "Log unknown voices",
            "settings.require_match": "Require speaker match (block unknowns)",
            "settings.passthrough_empty": "Passthrough when zero speakers enrolled",
            "settings.languages": "Advertised languages (comma-separated, e.g. de,en)",
            "settings.ph_languages": "leave empty for auto-detect from upstream",
            "settings.min_liveness": "Min liveness score (reject TV/background noise)",
            "settings.min_liveness_hint": "Voices below this score are classified as TV/noise and blocked. Set to 0 to disable. Default: 0.35.",
            "settings.auto_enroll": "Auto-enroll on match (aging/re-training)",
            "settings.auto_enroll_hint": "Automatically add fresh embeddings when a known speaker is recognized, so the model adapts over time.",
            "settings.save": "Save",
            "settings.refresh_langs": "Refresh from upstream",
            "settings.ping": "Ping upstream",
            "settings.saved": "Settings saved",
            "settings.listen": "Listen:",
            "settings.min_verify": "Min verify length: {sec}s (shorter utterances pass through)",
            "settings.ttl": "TTL: {h}h",
            "settings.ha_yes": "HA: configured",
            "settings.ha_no": "HA: not configured",

            // -- Settings hints --
            "hint.upstream_override": "Override active → {uri}. Clear the field and save to fall back to the compose-time default {default}.",
            "hint.upstream_default": "Using compose default {uri}. Enter a new host (e.g. 192.168.2.107:10300) to override — tcp:// is added automatically.",
            "hint.lang_override": "Override active → advertising: {langs}. Clear the field and save to fall back to upstream auto-detect.",
            "hint.lang_auto": "Auto-detected from upstream: {langs}. Override here if HA still can't see VoiceID.",
            "hint.lang_none": 'Upstream hasn\'t advertised any languages yet. Enter e.g. "de,en" to force VoiceID to appear in HA\'s pipeline picker.',

            // -- Ping --
            "ping.pinging": "Pinging…",
            "ping.ok": "✓ Upstream reachable",
            "ping.fail": "✗ Upstream unreachable",
            "ping.request_failed": "Ping request failed:",
            "ping.upstream_ok": "Upstream OK",
            "ping.upstream_unreachable": "Upstream unreachable",
            "ping.failed": "Ping failed: {err}",
            "ping.languages": "languages:",

            // -- Refresh --
            "refresh.refreshing": "Refreshing…",
            "refresh.languages": "Upstream languages: {langs}",
            "refresh.failed": "Refresh failed: {err}",

            // -- Restart --
            "service.title": "Service",
            "service.description": "Restart the VoiceID process. Docker will bring it back up automatically. Use this after changing the upstream URI or advertised languages if you're not sure the change has taken effect.",
            "service.restart_btn": "Restart VoiceID",
            "service.confirm": "Restart VoiceID now? The container will come back up automatically.",
            "service.sending": "Sending restart request…",
            "service.scheduled": "Restart scheduled. The page will reconnect in a few seconds…",
            "service.restarting": "Restarting…",
            "service.triggered": "Restart triggered (connection dropped — that's expected).",

            // -- Health --
            "health.status": "v{version} · {n} speakers",
            "health.failed": "Health check failed: {err}",

            // -- Generic --
            "generic.error": "Error: {err}",
            "generic.loading": "Loading…",
        },

        de: {
            // -- Nav / tabs --
            "nav.speakers": "Sprecher",
            "nav.verify": "Verifizieren",
            "nav.recognition": "Erkennungsprotokoll",
            "nav.unknown": "Unbekannt",
            "nav.settings": "Einstellungen",

            // -- Speakers tab --
            "speakers.enroll_title": "Sprecher registrieren",
            "speakers.enrolled_title": "Registrierte Sprecher",
            "speakers.add_to_existing": "Probe zu bestehendem Sprecher hinzufügen",
            "speakers.new_speaker": "— neuer Sprecher —",
            "speakers.name": "Name",
            "speakers.ha_user_id": "HA Benutzer-ID (optional — gilt nur für neue Sprecher)",
            "speakers.role": "Rolle (optional — gilt nur für neue Sprecher)",
            "speakers.none": "— keine —",
            "speakers.start_recording": "Aufnahme starten",
            "speakers.stop_recording": "Aufnahme stoppen",
            "speakers.idle": "Bereit",
            "speakers.recording": "Aufnahme läuft…",
            "speakers.or_upload": "…oder Audiodatei hochladen (WAV, MP3, MP4/M4A, OGG, FLAC, WebM)",
            "speakers.enroll_btn": "Registrieren",
            "speakers.uploading": "Hochladen…",
            "speakers.no_sample": "Bitte zuerst eine Probe aufnehmen oder hochladen.",
            "speakers.enrolled_ok": '"{name}" registriert — {n} Probe(n)',
            "speakers.speech_ratio": "Sprache {pct}%, {sec}s",
            "speakers.recorded_kb": "{kb} KB WAV aufgenommen",
            "speakers.mic_error": "Mikrofonfehler: {err}",
            "speakers.encode_error": "Kodierungsfehler: {err}",
            "speakers.no_speakers": "Noch keine Sprecher registriert.",
            "speakers.loading": "Laden…",
            "speakers.view_samples": "Proben anzeigen",
            "speakers.edit": "Bearbeiten",
            "speakers.delete_speaker": "Sprecher löschen",
            "speakers.confirm_delete": "Diesen Sprecher und alle Proben löschen?",
            "speakers.samples": "Proben",
            "speakers.loading_samples": "Proben laden…",
            "speakers.no_samples": "Keine Proben.",
            "speakers.no_filename": "(kein Dateiname)",
            "speakers.delete_sample": "Löschen",
            "speakers.saving": "Speichern…",
            "speakers.save": "Speichern",
            "speakers.cancel": "Abbrechen",
            "speakers.ha_clear_hint": "leer lassen zum Entfernen",
            "speakers.ph_name": "z.B. jonas",
            "speakers.ph_ha_id": "UUID aus Home Assistant",

            // -- Verify tab --
            "verify.title": "Probe verifizieren",
            "verify.description": "Lade eine WAV-Datei hoch (oder nimm eine im Sprecher-Tab auf), um sie gegen alle registrierten Sprecher zu testen. Nützlich zum Tuning des Schwellenwerts.",
            "verify.audio_file": "Audiodatei (WAV, MP3, MP4/M4A, OGG, FLAC, WebM)",
            "verify.btn": "Verifizieren",
            "verify.pick_file": "Bitte zuerst eine Datei auswählen.",
            "verify.scoring": "Auswertung…",
            "verify.match": "TREFFER: {name}",
            "verify.no_match": "KEIN TREFFER",
            "verify.best_distance": "Beste Distanz {d} · Schwellenwert {t}",
            "verify.col_speaker": "Sprecher",
            "verify.col_distance": "Distanz",
            "verify.no_speakers": "Noch keine Sprecher registriert.",

            // -- Recognition log tab --
            "rec.title": "Erkennungsprotokoll",
            "rec.description": "Jede abgeschlossene Wyoming-Sitzung wird hier protokolliert. Du siehst, welche Stimme wann erkannt wurde, was gesagt wurde und welchen Entscheidungspfad die Sitzung genommen hat.",
            "rec.outcome_filter": "Ergebnis",
            "rec.all": "— alle —",
            "rec.limit": "Limit",
            "rec.refresh": "Aktualisieren",
            "rec.insert_test": "Testeintrag",
            "rec.clear_log": "Protokoll leeren",
            "rec.no_events": "Noch keine Erkennungsereignisse. Sprich etwas in deinen Satelliten.",
            "rec.confirm_clear": "Das gesamte Erkennungsprotokoll löschen?",
            "rec.cleared": "{n} Einträge gelöscht",
            "rec.no_transcript": "(kein Transkript)",
            "rec.nearest": "nächster: {name}",
            "rec.unknown": "unbekannt",
            "rec.passthrough": "— (Durchleitung)",
            "rec.last_hours": "Letzte {h}h:",
            "rec.events_count": "{n} Ereignis(se)",
            "rec.no_events_window": "noch keine Ereignisse",
            "rec.recognised": "Erkannt:",

            // -- Outcomes --
            "outcome.match": "Treffer",
            "outcome.unknown-forwarded": "Unbekannt (weitergeleitet)",
            "outcome.blocked-no-match": "Blockiert — kein Treffer",
            "outcome.blocked-no-speakers": "Blockiert — keine Sprecher",
            "outcome.blocked-embed-failed": "Blockiert — Embedding fehlgeschlagen",
            "outcome.blocked-tv-noise": "Blockiert — TV/Hintergrund",
            "outcome.passthrough-short": "Durchleitung — zu kurz",
            "outcome.passthrough-no-speakers": "Durchleitung — keine Sprecher",
            "outcome.empty": "Leer",

            // -- Unknown tab --
            "unknown.title": "Unbekannte Proben",
            "unknown.include_tagged": "Markierte anzeigen",
            "unknown.refresh": "Aktualisieren",
            "unknown.cleanup": "Abgelaufene bereinigen",
            "unknown.no_samples": "Keine unbekannten Proben vorhanden.",
            "unknown.likely_tv": "Vermutlich TV",
            "unknown.likely_live": "Vermutlich live",
            "unknown.assign_ph": "Sprecher zuweisen",
            "unknown.assign_btn": "Zuweisen",
            "unknown.tag_tv": "Als TV markieren",
            "unknown.delete": "Löschen",
            "unknown.enter_name": "Bitte zuerst einen Sprechernamen eingeben.",
            "unknown.cleaned": "{n} abgelaufene Proben bereinigt",
            "unknown.distance": "Distanz",
            "unknown.best": "nächster:",
            "unknown.liveness": "Lebendigkeit",

            // -- Settings tab --
            "settings.title": "Laufzeiteinstellungen",
            "settings.upstream_uri": "Upstream-STT (Wyoming) URI",
            "settings.ph_upstream": "z.B. 192.168.2.107:10300",
            "settings.threshold": "Erkennungsschwelle (Kosinusdistanz, 0 = identisch)",
            "settings.skip_leading": "Anfang überspringen (Satellite-Benachrichtigungston vor Embedding abschneiden)",
            "settings.skip_hint": "STT erhält weiterhin das vollständige Audio. Typischer Wert: 1,0 s für ESPHome/Wyoming-Satelliten.",
            "settings.unknown_logging": "Unbekannte Stimmen protokollieren",
            "settings.require_match": "Sprecherübereinstimmung erzwingen (Unbekannte blockieren)",
            "settings.passthrough_empty": "Durchleitung wenn keine Sprecher registriert",
            "settings.languages": "Beworbene Sprachen (kommagetrennt, z.B. de,en)",
            "settings.ph_languages": "leer lassen für automatische Erkennung",
            "settings.min_liveness": "Mindest-Lebendigkeit (TV/Hintergrundgeräusche ablehnen)",
            "settings.min_liveness_hint": "Stimmen unter diesem Wert werden als TV/Hintergrund eingestuft und blockiert. Auf 0 setzen zum Deaktivieren. Standard: 0,35.",
            "settings.auto_enroll": "Automatische Nachregistrierung bei Treffer (Alterung/Nachtraining)",
            "settings.auto_enroll_hint": "Fügt automatisch frische Embeddings hinzu, wenn ein bekannter Sprecher erkannt wird, damit sich das Modell mit der Zeit anpasst.",
            "settings.save": "Speichern",
            "settings.refresh_langs": "Von Upstream aktualisieren",
            "settings.ping": "Upstream testen",
            "settings.saved": "Einstellungen gespeichert",
            "settings.listen": "Lauscht auf:",
            "settings.min_verify": "Mindestlänge: {sec}s (kürzere Äußerungen werden durchgeleitet)",
            "settings.ttl": "TTL: {h}h",
            "settings.ha_yes": "HA: konfiguriert",
            "settings.ha_no": "HA: nicht konfiguriert",

            // -- Settings hints --
            "hint.upstream_override": "Überschreibung aktiv → {uri}. Feld leeren und speichern für den Compose-Standard {default}.",
            "hint.upstream_default": "Compose-Standard {uri}. Neuen Host eingeben (z.B. 192.168.2.107:10300) zum Überschreiben — tcp:// wird automatisch ergänzt.",
            "hint.lang_override": "Überschreibung aktiv → bewirbt: {langs}. Feld leeren und speichern für automatische Erkennung.",
            "hint.lang_auto": "Automatisch erkannt vom Upstream: {langs}. Hier überschreiben, falls HA VoiceID nicht sieht.",
            "hint.lang_none": 'Upstream hat noch keine Sprachen gemeldet. Trage z.B. "de,en" ein, damit VoiceID in HAs Pipeline-Auswahl erscheint.',

            // -- Ping --
            "ping.pinging": "Teste…",
            "ping.ok": "✓ Upstream erreichbar",
            "ping.fail": "✗ Upstream nicht erreichbar",
            "ping.request_failed": "Ping fehlgeschlagen:",
            "ping.upstream_ok": "Upstream OK",
            "ping.upstream_unreachable": "Upstream nicht erreichbar",
            "ping.failed": "Ping fehlgeschlagen: {err}",
            "ping.languages": "Sprachen:",

            // -- Refresh --
            "refresh.refreshing": "Aktualisiere…",
            "refresh.languages": "Upstream-Sprachen: {langs}",
            "refresh.failed": "Aktualisierung fehlgeschlagen: {err}",

            // -- Restart --
            "service.title": "Dienst",
            "service.description": "VoiceID-Prozess neu starten. Docker startet ihn automatisch neu. Verwende dies nach Änderungen an der Upstream-URI oder den beworbenen Sprachen, wenn du unsicher bist, ob die Änderung wirkt.",
            "service.restart_btn": "VoiceID neustarten",
            "service.confirm": "VoiceID jetzt neustarten? Der Container kommt automatisch zurück.",
            "service.sending": "Neustart wird angefordert…",
            "service.scheduled": "Neustart geplant. Die Seite verbindet sich in wenigen Sekunden neu…",
            "service.restarting": "Neustart…",
            "service.triggered": "Neustart ausgelöst (Verbindung unterbrochen — das ist normal).",

            // -- Health --
            "health.status": "v{version} · {n} Sprecher",
            "health.failed": "Statusabfrage fehlgeschlagen: {err}",

            // -- Generic --
            "generic.error": "Fehler: {err}",
            "generic.loading": "Laden…",
        },
    };

    let _locale = localStorage.getItem("voiceid_lang") || "auto";
    let _resolved = _resolve(_locale);

    function _resolve(loc) {
        if (loc && loc !== "auto" && TRANSLATIONS[loc]) return loc;
        // Pick from browser language.
        const nav = (navigator.language || "en").split("-")[0].toLowerCase();
        return TRANSLATIONS[nav] ? nav : "en";
    }

    function _format(template, params) {
        if (!params) return template;
        return template.replace(/\{(\w+)\}/g, (_, key) =>
            params[key] !== undefined ? String(params[key]) : `{${key}}`
        );
    }

    /**
     * Translate a key. Returns the localised string, with optional
     * placeholder substitution. Falls back to English, then to the
     * raw key.
     */
    function t(key, params) {
        const dict = TRANSLATIONS[_resolved] || TRANSLATIONS.en;
        const val = dict[key] ?? TRANSLATIONS.en[key] ?? key;
        return _format(val, params);
    }

    /** Return the active locale code ("en" or "de"). */
    function locale() {
        return _resolved;
    }

    /** Return the user preference ("auto", "en", "de"). */
    function preference() {
        return _locale;
    }

    /** List available locale codes. */
    function available() {
        return Object.keys(TRANSLATIONS);
    }

    /** Switch locale. Persists to localStorage. */
    function setLocale(loc) {
        _locale = loc;
        _resolved = _resolve(loc);
        localStorage.setItem("voiceid_lang", loc);
        translatePage();
    }

    /**
     * Scan the DOM for data-i18n / data-i18n-ph / data-i18n-html
     * attributes and replace content with the current locale's strings.
     */
    function translatePage() {
        document.querySelectorAll("[data-i18n]").forEach((el) => {
            el.textContent = t(el.dataset.i18n);
        });
        document.querySelectorAll("[data-i18n-ph]").forEach((el) => {
            el.placeholder = t(el.dataset.i18nPh);
        });
        document.querySelectorAll("[data-i18n-html]").forEach((el) => {
            el.innerHTML = t(el.dataset.i18nHtml);
        });
    }

    /** Build outcome labels using current locale. */
    function outcomeLabels() {
        return {
            "match": { label: t("outcome.match"), cls: "ok" },
            "unknown-forwarded": { label: t("outcome.unknown-forwarded"), cls: "warn" },
            "blocked-no-match": { label: t("outcome.blocked-no-match"), cls: "err" },
            "blocked-no-speakers": { label: t("outcome.blocked-no-speakers"), cls: "err" },
            "blocked-embed-failed": { label: t("outcome.blocked-embed-failed"), cls: "err" },
            "blocked-tv-noise": { label: t("outcome.blocked-tv-noise"), cls: "err" },
            "passthrough-short": { label: t("outcome.passthrough-short"), cls: "" },
            "passthrough-no-speakers": { label: t("outcome.passthrough-no-speakers"), cls: "" },
            "empty": { label: t("outcome.empty"), cls: "" },
        };
    }

    return { t, locale, preference, available, setLocale, translatePage, outcomeLabels };
})();

// Global shortcut
const t = I18N.t;
