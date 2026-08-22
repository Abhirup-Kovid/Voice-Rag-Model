var LATENCY_TARGET_MS = 200;
var MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];

/* Prevent auto-scroll to a stale #hash on load */
if ("scrollRestoration" in history) history.scrollRestoration = "manual";
if (location.hash) history.replaceState(null, "", location.pathname + location.search);
window.scrollTo(0, 0);

/* ── Global state ──────────────────────────────────────── */
var state = {
    dialect: { lang: null, stt: "unknown" },
    lastSources: [],
    lastResult: null
};

var recordBtn = document.getElementById("recordBtn");
var statusEl = document.getElementById("status");
var textForm = document.getElementById("textForm");
var textInput = document.getElementById("textInput");
var apiDot = document.getElementById("apiDot");
var apiStatus = document.getElementById("apiStatus");
var stagesPanel = document.getElementById("stagesPanel");
var answerEl = document.getElementById("answerText");
statusEl.textContent = "READY // AUTO-LANG";

var mediaRecorder;
var recChunks = [];
var audioCtx, analyser, waveRaf, mediaStream;
var currentAnswer = "";
var lastQuery = "";
var toastTimer = null;

/* ── Scroll reveal ─────────────────────────────────────── */
(function () {
    var els = document.querySelectorAll(".reveal");
    if (!("IntersectionObserver" in window) || els.length === 0) {
        els.forEach(function (el) { el.classList.add("in"); });
        return;
    }
    var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add("in");
                io.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12 });
    els.forEach(function (el) { io.observe(el); });
})();

/* ── Health check (writes footer ping) ─────────────────── */
function checkHealth() {
    var t0 = performance.now();
    fetch("/api/health", { cache: "no-store" })
        .then(function(res) {
            var ping = Math.round(performance.now() - t0);
            var sysPing = document.getElementById("sysPing");
            var sysState = document.getElementById("sysState");
            if (res.ok) {
                apiDot.className = "api-led"; apiStatus.textContent = "PROTOCOL ONLINE";
                apiStatus.className = "font-mono text-[10px] tracking-[0.2em] text-slate-400";
                if (sysPing) { sysPing.textContent = ping; sysPing.className = "text-emerald-300"; }
                if (sysState) { sysState.className = "sys-ok"; sysState.textContent = "OPERATIONAL"; }
            } else {
                apiDot.className = "api-led down"; apiStatus.textContent = "DISCONNECTED";
                apiStatus.className = "font-mono text-[10px] tracking-[0.2em] text-rose-400";
                if (sysPing) { sysPing.textContent = "ERR"; sysPing.className = "text-rose-400"; }
                if (sysState) { sysState.className = "sys-degraded"; sysState.textContent = "DEGRADED"; }
            }
        })
        .catch(function() {
            apiDot.className = "api-led down"; apiStatus.textContent = "DISCONNECTED";
            apiStatus.className = "font-mono text-[10px] tracking-[0.2em] text-rose-400";
            var sysPing = document.getElementById("sysPing");
            var sysState = document.getElementById("sysState");
            if (sysPing) { sysPing.textContent = "ERR"; sysPing.className = "text-rose-400"; }
            if (sysState) { sysState.className = "sys-degraded"; sysState.textContent = "DEGRADED"; }
        });
}
checkHealth();
setInterval(checkHealth, 15000);

/* ── Audio capture ─────────────────────────────────────── */
function pickMimeType() {
    if (typeof MediaRecorder === "undefined") return "";
    for (var i = 0; i < MIME_CANDIDATES.length; i++) {
        if (MediaRecorder.isTypeSupported(MIME_CANDIDATES[i])) return MIME_CANDIDATES[i];
    }
    return "";
}
function extForMime(mime) {
    if (mime.indexOf("webm") !== -1) return "webm";
    if (mime.indexOf("mp4") !== -1) return "mp4";
    if (mime.indexOf("ogg") !== -1) return "ogg";
    return "wav";
}
function setupRecorder() {
    return navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
        mediaStream = stream;
        var mimeType = pickMimeType();
        var opts = mimeType ? { mimeType: mimeType } : undefined;
        try { mediaRecorder = new MediaRecorder(stream, opts); } catch(e) { mediaRecorder = new MediaRecorder(stream); }
        mediaRecorder.ondataavailable = function(e) { if (e.data.size > 0) recChunks.push(e.data); };
        mediaRecorder.onstop = function() {
            var mime = mediaRecorder.mimeType || "audio/webm";
            var blob = new Blob(recChunks, { type: mime });
            recChunks = [];
            sendAudio(blob, mime);
        };
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        var source = audioCtx.createMediaStreamSource(stream);
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
    });
}

/* ── Recording controls ────────────────────────────────── */
function startRecording() {
    if (recordBtn.classList.contains("recording")) return;
    setupRecorder().then(function() {
        recChunks = [];
        mediaRecorder.start();
        recordBtn.classList.add("recording");
        textForm.classList.add("speaking");
        document.getElementById("siteTop").classList.add("recording");
        recordBtn.setAttribute("aria-pressed", "true");
        recordBtn.querySelector(".mic-label").textContent = "RELEASE TO SEND";
        statusEl.innerHTML = '<span class="spinner"></span>LISTENING // CAPTURE ACTIVE';
        setStage("listening", "active", "mic live");
    }).catch(function() {
        statusEl.textContent = "MIC PERMISSION DENIED - USE THE TEXT BAR BELOW.";
    });
}
function stopRecording() {
    if (!mediaRecorder || mediaRecorder.state !== "recording") return;
    mediaRecorder.stop();
    recordBtn.classList.remove("recording");
    textForm.classList.remove("speaking");
    document.getElementById("siteTop").classList.remove("recording");
    recordBtn.setAttribute("aria-pressed", "false");
    recordBtn.querySelector(".mic-label").textContent = "HOLD TO TALK";
    statusEl.innerHTML = '<span class="spinner"></span>TRANSCRIBING + RETRIEVING + SYNTHESIZING...';
    setStage("listening", "done", "captured");
}
recordBtn.addEventListener("mousedown", function(e) { e.preventDefault(); startRecording(); });
recordBtn.addEventListener("touchstart", function(e) { e.preventDefault(); startRecording(); }, { passive: false });
recordBtn.addEventListener("mouseup", stopRecording);
recordBtn.addEventListener("mouseleave", function() { if (mediaRecorder && mediaRecorder.state === "recording") stopRecording(); });
recordBtn.addEventListener("touchend", function(e) { e.preventDefault(); stopRecording(); });
recordBtn.addEventListener("keydown", function(e) {
    if ((e.key === " " || e.key === "Enter") && !e.repeat) { e.preventDefault(); startRecording(); }
});
recordBtn.addEventListener("keyup", function(e) {
    if (e.key === " " || e.key === "Enter") { e.preventDefault(); stopRecording(); }
});

/* ── Query execution ───────────────────────────────────── */
function sendAudio(blob, mime) {
    pendingVoiceBlob = blob;
    resetUI();
    var formData = new FormData();
    formData.append("file", blob, "query." + extForMime(mime));
    if (state.dialect.stt) formData.append("language", state.dialect.stt);
    fetch("/api/voice", { method: "POST", body: formData })
        .then(function(res) { return res.json(); })
        .then(function(data) {
            if (data.stream_id) { connectSSE(data.stream_id); }
            else { statusEl.textContent = "ERROR: NO STREAM ID."; showToast("Backend did not return a stream.", retryLastVoice); }
        })
        .catch(function() { statusEl.textContent = "REQUEST FAILED - CHECK BACKEND."; showToast("Request failed. Check backend.", retryLastVoice); });
}

function connectSSE(streamId) {
    var es = new EventSource("/api/stream/" + streamId);
    var firstToken = false;
    showStages();
    es.addEventListener("stt", function(e) {
        var data = JSON.parse(e.data);
        setStage("listening", "done", data.stt_ms + "ms");
        setStage("transcribing", "done", "recognized");
        setStage("retrieving", "active", "searching");
        statusEl.innerHTML = '<span class="spinner"></span>STT DONE (' + data.stt_ms + 'ms) // RETRIEVING...';
        textInput.value = data.transcript || "";
        showTranscript(data.transcript || "");
    });
    es.addEventListener("token", function(e) {
        var data = JSON.parse(e.data);
        if (!firstToken) {
            firstToken = true;
            setStage("retrieving", "done", "");
            setStage("grounding", "done", "passed");
            setStage("answering", "active", "streaming");
        }
        appendAnswer(data.text || "");
    });
    es.addEventListener("done", function(e) {
        es.close();
        var data = JSON.parse(e.data);
        if (data.error) {
            statusEl.textContent = "ERROR: " + data.error;
            showToast("Could not process query: " + data.error, retryLastVoice);
            if (!firstToken) {
                setStage("retrieving", "skipped", "-");
                setStage("grounding", "failed", "no match");
                setStage("answering", "skipped", "-");
            }
        } else if (data.data) {
            renderResult(data.data);
        }
    });
    es.onerror = function() {
        es.close();
        statusEl.textContent = "STREAM CONNECTION LOST.";
        showToast("Connection lost. Check the backend.", null);
    };
}

var pendingVoiceBlob = null;
function retryLastVoice() {
    if (pendingVoiceBlob) sendAudio(pendingVoiceBlob, pendingVoiceBlob.type);
}

textForm.addEventListener("submit", function(e) {
    e.preventDefault();
    var q = textInput.value.trim();
    if (q) sendTextQuery(q);
});

/* Sample query cards — run the probe on the agent */
document.querySelectorAll(".q-card").forEach(function(card) {
    card.addEventListener("click", function() {
        var q = card.dataset.query;
        textInput.value = q;
        document.getElementById("agent").scrollIntoView({ behavior: "smooth", block: "start" });
        setTimeout(function() { sendTextQuery(q); }, 350);
    });
});

function sendTextQuery(query) {
    lastQuery = query;
    resetUI();
    showStages();
    showTranscript(query);
    statusEl.innerHTML = '<span class="spinner"></span>RETRIEVING + SYNTHESIZING...';
    setStage("listening", "skipped", "text input");
    setStage("transcribing", "active", "parsing");
    setTimeout(function () { setStage("transcribing", "done", "parsed"); setStage("retrieving", "active", "searching"); }, 250);
    setTimeout(function () { setStage("retrieving", "done", "..."); setStage("grounding", "active", "scoring"); }, 700);
    fetch("/api/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, language: state.dialect.lang })
    })
        .then(function(res) { return res.json(); })
        .then(function(data) { renderResult(data); })
        .catch(function() {
            statusEl.textContent = "REQUEST FAILED - CHECK BACKEND.";
            showToast("Request failed. Check backend.", function () { sendTextQuery(lastQuery); });
        });
}

/* ── UI reset / live stage strip ───────────────────────── */
function resetUI() {
    currentAnswer = "";
    answerEl.innerHTML = '<span class="spinner"></span>';
    document.getElementById("citePills").innerHTML = "";
    document.getElementById("transcriptPanel").hidden = true;
    document.getElementById("answerPanel").hidden = true;
    document.getElementById("sourcesPanel").hidden = true;
    document.getElementById("latencyPanel").hidden = true;
    document.getElementById("badge").className = "badge";
    document.getElementById("badge").textContent = "";
    resetStages();
}
function showStages() { stagesPanel.hidden = false; }
function setStage(name, stateName, text) {
    var el = stagesPanel.querySelector('.stage[data-stage="' + name + '"]');
    if (!el) return;
    el.classList.remove("active", "done", "failed", "skipped");
    if (stateName) el.classList.add(stateName);
    var ms = el.querySelector(".stage-ms");
    if (ms) ms.textContent = text;
}
function resetStages() {
    document.querySelectorAll(".stage").forEach(function (el) {
        el.classList.remove("active", "done", "failed", "skipped");
    });
}

/* ── Flip cards (how it works) ─────────────────────────── */
document.querySelectorAll(".flip-card").forEach(function(card) {
    card.addEventListener("click", function() {
        card.classList.toggle("flipped");
    });
});

/* ── Toast ─────────────────────────────────────────────── */
function showToast(message, retryFn) {
    var toast = document.getElementById("toast");
    toast.hidden = false;
    toast.innerHTML = '<span class="toast-msg">' + message + '</span>' +
        (retryFn ? '<button type="button" class="toast-retry">TAP TO RETRY</button>' : "") +
        '<button type="button" class="toast-close" aria-label="dismiss">&times;</button>';
    var retryBtn = toast.querySelector(".toast-retry");
    if (retryBtn) retryBtn.addEventListener("click", function() { hideToast(); retryFn(); });
    toast.querySelector(".toast-close").addEventListener("click", hideToast);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(hideToast, 5000);
}
function hideToast() {
    var toast = document.getElementById("toast");
    toast.hidden = true;
    clearTimeout(toastTimer);
}

/* ── Citation pills + evidence cross-hover ─────────────── */
function renderCitePills(sources) {
    var wrap = document.getElementById("citePills");
    wrap.innerHTML = "";
    if (!sources || sources.length === 0) return;
    sources.forEach(function(s, i) {
        var pill = document.createElement("span");
        pill.className = "cite-pill";
        pill.dataset.idx = i;
        pill.title = "doc " + (s.doc_id || "?") + " · " + (s.strategy || "fixed") + " · score " + (s.score || 0).toFixed(3);
        pill.textContent = (s.doc_id || "?");
        pill.addEventListener("mouseenter", function() { highlightEvidence(i); pill.classList.add("cite-active"); });
        pill.addEventListener("mouseleave", function() { clearEvidenceHighlight(); pill.classList.remove("cite-active"); });
        pill.addEventListener("click", function() {
            var card = document.querySelector('.evidence-card[data-idx="' + i + '"]');
            if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
        });
        wrap.appendChild(pill);
    });
}

function renderEvidencePath(evidencePath, sources) {
    var panel = document.getElementById("evidencePathPanel");
    var list = document.getElementById("evidencePathList");
    if (!panel || !list) return;
    list.innerHTML = "";
    if (!evidencePath || evidencePath.length === 0) {
        panel.hidden = true;
        return;
    }
    panel.hidden = false;
    evidencePath.forEach(function(e) {
        var li = document.createElement("li");
        li.className = "evidence-path-item";
        var conf = e.confidence || 0;
        var confClass = conf >= 0.3 ? "ev-high" : (conf >= 0.15 ? "ev-med" : "ev-low");
        li.innerHTML =
            '<span class="ev-sentence">"' + e.sentence + '"</span>' +
            '<span class="ev-arrow">→</span>' +
            '<span class="ev-source-ref">' + e.source_id + '</span>' +
            '<span class="ev-confidence ' + confClass + '">' + Math.round(conf * 100) + '% match</span>';
        li.addEventListener("mouseenter", function() {
            highlightEvidence(e.source_idx);
        });
        li.addEventListener("mouseleave", function() {
            clearEvidenceHighlight();
        });
        list.appendChild(li);
    });
}

function logEscalation(data) {
    if (!data.guardrail || !data.guardrail.needs_escalation) return;
    fetch("/api/escalate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            query_id: data.query_id,
            query: lastQuery,
            answer: data.answer,
            reason: data.guardrail.reason,
            evidence_score: data.guardrail.evidence_score,
        })
    }).catch(function() {});
}
function highlightEvidence(idx) {
    document.querySelectorAll(".evidence-card").forEach(function(c) {
        c.classList.toggle("ev-active", parseInt(c.dataset.idx, 10) === idx);
    });
}
function clearEvidenceHighlight() {
    document.querySelectorAll(".evidence-card").forEach(function(c) { c.classList.remove("ev-active"); });
}

function showTranscript(text) {
    document.getElementById("transcriptText").textContent = text;
    document.getElementById("transcriptPanel").hidden = false;
}

function appendAnswer(text) {
    if (currentAnswer === "") answerEl.innerHTML = "";
    currentAnswer += text;
    answerEl.textContent = currentAnswer;
    var cursor = document.createElement("span");
    cursor.className = "cursor";
    answerEl.appendChild(cursor);
}

/* ── Result rendering ──────────────────────────────────── */
function renderResult(data) {
    state.lastResult = data;
    statusEl.textContent = "";
    stagesPanel.hidden = false;

    if (data.answer && !currentAnswer) {
        answerEl.textContent = data.answer;
    } else if (!currentAnswer) {
        answerEl.textContent = data.answer || data.error || "NO ANSWER";
    }
    document.getElementById("answerPanel").hidden = false;

    var badge = document.getElementById("badge");
    var status = "ok";
    if (data.guardrail) {
        if (data.guardrail.unsafe) status = "unsafe";
        else if (data.guardrail.off_topic) status = "offtopic";
        else if (data.guardrail.needs_escalation) status = "escalation";
        else if (!data.guardrail.grounded) status = "ungrounded";
        else if (data.guardrail.refused) status = "refused";
        else if (!data.guardrail.passed) status = "ungrounded";
    }
    var BADGES = {
        ok: ["GROUNDED & VERIFIED", "badge-ok"],
        offtopic: ["OUT OF CORPUS // NO SUPPORT", "badge-warn"],
        ungrounded: ["NOT CONFIDENTLY GROUNDED", "badge-warn"],
        escalation: ["FLAGGED FOR REVIEW // LOW EVIDENCE", "badge-warn"],
        refused: ["REFUSED // OUTSIDE KNOWLEDGE BASE", "badge-warn"],
        unsafe: ["BLOCKED BY SAFETY GUARDRAIL", "badge-err"],
        error: ["PIPELINE ERROR", "badge-err"]
    };
    var b = BADGES[status] || ["UNKNOWN", "badge-err"];
    badge.textContent = b[0];
    badge.className = "badge " + b[1];

    renderSources(data.sources || []);
    renderCitePills(data.sources || []);
    renderEvidencePath(data.evidence_path || [], data.sources || []);
    renderLatency(data);
    renderStages(data);
    logEscalation(data);
    document.getElementById("latencyPanel").hidden = false;

    var resultsBox = document.getElementById("results");
    if (resultsBox && resultsBox.dataset.shown !== "1") {
        resultsBox.dataset.shown = "1";
        setTimeout(function() {
            resultsBox.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 80);
    }
}

function renderSources(sources) {
    var sourcesList = document.getElementById("sourcesList");
    sourcesList.innerHTML = "";
    if (sources.length === 0) {
        document.getElementById("sourcesPanel").hidden = true;
        return;
    }
    document.getElementById("sourcesPanel").hidden = false;
    sources.forEach(function(s, i) {
        var li = document.createElement("li");
        li.className = "evidence-card";
        li.dataset.idx = i;
        var head = document.createElement("div");
        head.className = "ev-head";
        head.innerHTML =
            '<span class="ev-ref">[' + (i + 1) + ']</span>' +
            '<span class="ev-score">Score: ' + (s.score || 0).toFixed(3) + ' | ' + Math.round((s.score || 0) * 100) + '% Match</span>' +
            '<span class="ev-lang">' + (s.language || "?").toUpperCase() + '</span>' +
            '<span class="ev-strategy">' + (s.strategy || "fixed") + '</span>';
        var gauge = document.createElement("div");
        gauge.className = "ev-gauge";
        var fill = document.createElement("div");
        fill.className = "ev-gauge-fill";
        fill.style.width = "0%";
        gauge.appendChild(fill);
        var text = document.createElement("p");
        text.className = "ev-text";
        text.textContent = s.text.length > 260 ? s.text.slice(0, 260) + "..." : s.text;
        li.appendChild(head);
        li.appendChild(gauge);
        li.appendChild(text);
        sourcesList.appendChild(li);
        setTimeout(function() { fill.style.width = Math.round((s.score || 0) * 100) + "%"; }, 60 + i * 120);
    });
}

/* ── Latency waterfall profiler ────────────────────────── */
function renderLatency(data) {
    var timings = data.latency || {};
    var total = timings.total_ms || 0;
    var stt = timings.stt_ms || 0;
    var ret = timings.retrieve_ms || 0;
    var llmTotal = timings.llm_total_ms || 0;
    var llmTtft = timings.llm_first_token_ms || 0;
    var llmStream = Math.max(0, llmTotal - llmTtft);

    var bar = document.getElementById("latencyBar");
    bar.innerHTML = "";
    var segs = [
        { cls: "stt", ms: stt, label: "STT network" },
        { cls: "retrieval", ms: ret, label: "dense vector retrieval" },
        { cls: "llm_ttft", ms: llmTtft, label: "LLM time-to-first-token" },
        { cls: "llm_stream", ms: llmStream, label: "token stream" }
    ];
    var denominator = total || (stt + ret + llmTotal) || 1;
    segs.forEach(function(seg) {
        if (seg.ms <= 0) return;
        var div = document.createElement("div");
        div.className = "latency-seg " + seg.cls;
        div.style.width = ((seg.ms / denominator) * 100).toFixed(2) + "%";
        div.title = seg.label + ": " + seg.ms + "ms";
        bar.appendChild(div);
    });

    var target = document.getElementById("latencyTarget");
    var lines = [];
    if (ret > 0) {
        var pass = ret < LATENCY_TARGET_MS;
        lines.push("retrieval: <strong class=\"" + (pass ? "pass" : "fail") + "\">" +
            ret + "ms - " + (pass ? "UNDER" : "OVER") + " the " + LATENCY_TARGET_MS + "ms target</strong>");
    }
    if (llmTtft) lines.push("LLM time to first token: <strong class=\"pass\">" + llmTtft + "ms</strong>");
    if (llmStream > 0) lines.push("token stream completion: <strong>" + llmStream + "ms</strong>");
    target.innerHTML = lines.join("<br/>");

    var latencyList = document.getElementById("latencyList");
    latencyList.innerHTML = "";
    var STAGE_LABEL = {
        stt_ms: "speech-to-text (network)",
        retrieve_ms: "chunking + vector retrieval",
        guardrail_ms: "guardrail checks",
        llm_first_token_ms: "LLM time to first token",
        llm_total_ms: "LLM generation (full)",
        total_ms: "total round trip"
    };
    var ORDER = ["stt_ms", "retrieve_ms", "guardrail_ms", "llm_first_token_ms", "llm_total_ms", "total_ms"];
    ORDER.forEach(function(key) {
        var val = timings[key];
        if (val === undefined || val === null) return;
        var li = document.createElement("li");
        li.innerHTML = "<span class=\"dot-" + key + "\">" + (STAGE_LABEL[key] || key) + "</span><span>" + val + "ms</span>";
        latencyList.appendChild(li);
    });

    var totalEl = document.getElementById("totalLatency");
    var totalMeta = document.getElementById("totalLatencyMeta");
    if (totalEl) totalEl.textContent = (total || 0) + " ms";
    if (totalMeta) {
        var g = data.guardrail || {};
        totalMeta.textContent = "QUERY " + (data.query_id || "?").slice(0, 8) +
            " · CACHED: " + (data.cached ? "YES" : "NO") +
            " · GROUNDED: " + (g.grounded ? "TRUE" : "FALSE");
    }
}

/* ── Live stage strip state machine ────────────────────── */
function renderStages(data) {
    var timings = {};
    var lat = data.latency || {};
    if (lat.stt_ms) timings.listening = lat.stt_ms;
    if (lat.retrieve_ms) timings.retrieving = lat.retrieve_ms;
    if (lat.llm_total_ms) timings.answering = lat.llm_total_ms;
    var guardrail = data.guardrail || {};
    resetStages();

    if (timings.listening) {
        setStage("listening", "done", timings.listening + "ms");
        setStage("transcribing", "done", "recognized");
    } else {
        setStage("listening", "skipped", "text input");
        setStage("transcribing", "skipped", "text input");
    }

    if (guardrail.unsafe || guardrail.off_topic) {
        setStage("grounding", "failed", guardrail.unsafe ? "BLOCKED" : "OUT OF CORPUS");
        setStage("retrieving", "skipped", "-");
        setStage("answering", "skipped", "-");
        showToast(
            guardrail.unsafe
                ? "SAFETY BOUNDARY VIOLATION - query blocked."
                : "QUERY OUT OF CORPUS - no supporting evidence found.",
            null
        );
        return;
    }
    setStage("grounding", "done", "passed");

    if (!timings.retrieving) {
        setStage("retrieving", "skipped", "-");
        setStage("answering", "skipped", "-");
        return;
    }
    setStage("retrieving", "done", timings.retrieving + "ms");

    if (!guardrail.passed) {
        setStage("grounding", "failed", "ungrounded");
    } else {
        setStage("grounding", "done", "passed");
    }

    if (timings.answering) {
        var ttft = lat.llm_first_token_ms ? lat.llm_first_token_ms + "ms -> 1st tok" : timings.answering + "ms";
        setStage("answering", "done", ttft);
    } else {
        setStage("answering", "skipped", "-");
    }
}