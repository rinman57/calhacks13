if (!globalThis.__ytExtractLoaded) {
  globalThis.__ytExtractLoaded = true;

  const SERVER = "http://127.0.0.1:8765";
  const FOLLOW_KEY = "followVideos";
  const STAGES = ["metadata", "transcript", "audio", "speech", "claims", "research", "assess"];

  const state = {
    videoId: null,
    follow: false,
    jobId: null,
    poll: 0,
    serverOk: false,
    panelOpen: false,
    cards: [],
    activeCueId: null,
    starting: false,
    attempted: false,
    requestSeq: 0,
    processing: false,
    manuallyOpened: new Set(),
    manuallyClosed: new Set(),
  };
  const jobsByVideo = new Map();
  const cardNodes = new Map();
  let player = null;

  function addText(parent, tag, text) {
    const node = document.createElement(tag);
    node.textContent = text;
    parent.appendChild(node);
    return node;
  }

  const logoUrl = chrome.runtime.getURL("icons/logo.png");
  const panel = document.createElement("aside");
  panel.id = "yt-extract-panel";
  panel.hidden = true;
  const brand = document.createElement("div");
  brand.className = "yt-extract-brand";
  const logo = document.createElement("img");
  logo.className = "yt-extract-logo";
  logo.alt = "";
  logo.src = logoUrl;
  brand.appendChild(logo);
  addText(brand, "h2", "Video checks");
  panel.appendChild(brand);
  const close = document.createElement("button");
  close.type = "button"; close.className = "yt-extract-close"; close.textContent = "×";
  close.setAttribute("aria-label", "Close claim list");
  close.addEventListener("click", () => setPanelOpen(false));
  panel.appendChild(close);
  const idLine = addText(panel, "p", "");
  idLine.className = "yt-extract-id";
  const note = addText(
    panel,
    "p",
    "Saves the transcript, then checks its claims against outside sources.",
  );
  note.className = "yt-extract-note";
  const button = document.createElement("button");
  button.className = "yt-extract-go";
  button.type = "button";
  button.textContent = "Extract";
  button.disabled = true;
  panel.appendChild(button);
  const followLabel = document.createElement("label");
  followLabel.className = "yt-extract-follow";
  const followBox = document.createElement("input");
  followBox.type = "checkbox";
  followLabel.appendChild(followBox);
  addText(followLabel, "span", "Run this on each video played in this tab");
  panel.appendChild(followLabel);
  const offline = addText(panel, "p", "");
  offline.className = "yt-extract-offline";
  offline.hidden = true;
  const statusLine = addText(panel, "p", "");
  statusLine.className = "yt-extract-status";
  statusLine.setAttribute("role", "status");
  statusLine.hidden = true;
  const rows = document.createElement("ul");
  rows.hidden = true;
  const rowValues = {};
  for (const [stage, label] of [
    ["metadata", "Details"],
    ["transcript", "Captions"],
    ["audio", "Audio"],
    ["speech", "Speech-to-text"],
    ["claims", "Claims"],
    ["research", "Research"],
    ["assess", "Verdicts"],
  ]) {
    const item = document.createElement("li");
    addText(item, "span", label);
    rowValues[stage] = addText(item, "span", "");
    rows.appendChild(item);
  }
  panel.appendChild(rows);
  const warning = addText(panel, "p", "");
  warning.className = "yt-extract-warning";
  warning.hidden = true;
  const preview = document.createElement("pre");
  preview.className = "yt-extract-preview";
  preview.hidden = true;
  panel.appendChild(preview);
  const verdicts = document.createElement("div");
  verdicts.className = "yt-extract-verdicts";
  verdicts.hidden = true;
  panel.appendChild(verdicts);
  const settings = document.createElement("details");
  settings.className = "yt-extract-settings";
  addText(settings, "summary", "Video check settings");
  for (const node of [note, button, followLabel, offline, rows]) settings.appendChild(node);
  panel.appendChild(settings);
  const launcher = document.createElement("button");
  launcher.id = "yt-extract-launcher"; launcher.type = "button"; launcher.hidden = true;
  const launcherLogo = document.createElement("img");
  launcherLogo.className = "yt-extract-logo";
  launcherLogo.alt = "";
  launcherLogo.src = logoUrl;
  const launcherLabel = document.createElement("span");
  launcherLabel.className = "yt-extract-launcher-label";
  launcherLabel.textContent = "Video checks";
  launcher.append(launcherLogo, launcherLabel);
  launcher.setAttribute("aria-controls", "yt-extract-panel");
  launcher.setAttribute("aria-expanded", "false");
  launcher.setAttribute("aria-haspopup", "true");
  launcher.addEventListener("click", () => setPanelOpen(!state.panelOpen));
  const announcement = document.createElement("span");
  announcement.className = "yt-extract-sr-only";
  announcement.setAttribute("aria-live", "polite");
  panel.addEventListener("keydown", (event) => {
    if (event.key === "Escape") { event.preventDefault(); setPanelOpen(false); }
  });
  document.documentElement.append(launcher, panel, announcement);

  function setPanelOpen(open) {
    state.panelOpen = open;
    panel.hidden = !open || !state.videoId;
    launcher.setAttribute("aria-expanded", String(open));
    if (open) {
      close.focus();
      cardNodes.get(state.activeCueId)?.scrollIntoView({ block: "nearest" });
    } else launcher.focus();
  }

  function syncCue() {
    const next = document.querySelector("video");
    if (next !== player) {
      if (player) for (const name of ["timeupdate", "seeked", "loadedmetadata"]) player.removeEventListener(name, syncCue);
      player = next;
      if (player) for (const name of ["timeupdate", "seeked", "loadedmetadata"]) player.addEventListener(name, syncCue);
    }
    const time = player?.currentTime || 0;
    const active = player ? [...state.cards].reverse().find((card) => {
      const start = Number(card.start_seconds);
      const end = Math.max(Number(card.end_seconds) || start, start + 10);
      return Number.isFinite(start) && time >= start && time <= end;
    }) : null;
    const changed = state.activeCueId !== (active?.claim_id || null);
    // A manual close suppresses auto-opening for this encounter only. Seeking
    // away and back should open the current timestamp again.
    if (changed) state.manuallyClosed.clear();
    state.activeCueId = active?.claim_id || null;
    launcher.classList.toggle("is-cue", Boolean(active));
    launcher.classList.toggle("has-flags", state.cards.length > 0);
    launcherLabel.textContent = active ? `${clock(active.start_seconds)} · ${capWords(active.headline || flagLabel(active), 8)}` : state.cards.length ? `${state.cards.length} flagged claim${state.cards.length === 1 ? "" : "s"}${state.processing ? " · checking more…" : ""}` : state.processing ? "Checking claims…" : "Video checks";
    const label = active ? `Flag at ${clock(active.start_seconds)}. ${active.headline || flagLabel(active)}. Open available claims.` : "Open video checks and available claims.";
    launcher.setAttribute("aria-label", label + (state.processing ? " Claim information is still being generated." : ""));
    launcher.setAttribute("aria-busy", String(state.processing));
    launcher.classList.toggle("is-loading", state.processing);
    launcher.title = launcher.getAttribute("aria-label");
    if (changed) announcement.textContent = active ? launcher.getAttribute("aria-label") : "";
    for (const [id, node] of cardNodes) {
      node.classList.toggle("at-playback", id === state.activeCueId);
      const shouldOpen = state.manuallyOpened.has(id) || (id === state.activeCueId && !state.manuallyClosed.has(id));
      if (node.open !== shouldOpen) node.open = shouldOpen;
    }
  }

  function flagLabel(card) {
    return card.verdict === "contradicted" ? "Evidence contradicts this" : "Possibly misleading";
  }


  button.addEventListener("click", () => startExtract(state.videoId));
  followBox.addEventListener("change", () => {
    state.follow = followBox.checked;
    saveFollow(state.follow);
    if (state.follow && state.videoId && !state.jobId) startExtract(state.videoId);
  });

  chromeListen();
  restoreFollow().then(() => {
    followBox.checked = state.follow;
    watchNavigation();
    refreshPage();
    checkServer();
    setInterval(checkServer, 4000);
    setInterval(syncCue, 250);
  });

  function chromeListen() {
    if (!inExtension()) return;
    chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
      if (message?.type === "show-job" && message.job?.job_id) {
        adoptJob(message.job);
        sendResponse({ ok: true });
        return;
      }
      if (message?.type === "extract-current") {
        if (!state.videoId) {
          sendResponse({ ok: false, error: "Open a YouTube video first." });
          return;
        }
        startExtract(state.videoId);
        sendResponse({ ok: true, videoId: state.videoId });
      }
    });
  }

  function watchNavigation() {
    let href = location.href;
    document.addEventListener("yt-navigate-finish", refreshPage);
    setInterval(() => {
      if (location.href !== href) {
        href = location.href;
        refreshPage();
      }
    }, 500);
  }

  function refreshPage() {
    const videoId = currentVideoId();
    launcher.hidden = videoId === null;
    panel.hidden = videoId === null || !state.panelOpen;
    if (videoId === null) {
      stopPoll();
      state.videoId = null;
      state.jobId = null;
      state.requestSeq += 1; state.panelOpen = false; state.starting = false;
      showIdle();
      return;
    }
    if (videoId === state.videoId) return;
    state.videoId = videoId;
    state.requestSeq += 1; state.panelOpen = false; panel.hidden = true;
    state.starting = false; state.attempted = false;
    showIdle();
    idLine.textContent = pageTitle(videoId);
    const existing = jobsByVideo.get(videoId);
    if (existing) {
      state.jobId = existing;
      pollJob();
      return;
    }
    stopPoll();
    state.jobId = null;
    showIdle();
    if (state.follow) startExtract(videoId);
  }

  function adoptJob(job) {
    const videoId = job.video_id || currentVideoId();
    if (!videoId || videoId !== currentVideoId()) return;
    if (state.jobId !== job.job_id) showIdle();
    state.videoId = videoId;
    state.serverOk = true;
    state.jobId = job.job_id;
    jobsByVideo.set(videoId, job.job_id);
    launcher.hidden = false;
    setPanelOpen(true);
    idLine.textContent = job.result?.title || pageTitle(videoId);
    showJob(job);
    if (job.status === "running") pollJob();
  }

  function startExtract(videoId) {
    if (!videoId || !state.serverOk || state.starting) return;
    const requestSeq = ++state.requestSeq;
    stopPoll();
    state.jobId = null;
    showIdle();
    state.starting = true; state.attempted = true;
    setBusy(true);
    showRunning("starting");
    requestExtract(videoId)
      .then((job) => {
        if (state.videoId !== videoId || state.requestSeq !== requestSeq) return;
        state.starting = false;
        state.jobId = job.job_id;
        jobsByVideo.set(videoId, job.job_id);
        showJob(job);
        pollJob();
      })
      .catch((error) => {
        if (state.videoId !== videoId || state.requestSeq !== requestSeq) return;
        state.starting = false;
        showFailure(error instanceof Error ? error.message : String(error));
      });
  }

  function pollJob() {
    stopPoll();
    const videoId = state.videoId;
    const jobId = state.jobId;
    if (!jobId) return;
    const tick = () => {
      requestJob(jobId)
        .then((job) => {
          if (state.videoId !== videoId || state.jobId !== jobId) return;
          showJob(job);
          if (job.status === "running") state.poll = window.setTimeout(tick, 1000);
        })
        .catch((error) => {
          if (state.videoId !== videoId || state.jobId !== jobId) return;
          showFailure(error instanceof Error ? error.message : String(error));
        });
    };
    tick();
  }

  function showIdle() {
    setBusy(false);
    statusLine.hidden = true;
    rows.hidden = true;
    warning.hidden = true;
    preview.hidden = true;
    verdicts.hidden = true;
    verdicts.replaceChildren();
    state.cards = []; cardNodes.clear(); state.activeCueId = null;
    state.manuallyOpened.clear(); state.manuallyClosed.clear();
    announcement.textContent = "";
    syncCue();
    clearRows();
  }

  function showRunning(stage) {
    statusLine.hidden = false;
    rows.hidden = true;
    warning.hidden = true;
    preview.hidden = true;
    verdicts.hidden = state.cards.length === 0;
    statusLine.textContent = stageLabel(stage);
    for (const name of STAGES) {
      const index = STAGES.indexOf(name);
      const current = STAGES.indexOf(stage);
      if (stage === "starting" || current < 0) setRow(name, "Waiting");
      else if (index < current) setRow(name, "Done");
      else if (index === current) setRow(name, "Working");
      else setRow(name, "Waiting");
    }
  }

  function showJob(job) {
    if (job.status === "running") {
      setBusy(true);
      showRunning(job.stage);
      if (job.result) {
        if (job.result.title) idLine.textContent = job.result.title;
        renderVerdicts(job.result);
      }
      return;
    }
    setBusy(false);
    statusLine.hidden = false;
    rows.hidden = true;
    if (job.status === "error" && !job.result) {
      showFailure(job.error || "Extraction failed.");
      return;
    }
    const result = job.result || {};
    const title = result.title || pageTitle(job.video_id);
    idLine.textContent = title;
    statusLine.textContent = job.status === "done" ? "Saved" : "Finished with a problem";
    fillFinalRows(result);
    const captionNote = captionMessage(result);
    const demandNote = geminiDemandSpike(result)
      ? "The claim service is busy. Please try again shortly."
      : "";
    const verifyNote = verificationMessage(result);
    const visibleNote = verifyNote || demandNote || captionNote;
    warning.hidden = !visibleNote;
    warning.textContent = visibleNote;
    renderVerdicts(result);
    const lines = (result.verdicts || []).length
      ? []
      : ["No misinformation flags to show. This does not establish that every statement is true."];
    preview.hidden = lines.length === 0;
    preview.textContent = lines.join("\n");
  }

  function showFailure(message) {
    setBusy(false);
    statusLine.hidden = false;
    statusLine.textContent = "Could not extract";
    warning.hidden = false;
    warning.textContent = message;
    verdicts.hidden = state.cards.length === 0;
  }

  function fillFinalRows(result) {
    const files = result.files || {};
    setRow("metadata", files.metadata ? "Saved" : "Missing");
    const fromSpeech = result.transcript_source === "cartesia";
    setRow("transcript", !fromSpeech && files.transcript ? "Saved" : "Not saved");
    setRow("speech", fromSpeech ? "Saved" : files.transcript ? "Not needed" : "Not saved");
    setRow(
      "audio",
      files.audio ? formatBytes(files.audio.bytes) : files.transcript && !fromSpeech ? "Not needed" : "Missing",
    );
    setRow("claims", claimStatus(result));
    setRow("research", result.files?.evidence ? "Saved" : "Not saved");
    setRow(
      "assess",
      result.files?.verdict ? `${result.assessed_count || 0} assessed` : "Not saved",
    );
  }

  function claimStatus(result) {
    if (result.files?.claims) return `${result.checkable_count || 0} to research`;
    if (geminiDemandSpike(result)) return "Gemini demand spike";
    return "Not saved";
  }

  function geminiDemandSpike(result) {
    return /high demand|spikes in demand/i.test((result.warnings || []).join(" "));
  }

  function verificationMessage(result) {
    const notes = (result.warnings || []).filter((item) => typeof item === "string");
    if (notes.some((item) => /^(Research|Assessment) failed\b/.test(item))) {
      return "Some claims could not be checked. Available results are shown.";
    }
    if (notes.some((item) => /^Research skipped\b/.test(item))) {
      return "Research could not start. Check the local server setup.";
    }
    if (notes.some((item) => /^Claim check failed\b/.test(item))) {
      return "Claims could not be generated. Please try again.";
    }
    // Partial fetch diagnostics stay in the saved files, not the viewer UI.
    return "";
  }

  function renderVerdicts(result) {
    const cards = [...(result.verdicts || [])].sort((a, b) => Number(a.start_seconds || 0) - Number(b.start_seconds || 0));
    state.cards = cards;
    verdicts.hidden = cards.length === 0;
    const present = new Set(cards.map((card) => card.claim_id));
    for (const [id, node] of cardNodes) {
      if (!present.has(id)) { node.remove(); cardNodes.delete(id); }
    }
    for (const card of cards) {
      let article = cardNodes.get(card.claim_id);
      if (!article) {
        article = document.createElement("details");
        article.className = "yt-extract-verdict";
        article.dataset.claimId = card.claim_id;
        const summary = document.createElement("summary");
        summary.addEventListener("click", (event) => {
          // Handle user toggles explicitly so programmatic playback updates do
          // not accidentally pin every automatically opened section.
          event.preventDefault();
          const open = !article.open;
          if (open) {
            state.manuallyOpened.add(card.claim_id);
            state.manuallyClosed.delete(card.claim_id);
          } else {
            state.manuallyOpened.delete(card.claim_id);
            state.manuallyClosed.add(card.claim_id);
          }
          article.open = open;
        });
        const seek = document.createElement("button");
        seek.type = "button"; seek.className = "yt-extract-time";
        seek.addEventListener("click", (event) => {
          event.preventDefault(); event.stopPropagation();
          const current = state.cards.find((item) => item.claim_id === card.claim_id);
          if (current) seekTo(current.start_seconds);
        });
        summary.appendChild(seek);
        const description = document.createElement("span");
        description.className = "yt-extract-flag-description";
        summary.appendChild(description);
        article.appendChild(summary);
        const body = document.createElement("div"); body.className = "yt-extract-detail";
        article.appendChild(body);
        cardNodes.set(card.claim_id, article);
      }
      const signature = JSON.stringify(card);
      if (article.dataset.signature !== signature) {
        article.dataset.signature = signature;
        const seek = article.querySelector(".yt-extract-time");
        seek.textContent = clock(card.start_seconds);
        seek.setAttribute("aria-label", `Seek to ${clock(card.start_seconds)}`);
        const description = article.querySelector(".yt-extract-flag-description");
        description.textContent = card.headline || flagLabel(card);
        description.title = `${flagLabel(card)}: ${card.headline || ""}`;
        const body = article.querySelector(".yt-extract-detail");
        body.replaceChildren();
        addText(body, "p", flagLabel(card)).className = "yt-extract-label";
        addText(body, "blockquote", card.text || "");
        addParagraph(body, shortExplanation(card.explanation || card.context));
        const links = document.createElement("ul");
        for (const source of card.sources || []) {
          const href = safeHttpUrl(source.url);
          if (!href) continue;
          const item = document.createElement("li");
          const link = addText(item, "a", source.title || href);
          link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer";
          links.appendChild(item);
        }
        if (links.childNodes.length) body.appendChild(links);
      }
      // Moving existing nodes preserves their native expanded state and focus.
      const position = cards.indexOf(card);
      if (verdicts.children[position] !== article) verdicts.insertBefore(article, verdicts.children[position] || null);
    }
    syncCue();
  }

  function addParagraph(parent, text) {
    if (!text) return;
    const node = document.createElement("p");
    node.textContent = text;
    parent.appendChild(node);
  }

  function shortExplanation(value) {
    const text = String(value || "").replace(/\s+/g, " ").trim();
    if (!text) return "";
    if (typeof Intl.Segmenter === "function") {
      const sentences = new Intl.Segmenter("en", { granularity: "sentence" }).segment(text);
      return Array.from(sentences, (item) => item.segment).slice(0, 3).join("").trim();
    }
    return text.split(/(?<=[.!?])\s+/).slice(0, 3).join(" ");
  }

  function seekTo(seconds) {
    const video = document.querySelector("video");
    const time = Number(seconds);
    if (!video || !Number.isFinite(time)) return;
    video.currentTime = time;
    video.play().catch(() => {});
  }

  function capWords(text, limit) {
    const words = String(text || "").trim().split(/\s+/).filter(Boolean);
    if (words.length <= limit) return words.join(" ");
    const clipped = words.slice(0, limit);
    const joined = clipped.join(" ");
    const end = Math.max(joined.lastIndexOf("."), joined.lastIndexOf("!"), joined.lastIndexOf("?"));
    if (end >= 0 && joined.slice(0, end + 1).trim().split(/\s+/).length >= 8) {
      return joined.slice(0, end + 1);
    }
    return `${joined}…`;
  }

  function clock(seconds) {
    const total = Number.isFinite(Number(seconds)) ? Math.max(0, Math.round(Number(seconds))) : 0;
    const minutes = Math.floor(total / 60);
    const secs = total % 60;
    return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  function safeHttpUrl(value) {
    try {
      const url = new URL(value);
      if (url.protocol === "https:" || url.protocol === "http:") return url.href;
    } catch {
      return "";
    }
    return "";
  }

  function setRow(stage, text) {
    rowValues[stage].textContent = text;
  }

  function clearRows() {
    for (const name of STAGES) setRow(name, "");
  }

  function setBusy(busy) {
    const changed = state.processing !== busy;
    state.processing = busy;
    statusLine.classList.toggle("is-loading", busy);
    panel.classList.toggle("is-loading", busy);
    button.disabled = busy || !state.serverOk;
    button.textContent = busy ? "Extracting" : "Extract";
    syncCue();
    if (changed && state.activeCueId) announcement.textContent = launcher.getAttribute("aria-label");
  }

  async function checkServer() {
    try {
      const health = await requestHealth();
      state.serverOk = Boolean(health.ok);
    } catch {
      state.serverOk = false;
    }
    offline.hidden = state.serverOk;
    offline.textContent = state.serverOk
      ? ""
      : "Start the extractor first: python -m extract.server";
    if (!state.jobId && !state.starting) button.disabled = !state.serverOk;
    if (state.serverOk && state.follow && state.videoId && !state.jobId && !state.starting && !state.attempted) startExtract(state.videoId);
  }

  function requestHealth() {
    return callServer({ type: "health" }, "/health");
  }

  function requestExtract(videoId) {
    const url = `https://www.youtube.com/watch?v=${videoId}`;
    return callServer({ type: "extract", url }, "/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
  }

  function requestJob(jobId) {
    return callServer({ type: "job", jobId }, `/jobs/${encodeURIComponent(jobId)}`);
  }

  async function callServer(message, path, init) {
    if (inExtension()) {
      const response = await chrome.runtime.sendMessage(message);
      if (response && response.ok === false) throw new Error(response.error || "Extractor request failed.");
      if (!response) throw new Error("The extension did not answer.");
      return response;
    }
    const response = await fetch(`${SERVER}${path}`, init);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok && response.status !== 202) {
      throw new Error(payload.error || `Extractor returned ${response.status}.`);
    }
    return payload;
  }

  function captionMessage(result) {
    if (result.files?.transcript) return "";
    const warnings = (result.warnings || []).join(" ");
    const languages = result.caption_languages || {};
    const listed = [...new Set([...(languages.manual || []), ...(languages.automatic || [])])];
    if (/429|IpBlocked|blocked/i.test(warnings) && listed.length) {
      return `Captions are listed (${listed.join(", ")}), but YouTube blocked the download.`;
    }
    if (listed.length) return "Captions were listed, but none were saved.";
    return "No captions were saved.";
  }

  function stageLabel(stage) {
    if (stage === "metadata") return "Reading the video details";
    if (stage === "transcript") return "Fetching captions";
    if (stage === "audio") return "Saving audio";
    if (stage === "speech") return "Transcribing the audio";
    if (stage === "claims") return "Flagging claims";
    if (stage === "research") return "Researching claims";
    if (stage === "assess") return "Assessing evidence";
    return "Starting extraction";
  }

  function pageTitle(videoId) {
    const title = document.title.replace(/ - YouTube$/, "").trim();
    return title && title !== "YouTube" ? title : videoId;
  }

  function currentVideoId() {
    const url = new URL(location.href);
    const host = url.hostname.replace(/^www\./, "");
    if (host !== "youtube.com" && host !== "m.youtube.com") return null;
    if (url.pathname === "/watch") return url.searchParams.get("v");
    const parts = url.pathname.split("/").filter(Boolean);
    if (["shorts", "live", "embed", "v"].includes(parts[0]) && parts[1]) return parts[1];
    return null;
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function stopPoll() {
    if (state.poll) window.clearTimeout(state.poll);
    state.poll = 0;
  }

  function inExtension() {
    return typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.id;
  }

  function restoreFollow() {
    if (!inExtension()) return Promise.resolve();
    return chrome.storage.local.get(FOLLOW_KEY).then((stored) => {
      state.follow = stored[FOLLOW_KEY] !== false;
    });
  }

  function saveFollow(value) {
    if (!inExtension()) return;
    chrome.storage.local.set({ [FOLLOW_KEY]: value });
  }
}
