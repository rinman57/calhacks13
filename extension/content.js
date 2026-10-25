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
  };
  const jobsByVideo = new Map();

  function addText(parent, tag, text) {
    const node = document.createElement(tag);
    node.textContent = text;
    parent.appendChild(node);
    return node;
  }

  const panel = document.createElement("aside");
  panel.id = "yt-extract-panel";
  panel.hidden = true;
  addText(panel, "h2", "Extract this video");
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
  document.documentElement.appendChild(panel);

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
    panel.hidden = videoId === null;
    if (videoId === null) {
      stopPoll();
      state.videoId = null;
      state.jobId = null;
      return;
    }
    if (videoId === state.videoId) return;
    state.videoId = videoId;
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
    if (!videoId) return;
    state.videoId = videoId;
    state.serverOk = true;
    state.jobId = job.job_id;
    jobsByVideo.set(videoId, job.job_id);
    panel.hidden = false;
    idLine.textContent = job.result?.title || pageTitle(videoId);
    showJob(job);
    if (job.status === "running") pollJob();
  }

  function startExtract(videoId) {
    if (!videoId || !state.serverOk) return;
    stopPoll();
    state.jobId = null;
    setBusy(true);
    showRunning("starting");
    requestExtract(videoId)
      .then((job) => {
        if (state.videoId !== videoId) return;
        state.jobId = job.job_id;
        jobsByVideo.set(videoId, job.job_id);
        showJob(job);
        pollJob();
      })
      .catch((error) => {
        if (state.videoId !== videoId) return;
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
          if (state.videoId !== videoId) return;
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
    clearRows();
  }

  function showRunning(stage) {
    statusLine.hidden = false;
    rows.hidden = false;
    warning.hidden = true;
    preview.hidden = true;
    verdicts.hidden = true;
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
      return;
    }
    setBusy(false);
    statusLine.hidden = false;
    rows.hidden = false;
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
      ? "Gemini is experiencing a demand spike, so claims were not saved. Try again in a little while."
      : "";
    const verifyNote = verificationMessage(result);
    warning.hidden = !captionNote && !demandNote && !verifyNote && !(result.warnings || []).length;
    warning.textContent = verifyNote || demandNote || captionNote || (result.warnings || [])[0] || "";
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
    verdicts.hidden = true;
    verdicts.replaceChildren();
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
    return (result.warnings || []).find((item) => /^Research|^Assessment/.test(item)) || "";
  }

  function renderVerdicts(result) {
    verdicts.replaceChildren();
    const cards = result.verdicts || [];
    verdicts.hidden = cards.length === 0;
    for (const card of cards) {
      const article = document.createElement("article");
      article.className = "yt-extract-verdict";
      const heading = document.createElement("p");
      heading.className = "yt-extract-verdict-title";
      const seek = document.createElement("button");
      seek.type = "button";
      seek.textContent = clock(card.start_seconds);
      seek.addEventListener("click", () => seekTo(card.start_seconds));
      heading.appendChild(seek);
      heading.appendChild(document.createTextNode(` ${card.verdict || card.assessment_status || "Not assessed"}`));
      article.appendChild(heading);
      addParagraph(article, card.text);
      addParagraph(article, capWords(card.context, 30));
      if ((card.limitations || []).length) {
        const list = document.createElement("ul");
        for (const limitation of card.limitations) {
          const item = document.createElement("li");
          item.textContent = limitation;
          list.appendChild(item);
        }
        article.appendChild(list);
      }
      const links = document.createElement("ul");
      for (const source of card.sources || []) {
        const href = safeHttpUrl(source.url);
        if (!href) continue;
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = source.title || href;
        item.appendChild(link);
        links.appendChild(item);
      }
      if (links.childNodes.length) article.appendChild(links);
      if (card.error) addParagraph(article, card.error);
      verdicts.appendChild(article);
    }
  }

  function addParagraph(parent, text) {
    if (!text) return;
    const node = document.createElement("p");
    node.textContent = text;
    parent.appendChild(node);
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
    button.disabled = busy || !state.serverOk;
    button.textContent = busy ? "Extracting" : "Extract";
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
    if (!state.jobId) button.disabled = !state.serverOk;
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
