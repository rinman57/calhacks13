const SERVER = "http://127.0.0.1:8765";

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  handle(message)
    .then(sendResponse)
    .catch((error) => {
      sendResponse({ ok: false, error: error instanceof Error ? error.message : String(error) });
    });
  return true;
});

async function handle(message) {
  if (!message || message.type === "health") {
    return getJson("/health");
  }
  if (message.type === "extract") {
    return postJson("/extract", { url: message.url });
  }
  if (message.type === "extract-tab") {
    if (!message.url || !String(message.url).includes("youtube.com/")) {
      return { ok: false, error: "Open a YouTube video first." };
    }
    const job = await postJson("/extract", { url: message.url });
    if (message.tabId) await showOnPage(message.tabId, job);
    return { ok: true, job };
  }
  if (message.type === "job") {
    return getJson(`/jobs/${encodeURIComponent(message.jobId)}`);
  }
  return { ok: false, error: "Unknown request." };
}

async function showOnPage(tabId, job) {
  if (await notifyPage(tabId, job)) return;
  await chrome.scripting.insertCSS({ target: { tabId }, files: ["content.css"] });
  await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
  await notifyPage(tabId, job);
}

function notifyPage(tabId, job) {
  return chrome.tabs.sendMessage(tabId, { type: "show-job", job }).then(
    (response) => response?.ok === true,
    () => false,
  );
}

async function getJson(path) {
  const response = await fetch(`${SERVER}${path}`);
  return readResponse(response);
}

async function postJson(path, body) {
  const response = await fetch(`${SERVER}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readResponse(response);
}

async function readResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok && response.status !== 202) {
    throw new Error(payload.error || `Extractor returned ${response.status}.`);
  }
  return payload;
}
