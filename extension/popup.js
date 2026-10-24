const status = document.querySelector("#status");
const button = document.querySelector("#extract");

chrome.runtime.sendMessage({ type: "health" }).then((health) => {
  if (!health?.ok) {
    status.className = "warn";
    status.textContent = "Start the extractor first: python -m extract.server";
    return;
  }
  status.textContent = "This saves the open YouTube video. Playback can keep going.";
  button.disabled = false;
}).catch(() => {
  status.className = "warn";
  status.textContent = "Start the extractor first: python -m extract.server";
});

button.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id || !tab.url?.includes("youtube.com")) {
    status.className = "warn";
    status.textContent = "Open a YouTube video, then come back to this button.";
    return;
  }
  button.disabled = true;
  try {
    const response = await chrome.runtime.sendMessage({
      type: "extract-tab",
      tabId: tab.id,
      url: tab.url,
    });
    if (!response?.ok) throw new Error(response?.error || "Extraction did not start.");
    status.className = "";
    status.textContent = "Saving this video. Progress shows on the page.";
  } catch (error) {
    status.className = "warn";
    status.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    button.disabled = false;
  }
});
