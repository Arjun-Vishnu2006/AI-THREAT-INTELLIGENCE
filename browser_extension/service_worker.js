const API = 'http://127.0.0.1:8000/api/v1';

async function checkUrl(value) {
  let url;
  try { url = new URL(value); } catch { return { known: [], error: 'Invalid URL' }; }
  if (!['http:', 'https:'].includes(url.protocol)) return { known: [], skipped: true };
  const indicators = [{ type: 'url', value: url.href }];
  if (url.hostname) indicators.push({ type: 'domain', value: url.hostname });
  const response = await fetch(`${API}/intel/check`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ indicators }),
  });
  if (!response.ok) throw new Error(`Local API returned HTTP ${response.status}`);
  const result = await response.json();
  return { known: (result.assessments || []).filter((item) => item.knowledge_status === 'KNOWN'), skipped: false };
}

function notifyMatch(urlValue, known) {
  if (!known.length) return;
  let host = 'URL';
  try { host = new URL(urlValue).hostname || host; } catch { /* Keep the generic label. */ }
  const label = known.map((item) => item.indicator_type).join(', ');
  chrome.notifications.create({
    type: 'basic', iconUrl: 'icon.png', title: 'Local IOC match',
    message: `${host} matched the configured local dataset (${label}). Review the source before proceeding.`,
    priority: 2,
  });
}

async function backgroundCheck(urlValue) {
  try {
    const result = await checkUrl(urlValue);
    notifyMatch(urlValue, result.known);
  } catch (error) {
    console.debug('Signal Trace local lookup unavailable:', error.message);
  }
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'signal-trace-check-link', title: 'Check link with Signal Trace', contexts: ['link'],
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId === 'signal-trace-check-link' && info.linkUrl) backgroundCheck(info.linkUrl);
});

function registerOptionalListeners() {
  if (chrome.webNavigation?.onCommitted && !chrome.webNavigation.onCommitted.hasListener?.(onCommitted)) {
    chrome.webNavigation.onCommitted.addListener(onCommitted);
  }
  if (chrome.downloads?.onCreated && !chrome.downloads.onCreated.hasListener?.(onDownloadCreated)) {
    chrome.downloads.onCreated.addListener(onDownloadCreated);
  }
}

async function onCommitted(details) {
  if (details.frameId !== 0) return;
  if (!(await chrome.permissions.contains({ permissions: ['webNavigation'] }))) return;
  let parsed;
  try { parsed = new URL(details.url); } catch { return; }
  if (!['http:', 'https:'].includes(parsed.protocol) || ['localhost', '127.0.0.1'].includes(parsed.hostname)) return;
  backgroundCheck(details.url);
}

async function onDownloadCreated(download) {
  if (!(await chrome.permissions.contains({ permissions: ['downloads'] }))) return;
  if (download.url) backgroundCheck(download.url);
}

registerOptionalListeners();
chrome.permissions.onAdded.addListener(registerOptionalListeners);
