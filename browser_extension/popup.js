const API = 'http://127.0.0.1:8000/api/v1';
const resultBox = document.querySelector('#result');

function show(text, kind = '') {
  resultBox.className = `result ${kind}`;
  resultBox.textContent = text;
}

async function check(value) {
  const url = new URL(value);
  const indicators = [{ type: 'url', value: url.href }, { type: 'domain', value: url.hostname }];
  const [analysisResponse, intelResponse] = await Promise.all([
    fetch(`${API}/analyze/url`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ submitted_value: value }) }),
    fetch(`${API}/intel/check`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ indicators }) }),
  ]);
  if (!analysisResponse.ok || !intelResponse.ok) throw new Error('Could not reach the local API. Is Uvicorn running on port 8000?');
  const [analysis, intelligence] = await Promise.all([analysisResponse.json(), intelResponse.json()]);
  const matches = (intelligence.assessments || []).filter((item) => item.knowledge_status === 'KNOWN');
  const host = analysis.features?.['url.host'] || url.hostname;
  const details = `${host} · ${Object.keys(analysis.features || {}).length} extracted URL features · ${matches.length ? `KNOWN in local dataset (${matches.map((item) => item.indicator_type).join(', ')})` : 'no exact local IOC match'}. This is not a verdict about safety.`;
  show(details, matches.length ? 'match' : '');
}

document.querySelector('#check-page').addEventListener('click', async () => {
  show('Checking…');
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.url) throw new Error('The current tab URL is unavailable.');
    await check(tab.url);
  } catch (error) { show(error.message, 'error'); }
});

async function updateButtons() {
  const pages = await chrome.permissions.contains({ permissions: ['webNavigation'], origins: ['http://*/*', 'https://*/*'] });
  const downloads = await chrome.permissions.contains({ permissions: ['downloads'] });
  document.querySelector('#toggle-pages').textContent = pages ? 'Disable page checks' : 'Enable page checks';
  document.querySelector('#toggle-downloads').textContent = downloads ? 'Disable download URL checks' : 'Enable download URL checks';
}

document.querySelector('#toggle-pages').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  try {
    const granted = await chrome.permissions.contains({ permissions: ['webNavigation'], origins: ['http://*/*', 'https://*/*'] });
    if (granted) await chrome.permissions.remove({ permissions: ['webNavigation'], origins: ['http://*/*', 'https://*/*'] });
    else await chrome.permissions.request({ permissions: ['webNavigation'], origins: ['http://*/*', 'https://*/*'] });
  } finally { button.disabled = false; await updateButtons(); }
});

document.querySelector('#toggle-downloads').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  try {
    const granted = await chrome.permissions.contains({ permissions: ['downloads'] });
    if (granted) await chrome.permissions.remove({ permissions: ['downloads'] });
    else await chrome.permissions.request({ permissions: ['downloads'] });
  } finally { button.disabled = false; await updateButtons(); }
});

updateButtons();
