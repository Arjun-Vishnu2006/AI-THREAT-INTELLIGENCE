(() => {
  const API = '/api/v1';
  const $ = (selector) => document.querySelector(selector);
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  };
  const dateText = (value) => {
    if (!value) return 'Time unavailable';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
  };
  const short = (value, max = 60) => {
    const text = String(value ?? '');
    return text.length > max ? `${text.slice(0, max - 1)}…` : text;
  };
  let toastTimer;
  function toast(message) {
    const box = $('#toast');
    box.textContent = message;
    box.classList.add('visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => box.classList.remove('visible'), 4200);
  }
  async function api(path, options = {}) {
    const response = await fetch(`${API}${path}`, {
      ...options,
      headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(options.headers || {}) },
    });
    const raw = await response.text();
    let data;
    try { data = raw ? JSON.parse(raw) : null; } catch { data = raw; }
    if (!response.ok) {
      const detail = data && typeof data === 'object' ? data.detail : data;
      throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`);
    }
    return data;
  }
  function setHealth(ok, label) {
    $('#health-label').textContent = label;
    $('#health-dot').className = `pulse-dot ${ok ? 'ok' : 'bad'}`;
  }
  async function loadHealth() {
    try { const result = await api('/health'); setHealth(result.status === 'ok', result.status === 'ok' ? 'Connected' : 'Service issue'); }
    catch { setHealth(false, 'Not reachable'); }
  }
  function renderAlert(alert) {
    const row = el('article', 'alert-row');
    row.tabIndex = 0;
    row.setAttribute('role', 'button');
    const top = el('div', 'alert-row-top');
    top.append(el('span', 'alert-title', alert.title || 'Untitled alert'));
    top.append(el('span', `alert-severity ${alert.severity === 'medium' ? 'medium' : ''}`, alert.severity || 'alert'));
    const meta = el('div', 'alert-meta');
    meta.append(el('span', 'alert-status', alert.status || 'open'));
    meta.append(el('time', '', dateText(alert.triggered_at)));
    row.append(top, meta);
    const open = () => openAlert(alert.id);
    row.addEventListener('click', open);
    row.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); } });
    return row;
  }
  async function loadDashboard() {
    await loadHealth();
    try {
      const [alerts, sessions] = await Promise.all([api('/alerts?limit=200'), api('/sessions?limit=12')]);
      const openCount = alerts.filter((item) => item.status === 'open' || item.status === 'acknowledged').length;
      $('#stat-open-alerts').textContent = openCount;
      $('#sidebar-alert-count').textContent = openCount;
      $('#stat-analyses').textContent = sessions.length;
      const list = $('#recent-alerts');
      list.replaceChildren();
      if (!alerts.length) list.append(el('div', 'quiet', 'No alerts recorded. Recent session analyses will appear here.'));
      else alerts.slice(0, 6).forEach((alert) => list.append(renderAlert(alert)));

      let novelty = 0;
      let high = 0;
      const latestSessions = sessions.slice(0, 8);
      const histories = await Promise.all(latestSessions.map((session) => api(`/sessions/${encodeURIComponent(session.id)}/history`).catch(() => null)));
      for (const history of histories.filter(Boolean)) {
        const evidence = history.evidence || [];
        const newestNovelty = [...evidence].reverse().find((item) => item.kind === 'novelty_assessment');
        const newestExposure = [...evidence].reverse().find((item) => item.kind === 'exposure_assessment');
        if (newestNovelty && newestNovelty.payload?.novelty_status === 'NOVEL') novelty += 1;
        if ((newestExposure?.payload?.category || newestExposure?.payload?.prediction?.category) === 'HIGH') high += 1;
      }
      $('#stat-novelty').textContent = novelty;
      $('#stat-high-exposure').textContent = high;
    } catch (error) {
      $('#recent-alerts').replaceChildren(el('div', 'quiet', error.message));
      toast(error.message);
    }
  }
  function readAsBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
      reader.onload = () => resolve(String(reader.result).split(',')[1] || '');
      reader.readAsDataURL(file);
    });
  }
  let cameraStream = null;
  function stopCamera() {
    if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
    $('#camera-video').srcObject = null;
    $('#camera-tools').classList.add('hidden');
    $('#camera-open').classList.remove('hidden');
  }
  async function openCamera() {
    if (!navigator.mediaDevices?.getUserMedia) { toast('Camera capture is unavailable here. Use HTTPS or localhost and a supported browser.'); return; }
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
      $('#camera-video').srcObject = cameraStream;
      $('#camera-tools').classList.remove('hidden');
      $('#camera-open').classList.add('hidden');
      await $('#camera-video').play();
    } catch (error) { stopCamera(); toast(`Camera access was not available: ${error.message}`); }
  }
  async function captureCameraFrame() {
    const video = $('#camera-video');
    if (!video.videoWidth || !video.videoHeight) { toast('Camera is still starting. Try capture again in a moment.'); return; }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
    if (!blob) { toast('Could not capture a QR image.'); return; }
    const captured = new File([blob], 'camera-qr.jpg', { type: 'image/jpeg', lastModified: Date.now() });
    const transfer = new DataTransfer();
    transfer.items.add(captured);
    $('#qr-input').files = transfer.files;
    stopCamera();
    toast('Camera image added. Submit the investigation to decode and analyze it.');
  }
  async function sha256(file) {
    if (!crypto.subtle) return null;
    const bytes = await file.arrayBuffer();
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
  }
  async function addStoredArtifact(sessionId, kind, file, submittedValue) {
    const artifact = await api(`/sessions/${encodeURIComponent(sessionId)}/artifacts`, {
      method: 'POST', body: JSON.stringify({
        kind, name: file?.name || null, media_type: file?.type || null,
        content_hash: file ? await sha256(file).catch(() => null) : null,
        attributes: { intake: 'dashboard' },
      }),
    });
    const requestItem = { kind, artifact_id: artifact.id };
    if (kind === 'url' || kind === 'email') requestItem.submitted_value = submittedValue;
    else {
      requestItem.content_base64 = await readAsBase64(file);
      requestItem.name = file.name;
      requestItem.media_type = file.type || null;
    }
    return requestItem;
  }
  function showResult(message, kind = 'success') {
    const box = $('#analysis-result');
    box.className = `analysis-result ${kind}`;
    box.textContent = message;
    box.classList.remove('hidden');
  }
  async function submitAnalysis(event) {
    event.preventDefault();
    const url = $('#url-input').value.trim();
    const email = $('#email-input').value.trim();
    const file = $('#file-input').files[0];
    const qr = $('#qr-input').files[0];
    if (!url && !email && !file && !qr) { toast('Add at least one URL, email, file, or QR image.'); return; }
    const button = $('#analyze-button');
    button.disabled = true;
    button.querySelector('span:first-child').textContent = 'Analyzing…';
    $('#analysis-result').classList.add('hidden');
    try {
      const session = await api('/sessions', { method: 'POST', body: JSON.stringify({ source: 'dashboard', metadata: { interface: 'phase-8-dashboard' } }) });
      const inputs = [];
      if (url) inputs.push(await addStoredArtifact(session.id, 'url', null, url));
      if (email) inputs.push(await addStoredArtifact(session.id, 'email', null, email));
      if (file) inputs.push(await addStoredArtifact(session.id, 'file', file));
      if (qr) inputs.push(await addStoredArtifact(session.id, 'qr', qr));
      const result = await api('/risk/assess', { method: 'POST', body: JSON.stringify({ session_id: session.id, artifacts: inputs }) });
      const prediction = result.exposure_prediction || {};
      const category = prediction.category || prediction.exposure_category || 'Recorded';
      const risk = result.risk_assessment?.risk_level || '—';
      const alert = result.alert_id ? `Alert ${result.alert_id} was created.` : 'No alert was recommended by the configured evidence policy.';
      showResult(`Investigation saved. Exposure category: ${category}. Evidence risk: ${risk}. ${alert} Session: ${session.id}`);
      $('#session-id-input').value = session.id;
      await loadSession(session.id);
      await loadDashboard();
    } catch (error) {
      showResult(`${error.message} The session record may still exist. Check that the Phase 5 novelty and Phase 6 exposure models are registered, then retry with a new analysis.`, 'error');
    } finally {
      button.disabled = false;
      button.querySelector('span:first-child').textContent = 'Run investigation';
    }
  }
  function latest(history, kind) {
    return [...(history.evidence || [])].reverse().find((item) => item.kind === kind);
  }
  function renderSummary(history) {
    const risk = latest(history, 'risk_assessment')?.payload || {};
    const exposure = latest(history, 'exposure_assessment')?.payload || {};
    const novelty = latest(history, 'novelty_assessment')?.payload || {};
    const values = [
      ['SESSION', history.session_id],
      ['EVIDENCE RISK', risk.risk_level || 'Not assessed'],
      ['EXPOSURE CATEGORY', exposure.category || exposure.prediction?.category || 'Not assessed'],
      ['NOVELTY', novelty.novelty_status || novelty.status || (novelty.novelty ? 'Assessed' : 'Not assessed')],
    ];
    const parent = $('#investigation-summary');
    parent.replaceChildren();
    for (const [name, value] of values) {
      const chip = el('div', 'summary-chip');
      chip.append(el('span', '', name), el('strong', '', value));
      parent.append(chip);
    }
  }
  function renderNovelty(history) {
    const holder = $('#novelty-bars');
    holder.replaceChildren();
    const record = latest(history, 'novelty_assessment');
    const payload = record?.payload || {};
    const layers = payload.layers || payload.layer_results || {};
    if (!record || !Object.keys(layers).length) { holder.append(el('p', 'quiet', 'No novelty layer evidence recorded for this session.')); return; }
    for (const [name, dataValue] of Object.entries(layers)) {
      const data = dataValue || {};
      const p = Number(data.p_value ?? data.p_value_adjusted ?? data.conformal_p_value);
      const row = el('div', 'novelty-row');
      row.append(el('span', 'novelty-name', name.replaceAll('_', ' ')));
      const track = el('div', 'novelty-track');
      const fill = el('div', 'novelty-fill');
      const strength = Number.isFinite(p) ? Math.max(5, Math.min(100, -Math.log10(Math.max(p, 0.000001)) * 20)) : 0;
      fill.style.width = `${strength}%`;
      track.append(fill);
      row.append(track, el('span', 'novelty-p', Number.isFinite(p) ? `p=${p.toPrecision(2)}` : 'n/a'));
      holder.append(row);
      const evidence = data.evidence || data.features || [];
      const detail = typeof evidence === 'string' ? evidence : Array.isArray(evidence) ? evidence.map((item) => typeof item === 'string' ? item : item.feature || item.name || JSON.stringify(item)).slice(0, 3).join(' · ') : Object.keys(evidence).slice(0, 4).join(' · ');
      if (detail) holder.append(el('div', 'layer-detail', short(detail, 180)));
    }
  }
  const svgNS = 'http://www.w3.org/2000/svg';
  function svg(tag, attrs = {}, text) {
    const node = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = String(text);
    return node;
  }
  function renderGraph(history) {
    const canvas = $('#graph-canvas');
    canvas.replaceChildren();
    const payload = latest(history, 'graph_assessment')?.payload || {};
    const entities = payload.entities || [];
    const relationships = payload.relationships || [];
    if (!entities.length) {
      canvas.append(el('div', 'graph-empty', 'No graph entities recorded for this session.'));
      $('#graph-caption').textContent = 'Waiting for graph evidence';
      return;
    }
    const width = Math.max(560, Math.min(entities.length * 130, 1100));
    const height = 240;
    const positions = entities.map((entity, index) => {
      const angle = -Math.PI / 2 + 2 * Math.PI * index / entities.length;
      return { x: width / 2 + Math.cos(angle) * Math.min(width * .37, 350), y: height / 2 + Math.sin(angle) * 78 };
    });
    const byIdentity = new Map();
    entities.forEach((entity, index) => {
      [entity.id, entity.value, entity.normalized_value].filter(Boolean).forEach((key) => byIdentity.set(String(key), index));
    });
    const graph = svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': 'Evidence relationship graph' });
    for (const relation of relationships) {
      const source = relation.source ?? relation.source_id ?? relation.from;
      const target = relation.target ?? relation.target_id ?? relation.to;
      const a = byIdentity.get(String(source));
      const b = byIdentity.get(String(target));
      if (a === undefined || b === undefined) continue;
      const p1 = positions[a], p2 = positions[b];
      graph.append(svg('line', { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y, class: 'edge-line' }));
      graph.append(svg('text', { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 - 4, class: 'edge-label' }, relation.type || relation.kind || relation.relationship || 'RELATED'));
    }
    entities.forEach((entity, index) => {
      const point = positions[index];
      const type = entity.type || entity.kind || 'entity';
      const colors = type === 'artifact' ? ['#2b5260', '#6ed4cb'] : type === 'domain' || type === 'url' ? ['#403665', '#ab98fa'] : ['#3c465b', '#a2b0c8'];
      const group = svg('g', { class: 'graph-node' });
      group.append(svg('circle', { cx: point.x, cy: point.y, r: 13, fill: colors[0], stroke: colors[1] }));
      group.append(svg('text', { x: point.x, y: point.y + 3 }, String(index + 1)));
      group.append(svg('text', { x: point.x, y: point.y + 29 }, short(entity.value ?? entity.normalized_value ?? entity.label ?? type, 26)));
      group.append(svg('text', { x: point.x, y: point.y + 41, class: 'node-type' }, type));
      graph.append(group);
    });
    canvas.append(graph);
    $('#graph-caption').textContent = `${entities.length} entities · ${relationships.length} observed relationships`;
  }
  function renderPaths(history) {
    const holder = $('#path-list');
    holder.replaceChildren();
    const graph = latest(history, 'graph_assessment')?.payload || {};
    const paths = graph.paths || [];
    $('#path-count').textContent = `${paths.length} PATH${paths.length === 1 ? '' : 'S'}`;
    if (!paths.length) { holder.append(el('p', 'quiet', 'No artifact-to-artifact paths recorded.')); return; }
    paths.slice(0, 30).forEach((path) => {
      const row = el('div', 'path-row');
      const sequence = el('div', 'path-sequence');
      const nodes = path.nodes || path.entities || path.path || [];
      nodes.forEach((item, index) => {
        if (index) sequence.append(el('span', 'path-arrow', '→'));
        const label = typeof item === 'object' ? item.value || item.normalized_value || item.type || 'entity' : item;
        sequence.append(el('span', 'path-node', short(label, 32)));
      });
      row.append(sequence);
      const types = path.relationships || path.relationship_types || (path.edges || []).flatMap((edge) => edge.relationships || [edge.kind || edge.type || 'RELATED_TO']);
      if (types.length) row.append(el('div', 'path-kind', types.map((edge) => typeof edge === 'string' ? edge : edge.type || edge.kind || 'RELATED_TO').join(' · ')));
      holder.append(row);
    });
  }
  function renderTimeline(history) {
    const holder = $('#timeline-list');
    holder.replaceChildren();
    const events = [];
    for (const item of history.evidence || []) events.push({ time: item.observed_at, title: item.summary || item.kind, meta: `${item.kind} · ${item.producer || 'system'}` });
    for (const item of history.findings || []) events.push({ time: item.created_at, title: item.title, meta: `finding · ${item.status || 'observed'}` });
    for (const item of history.alert_events || []) events.push({ time: item.created_at, title: `Alert ${item.event_type}: ${item.status || ''}`, meta: item.note || item.actor || 'alert lifecycle' });
    events.sort((a, b) => new Date(b.time || 0) - new Date(a.time || 0));
    $('#timeline-count').textContent = `${events.length} EVENTS`;
    if (!events.length) { holder.append(el('p', 'quiet', 'No evidence history has been recorded yet.')); return; }
    events.slice(0, 40).forEach((event) => {
      const item = el('div', 'timeline-item');
      const rail = el('div', 'timeline-rail');
      rail.append(el('i', 'timeline-dot'));
      item.append(rail);
      const body = el('div', 'timeline-body');
      body.append(el('div', 'timeline-title', event.title), el('div', 'timeline-meta', `${dateText(event.time)} · ${event.meta}`));
      item.append(body);
      holder.append(item);
    });
  }
  function renderAlertHistory(alert) {
    const card = $('#alert-history-card');
    const holder = $('#alert-history-list');
    holder.replaceChildren();
    const events = alert.alert_history || [];
    if (!events.length) holder.append(el('p', 'quiet', 'No lifecycle events returned for this alert.'));
    events.forEach((event) => {
      const row = el('div', 'history-event');
      row.append(el('time', '', dateText(event.created_at)), el('span', '', `${event.event_type}: ${event.previous_status ? `${event.previous_status} → ` : ''}${event.status || ''}${event.note ? ` · ${event.note}` : ''}`));
      holder.append(row);
    });
    if (alert.alert?.id) {
      const controls = el('div', 'alert-controls');
      const status = alert.alert.status;
      const options = status === 'open' ? ['acknowledged', 'resolved', 'false_positive']
        : status === 'acknowledged' ? ['open', 'resolved', 'false_positive'] : ['open'];
      for (const nextStatus of options) {
        const button = el('button', 'button button-secondary', nextStatus.replace('_', ' '));
        button.type = 'button';
        button.addEventListener('click', async () => {
          button.disabled = true;
          try {
            await api(`/alerts/${encodeURIComponent(alert.alert.id)}`, {
              method: 'PATCH', body: JSON.stringify({ status: nextStatus, actor: 'dashboard' }),
            });
            toast(`Alert status changed to ${nextStatus.replace('_', ' ')}.`);
            const refreshed = await api(`/alerts/${encodeURIComponent(alert.alert.id)}/investigation`);
            renderAlertHistory(refreshed);
            await loadDashboard();
          } catch (error) { toast(error.message); button.disabled = false; }
        });
        controls.append(button);
      }
      holder.append(controls);
    }
    card.classList.remove('hidden');
  }
  async function loadSession(sessionId, alertData = null) {
    try {
      const history = await api(`/sessions/${encodeURIComponent(sessionId)}/history`);
      $('#investigation-empty').classList.add('hidden');
      $('#investigation-content').classList.remove('hidden');
      renderSummary(history);
      renderNovelty(history);
      renderGraph(history);
      renderPaths(history);
      renderTimeline(history);
      $('#alert-history-card').classList.add('hidden');
      if (alertData) renderAlertHistory(alertData);
      $('#investigation').scrollIntoView({ behavior: 'smooth', block: 'start' });
      return history;
    } catch (error) { toast(`Could not load session: ${error.message}`); return null; }
  }
  async function openAlert(alertId) {
    try {
      const investigation = await api(`/alerts/${encodeURIComponent(alertId)}/investigation`);
      $('#session-id-input').value = investigation.alert.session_id;
      await loadSession(investigation.alert.session_id, investigation);
    } catch (error) { toast(error.message); }
  }
  $('#analysis-form').addEventListener('submit', submitAnalysis);
  $('#session-lookup').addEventListener('submit', (event) => {
    event.preventDefault();
    const id = $('#session-id-input').value.trim();
    if (id) loadSession(id);
  });
  $('#refresh-button').addEventListener('click', loadDashboard);
  $('#camera-open').addEventListener('click', openCamera);
  $('#camera-close').addEventListener('click', stopCamera);
  $('#camera-capture').addEventListener('click', captureCameraFrame);
  window.addEventListener('pagehide', stopCamera);
  $('#today-label').textContent = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date());
  loadDashboard();
})();
