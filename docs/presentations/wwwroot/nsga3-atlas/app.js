import { STAGES, StudySession, requestJson } from './model.js';
import { CodeView } from './code-view.js';

const $ = id => document.getElementById(id);
const session = new StudySession();
const initialLocation = new URLSearchParams(location.hash.slice(1));
const state = { stage: 0, niche: 0, selected: null, playing: false, timer: 0, speed: 1, scene: null, bundle: null, code: null, refVersion: 0, refTimer: 0, sourceStage: null, flowMode: 'calls' };
const fmt = (number, digits = 3) => Number.isFinite(number) ? number.toFixed(digits) : '—';
const vector = values => `[${values.map(v => fmt(v)).join(', ')}]`;
const esc = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const stage = () => STAGES[state.stage];

function status(text, kind = 'online') {
  $('engine-status').textContent = text;
  $('engine-status').dataset.state = kind;
}
function setBusy(busy) {
  $('new-run').disabled = busy;
  ['step-stage', 'step-generation'].forEach(id => { $(id).disabled = busy || !session.id; });
  // Pause remains available while the synchronous backend finishes its current request.
  $('play').disabled = !session.id && !state.playing;
  $('history').disabled = busy || session.history.length < 2;
  $('niche-step').disabled = busy;
  document.querySelectorAll('button[data-stage]').forEach(button => { button.disabled = busy; });
  ['ref-mode', 'ref-f1', 'ref-f2', 'ref-f3', 'ref-divisions'].forEach(id => { $(id).disabled = busy; });
  $('settings-form').setAttribute('aria-busy', String(busy));
}
function error(error) {
  pause();
  $('run-error').hidden = false;
  $('run-error').textContent = `${error.message || error} Recorded results remain available. Retry a step to reconnect, or create a new run.`;
  status('Connection needs attention', 'offline');
}
function clearError() { $('run-error').hidden = true; $('run-error').textContent = ''; }

function options() {
  return {
    stage: stage().id,
    selectedId: state.selected,
    nicheIndex: state.niche,
    showSurface: $('show-surface').checked,
    showReferences: $('show-references').checked,
    showGrid: $('show-grid').checked,
    pointScale: Number($('point-scale').value),
    objectiveScale: Number($('objective-scale').value)
  };
}

function expressionValues(id, s) {
  if (!s) return 'Waiting for a real C# generation';
  const selected = s.candidates.find(p => p.id === state.selected);
  switch (id) {
    case 'initialize': return `N = ${s.config.populationSize} · seed = ${s.config.seed} · 12 decision variables`;
    case 'variation': return `|Q| = ${s.offspring.length} · p(mutation) = ${s.config.mutationProbability}`;
    case 'combine': return `${s.parents.length} parents + ${s.offspring.length} offspring = ${s.candidates.length} candidates`;
    case 'rank': return `${s.selection.fronts.length} fronts · ${s.selection.acceptedIds.length} accepted · ${s.selection.splittingFrontIds.length} borderline`;
    case 'normalize': return `z* = ${vector(s.selection.idealPoint)}\na = ${vector(s.selection.intercepts)}`;
    case 'associate': return selected ? `Trial #${selected.id} → ray ${selected.referenceIndex + 1} · d⊥ = ${fmt(selected.distance, 5)}` : `${s.referenceDirections.length} rays · select a trial to inspect its distance`;
    case 'niche': {
      const pick = s.selection.nicheSteps[state.niche];
      return pick ? `Pick ${state.niche + 1}/${s.selection.nicheSteps.length} · ray ${pick.referenceIndex + 1}\nρ: ${pick.occupancyBefore} → ${pick.occupancyBefore + 1} · trial #${pick.trialId}` : 'Earlier fronts fill the population';
    }
    case 'survive': return `${s.survivors.length} actual survivors · mean radial gap = ${fmt(s.metrics.meanDistanceToFront, 4)}`;
    default: return '';
  }
}

function createExpressions() {
  $('expression-list').innerHTML = STAGES.map((item, index) => `<button type="button" class="expression-row" data-stage="${item.id}" aria-pressed="${index === state.stage}"><span class="expression-index">${index + 1}</span><span class="expression-body"><span class="expression-title">${esc(item.title)}</span><span class="expression-formula">${esc(item.formula)}</span><span class="expression-values" id="expression-${item.id}"></span></span></button>`).join('');
  $('timeline').innerHTML = STAGES.map((item, index) => `<button type="button" class="stage-button" data-stage="${item.id}" aria-pressed="${index === state.stage}"><span>${String(index + 1).padStart(2, '0')}</span>${esc(item.title)}</button>`).join('');
  document.querySelectorAll('[data-stage]').forEach(button => button.addEventListener('click', () => {
    if (session.busy) return;
    pause();
    clearReference();
    setStage(STAGES.findIndex(s => s.id === button.dataset.stage));
  }));
}

function mappedStage() { return state.bundle?.stages.find(item => item.id === stage().id); }
function synchronizeSource(force = false) {
  const mapped = mappedStage();
  if (!state.code || !mapped || (!force && state.sourceStage === mapped.id)) return;
  state.sourceStage = mapped.id;
  state.code.showFile(mapped.fileId, mapped.startLine, mapped);
  renderFlow();
}

function renderReadouts() {
  const s = session.snapshot;
  document.querySelectorAll('[data-stage]').forEach(button => {
    const index = STAGES.findIndex(item => item.id === button.dataset.stage);
    button.classList.toggle('is-active', index === state.stage);
    button.classList.toggle('is-past', index < state.stage);
    button.setAttribute('aria-pressed', String(index === state.stage));
  });
  STAGES.forEach(item => { $(`expression-${item.id}`).textContent = expressionValues(item.id, s); });
  $('scene-caption').textContent = `${stage().title} · ${stage().description} ${s ? `Generation ${s.generation}; geometry reconstructed from the real C# selection.` : ''}`;
  const legend = document.querySelector('.scene-legend');
  const entries = stage().id === 'rank' ? [['#d9f57a', 'Front 0'], ['#b5a1fa', 'Front 1'], ['#ff947f', 'Front 2'], ['#70bfd2', 'Later fronts']]
    : ['normalize', 'associate'].includes(stage().id) ? [['#d9f57a', 'Earlier fronts'], ['#b5a1fa', 'Borderline front']]
      : stage().id === 'niche' ? [['#d9f57a', 'Kept so far'], ['#fff2d4', 'Current pick'], ['#b5a1fa', 'Active niche']]
        : stage().id === 'survive' ? [['#d9f57a', 'Actual survivors']]
          : [['#ff947f', 'Offspring'], ['#b5a1fa', 'Parents']];
  if (legend) legend.innerHTML = entries.map(([color, title]) => `<span><i class="legend-dot" style="background:${color}"></i>${title}</span>`).join('');
  if (!s) return;
  $('metric-generation').textContent = String(s.generation).padStart(3, '0');
  $('metric-evaluations').textContent = s.evaluations.toLocaleString();
  $('metric-gap').textContent = fmt(s.metrics.meanDistanceToFront, 4);
  $('metric-coverage').textContent = `${s.metrics.occupiedDirections} / ${s.metrics.referenceDirectionCount}`;
  $('history').max = session.history.length - 1;
  $('history').value = session.index;
  $('history').disabled = session.busy || session.history.length < 2;
  $('history-label').textContent = `Generation ${s.generation} / ${session.latest.generation}${session.index < session.history.length - 1 ? ' · recorded replay' : ' · latest'}`;
  const picks = s.selection.nicheSteps;
  $('niche-step').closest('.niche-control').hidden = stage().id !== 'niche';
  $('niche-step').max = Math.max(0, picks.length - 1);
  $('niche-step').value = state.niche;
  $('niche-step').hidden = stage().id !== 'niche' || !picks.length;
  $('niche-label').hidden = stage().id !== 'niche';
  $('niche-label').textContent = picks.length ? `Survivor pick ${state.niche + 1} / ${picks.length}` : 'No borderline picks';
  renderSelected();
}

function renderSelected() {
  const s = session.snapshot;
  const picker = $('trial-picker');
  if (picker && s && picker.dataset.generation !== `${s.runId}:${s.generation}`) {
    picker.replaceChildren(new Option('Choose a trial…', ''), ...s.candidates.map(p => new Option(`#${p.id} · rank ${p.rank} · ${p.origin}`, String(p.id))));
    picker.dataset.generation = `${s.runId}:${s.generation}`;
  }
  if (picker) picker.value = state.selected === null ? '' : String(state.selected);
  const p = s?.candidates.find(p => p.id === state.selected);
  if (!p) {
    $('selected-trial').innerHTML = '<span class="selection-label">INSPECT A SOLUTION</span><p>Click a sphere to see its objectives, front rank, reference ray, and actual survival decision.</p>';
    return;
  }
  $('selected-trial').innerHTML = `<span class="selection-label">TRIAL #${p.id} · ${p.origin.toUpperCase()}</span><dl><dt>Objectives</dt><dd>${vector(p.values)}</dd><dt>Normalized</dt><dd>${vector(p.normalized)}</dd><dt>Front / ray</dt><dd>${p.rank} / ${p.referenceIndex + 1}</dd><dt>Perpendicular distance</dt><dd>${fmt(p.distance, 6)}</dd><dt>Survival</dt><dd>${p.selected ? 'Selected by OptunaSharp' : 'Not selected'}</dd></dl>`;
}

function renderScene() {
  if (session.snapshot && state.scene?.supported) state.scene.setSnapshot(session.snapshot, options());
}
function setStage(index, { synchronize = true } = {}) {
  if (index < 0 || index >= STAGES.length) return;
  state.stage = index;
  state.niche = 0;
  if (stage().id === 'niche') state.selected = session.snapshot?.selection.nicheSteps[0]?.trialId ?? state.selected;
  renderReadouts();
  renderScene();
  if (synchronize) synchronizeSource(true);
  renderFlow();
}

function selectTrial(value) {
  state.selected = value == null ? null : typeof value === 'object' ? value.id : value;
  state.scene?.setSelected(state.selected);
  renderReadouts();
  // A selection is an inspection action, never another optimization request.
}

function renderCalls() {
  const mapped = mappedStage();
  const currentId = state.code ? state.code.symbolSelect.value : mapped?.symbolId;
  const lookup = id => {
    const file = state.bundle.files.find(f => f.symbols.some(s => s.id === id));
    return file ? { file, symbol: file.symbols.find(s => s.id === id) } : null;
  };
  const current = lookup(currentId);
  if (!current) { $('flow-graph').innerHTML = '<text x="20" y="40">Select a function to inspect its local calls.</text>'; return; }
  const relations = state.bundle.calls || [];
  const callers = [...new Set(relations.filter(c => c.to === currentId).map(c => c.from))].map(lookup).filter(Boolean);
  const callees = [...new Set(relations.filter(c => c.from === currentId).map(c => c.to))].map(lookup).filter(Boolean);
  const topRows = Math.ceil(callers.length / 2), bottomRows = Math.ceil(callees.length / 2);
  const centerY = 45 + topRows * 98;
  const height = centerY + 140 + bottomRows * 98;
  $('flow-graph').setAttribute('viewBox', `0 0 440 ${height}`);
  const wrap = (name, length) => {
    const words = name.replace(/([a-z])([A-Z])/g, '$1 $2').split(' ');
    const lines = [''];
    for (const word of words) {
      if (lines.at(-1).length && lines.at(-1).length + word.length + 1 > length) lines.push('');
      lines[lines.length - 1] += `${lines.at(-1) ? ' ' : ''}${word}`;
    }
    return lines.slice(0, 2);
  };
  const box = (entry, x, y, width, selected = false) => {
    const nameLines = wrap(entry.symbol.name, selected ? 43 : 22);
    return `<g class="flow-node${selected ? ' is-active' : ''}" data-call-file="${esc(entry.file.id)}" data-call-symbol="${esc(entry.symbol.id)}" tabindex="0" role="button" aria-label="Open ${esc(entry.symbol.name)}, ${esc(entry.file.name)} line ${entry.symbol.startLine}"><title>${esc(entry.symbol.name)} · ${esc(entry.file.path)}:${entry.symbol.startLine}</title><rect x="${x}" y="${y}" width="${width}" height="72" rx="4"/>${nameLines.map((name, i) => `<text x="${x + 12}" y="${y + 22 + i * 19}" class="call-title">${esc(name)}</text>`).join('')}<text x="${x + 12}" y="${y + 60}" class="call-location">${esc(entry.file.id)} · L${entry.symbol.startLine}</text></g>`;
  };
  let content = `<title>Actual local call references for ${esc(current.symbol.name)}</title><defs><marker id="xref-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor"/></marker></defs><text class="call-caption" x="20" y="22">CALLED BY · ${callers.length}</text>`;
  callers.forEach((entry, i) => {
    const x = 20 + (i % 2) * 208, y = 38 + Math.floor(i / 2) * 98;
    content += `<path class="flow-edge" d="M${x + 95} ${y + 72}V${centerY - 10}H220V${centerY}" marker-end="url(#xref-arrow)"/>` + box(entry, x, y, 190);
  });
  content += box(current, 30, centerY, 380, true);
  content += `<text class="call-caption" x="20" y="${centerY + 110}">CALLS · ${callees.length}</text>`;
  callees.forEach((entry, i) => {
    const x = 20 + (i % 2) * 208, y = centerY + 130 + Math.floor(i / 2) * 98;
    content += `<path class="flow-edge" d="M220 ${centerY + 72}V${y - 9}H${x + 95}V${y}" marker-end="url(#xref-arrow)"/>` + box(entry, x, y, 190);
  });
  if (!callees.length) content += `<text class="call-caption" x="20" y="${centerY + 140}">No curated local outgoing calls.</text>`;
  $('flow-graph').innerHTML = content;
  $('flow-graph').querySelectorAll('[data-call-symbol]').forEach(node => {
    const activate = () => state.code.showSymbol(node.dataset.callFile, node.dataset.callSymbol);
    node.addEventListener('click', activate);
    node.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); activate(); } });
  });
}

function renderFlow() {
  if (!state.bundle) return;
  if (state.flowMode === 'calls') { renderCalls(); return; }
  const width = 570, rowHeight = 85, x = 80, boxWidth = 410, boxHeight = 62;
  const height = STAGES.length * rowHeight + 50;
  $('flow-graph').setAttribute('viewBox', `0 0 ${width} ${height}`);
  let content = '<title>NSGA-III semantic control flow with exact C# source references</title><defs><marker id="flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor"/></marker></defs>';
  for (let i = 0; i < STAGES.length; i++) {
    const item = STAGES[i];
    const mapped = state.bundle.stages.find(s => s.id === item.id);
    const file = state.bundle.files.find(f => f.id === mapped?.fileId);
    const symbol = file?.symbols.find(s => s.id === mapped?.symbolId);
    const y = 20 + i * rowHeight;
    if (i < STAGES.length - 1) content += `<path class="flow-edge" d="M285 ${y + boxHeight}V${y + rowHeight}" marker-end="url(#flow-arrow)"/>`;
    content += `<g class="flow-node${i === state.stage ? ' is-active' : ''}" data-flow-stage="${item.id}" tabindex="0" role="button" aria-label="${esc(item.title)}: open ${esc(symbol?.name || '')} at line ${mapped?.startLine || 1}"><rect x="${x}" y="${y}" width="${boxWidth}" height="${boxHeight}" rx="4"/><text x="${x + 15}" y="${y + 24}" class="flow-title">${i + 1}. ${esc(item.title)}</text><text x="${x + 15}" y="${y + 46}" class="flow-label">${esc(symbol?.name || mapped?.symbolId || '')} · L${mapped?.startLine || '?'}</text></g>`;
  }
  content += `<path class="flow-edge loop-edge" d="M${x + boxWidth} ${20 + 7 * rowHeight + 31}H540V${20 + rowHeight + 31}H${x + boxWidth}" marker-end="url(#flow-arrow)"/><text class="flow-label" x="552" y="400" transform="rotate(-90 552 400)">next generation</text>`;
  $('flow-graph').innerHTML = content;
  $('flow-graph').querySelectorAll('[data-flow-stage]').forEach(node => {
    const activate = () => {
      if (session.busy) return;
      pause();
      clearReference();
      setStage(STAGES.findIndex(s => s.id === node.dataset.flowStage));
    };
    node.addEventListener('click', activate);
    node.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(); } });
  });
}

async function newRun(event) {
  event?.preventDefault();
  if (session.busy || !$('settings-form').reportValidity()) return;
  pause();
  clearReference();
  setBusy(true);
  clearError();
  status('Evaluating initial C# population', 'busy');
  try {
    await session.create({ populationSize: Number($('population-size').value), divisions: Number($('divisions').value), seed: Number($('seed').value), mutationProbability: Number($('mutation').value), problem: 'dtlz2' });
    state.selected = null;
    status('Live · OptunaSharp / C#');
    setStage(0);
  } catch (failure) { error(failure); }
  finally { setBusy(false); }
}

async function advanceGeneration(startStage = 1) {
  if (session.busy || !session.id) return false;
  clearReference();
  if (session.index === session.history.length - 1 && session.latest.generation >= 250) {
    pause(); status('Run complete · create a new study'); return false;
  }
  setBusy(true);
  clearError();
  status('Computing next C# generation', 'busy');
  try {
    const result = await session.advance();
    if (result) {
      state.selected = null;
      setStage(startStage);
      status('Live · OptunaSharp / C#');
    }
    return Boolean(result);
  } catch (failure) { error(failure); return false; }
  finally { setBusy(false); }
}

async function step() {
  if (session.busy || !session.id) return;
  clearReference();
  if (state.stage === 0 || state.stage === 7) await advanceGeneration(1);
  else if (stage().id === 'niche' && state.niche < session.snapshot.selection.nicheSteps.length - 1) {
    state.niche++;
    state.selected = session.snapshot.selection.nicheSteps[state.niche].trialId;
    renderReadouts(); renderScene();
  } else setStage(state.stage + 1);
}
function pause() {
  state.playing = false;
  clearTimeout(state.timer);
  $('play').textContent = '▶ Play';
  $('play').setAttribute('aria-pressed', 'false');
}
function schedule() {
  clearTimeout(state.timer);
  if (!state.playing) return;
  const delay = (stage().id === 'niche' ? 430 : 1750) / state.speed;
  state.timer = setTimeout(async () => { if (state.playing) { await step(); schedule(); } }, delay);
}
function play() {
  if (state.playing) { pause(); return; }
  if (!session.id || session.busy) return;
  clearReference();
  state.playing = true;
  $('play').textContent = 'Ⅱ Pause';
  $('play').setAttribute('aria-pressed', 'true');
  step().then(schedule);
}

function clearReference() {
  if ($('ref-mode').checked) {
    $('ref-mode').checked = false;
    state.refVersion++;
    clearTimeout(state.refTimer);
    state.scene?.clearReferenceLab();
    $('ref-result').textContent = 'Enable the vector laboratory to inspect a point against the reference rays.';
  }
}
async function referenceLab() {
  if (!$('ref-mode').checked) {
    state.scene?.clearReferenceLab(); renderReadouts(); renderScene(); synchronizeSource(true); return;
  }
  pause();
  const version = ++state.refVersion;
  const point = ['ref-f1', 'ref-f2', 'ref-f3'].map(id => Number($(id).value));
  const divisions = Number($('ref-divisions').value);
  $('ref-result').textContent = 'Computing perpendicular association in C#…';
  try {
    const result = await requestJson('/api/nsga3/reference-lab', { point, divisions });
    if (version !== state.refVersion || !$('ref-mode').checked) return;
    state.scene?.setReferenceLab(result);
    $('ref-result').textContent = `f̄ = ${vector(result.point)} · ray ${result.referenceIndex + 1}/${result.referenceDirections.length}\nd⊥ = ${fmt(result.distance, 6)} · projection = ${vector(result.projection)}`;
    $('scene-caption').textContent = 'Reference-vector laboratory · exact C# association. This vector is independent of the current study; its coordinates need not sum to one.';
    const mapped = state.bundle?.stages.find(s => s.id === 'associate');
    if (mapped) state.code?.showFile(mapped.fileId, mapped.startLine, mapped);
  } catch (failure) {
    if (version === state.refVersion) $('ref-result').textContent = `Geometry unavailable: ${failure.message}`;
  }
}

async function loadSource() {
  try {
    state.bundle = await requestJson('/nsga3-atlas/source-data.json', undefined, 'GET');
    state.code = new CodeView(state.bundle, location => {
      const params = new URLSearchParams({ stage: stage().id, file: location.fileId, line: String(location.line) });
      history.replaceState(null, '', `#${params}`);
      renderFlow();
    });
    $('source-status').textContent = `${state.bundle.files.length} complete files · ${state.bundle.totalLines.toLocaleString()} lines`;
    synchronizeSource(true);
    renderReadouts();
    renderFlow();
  } catch (failure) {
    $('source-status').textContent = `Source document unavailable: ${failure.message}. Run node docs/presentations/build-sourcebook.mjs.`;
  }
}

async function createScene() {
  try {
    state.scene?.dispose();
    state.scene = null;
    const { AlgorithmScene } = await import('./scene.js');
    state.scene = new AlgorithmScene({
      container: $('scene-container'),
      onSelect: selectTrial,
      onStatus: result => {
        $('renderer-status').textContent = result.ready ? (result.renderer || 'Three.js / WebGL 2') : (result.message || 'WebGL 2 is unavailable');
        $('renderer-status').dataset.ready = String(Boolean(result.ready));
        $('scene-empty').hidden = Boolean(result.ready);
      }
    });
    renderScene();
    if ($('ref-mode').checked) referenceLab();
  } catch (failure) {
    $('renderer-status').textContent = `Three.js could not initialize: ${failure.message}`;
    $('scene-empty').hidden = false;
  }
}

function chooseView(view) {
  document.body.dataset.view = view;
  document.querySelectorAll('[data-view]').forEach(button => {
    if (button === document.body) return;
    button.setAttribute('aria-pressed', String(button.dataset.view === view));
  });
  requestAnimationFrame(() => state.scene?.resize());
}
function chooseInspector(panel) {
  $('source-pane').hidden = panel !== 'source';
  $('flow-pane').hidden = panel !== 'flow';
  $('code-panel').classList.toggle('showing-flow', panel === 'flow');
  document.querySelectorAll('[data-inspector]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.inspector === panel)));
}

$('settings-form').addEventListener('submit', newRun);
$('play').addEventListener('click', play);
$('step-stage').addEventListener('click', () => { pause(); step(); });
$('step-generation').addEventListener('click', () => { pause(); advanceGeneration(7); });
$('speed').addEventListener('input', event => { state.speed = Number(event.target.value); if (state.playing) schedule(); });
$('history').addEventListener('input', event => {
  pause(); clearReference(); session.inspect(Number(event.target.value)); state.selected = null;
  setStage(session.snapshot?.generation === 0 ? 0 : 7);
});
$('niche-step').addEventListener('input', event => {
  pause(); state.niche = Number(event.target.value);
  state.selected = session.snapshot?.selection.nicheSteps[state.niche]?.trialId ?? null;
  renderReadouts(); renderScene();
});
['show-surface', 'show-references', 'show-grid', 'point-scale', 'objective-scale'].forEach(id => $(id).addEventListener('input', () => state.scene?.setOptions(options())));
document.querySelectorAll('[data-camera]').forEach(button => button.addEventListener('click', () => {
  state.scene?.setCameraPreset(button.dataset.camera);
  document.querySelectorAll('[data-camera]').forEach(control => control.setAttribute('aria-pressed', String(control === button)));
}));
$('ref-mode').addEventListener('change', () => { state.refVersion++; referenceLab(); });
['ref-f1', 'ref-f2', 'ref-f3', 'ref-divisions'].forEach(id => $(id).addEventListener('input', () => {
  if (!$('ref-mode').checked) $('ref-mode').checked = true;
  state.refVersion++;
  clearTimeout(state.refTimer);
  state.refTimer = setTimeout(referenceLab, 100);
}));
$('retry-renderer').addEventListener('click', createScene);
document.querySelectorAll('button[data-view]').forEach(button => button.addEventListener('click', () => chooseView(button.dataset.view)));
document.querySelectorAll('[data-inspector]').forEach(button => button.addEventListener('click', () => chooseInspector(button.dataset.inspector)));
$('present-mode').addEventListener('click', () => {
  document.body.classList.toggle('presentation-mode');
  $('present-mode').setAttribute('aria-pressed', String(document.body.classList.contains('presentation-mode')));
  requestAnimationFrame(() => state.scene?.resize());
});
$('full-screen').addEventListener('click', async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
  catch { $('renderer-status').textContent = 'Fullscreen is unavailable here. The scene remains interactive.'; }
});
document.addEventListener('keydown', event => {
  if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || event.target.closest('input,select,textarea,button,a,[contenteditable],canvas')) return;
  if (event.key === ' ') { event.preventDefault(); play(); }
  if (event.key === 'ArrowRight') { event.preventDefault(); pause(); step(); }
  if (event.key === 'ArrowLeft' && !session.busy) { event.preventDefault(); pause(); setStage(Math.max(0, state.stage - 1)); }
});
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
window.addEventListener('pagehide', event => {
  pause();
  if (!event.persisted) {
    state.scene?.dispose();
    if (session.id) fetch(`/api/nsga3/runs/${encodeURIComponent(session.id)}`, { method: 'DELETE', keepalive: true }).catch(() => {});
  }
});

const trialLabel = document.createElement('label');
trialLabel.className = 'trial-picker-label';
trialLabel.textContent = 'Inspect an evaluated trial';
const trialPicker = document.createElement('select');
trialPicker.id = 'trial-picker';
trialPicker.setAttribute('aria-label', 'Inspect an evaluated trial');
trialLabel.append(trialPicker);
$('selected-trial').before(trialLabel);
trialPicker.addEventListener('change', () => selectTrial(trialPicker.value === '' ? null : Number(trialPicker.value)));
const flowModes = document.createElement('div');
flowModes.className = 'flow-mode-switch';
flowModes.setAttribute('role', 'group');
flowModes.setAttribute('aria-label', 'Graph relationship');
flowModes.innerHTML = '<button type="button" data-flow-mode="calls" aria-pressed="true">Function references</button><button type="button" data-flow-mode="stages" aria-pressed="false">Algorithm stages</button>';
$('flow-graph').before(flowModes);
$('flow-graph').setAttribute('role', 'group');
$('flow-graph').setAttribute('aria-label', 'Interactive source relationship graph');
flowModes.addEventListener('click', event => {
  const button = event.target.closest('[data-flow-mode]');
  if (!button) return;
  state.flowMode = button.dataset.flowMode;
  flowModes.querySelectorAll('button').forEach(control => control.setAttribute('aria-pressed', String(control === button)));
  renderFlow();
});
createExpressions();
renderReadouts();
setBusy(true);
chooseView(document.body.dataset.view || 'all');
chooseInspector('source');
await Promise.allSettled([loadSource(), createScene(), newRun()]);
// Restore navigation once, after independent source/renderer/study bootstrap resolves.
const requestedStage = STAGES.findIndex(item => item.id === initialLocation.get('stage'));
if (requestedStage >= 0) setStage(requestedStage);
if (initialLocation.get('file')) state.code?.showFile(initialLocation.get('file'), Number(initialLocation.get('line')) || 1);
