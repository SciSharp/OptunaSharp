/* NSGA-III field guide. The two teaching labs are separate from the real run.
 * Actual evolution and recorded survivor selections always come from OptunaSharp.
 * No third-party runtime is required. */
(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const C = { lime: '#d8f276', coral: '#ed947d', lavender: '#b7a9e8', paper: '#edece2', muted: '#89927a', grid: '#343c2b', ink: '#11130f' };
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = t => 1 - Math.pow(1 - clamp(t, 0, 1), 3);
  const fixed = (n, digits = 2) => Number.isFinite(n) ? n.toFixed(digits) : '—';
  const esc = text => String(text).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]));
  const path = points => points.map((p, i) => `${i ? 'L' : 'M'}${fixed(p[0], 2)},${fixed(p[1], 2)}`).join(' ');
  const line = (a, b, stroke, opacity = 1, width = 1, extra = '') => `<path d="${path([a, b])}" fill="none" stroke="${stroke}" stroke-width="${width}" opacity="${opacity}" ${extra}/>`;
  const circle = (p, r, color, extra = '') => `<circle cx="${fixed(p[0], 2)}" cy="${fixed(p[1], 2)}" r="${r}" fill="${color}" ${extra}/>`;
  const label = (p, text, color = C.muted, size = 10, extra = '') => `<text x="${fixed(p[0], 2)}" y="${fixed(p[1], 2)}" fill="${color}" font-size="${size}" ${extra}>${esc(text)}</text>`;
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const norm = v => Math.sqrt(dot(v, v));
  const scale3 = (v, s) => v.map(x => x * s);
  const svgPoint = (svg, event) => {
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const matrix = svg.getScreenCTM();
    return matrix ? point.matrixTransform(matrix.inverse()) : { x: 0, y: 0 };
  };

  async function api(url, body, method = 'POST') {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch(url, {
        method,
        headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal
      });
      if (!response.ok) {
        const content = await response.text();
        let message = '';
        try { const error = JSON.parse(content); message = error.detail || error.error || error.title || ''; } catch { /* Non-JSON host errors get a safe status message. */ }
        throw new Error(message || `The backend returned HTTP ${response.status}.`);
      }
      return response.status === 204 ? null : await response.json();
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('The backend did not respond within 30 seconds.');
      throw error;
    } finally { clearTimeout(timeout); }
  }

  // An intentionally illustrative cover: a positive-octant Pareto surface.
  function renderHero() {
    const project = ([x, y, z]) => [355 + 228 * (y - x), 507 - 99 * (x + y) - 277 * z];
    const sphere = (a, b) => [Math.cos(a) * Math.cos(b), Math.cos(a) * Math.sin(b), Math.sin(a)];
    let svg = '<title id="hero-art-title">An illustrative curved Pareto frontier with reference rays</title><defs><radialGradient id="hero-haze"><stop stop-color="#d8f276" stop-opacity=".065"/><stop offset="1" stop-color="#d8f276" stop-opacity="0"/></radialGradient></defs>';
    svg += '<ellipse cx="362" cy="326" rx="300" ry="272" fill="url(#hero-haze)"/>';
    const o = project([0, 0, 0]);
    const axes = [[1.38, 0, 0], [0, 1.38, 0], [0, 0, 1.34]];
    for (const axis of axes) svg += line(o, project(axis), '#6b7751', .45, .75);
    for (let i = 0; i <= 8; i++) {
      const a = i * Math.PI / 16;
      const meridian = [], latitude = [];
      for (let j = 0; j <= 48; j++) {
        const b = j * Math.PI / 96;
        meridian.push(project(sphere(b, a)));
        latitude.push(project(sphere(a, b)));
      }
      svg += `<path d="${path(meridian)}" stroke="#96a963" opacity="${i === 0 || i === 8 ? .7 : .32}" stroke-width=".65" fill="none"/>`;
      svg += `<path d="${path(latitude)}" stroke="#96a963" opacity="${i === 0 ? .7 : .26}" stroke-width=".65" fill="none"/>`;
    }
    for (let i = 0; i < 23; i++) {
      const a = .11 + (((i * 29) % 97) / 97) * 1.27;
      const b = .06 + (((i * 47 + 13) % 101) / 101) * 1.41;
      const xyz = sphere(a, b), p = project(xyz);
      if (i % 3 === 0) svg += line(o, project(scale3(xyz, 1.12)), C.lime, .11, .8);
      svg += circle(p, 3.1, C.lime, 'opacity=".9"');
      if (i % 5 === 0) svg += circle(p, 8, 'none', `stroke="${C.lime}" stroke-width=".6" opacity=".25"`);
    }
    for (let i = 0; i < 18; i++) {
      const xyz = sphere(.1 + ((i * 17) % 37) / 37 * 1.14, .14 + ((i * 11) % 31) / 31 * 1.14);
      const p = project(scale3(xyz, 1.12 + (i % 5) * .09));
      svg += circle(p, 1.8, i % 3 === 0 ? C.coral : C.lavender, 'opacity=".7"');
    }
    svg += circle(o, 2.5, C.paper, 'opacity=".5"');
    svg += label([34, 382], 'f₁', C.muted, 13) + label([670, 382], 'f₂', C.muted, 13) + label([364, 130], 'f₃', C.muted, 13);
    svg += label([366, 528], '0', C.muted, 9);
    svg += label([519, 186], 'DIVERSITY', C.lime, 8, 'letter-spacing="1.5"') + line([509, 192], [464, 232], C.lime, .35, .65);
    svg += label([89, 295], 'CONVERGENCE', C.lavender, 8, 'letter-spacing="1.5"') + line([184, 291], [238, 313], C.lavender, .35, .65);
    $('hero-plot').innerHTML = svg;
  }

  // Lab 1: exact, local Pareto dominance with draggable and keyboard-operable points.
  const initialPoints = [[.16, .77], [.25, .51], [.4, .32], [.64, .19], [.83, .13], [.4, .73], [.59, .55], [.78, .42], [.87, .72], [.29, .87], [.68, .83], [.53, .39]];
  let tradeoffs = [], selectedPoint = 0, dragPareto = null;
  const paretoPlot = $('pareto-plot');
  const P = { x: 104, y: 34, w: 567, h: 442 };
  const projectPareto = p => [P.x + p.x * P.w, P.y + (1 - p.y) * P.h];
  const dominates = (a, b) => a.x <= b.x && a.y <= b.y && (a.x < b.x || a.y < b.y);
  function rankPoints() {
    let remaining = tradeoffs.slice(), rank = 0;
    while (remaining.length) {
      const front = remaining.filter(a => !remaining.some(b => dominates(b, a)));
      front.forEach(p => { p.rank = rank; });
      const frontIds = new Set(front.map(p => p.id));
      remaining = remaining.filter(p => !frontIds.has(p.id));
      rank++;
    }
  }
  function renderPareto(restoreFocus = false) {
    rankPoints();
    const selected = tradeoffs.find(p => p.id === selectedPoint) || tradeoffs[0];
    const sp = projectPareto(selected);
    const focused = restoreFocus ? selected.id : null;
    let s = '';
    for (let i = 0; i <= 5; i++) {
      const v = i / 5;
      s += line([P.x + v * P.w, P.y], [P.x + v * P.w, P.y + P.h], C.grid, .8, .65);
      s += line([P.x, P.y + v * P.h], [P.x + P.w, P.y + v * P.h], C.grid, .8, .65);
      s += label([P.x + v * P.w, P.y + P.h + 25], fixed(v, 1), '#727d63', 10, 'text-anchor="middle"');
      s += label([P.x - 19, P.y + (1 - v) * P.h + 4], fixed(v, 1), '#727d63', 10, 'text-anchor="end"');
    }
    s += `<rect x="${sp[0]}" y="${P.y}" width="${P.x + P.w - sp[0]}" height="${sp[1] - P.y}" fill="${C.coral}" opacity=".065"/>`;
    s += line([sp[0], P.y], [sp[0], sp[1]], C.coral, .3, .7, 'stroke-dasharray="3 5"');
    s += line([sp[0], sp[1]], [P.x + P.w, sp[1]], C.coral, .3, .7, 'stroke-dasharray="3 5"');
    if (selected.x < .62 && selected.y < .8) s += label([sp[0] + 13, P.y + 21], 'WORSE IN BOTH', C.coral, 8, 'opacity=".75" letter-spacing=".6"');
    const front = tradeoffs.filter(p => p.rank === 0).sort((a, b) => a.x - b.x);
    if (front.length > 1) s += `<path d="${path(front.map(projectPareto))}" fill="none" stroke="${C.lime}" stroke-width="1" stroke-dasharray="4 5" opacity=".55"/>`;
    s += label([P.x + P.w / 2, 534], 'f₁  /  MINIMIZE ←', C.muted, 10, 'text-anchor="middle" letter-spacing=".6"');
    s += `<text x="28" y="${P.y + P.h / 2}" transform="rotate(-90 28 ${P.y + P.h / 2})" fill="${C.muted}" font-size="10" text-anchor="middle" letter-spacing=".6">f₂  /  MINIMIZE ←</text>`;
    s += label([P.x + 14, P.y + P.h - 15], 'BETTER ↙', C.lime, 9, 'opacity=".65"');
    for (const p of tradeoffs) {
      const xy = projectPareto(p), active = p.id === selectedPoint, fill = p.rank === 0 ? C.lime : dominates(selected, p) ? C.coral : '#82906d';
      s += `<g class="pareto-point" data-point-id="${p.id}" tabindex="0" role="button" aria-label="Point ${p.name}: f1 ${fixed(p.x)}, f2 ${fixed(p.y)}, rank ${p.rank}. Use arrow keys to move, Tab for the next point.">`;
      s += circle(xy, 16, 'transparent');
      s += circle(xy, active ? 12 : 10, 'none', `class="point-focus" stroke="${active ? C.paper : 'transparent'}" stroke-width=".8" opacity=".8"`);
      s += circle(xy, active ? 5 : 4, fill);
      s += label([xy[0] + 11, xy[1] - 10], p.name, active ? C.paper : '#9ba58d', 9);
      s += '</g>';
    }
    paretoPlot.innerHTML = s;
    $('pareto-selected-label').textContent = `SELECTED / ${selected.name}`;
    $('pareto-f1').textContent = fixed(selected.x);
    $('pareto-f2').textContent = fixed(selected.y);
    $('pareto-rank').textContent = selected.rank;
    const beats = tradeoffs.filter(p => dominates(selected, p)).length;
    const beaten = tradeoffs.filter(p => dominates(p, selected)).length;
    $('pareto-description').textContent = `${beats ? `Dominates ${beats} choice${beats === 1 ? '' : 's'}.` : 'Dominates no other choice.'} ${beaten ? `Dominated by ${beaten}.` : 'No choice dominates it.'}`;
    if (focused !== null) paretoPlot.querySelector(`[data-point-id="${focused}"]`)?.focus({ preventScroll: true });
  }
  function resetPareto() {
    tradeoffs = initialPoints.map(([x, y], id) => ({ x, y, id, name: String.fromCharCode(65 + id), rank: 0 }));
    selectedPoint = 0;
    renderPareto();
  }
  function movePareto(event) {
    const p = tradeoffs.find(p => p.id === dragPareto);
    if (!p) return;
    const xy = svgPoint(paretoPlot, event);
    p.x = clamp((xy.x - P.x) / P.w, .025, .975);
    p.y = clamp(1 - (xy.y - P.y) / P.h, .025, .975);
    renderPareto();
  }
  paretoPlot.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    const target = event.target.closest('[data-point-id]');
    if (target) {
      selectedPoint = Number(target.dataset.pointId);
    } else {
      const xy = svgPoint(paretoPlot, event);
      if (xy.x < P.x || xy.x > P.x + P.w || xy.y < P.y || xy.y > P.y + P.h || tradeoffs.length >= 24) return;
      selectedPoint = Math.max(...tradeoffs.map(p => p.id)) + 1;
      tradeoffs.push({ id: selectedPoint, name: String.fromCharCode(65 + selectedPoint), x: (xy.x - P.x) / P.w, y: 1 - (xy.y - P.y) / P.h, rank: 0 });
    }
    dragPareto = selectedPoint;
    paretoPlot.setPointerCapture(event.pointerId);
    renderPareto(true);
    event.preventDefault();
  });
  paretoPlot.addEventListener('pointermove', event => { if (dragPareto !== null) movePareto(event); });
  const endPareto = () => { if (dragPareto !== null) { dragPareto = null; renderPareto(true); } };
  paretoPlot.addEventListener('pointerup', endPareto);
  paretoPlot.addEventListener('pointercancel', endPareto);
  paretoPlot.addEventListener('focusin', event => {
    const target = event.target.closest('[data-point-id]');
    if (!target || selectedPoint === Number(target.dataset.pointId)) return;
    selectedPoint = Number(target.dataset.pointId);
    renderPareto(true);
  });
  paretoPlot.addEventListener('keydown', event => {
    const target = event.target.closest('[data-point-id]');
    if (!target) return;
    const p = tradeoffs.find(p => p.id === Number(target.dataset.pointId));
    const step = event.shiftKey ? .05 : .01;
    if (event.key === 'ArrowLeft') p.x = clamp(p.x - step, .025, .975);
    else if (event.key === 'ArrowRight') p.x = clamp(p.x + step, .025, .975);
    else if (event.key === 'ArrowUp') p.y = clamp(p.y + step, .025, .975);
    else if (event.key === 'ArrowDown') p.y = clamp(p.y - step, .025, .975);
    else return;
    event.preventDefault();
    event.stopPropagation();
    selectedPoint = p.id;
    renderPareto(true);
  });
  $('pareto-reset').addEventListener('click', resetPareto);

  // Lab 2: backend geometry, with an explicitly labelled local preview while offline.
  let referencePoint = [.45, .35, .2], referenceDivisions = 6, referenceResult;
  let referenceRequest = 0, referenceTimer = 0, dragReference = false;
  const referencePlot = $('reference-plot');
  const refProject = ([x, y, z]) => [390 + 312 * (y - x), 479 - 69 * (x + y) - 351 * z];
  function localReferenceResult() {
    const references = [];
    for (let a = referenceDivisions; a >= 0; a--) for (let b = referenceDivisions - a; b >= 0; b--) references.push([a / referenceDivisions, b / referenceDivisions, (referenceDivisions - a - b) / referenceDivisions]);
    let referenceIndex = 0, distance = Infinity, projection = [0, 0, 0];
    references.forEach((r, index) => {
      const q = scale3(r, dot(referencePoint, r) / dot(r, r));
      const d = norm(referencePoint.map((v, j) => v - q[j]));
      if (d < distance) { distance = d; referenceIndex = index; projection = q; }
    });
    return { referenceDirections: references, point: referencePoint.slice(), referenceIndex, distance, projection };
  }
  function renderReference() {
    const result = referenceResult || localReferenceResult();
    const vertices = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(refProject), origin = refProject([0, 0, 0]);
    const candidate = refProject(result.point), ray = result.referenceDirections[result.referenceIndex], endpoint = refProject(ray), projection = refProject(result.projection);
    let s = `<defs><linearGradient id="simplex-shade" x1="0" x2="0" y1="0" y2="1"><stop stop-color="#d8f276" stop-opacity=".055"/><stop offset="1" stop-color="#d8f276" stop-opacity=".01"/></linearGradient></defs>`;
    s += `<path d="${path([...vertices, vertices[0]])}" fill="url(#simplex-shade)" stroke="#536241" stroke-width="1"/>`;
    for (let i = 1; i < referenceDivisions; i++) {
      const v = i / referenceDivisions;
      s += line(refProject([v, 1 - v, 0]), refProject([v, 0, 1 - v]), C.grid, .6, .65);
      s += line(refProject([1 - v, v, 0]), refProject([0, v, 1 - v]), C.grid, .6, .65);
      s += line(refProject([1 - v, 0, v]), refProject([0, 1 - v, v]), C.grid, .6, .65);
    }
    result.referenceDirections.forEach((r, index) => {
      const xy = refProject(r);
      if (index % Math.max(1, Math.floor(result.referenceDirections.length / 12)) === 0) s += line(origin, xy, '#698342', .13, .7);
      s += circle(xy, index === result.referenceIndex ? 4 : 2.2, index === result.referenceIndex ? C.lime : '#6d7e55', `opacity="${index === result.referenceIndex ? 1 : .8}"`);
    });
    vertices.forEach(v => { s += line(origin, v, '#66784d', .35, .85); });
    s += line(origin, refProject(scale3(ray, 1.14)), C.lime, .9, 1.5);
    s += line(candidate, projection, C.lavender, 1, 1.5, 'stroke-dasharray="4 3"');
    // The right-angle glyph is constructed in three dimensions before projection.
    if (result.distance > .0001) {
      const alongRay = scale3(ray, .034 / norm(ray));
      const towardPoint = scale3(result.point.map((v, i) => v - result.projection[i]), .034 / result.distance);
      const q1 = result.projection.map((v, i) => v + alongRay[i]);
      const q2 = q1.map((v, i) => v + towardPoint[i]);
      const q3 = result.projection.map((v, i) => v + towardPoint[i]);
      s += `<path d="${path([q1, q2, q3].map(refProject))}" fill="none" stroke="${C.lavender}" stroke-width="1"/>`;
    }
    s += circle(endpoint, 10, 'none', `stroke="${C.lime}" stroke-width=".65" opacity=".7"`);
    s += circle(projection, 3.5, C.ink, `stroke="${C.lavender}" stroke-width="1.3"`);
    s += circle(origin, 3, '#889575');
    s += label([origin[0] + 13, origin[1] + 18], 'NORMALIZED ORIGIN', C.muted, 8, 'letter-spacing=".5"');
    s += label([vertices[0][0] - 4, vertices[0][1] + 28], 'f₁ = 1', C.coral, 11, 'text-anchor="middle"');
    s += label([vertices[1][0] + 4, vertices[1][1] + 28], 'f₂ = 1', C.lavender, 11, 'text-anchor="middle"');
    s += label([vertices[2][0], vertices[2][1] - 24], 'f₃ = 1', C.lime, 11, 'text-anchor="middle"');
    s += label([577, 169], 'f₁ + f₂ + f₃ = 1', C.muted, 9, 'class="plot-annotation"');
    s += label([577, 186], 'REFERENCE ENDPOINTS', '#5f6b51', 7, 'letter-spacing=".6" class="plot-annotation"');
    s += `<g id="reference-candidate" role="img" aria-label="Your solution at ${result.point.map(v => fixed(v)).join(', ')}">`;
    s += circle(candidate, 19, 'transparent') + circle(candidate, 11, 'none', `stroke="${C.coral}" stroke-width=".8" opacity=".6"`) + circle(candidate, 5, C.coral);
    s += label([candidate[0] + 15, candidate[1] - 12], 'YOU', C.coral, 8, 'letter-spacing="1"') + '</g>';
    if (result.distance > .025) s += label([(candidate[0] + projection[0]) / 2 + 12, (candidate[1] + projection[1]) / 2 + 14], 'd⊥', C.lavender, 11);
    referencePlot.innerHTML = s;
    $('reference-count').textContent = result.referenceDirections.length;
    $('reference-distance').textContent = fixed(result.distance, 4);
    $('reference-ray-label').textContent = `Ray ${result.referenceIndex + 1} / ${result.referenceDirections.length}`;
    $('reference-divisions-value').textContent = referenceDivisions;
    referencePoint.forEach((v, i) => {
      $(`reference-f${i + 1}`).value = String(Math.round(v * 100));
      $(`reference-f${i + 1}-value`).textContent = fixed(v);
    });
  }
  async function requestReference() {
    const requestId = ++referenceRequest;
    try {
      const result = await api('/api/nsga3/reference-lab', { divisions: referenceDivisions, point: referencePoint.slice() });
      if (requestId !== referenceRequest) return;
      referenceResult = result;
      $('reference-status').textContent = 'Geometry computed by OptunaSharp';
      $('reference-status').className = 'section-category online';
      renderReference();
    } catch {
      if (requestId !== referenceRequest) return;
      $('reference-status').textContent = 'Offline · local geometry preview';
      $('reference-status').className = 'section-category offline';
    }
  }
  function updateReference() {
    referenceRequest++;
    clearTimeout(referenceTimer);
    referenceResult = localReferenceResult();
    renderReference();
    $('reference-status').textContent = 'Local preview · checking with backend';
    $('reference-status').className = 'section-category';
    referenceTimer = setTimeout(requestReference, 130);
  }
  $('reference-divisions').addEventListener('input', event => { referenceDivisions = Number(event.target.value); updateReference(); });
  for (let j = 0; j < 3; j++) $(`reference-f${j + 1}`).addEventListener('input', event => {
    const value = Number(event.target.value) / 100;
    const rest = referencePoint.reduce((sum, v, i) => sum + (i === j ? 0 : v), 0);
    referencePoint = referencePoint.map((v, i) => i === j ? value : rest > 0 ? v / rest * (1 - value) : (1 - value) / 2);
    updateReference();
  });
  function moveReference(event) {
    const p = svgPoint(referencePlot, event);
    const z = (410 - p.y) / 282;
    const difference = (p.x - 390) / 312;
    const raw = [(1 - z - difference) / 2, (1 - z + difference) / 2, z].map(v => Math.max(0, v));
    const sum = raw.reduce((a, b) => a + b, 0);
    referencePoint = raw.map(v => v / sum);
    updateReference();
  }
  referencePlot.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    const p = svgPoint(referencePlot, event);
    if (p.y < 102 || p.y > 440 || p.x < 55 || p.x > 725) return;
    dragReference = true;
    referencePlot.setPointerCapture(event.pointerId);
    moveReference(event);
    event.preventDefault();
  });
  referencePlot.addEventListener('pointermove', event => { if (dragReference) moveReference(event); });
  referencePlot.addEventListener('pointerup', () => { dragReference = false; });
  referencePlot.addEventListener('pointercancel', () => { dragReference = false; });

  // Real run state. The server advances full generations; visual phases replay them.
  const phases = [
    ['START WITH VARIATION', 'Sample the possibilities.', 'OptunaSharp draws and evaluates the initial population. The search has somewhere to begin.'],
    ['EXPLORE THE DECISION SPACE', 'Create new possibilities.', 'Crossover recombines decision variables. Mutation perturbs them. OptunaSharp evaluates the new offspring on all three objectives.'],
    ['LET GENERATIONS COMPETE', 'Bring everyone together.', 'Combine the current parents and newly evaluated offspring. Selection now has twice the population to choose from.'],
    ['PREFER STRONG TRADEOFFS', 'Peel back the fronts.', 'Non-dominated sorting groups candidates by rank. Earlier fronts survive first; the last front that can fit only partly needs a tie-break.'],
    ['PUT OBJECTIVES IN SCALE', 'Align the geometry.', 'Subtract the ideal point and scale by estimated intercepts. This keeps objective scale from deciding which reference ray is closest.'],
    ['FIND A DIRECTION', 'Follow the nearest ray.', 'Each eligible normalized candidate associates with the reference ray at the smallest perpendicular distance. Directions represent different tradeoffs.'],
    ['PROTECT THE GAPS', 'Fill the quiet corners.', 'Pick a least occupied eligible niche. If it is empty, choose its closest candidate. Otherwise choose a remaining candidate using the seeded shuffle.'],
    ['CARRY THE VARIETY FORWARD', 'A new generation survives.', 'The selected population becomes the parents of the next generation. Repeat to improve proximity while preserving different tradeoffs.']
  ];
  const run = { id: null, history: [], inspected: 0, phase: 0, playing: false, busy: false, speed: 1, angle: 0, tick: 0, animation: 0, transitionStart: 0, from: new Map(), current: new Map(), phaseProgress: 1, rawLimit: 2.5, nicheIndex: -1, requestVersion: 0, needsReconcile: false };
  const evolutionPlot = $('evolution-plot');
  const currentSnapshot = () => run.history[run.inspected] || null;
  function setBackendStatus(text, state) {
    $('backend-status').className = `connection-status ${state || ''}`;
    $('backend-status').replaceChildren();
    $('backend-status').append(document.createElement('i'), document.createTextNode(text));
  }
  function setBusy(value) {
    run.busy = value;
    $('initialize-run').disabled = value;
    ['play-run', 'step-phase', 'next-generation'].forEach(id => { $(id).disabled = value || !run.id; });
    $('history-generation').disabled = value || run.history.length < 2;
    $('export-run').disabled = !run.history.length;
    evolutionPlot.setAttribute('aria-busy', String(value));
  }
  function pauseRun() {
    run.playing = false;
    clearTimeout(run.tick);
    $('play-run').innerHTML = '<span aria-hidden="true">▶</span> Play';
    $('play-run').setAttribute('aria-label', 'Play algorithm animation');
  }
  function showRunError(error) {
    pauseRun();
    setBackendStatus('Backend unavailable', 'offline');
    $('run-error').textContent = `${error.message || 'Unable to reach the backend.'} ${run.id ? 'Your recorded generations are preserved. Try Step phase again, or start a new run.' : 'Run the presentation host and select New run to reconnect. No simulated run data is shown.'}`;
    $('run-error').hidden = false;
    if (!run.id) {
      $('empty-state').hidden = false;
      $('empty-state').querySelector('span').textContent = 'Waiting for OptunaSharp';
      $('empty-state').querySelector('p').textContent = 'Start the local presentation host, then choose New run.';
    }
  }
  function clearRunError() { $('run-error').hidden = true; $('run-error').textContent = ''; }
  function validateSnapshot(snapshot) {
    if (!snapshot || typeof snapshot.runId !== 'string' || !Array.isArray(snapshot.candidates) || !Array.isArray(snapshot.survivors) || !snapshot.selection || !snapshot.metrics) throw new Error('The backend returned an unexpected run snapshot.');
    return snapshot;
  }
  async function initializeRun(event) {
    event?.preventDefault();
    if (run.busy || !$('run-config').reportValidity()) return;
    pauseRun();
    setBusy(true);
    clearRunError();
    setBackendStatus('Initializing real OptunaSharp trials');
    const oldId = run.id, requestVersion = ++run.requestVersion;
    try {
      const snapshot = validateSnapshot(await api('/api/nsga3/runs', {
        populationSize: Number($('population-size').value),
        divisions: Number($('run-divisions').value),
        seed: Number($('run-seed').value),
        mutationProbability: Number($('mutation').value),
        problem: 'dtlz2'
      }));
      if (requestVersion !== run.requestVersion) return;
      run.id = snapshot.runId;
      run.history = [snapshot];
      run.inspected = 0;
      run.needsReconcile = false;
      run.current.clear();
      run.rawLimit = Math.max(1.35, ...snapshot.candidates.flatMap(p => p.values)) * 1.1;
      $('empty-state').hidden = true;
      setBackendStatus('Live · OptunaSharp / C#', 'online');
      setPhase(0);
      if (oldId && oldId !== run.id) api(`/api/nsga3/runs/${encodeURIComponent(oldId)}`, undefined, 'DELETE').catch(() => { /* The new run is usable even if an expired previous run cannot be closed. */ });
    } catch (error) { showRunError(error); }
    finally { if (requestVersion === run.requestVersion) setBusy(false); }
  }
  async function fetchNextGeneration() {
    if (run.busy || !run.id) return false;
    setBusy(true);
    clearRunError();
    setBackendStatus('Computing the next generation');
    const requestVersion = run.requestVersion;
    function acceptSnapshot(snapshot) {
      if (snapshot.runId !== run.id) throw new Error('The backend returned a different run. Start a new run to reconnect.');
      if (snapshot.generation <= run.history.at(-1).generation) return false;
      run.history.push(snapshot);
      run.inspected = run.history.length - 1;
      run.rawLimit = Math.max(run.rawLimit, ...snapshot.candidates.flatMap(p => p.values).map(v => v * 1.1));
      setBackendStatus('Live · OptunaSharp / C#', 'online');
      return true;
    }
    try {
      // A client timeout does not necessarily cancel server computation. Reconcile
      // an uncertain prior request before another POST can advance the run again.
      if (run.needsReconcile) {
        const recovered = validateSnapshot(await api(`/api/nsga3/runs/${encodeURIComponent(run.id)}`, undefined, 'GET'));
        run.needsReconcile = false;
        if (requestVersion !== run.requestVersion) return false;
        if (acceptSnapshot(recovered)) return true;
      }
      run.needsReconcile = true;
      const snapshot = validateSnapshot(await api(`/api/nsga3/runs/${encodeURIComponent(run.id)}/step`, {}));
      if (requestVersion !== run.requestVersion) return false;
      run.needsReconcile = false;
      return acceptSnapshot(snapshot);
    } catch (error) {
      if (run.needsReconcile && requestVersion === run.requestVersion) {
        try {
          const recovered = validateSnapshot(await api(`/api/nsga3/runs/${encodeURIComponent(run.id)}`, undefined, 'GET'));
          if (requestVersion !== run.requestVersion) return false;
          run.needsReconcile = false;
          if (acceptSnapshot(recovered)) return true;
        } catch { /* Keep needsReconcile true. The next attempt must read before advancing. */ }
      }
      showRunError(error);
      return false;
    }
    finally { if (requestVersion === run.requestVersion) setBusy(false); }
  }
  async function stepPhase() {
    if (run.busy || !run.id) return;
    // History inspection does not rewind server state. Replay recorded generations first.
    if (run.phase === 0 || run.phase === 7) {
      if (run.inspected < run.history.length - 1) {
        run.inspected++;
        setPhase(1);
      } else if (await fetchNextGeneration()) setPhase(1);
    } else setPhase(run.phase + 1);
  }
  function schedulePlayback() {
    clearTimeout(run.tick);
    if (!run.playing) return;
    run.tick = setTimeout(async () => {
      if (!run.playing) return;
      await stepPhase();
      if (run.playing) schedulePlayback();
    }, 1750 / run.speed);
  }
  function playRun() {
    if (run.playing) { pauseRun(); return; }
    if (!run.id || run.busy) return;
    run.playing = true;
    $('play-run').innerHTML = '<span aria-hidden="true">Ⅱ</span> Pause';
    $('play-run').setAttribute('aria-label', 'Pause algorithm animation');
    stepPhase().then(schedulePlayback);
  }
  function updateRunReadouts() {
    const s = currentSnapshot();
    if (!s) return;
    $('plot-generation').textContent = `GENERATION ${String(s.generation).padStart(3, '0')}`;
    $('plot-population').textContent = `${s.config.populationSize} SURVIVORS / ${s.referenceDirections.length} DIRECTIONS`;
    $('metric-generation').textContent = String(s.generation).padStart(3, '0');
    $('metric-evaluations').textContent = s.evaluations.toLocaleString();
    $('metric-distance').textContent = fixed(s.metrics.meanDistanceToFront, 4);
    $('metric-coverage').textContent = `${s.metrics.occupiedDirections} / ${s.metrics.referenceDirectionCount}`;
    $('metric-nondominated').textContent = `${s.metrics.nonDominatedCount} / ${s.config.populationSize}`;
    $('history-generation').max = run.history.length - 1;
    $('history-generation').value = run.inspected;
    $('history-generation').disabled = run.history.length < 2 || run.busy;
    $('history-generation-label').textContent = `${s.generation} / ${run.history.at(-1).generation}`;
    $('history-description').textContent = run.inspected === run.history.length - 1 ? 'Latest recorded generation. Play continues the actual run.' : 'Inspecting a recorded generation. Play replays history before continuing the actual run.';
    $('normalization-note').replaceChildren();
    const strong = document.createElement('strong'); strong.textContent = 'Normalization: ';
    $('normalization-note').append(strong, document.createTextNode(s.selection.normalization));
    $('export-run').disabled = false;
    renderConvergence();
  }
  function setPhase(phase) {
    run.from = new Map(run.current);
    run.phase = phase;
    run.phaseProgress = reducedMotion ? 1 : 0;
    run.transitionStart = performance.now();
    run.nicheIndex = -1;
    const info = phases[phase];
    $('phase-kicker').textContent = info[0];
    $('phase-title').textContent = info[1];
    $('phase-description').textContent = info[2];
    $('phase-counter').textContent = `${String(phase).padStart(2, '0')} / 07`;
    document.querySelectorAll('#phase-list li').forEach((li, i) => {
      li.classList.toggle('active', i === phase);
      li.classList.toggle('passed', i < phase);
      if (i === phase) li.setAttribute('aria-current', 'step'); else li.removeAttribute('aria-current');
    });
    $('space-label').textContent = phase >= 4 && phase <= 6 ? 'OBJECTIVE SPACE / NORMALIZED VALUES' : 'OBJECTIVE SPACE / RAW VALUES';
    $('niche-detail').hidden = phase !== 6;
    if (phase === 6) $('niche-detail').textContent = 'Earlier accepted fronts establish the starting niche occupancy.';
    updateRunReadouts();
    animateEvolution();
  }

  function objectiveProject(v, limit) {
    const a = run.angle * Math.PI / 180;
    const side = (v[1] - v[0]) * Math.cos(a) + (v[0] + v[1]) * Math.sin(a) * .72;
    const depth = (v[0] + v[1]) * Math.cos(a) - (v[1] - v[0]) * Math.sin(a) * .8;
    const scale = 278 / limit;
    return [420 + side * scale, 505 - (.3 * depth + .98 * v[2]) * scale, depth - v[2] * .1];
  }
  function renderEvolution() {
    const snapshot = currentSnapshot();
    if (!snapshot) {
      evolutionPlot.innerHTML = '<title id="evolution-plot-title">Waiting for the live OptunaSharp population</title><desc id="evolution-plot-desc">No run data is displayed until the backend responds.</desc>';
      return;
    }
    const phase = run.phase, normalized = phase >= 4 && phase <= 6;
    const eligibleIds = new Set([...snapshot.selection.acceptedIds, ...snapshot.selection.splittingFrontIds]);
    const selectedIds = new Set(snapshot.selection.selectedIds);
    const acceptedIds = new Set(snapshot.selection.acceptedIds);
    const points = snapshot.candidates;
    const limit = normalized ? Math.max(1.25, ...points.filter(p => eligibleIds.has(p.id)).flatMap(p => p.normalized)) * 1.1 : run.rawLimit;
    const project = v => objectiveProject(v, limit);
    const origin = project([0, 0, 0]);
    const progress = ease(run.phaseProgress);
    let s = `<title id="evolution-plot-title">Generation ${snapshot.generation}, ${esc(phases[phase][1])}</title><desc id="evolution-plot-desc">${normalized ? 'Normalized' : 'Raw'} three-objective space. ${snapshot.candidates.length} recorded candidates and ${snapshot.survivors.length} actual survivors. Drag to rotate.</desc>`;
    // Subdued base grid, readable axes, and a true surface only in raw space.
    for (let i = 0; i <= 4; i++) {
      const v = limit * i / 4;
      s += line(project([v, 0, 0]), project([v, limit, 0]), C.grid, .52, .65);
      s += line(project([0, v, 0]), project([limit, v, 0]), C.grid, .52, .65);
    }
    if (!normalized) {
      for (let i = 0; i <= 8; i++) {
        const constant = i * Math.PI / 16, latitude = [], meridian = [];
        for (let j = 0; j <= 32; j++) {
          const t = j * Math.PI / 64;
          latitude.push(project([Math.cos(constant) * Math.cos(t), Math.cos(constant) * Math.sin(t), Math.sin(constant)]));
          meridian.push(project([Math.cos(t) * Math.cos(constant), Math.cos(t) * Math.sin(constant), Math.sin(t)]));
        }
        s += `<path d="${path(latitude)}" fill="none" stroke="#89a252" stroke-width=".75" opacity=".25"/><path d="${path(meridian)}" fill="none" stroke="#89a252" stroke-width=".75" opacity=".25"/>`;
      }
      const target = project([.4, .42, .814]);
      s += line(target, [target[0] + 74, target[1] - 24], C.lime, .3, .7);
      s += label([target[0] + 80, target[1] - 26], 'TRUE PARETO SURFACE', '#93a46e', 8, 'letter-spacing=".5" class="plot-annotation"');
      s += label([target[0] + 80, target[1] - 12], 'DTLZ2 · radius 1', '#667353', 7, 'class="plot-annotation"');
    }
    const axisEnd = [[limit * 1.08, 0, 0], [0, limit * 1.08, 0], [0, 0, limit * 1.18]];
    axisEnd.forEach((v, i) => {
      const p = project(v);
      s += line(origin, p, '#7b8866', .65, .8);
      const offset = i === 0 ? [-15, 17] : i === 1 ? [12, 17] : [13, -3];
      s += label([p[0] + offset[0], p[1] + offset[1]], ['f₁', 'f₂', 'f₃'][i], [C.coral, C.lavender, C.lime][i], 13);
      if (i < 2) s += label([p[0] + offset[0], p[1] + offset[1] + 20], 'min', '#667353', 8, 'class="plot-annotation"');
      for (let j = 1; j <= 3; j++) {
        const vTick = limit * j / 4, xyz = [0, 0, 0]; xyz[i] = vTick;
        const xy = project(xyz);
        s += label([xy[0] + (i === 0 ? -12 : 8), xy[1] + (i === 2 ? 3 : 15)], fixed(vTick, 1), '#657154', 8, i === 0 ? 'text-anchor="end"' : '');
      }
    });
    s += circle(origin, 2, C.muted) + label([origin[0] + 9, origin[1] + 19], '0', C.muted, 8);
    let nicheStep = null, nicheSelected = new Set(acceptedIds);
    if (phase === 6) {
      const steps = snapshot.selection.nicheSteps;
      const count = Math.min(steps.length, Math.max(1, Math.ceil(run.phaseProgress * steps.length)));
      steps.slice(0, count).forEach(step => nicheSelected.add(step.trialId));
      nicheStep = steps[count - 1] || null;
      if (run.nicheIndex !== count) {
        run.nicheIndex = count;
        $('niche-detail').textContent = nicheStep ? `Pick ${count}/${steps.length} · ray ${nicheStep.referenceIndex + 1} · occupancy ${nicheStep.occupancyBefore} → ${nicheStep.occupancyBefore + 1}\n${nicheStep.occupancyBefore === 0 ? `Closest candidate · distance ${fixed(nicheStep.distance, 4)}` : 'Occupied niche · seeded random choice'}` : 'All survivors fit in the accepted fronts; no additional niche picks are required.';
      }
    }
    if (phase === 5 || phase === 6) {
      snapshot.referenceDirections.forEach((r, index) => {
        const rayEnd = project(scale3(r, limit / Math.max(...r)));
        const active = nicheStep?.referenceIndex === index;
        s += line(origin, rayEnd, active ? C.lime : '#7d9452', active ? .8 : .15 * progress, active ? 1.6 : .65);
        if (active) s += circle(rayEnd, 4, 'none', `stroke="${C.lime}" stroke-width="1"`);
      });
    }
    if (phase === 3) {
      snapshot.selection.fronts.slice(0, 4).forEach((ids, rank) => {
        const xy = ids.map(id => points.find(p => p.id === id)).filter(Boolean).map(p => project(p.values)).sort((a, b) => a[0] - b[0]);
        if (xy.length >= 3) s += `<path d="${path(xy)}" fill="none" stroke="${[C.lime, C.lavender, C.coral, '#727e62'][rank]}" stroke-dasharray="3 5" opacity=".35" stroke-width=".8"/>`;
      });
      s += label([38, 64], `${snapshot.selection.fronts.length} NON-DOMINATED FRONTS`, C.lime, 9, 'letter-spacing=".8"');
      s += label([38, 84], `${snapshot.selection.acceptedIds.length} accepted · ${snapshot.selection.splittingFrontIds.length} in the splitting front`, C.muted, 8, 'class="plot-annotation"');
    }
    if (phase === 4) {
      s += label([38, 64], 'TRANSLATE → SCALE', C.lime, 9, 'letter-spacing=".8"');
      s += label([38, 84], `Ideal [${snapshot.selection.idealPoint.map(v => fixed(v)).join(', ')}]`, C.muted, 8, 'class="plot-annotation"');
      s += label([38, 101], `Intercepts [${snapshot.selection.intercepts.map(v => fixed(v)).join(', ')}]`, C.muted, 8, 'class="plot-annotation"');
    }
    if (phase === 1) {
      s += label([38, 64], `${snapshot.offspring.length} NEW EVALUATIONS`, C.coral, 9, 'letter-spacing=".8"');
      s += label([38, 83], 'Variation happens in decision space.', C.muted, 8, 'class="plot-annotation"');
      s += label([38, 98], 'Dots enter at their evaluated objectives.', '#667353', 8, 'class="plot-annotation"');
    }
    if (phase === 0) {
      s += label([38, 64], `${snapshot.config.populationSize} INITIAL SOLUTIONS`, C.lavender, 9, 'letter-spacing=".8"');
      s += label([38, 83], `Seed ${snapshot.config.seed} · real objective evaluations`, C.muted, 8, 'class="plot-annotation"');
    }
    if (phase === 7) {
      s += label([38, 64], `${snapshot.survivors.length} ACTUAL SURVIVORS`, C.lime, 9, 'letter-spacing=".8"');
      s += label([38, 83], 'Selected by the OptunaSharp sampler.', C.muted, 8, 'class="plot-annotation"');
    }
    const nextPositions = new Map();
    const projectedPoints = points.map(point => {
      const xyz = normalized ? point.normalized : point.values;
      const target = project(xyz);
      let opacity = 1, radius = 4.1, color = point.origin === 'parent' ? C.lavender : C.coral;
      if (phase === 0) color = C.lavender;
      if (phase === 1 && point.origin === 'parent') opacity = .45;
      if (phase === 3) color = [C.lime, C.lavender, C.coral, '#788369'][Math.min(point.rank, 3)];
      if (normalized && !eligibleIds.has(point.id)) opacity = .09;
      if (phase === 6) { color = nicheSelected.has(point.id) ? C.lime : color; opacity = nicheSelected.has(point.id) ? 1 : eligibleIds.has(point.id) ? .35 : .055; }
      if (phase === 7) { color = C.lime; opacity = selectedIds.has(point.id) ? 1 : .02; radius = selectedIds.has(point.id) ? 4.6 : 2.1; }
      let from = run.from.get(point.id);
      if (!from) from = { x: phase === 0 ? origin[0] : 100 + (point.id % 14) * 42, y: phase === 0 ? origin[1] : 559, opacity: 0, radius: 2 };
      const xy = { x: lerp(from.x, target[0], progress), y: lerp(from.y, target[1], progress), opacity: lerp(from.opacity, opacity, progress), radius: lerp(from.radius, radius, progress), color, point, depth: target[2] };
      nextPositions.set(point.id, xy);
      return xy;
    }).sort((a, b) => b.depth - a.depth);
    run.current = nextPositions;
    // Display perpendicular associations only for the eligible selection set.
    if (phase === 5) projectedPoints.forEach(p => {
      if (!eligibleIds.has(p.point.id)) return;
      const r = snapshot.referenceDirections[p.point.referenceIndex];
      const foot = project(scale3(r, dot(p.point.normalized, r) / dot(r, r)));
      s += line([p.x, p.y], foot, C.lavender, .28 * progress, .7, 'stroke-dasharray="2 3"');
    });
    for (const p of projectedPoints) {
      if (p.opacity < .015) continue;
      if (nicheStep?.trialId === p.point.id) s += circle([p.x, p.y], 12, 'none', `stroke="${C.lime}" stroke-width="1.2"`);
      s += `<g opacity="${fixed(p.opacity, 3)}"><title>Trial ${p.point.id} · f1 ${fixed(p.point.values[0], 4)} · f2 ${fixed(p.point.values[1], 4)} · f3 ${fixed(p.point.values[2], 4)} · rank ${p.point.rank} · ray ${p.point.referenceIndex + 1}${p.point.selected ? ' · selected' : ''}</title>`;
      s += circle([p.x, p.y], fixed(p.radius + 3.5, 2), p.color, 'opacity=".065"');
      s += circle([p.x, p.y], fixed(p.radius, 2), p.color, `stroke="${C.ink}" stroke-width=".9"`);
      if (phase === 3 && p.point.rank < 3) s += label([p.x + 7, p.y - 6], `r${p.point.rank}`, p.color, 7, 'opacity=".7"');
      s += '</g>';
    }
    evolutionPlot.innerHTML = s;
  }
  function animateEvolution() {
    cancelAnimationFrame(run.animation);
    const duration = reducedMotion ? 0 : (run.phase === 6 ? 1350 : 950) / run.speed;
    function frame(now) {
      run.phaseProgress = duration ? clamp((now - run.transitionStart) / duration, 0, 1) : 1;
      renderEvolution();
      if (run.phaseProgress < 1) run.animation = requestAnimationFrame(frame);
    }
    run.animation = requestAnimationFrame(frame);
  }

  function renderConvergence() {
    const history = run.history;
    const w = 606, h = 137, left = 84, top = 23;
    let s = '';
    const maxGap = history.length ? Math.max(...history.map(x => x.metrics.meanDistanceToFront), .01) * 1.12 : 1;
    const maxGen = Math.max(1, history.at(-1)?.generation || 1);
    const point = snapshot => [left + snapshot.generation / maxGen * w, top + h - snapshot.metrics.meanDistanceToFront / maxGap * h];
    for (let i = 0; i <= 3; i++) {
      const y = top + h * i / 3;
      s += line([left, y], [left + w, y], C.grid, .65, .7);
      s += label([left - 13, y + 3], fixed(maxGap * (1 - i / 3), 2), '#738062', 8, 'text-anchor="end"');
    }
    s += label([left, top + h + 23], '0', C.muted, 8, 'text-anchor="middle"');
    s += label([left + w, top + h + 23], `${maxGen} gen`, C.muted, 8, 'text-anchor="end"');
    if (history.length) {
      const pts = history.map(point);
      s += `<defs><linearGradient id="history-fill" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#d8f276" stop-opacity=".12"/><stop offset="1" stop-color="#d8f276" stop-opacity="0"/></linearGradient></defs>`;
      if (pts.length > 1) s += `<path d="${path([[pts[0][0], top + h], ...pts, [pts.at(-1)[0], top + h]])}Z" fill="url(#history-fill)"/><path d="${path(pts)}" fill="none" stroke="${C.lime}" stroke-width="1.5"/>`;
      const current = pts[run.inspected];
      s += line([current[0], top], [current[0], top + h], C.lime, .3, .75, 'stroke-dasharray="3 4"');
      s += circle(current, 4, C.lime, `stroke="${C.ink}" stroke-width="1.5"`);
      s += label([clamp(current[0], left + 32, left + w - 32), Math.max(13, current[1] - 13)], fixed(history[run.inspected].metrics.meanDistanceToFront, 4), C.lime, 8, 'text-anchor="middle"');
    } else s += label([left + w / 2, top + h / 2], 'A real run will leave a trace here.', C.muted, 10, 'text-anchor="middle"');
    $('convergence-plot').innerHTML = s;
  }
  let cameraDrag = null;
  function renderCameraChange() {
    cancelAnimationFrame(run.animation);
    run.phaseProgress = 1;
    renderEvolution();
  }
  evolutionPlot.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    cameraDrag = { x: event.clientX, angle: run.angle };
    evolutionPlot.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  evolutionPlot.addEventListener('pointermove', event => {
    if (!cameraDrag) return;
    run.angle = clamp(cameraDrag.angle + (event.clientX - cameraDrag.x) * .3, -90, 90);
    $('camera-angle').value = run.angle;
    // Camera interaction changes the view immediately; it does not modify trial data.
    renderCameraChange();
  });
  evolutionPlot.addEventListener('pointerup', () => { cameraDrag = null; });
  evolutionPlot.addEventListener('pointercancel', () => { cameraDrag = null; });
  $('camera-angle').addEventListener('input', event => { run.angle = Number(event.target.value); renderCameraChange(); });
  $('rotation-reset').addEventListener('click', () => { run.angle = 0; $('camera-angle').value = 0; renderCameraChange(); });
  $('run-config').addEventListener('submit', initializeRun);
  $('play-run').addEventListener('click', playRun);
  $('step-phase').addEventListener('click', () => { pauseRun(); stepPhase(); });
  $('next-generation').addEventListener('click', async () => {
    pauseRun();
    if (run.inspected < run.history.length - 1) { run.inspected++; setPhase(7); }
    else if (await fetchNextGeneration()) setPhase(7);
  });
  $('animation-speed').addEventListener('input', event => {
    run.speed = Number(event.target.value);
    $('animation-speed-label').textContent = `${run.speed}×`;
    if (run.playing) schedulePlayback();
  });
  $('history-generation').addEventListener('input', event => {
    pauseRun();
    run.inspected = Number(event.target.value);
    setPhase(run.history[run.inspected].generation === 0 ? 0 : 7);
  });
  $('export-run').addEventListener('click', () => {
    if (!run.history.length) return;
    const data = { format: 'OptunaSharp NSGA-III presentation history', version: 1, exportedAt: new Date().toISOString(), provenance: 'Actual OptunaSharp trials and selected survivor IDs. Selection geometry reconstructed on the backend for visual replay.', snapshots: run.history };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `nsga3-dtlz2-seed${run.history[0].config.seed}-generation${run.history.at(-1).generation}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pauseRun(); });
  window.addEventListener('pagehide', event => {
    pauseRun();
    cancelAnimationFrame(run.animation);
    if (run.id && !event.persisted) fetch(`/api/nsga3/runs/${encodeURIComponent(run.id)}`, { method: 'DELETE', keepalive: true }).catch(() => {});
  });

  renderHero();
  resetPareto();
  renderReference();
  requestReference();
  renderEvolution();
  renderConvergence();
  initializeRun();
})();
