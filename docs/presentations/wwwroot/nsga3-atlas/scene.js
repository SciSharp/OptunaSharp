/**
 * NSGA-III objective-space instrument. Three.js 0.183.2, local ES modules.
 * Geometry is a view of RunSnapshot/ReferenceLabResult; this file never selects,
 * mutates, evaluates, or invents an optimization trial.
 * Three.js API references: https://threejs.org/docs/pages/WebGLRenderer.html
 * and https://threejs.org/docs/pages/OrbitControls.html.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLORS = {
  lime: 0xd9f57a, coral: 0xff947f, violet: 0xb5a1fa, white: 0xfff2d4,
  grid: 0x625e52, surface: 0x769b82, shadow: 0x32342b,
};
const RANK_COLORS = [COLORS.lime, COLORS.violet, COLORS.coral, 0x70bfd2, 0xce91ab, 0x9e9985];
const AXIS_COLORS = [COLORS.coral, COLORS.violet, COLORS.lime];
const NORMALIZED_STAGES = new Set(['normalize', 'associate', 'niche']);
const DEFAULT_OPTIONS = {
  stage: 'survive', selectedId: null, nicheIndex: 0,
  showSurface: true, showReferences: true, showGrid: true,
  pointScale: 1, objectiveScale: 1, focus: 'all',
};
const finite3 = a => Array.isArray(a) && a.length >= 3 && a.slice(0, 3).every(Number.isFinite);
const vector = a => new THREE.Vector3(a[0], a[2], a[1]);
const fmt = v => Math.abs(v) >= 100 ? v.toPrecision(3) : Number(v.toFixed(3)).toString();
const ease = t => 1 - (1 - t) ** 3;
const colorCSS = c => `#${c.toString(16).padStart(6, '0')}`;

export class AlgorithmScene {
  constructor({ container, onSelect = () => {}, onStatus = () => {} }) {
    if (!container) throw new Error('AlgorithmScene requires a container.');
    this.container = container;
    this.onSelect = onSelect;
    this.onStatus = onStatus;
    this.options = { ...DEFAULT_OPTIONS };
    this.snapshot = null;
    this.lab = null;
    this.status = { ready: false, renderer: 'Three.js / WebGL2', message: 'Starting renderer' };
    this.supported = false;
    this.disposed = false;
    this.pointRecords = new Map();
    this.labels = [];
    this.frame = 0;
    this.animatingUntil = 0;
    this.cameraPreset = 'isometric';
    this.extent = 2;
    this.bounds = new THREE.Box3(new THREE.Vector3(), new THREE.Vector3(2, 2, 2));
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.mediaChange = () => { this.controls.enableDamping = !this.reducedMotion.matches; this._invalidate(); };

    if (getComputedStyle(container).position === 'static') container.style.position = 'relative';
    this.backdrop = document.createElement('div');
    this.backdrop.className = 'atlas-scene-backdrop';
    Object.assign(this.backdrop.style, {
      position: 'absolute', inset: '0', pointerEvents: 'none',
      background: 'radial-gradient(ellipse at 47% 53%, rgba(100,111,58,.13), transparent 58%), radial-gradient(ellipse at 83% 17%, rgba(110,85,145,.09), transparent 46%)',
    });
    container.append(this.backdrop);
    const canvas = document.createElement('canvas');
    canvas.className = 'atlas-scene-canvas';
    canvas.setAttribute('aria-label', 'Interactive three-dimensional objective space. Drag to orbit, scroll to zoom, shift-drag to pan. Click a trial to inspect its exact values.');
    canvas.setAttribute('role', 'img');
    canvas.tabIndex = 0;
    Object.assign(canvas.style, { display: 'block', width: '100%', height: '100%', position: 'absolute', inset: '0', touchAction: 'none', outlineOffset: '-3px' });
    try {
      const context = canvas.getContext('webgl2', { antialias: true, alpha: true, powerPreference: 'high-performance' });
      if (!context) throw new Error('WebGL2 is unavailable in this browser.');
      this.renderer = new THREE.WebGLRenderer({ canvas, context, antialias: true, alpha: true });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
      this.renderer.setClearColor(0x141610, 0);
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.2;
    } catch (error) {
      this._unavailable(error.message);
      return;
    }
    container.append(canvas);
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.001, 2000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = !this.reducedMotion.matches;
    this.controls.dampingFactor = 0.1;
    this.controls.rotateSpeed = 0.7;
    this.controls.zoomSpeed = 0.75;
    this.controls.panSpeed = 0.65;
    this.controls.minDistance = 0.12;
    this.controls.maxDistance = 100;
    this.controls.autoRotate = false;
    this.controlChange = () => this._invalidate();
    this.controls.addEventListener('change', this.controlChange);
    this.controls.addEventListener('start', this.controlChange);
    this.controls.addEventListener('end', this.controlChange);
    this.controls.listenToKeyEvents(canvas);
    this.reducedMotion.addEventListener('change', this.mediaChange);
    this.scene.add(new THREE.HemisphereLight(0xecf2c7, 0x4a365f, 2.5));
    const key = new THREE.DirectionalLight(0xffedb6, 4.2);
    key.position.set(-1, 5, 4);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xbca3ff, 2.8);
    rim.position.set(4, 1, -3);
    this.scene.add(rim);
    this.staticGroup = new THREE.Group();
    this.pointGroup = new THREE.Group();
    this.connectionGroup = new THREE.Group();
    this.scene.add(this.staticGroup, this.pointGroup, this.connectionGroup);
    this.sphereGeometry = new THREE.SphereGeometry(1, 18, 12);
    this.halo = new THREE.Mesh(new THREE.RingGeometry(0.82, 1, 48),
      new THREE.MeshBasicMaterial({ color: COLORS.white, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthTest: false, depthWrite: false }));
    this.halo.visible = false;
    this.halo.renderOrder = 20;
    this.scene.add(this.halo);

    this.labelLayer = document.createElement('div');
    this.labelLayer.className = 'atlas-scene-labels';
    this.labelLayer.setAttribute('aria-hidden', 'true');
    Object.assign(this.labelLayer.style, { position: 'absolute', inset: '0', overflow: 'hidden', pointerEvents: 'none' });
    container.append(this.labelLayer);
    this.caption = document.createElement('div');
    this.caption.className = 'atlas-scene-caption';
    Object.assign(this.caption.style, {
      position: 'absolute', left: '16px', bottom: '13px', right: '16px', pointerEvents: 'none',
      color: '#aaa996', font: '10px/1.7 ui-monospace, SFMono-Regular, Consolas, monospace', letterSpacing: '.03em',
    });
    container.append(this.caption);
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.pointerStart = null;
    this.pointerDown = event => { this.pointerStart = { x: event.clientX, y: event.clientY, button: event.button }; };
    this.pointerUp = event => this._pick(event);
    this.contextLost = event => { event.preventDefault(); this._unavailable('The WebGL2 context was lost. Reload the presentation to restore the 3D view.'); };
    canvas.addEventListener('pointerdown', this.pointerDown);
    canvas.addEventListener('pointerup', this.pointerUp);
    canvas.addEventListener('webglcontextlost', this.contextLost);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.supported = true;
    this.status = { ready: true, renderer: `Three.js r${THREE.REVISION} / WebGL2`, message: '3D objective-space renderer ready' };
    this.onStatus(this.status);
    this._rebuild();
    this.setCameraPreset('isometric');
    this.resize();
  }

  setSnapshot(snapshot, options = {}) {
    if (this.snapshot?.runId !== snapshot?.runId) this.hasFramed = false;
    this.snapshot = snapshot;
    this.options = { ...this.options, ...options };
    if (this.supported) this._rebuild();
  }

  setOptions(options = {}) {
    this.options = { ...this.options, ...options };
    if (this.supported) this._rebuild();
  }

  setStage(stage) { this.setOptions({ stage }); }
  setSelected(id) { this.setOptions({ selectedId: id }); }
  setReferenceLab(result) { this.lab = result; if (this.supported) this._rebuild(); }
  clearReferenceLab() { this.lab = null; if (this.supported) this._rebuild(); }

  setCameraPreset(preset = 'isometric') {
    if (!this.supported) return;
    this.cameraPreset = preset;
    const center = this.bounds.getCenter(new THREE.Vector3());
    const span = this.bounds.getSize(new THREE.Vector3()).length();
    const aspectPenalty = Math.max(1, 0.85 / Math.max(0.3, this.camera.aspect));
    const distance = Math.max(1.5, span * 1.6) * aspectPenalty;
    const direction = preset === 'top' ? new THREE.Vector3(0, 1, 0.0001)
      : preset === 'front' ? new THREE.Vector3(0, 0, 1)
        : new THREE.Vector3(1.28, 0.94, 1.4).normalize();
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(direction, distance);
    this.camera.near = Math.max(0.0001, distance / 2000);
    this.camera.far = Math.max(100, distance * 30);
    this.camera.updateProjectionMatrix();
    this.controls.maxDistance = Math.max(20, distance * 8);
    this.controls.update();
    this._invalidate();
  }

  resize() {
    if (!this.supported || this.disposed) return;
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    const oldAspect = this.camera.aspect;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    if ((oldAspect >= 0.75) !== (this.camera.aspect >= 0.75)) this.setCameraPreset(this.cameraPreset);
    this._invalidate();
  }

  _unavailable(message) {
    this.supported = false;
    this.status = { ready: false, renderer: 'Three.js / WebGL2 unavailable', message };
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    if (!this.errorElement) {
      this.errorElement = document.createElement('div');
      this.errorElement.className = 'atlas-scene-error';
      this.errorElement.setAttribute('role', 'status');
      Object.assign(this.errorElement.style, { position: 'absolute', inset: '24px', display: 'grid', placeContent: 'center', padding: '28px', color: '#eee9d7', background: '#191b15', textAlign: 'center', font: '15px/1.7 system-ui' });
      this.container.append(this.errorElement);
    }
    this.errorElement.textContent = `${message} The source code, exact trial data, and algorithm controls remain available.`;
    this.onStatus(this.status);
  }

  _isNormalized() { return !this.lab && NORMALIZED_STAGES.has(this.options.stage); }

  _scales() {
    const scale = this.options.objectiveScale;
    return (finite3(scale) ? scale : [scale, scale, scale]).map(v => Number.isFinite(v) && v > 0 ? v : 1);
  }

  _world(values) {
    const scales = this._scales();
    return vector(values.map((v, i) => v * scales[i]));
  }

  _surfacePoint(a, b) {
    let values = [Math.cos(a) * Math.cos(b), Math.cos(a) * Math.sin(b), Math.sin(a)];
    if (this._isNormalized()) {
      const selection = this.snapshot?.selection;
      if (selection) values = values.map((v, i) => selection.intercepts[i] === 0 ? 0 : (v - selection.idealPoint[i]) / selection.intercepts[i]);
    }
    return this._world(values);
  }

  _visiblePoints() {
    if (this.lab) return [{ id: 'reference-lab', values: this.lab.point, normalized: this.lab.point, rank: 0, referenceIndex: this.lab.referenceIndex, distance: this.lab.distance, origin: 'probe', selected: true }];
    if (!this.snapshot) return [];
    const s = this.snapshot;
    const stage = this.options.stage;
    let points;
    if (stage === 'initialize') points = s.parents.length ? s.parents : s.offspring;
    else if (stage === 'survive') points = s.survivors;
    else if (this._isNormalized()) {
      const considered = new Set([...s.selection.acceptedIds, ...s.selection.splittingFrontIds]);
      points = s.candidates.filter(p => considered.has(p.id));
    } else points = s.candidates;
    const focus = this.options.focus;
    if (focus === 'parents') points = points.filter(p => p.origin === 'parent');
    else if (focus === 'offspring') points = points.filter(p => p.origin === 'offspring');
    else if (focus === 'pareto' || focus === 'front') points = points.filter(p => p.rank === 0);
    else if (focus === 'survivors' || focus === 'selected') points = points.filter(p => p.selected);
    return points.filter(p => finite3(this._isNormalized() ? p.normalized : p.values));
  }

  _pointAppearance(point) {
    const stage = this.options.stage;
    if (this.lab) return { color: COLORS.coral, opacity: 1 };
    if (stage === 'rank') return { color: RANK_COLORS[Math.min(point.rank, RANK_COLORS.length - 1)], opacity: point.rank < 2 ? 1 : 0.55 };
    if (stage === 'initialize') return { color: point.origin === 'offspring' ? COLORS.coral : COLORS.violet, opacity: 1 };
    if (stage === 'survive') return { color: COLORS.lime, opacity: 1 };
    if (stage === 'variation') return { color: point.origin === 'offspring' ? COLORS.coral : COLORS.violet, opacity: point.origin === 'offspring' ? 1 : 0.36 };
    if (stage === 'combine') return { color: point.origin === 'offspring' ? COLORS.coral : COLORS.violet, opacity: 1 };
    if (stage === 'niche') {
      const stepIndex = Math.max(0, Math.min(this.options.nicheIndex, this.snapshot.selection.nicheSteps.length - 1));
      const steps = this.snapshot.selection.nicheSteps;
      const chosen = new Set([...this.snapshot.selection.acceptedIds, ...steps.slice(0, stepIndex + 1).map(s => s.trialId)]);
      if (steps[stepIndex]?.trialId === point.id) return { color: COLORS.white, opacity: 1 };
      if (chosen.has(point.id)) return { color: COLORS.lime, opacity: 1 };
      return { color: point.referenceIndex === steps[stepIndex]?.referenceIndex ? COLORS.violet : COLORS.shadow, opacity: point.referenceIndex === steps[stepIndex]?.referenceIndex ? 0.95 : 0.4 };
    }
    if (this._isNormalized()) return { color: this.snapshot.selection.acceptedIds.includes(point.id) ? COLORS.lime : COLORS.violet, opacity: 1 };
    return { color: point.rank === 0 ? COLORS.lime : COLORS.violet, opacity: 1 };
  }

  _rebuild() {
    if (this.disposed || !this.supported) return;
    const points = this._visiblePoints();
    const normalized = this._isNormalized();
    const newBounds = new THREE.Box3().expandByPoint(new THREE.Vector3());
    for (const p of points) newBounds.expandByPoint(this._world(normalized ? p.normalized : p.values));
    // Keep the unit-octant reference visible without letting near-singular intercepts
    // distort framing. All candidate coordinates, including negative ones, are fitted.
    if (!normalized) newBounds.expandByPoint(this._world([1, 1, 1]));
    if (this.lab?.projection && finite3(this.lab.projection)) newBounds.expandByPoint(this._world(this.lab.projection));
    const oldExtent = this.extent;
    this.bounds.copy(newBounds);
    this.extent = Math.max(0.5, newBounds.getSize(new THREE.Vector3()).length() / Math.sqrt(3));
    this.bounds.expandByScalar(this.extent * 0.08);
    this.pointRadius = this.extent * 0.015 * Math.max(0.25, Math.min(3, Number(this.options.pointScale) || 1));
    this._clearGroup(this.staticGroup);
    this._clearGroup(this.connectionGroup);
    this.labelLayer.replaceChildren();
    this.labels = [];
    this._makeFrame();
    if (this.options.showSurface && !this.lab) this._makeSurface();
    const references = this.lab?.referenceDirections || this.snapshot?.referenceDirections || [];
    if (this.options.showReferences && (normalized || this.lab)) this._makeReferences(references);
    if (normalized) this._makeNormalizationPlane();
    this._syncPoints(points);
    this._makeConnections(points, references);
    this._makeSelectedLabel(points);
    const basis = this.lab ? 'NORMALIZED PROBE' : normalized ? 'NORMALIZED OBJECTIVES' : 'OBJECTIVE SPACE';
    const extra = this.options.stage === 'normalize' && this.snapshot ? ' · origin = ideal point after translation' : '';
    const scales = this._scales();
    const scaleNote = scales.some(v => v !== scales[0]) ? ` · axis display scale ${scales.map(fmt).join(':')}` : '';
    this.caption.textContent = `${basis}  /  ${points.length} ${this.lab ? 'probe' : 'trials'}${extra}${scaleNote}   ·   drag orbit / scroll zoom / shift-drag pan`;
    this.canvas.setAttribute('aria-label', `${basis.toLowerCase()}, ${points.length} plotted ${this.lab ? 'probe' : 'trials'}. ${normalized ? 'Accepted fronts and splitting front only. ' : ''}Drag to orbit, scroll to zoom, shift-drag to pan. Trial details are also available in the inspector.`);
    if (!this.hasFramed || this.framedBasis !== basis || this.extent > oldExtent * 1.8 || this.extent < oldExtent * 0.45) {
      this.setCameraPreset(this.cameraPreset);
      this.hasFramed = true;
      this.framedBasis = basis;
    }
    this._invalidate();
  }

  _makeFrame() {
    const low = this.bounds.min;
    const high = this.bounds.max;
    const floor = Math.min(0, low.y);
    const gridVertices = [];
    const size = this.extent;
    const gridStep = this._tickStep(Math.max(high.x - low.x, high.z - low.z), 7);
    if (this.options.showGrid) {
      for (let i = Math.ceil(low.x / gridStep); i <= Math.floor(high.x / gridStep); i++) gridVertices.push(new THREE.Vector3(i * gridStep, floor, low.z), new THREE.Vector3(i * gridStep, floor, high.z));
      for (let i = Math.ceil(low.z / gridStep); i <= Math.floor(high.z / gridStep); i++) gridVertices.push(new THREE.Vector3(low.x, floor, i * gridStep), new THREE.Vector3(high.x, floor, i * gridStep));
      this._segments(gridVertices, COLORS.grid, 0.19, this.staticGroup);
      const floorMesh = new THREE.Mesh(new THREE.PlaneGeometry(high.x - low.x, high.z - low.z), new THREE.MeshBasicMaterial({ color: 0x777553, transparent: true, opacity: 0.025, side: THREE.DoubleSide, depthWrite: false }));
      floorMesh.rotation.x = -Math.PI / 2;
      floorMesh.position.set((high.x + low.x) / 2, floor - size * 0.005, (high.z + low.z) / 2);
      this.staticGroup.add(floorMesh);
    }
    const scales = this._scales();
    for (let axis = 0; axis < 3; axis++) {
      const worldAxis = [0, 2, 1][axis];
      const start = new THREE.Vector3();
      const end = new THREE.Vector3();
      start.setComponent(worldAxis, Math.min(0, low.getComponent(worldAxis)));
      end.setComponent(worldAxis, high.getComponent(worldAxis) + size * 0.06);
      this._segments([start, end], AXIS_COLORS[axis], 0.66, this.staticGroup);
      const arrow = new THREE.Mesh(new THREE.ConeGeometry(size * 0.012, size * 0.05, 12), new THREE.MeshBasicMaterial({ color: AXIS_COLORS[axis] }));
      const direction = new THREE.Vector3().setComponent(worldAxis, 1);
      arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
      arrow.position.copy(end);
      this.staticGroup.add(arrow);
      const labelPosition = end.clone().addScaledVector(direction, size * 0.05);
      const axisLabel = this._isNormalized() || this.lab ? `f̃${'₁₂₃'[axis]}` : `f${'₁₂₃'[axis]}`;
      this._label(axisLabel, labelPosition, { color: colorCSS(AXIS_COLORS[axis]), size: 14, weight: '700' });
      if (!this.options.showGrid) continue;
      const step = this._tickStep((end.getComponent(worldAxis) - start.getComponent(worldAxis)) / scales[axis], 4);
      for (let i = Math.ceil(start.getComponent(worldAxis) / scales[axis] / step); i <= Math.floor(end.getComponent(worldAxis) / scales[axis] / step); i++) {
        if (i === 0) continue;
        const tick = new THREE.Vector3().setComponent(worldAxis, i * step * scales[axis]);
        const sideAxis = worldAxis === 1 ? 0 : 1;
        const tickA = tick.clone().setComponent(sideAxis, -size * 0.01);
        const tickB = tick.clone().setComponent(sideAxis, size * 0.01);
        this._segments([tickA, tickB], COLORS.grid, 0.45, this.staticGroup);
        this._label(fmt(i * step), tick, { color: '#999d87', size: 9, offsetX: 8, offsetY: 12 });
      }
    }
    this._label(this._isNormalized() ? 'ideal → 0' : '0', new THREE.Vector3(), { color: '#c7c4ae', size: 10, offsetX: -14, offsetY: 12 });
    if (this.snapshot && !this.lab && !this._isNormalized() && ['rank', 'combine'].includes(this.options.stage)) {
      const ideal = this._world(this.snapshot.selection.idealPoint);
      this._cross(ideal, this.extent * 0.025, COLORS.white, this.staticGroup);
      this._label('ideal z*', ideal, { color: '#e0dcc2', size: 10, offsetY: -18 });
    }
  }

  _tickStep(range, count) {
    const raw = Math.max(0.00001, range / count);
    const power = 10 ** Math.floor(Math.log10(raw));
    const n = raw / power;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * power;
  }

  _makeSurface() {
    const divisions = 40;
    const vertices = [];
    const indices = [];
    for (let a = 0; a <= divisions; a++) {
      for (let b = 0; b <= divisions; b++) {
        const p = this._surfacePoint(a / divisions * Math.PI / 2, b / divisions * Math.PI / 2);
        vertices.push(p.x, p.y, p.z);
        if (a < divisions && b < divisions) {
          const index = a * (divisions + 1) + b;
          indices.push(index, index + 1, index + divisions + 1, index + 1, index + divisions + 2, index + divisions + 1);
        }
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, new THREE.MeshPhysicalMaterial({
      color: COLORS.surface, emissive: 0x253827, emissiveIntensity: 0.3,
      metalness: 0.18, roughness: 0.38, clearcoat: 0.6,
      transparent: true, opacity: 0.15, side: THREE.DoubleSide, depthWrite: false,
    }));
    mesh.renderOrder = -2;
    this.staticGroup.add(mesh);
    const lattice = [];
    for (let line = 0; line <= 12; line++) {
      const fixed = line / 12 * Math.PI / 2;
      for (let i = 0; i < 48; i++) {
        const a = i / 48 * Math.PI / 2;
        const b = (i + 1) / 48 * Math.PI / 2;
        lattice.push(this._surfacePoint(fixed, a), this._surfacePoint(fixed, b), this._surfacePoint(a, fixed), this._surfacePoint(b, fixed));
      }
    }
    this._segments(lattice, 0xb7ca8d, 0.16, this.staticGroup);
    const boundary = [];
    for (let i = 0; i < 64; i++) {
      const a = i / 64 * Math.PI / 2;
      const b = (i + 1) / 64 * Math.PI / 2;
      boundary.push(this._surfacePoint(0, a), this._surfacePoint(0, b),
        this._surfacePoint(a, 0), this._surfacePoint(b, 0),
        this._surfacePoint(a, Math.PI / 2), this._surfacePoint(b, Math.PI / 2));
    }
    this._segments(boundary, COLORS.lime, 0.36, this.staticGroup);
    // The label follows the actual transformed analytic surface, not a decorative shell.
    const labelAt = this._surfacePoint(0.15, 0.75);
    this._label(this._isNormalized() ? 'transformed DTLZ2 front' : 'DTLZ2 · ‖f‖₂ = 1', labelAt, { color: '#adb895', size: 10, offsetY: 21 });
  }

  _makeReferences(references) {
    const active = this._activeReference();
    const lines = [];
    const dots = [];
    const length = Math.max(1.05, this.extent * 1.25);
    for (let i = 0; i < references.length; i++) {
      if (!finite3(references[i])) continue;
      const direction = vector(references[i]).normalize();
      const endpointValues = [direction.x * length, direction.z * length, direction.y * length];
      const endpoint = this._world(endpointValues);
      lines.push(new THREE.Vector3(), endpoint);
      dots.push(this._world(references[i]));
      if (i === active) this._tube(new THREE.Vector3(), endpoint, this.extent * 0.0026, COLORS.violet, 0.95, this.connectionGroup);
    }
    this._segments(lines, COLORS.violet, this.lab || this._isNormalized() ? 0.16 : 0.08, this.staticGroup);
    if (dots.length) {
      const geometry = new THREE.BufferGeometry().setFromPoints(dots);
      this.staticGroup.add(new THREE.Points(geometry, new THREE.PointsMaterial({ color: COLORS.violet, size: 3, transparent: true, opacity: 0.4, sizeAttenuation: false, depthWrite: false })));
    }
  }

  _activeReference() {
    if (this.lab) return this.lab.referenceIndex;
    if (!this.snapshot || !this._isNormalized()) return -1;
    if (this.options.stage === 'niche') return this.snapshot.selection.nicheSteps[Math.max(0, Math.min(this.options.nicheIndex, this.snapshot.selection.nicheSteps.length - 1))]?.referenceIndex ?? -1;
    return this.snapshot.candidates.find(p => p.id === this.options.selectedId)?.referenceIndex ?? -1;
  }

  _makeNormalizationPlane() {
    const vertices = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(p => this._world(p));
    const geometry = new THREE.BufferGeometry().setFromPoints(vertices);
    geometry.setIndex([0, 1, 2]);
    const plane = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: COLORS.violet, side: THREE.DoubleSide, transparent: true, opacity: this.options.stage === 'normalize' ? 0.07 : 0.025, depthWrite: false }));
    this.staticGroup.add(plane);
    this._segments([vertices[0], vertices[1], vertices[1], vertices[2], vertices[2], vertices[0]], COLORS.violet, 0.23, this.staticGroup);
    if (this.options.stage === 'normalize') {
      this._label('unit-intercept plane', this._world([0.35, 0.45, 0.2]), { color: '#b1a7cb', size: 10, offsetY: -11 });
      this._cross(new THREE.Vector3(), this.extent * 0.023, COLORS.white, this.staticGroup);
    }
  }

  _syncPoints(points) {
    const now = performance.now();
    const duration = this.reducedMotion.matches ? 0 : 650;
    const ids = new Set(points.map(p => p.id));
    for (const [id, record] of this.pointRecords) {
      if (!ids.has(id)) {
        record.startPosition.copy(record.mesh.position);
        record.startOpacity = record.mesh.material.opacity;
        record.targetOpacity = 0;
        record.startRadius = record.mesh.scale.x;
        record.targetRadius = this.pointRadius * 0.35;
        record.started = now;
        record.duration = duration;
        record.removing = true;
      }
    }
    for (const point of points) {
      let record = this.pointRecords.get(point.id);
      const appearance = this._pointAppearance(point);
      const target = this._world(this._isNormalized() ? point.normalized : point.values);
      if (!record) {
        const material = new THREE.MeshStandardMaterial({ color: appearance.color, emissive: appearance.color, emissiveIntensity: 0.38, metalness: 0.18, roughness: 0.26, transparent: true, opacity: 0 });
        const mesh = new THREE.Mesh(this.sphereGeometry, material);
        mesh.position.copy(target);
        mesh.scale.setScalar(this.pointRadius * 0.15);
        mesh.userData.pointId = point.id;
        this.pointGroup.add(mesh);
        record = { mesh, startPosition: target.clone(), targetPosition: target.clone() };
        this.pointRecords.set(point.id, record);
      }
      record.mesh.material.color.setHex(appearance.color);
      record.mesh.material.emissive.setHex(appearance.color);
      record.point = point;
      record.startPosition.copy(record.mesh.position);
      record.targetPosition.copy(target);
      record.startOpacity = record.mesh.material.opacity;
      record.targetOpacity = appearance.opacity;
      record.startRadius = record.mesh.scale.x;
      record.targetRadius = this.pointRadius * (point.id === this.options.selectedId || this.lab ? 1.25 : 1);
      record.started = now;
      record.duration = duration;
      record.removing = false;
    }
    this.animatingUntil = now + duration;
  }

  _projection(point, direction) {
    const denominator = direction.reduce((sum, v) => sum + v * v, 0);
    const factor = denominator ? point.reduce((sum, v, i) => sum + v * direction[i], 0) / denominator : 0;
    return direction.map(v => factor * v);
  }

  _makeConnections(points, references) {
    if (this.lab) {
      const from = this._world(this.lab.point);
      const projection = this._world(this.lab.projection);
      this._dashed(from, projection, COLORS.coral, 0.95);
      this._cross(projection, this.extent * 0.02, COLORS.white, this.connectionGroup);
      this._label(`d⊥ = ${fmt(this.lab.distance)}`, from.clone().lerp(projection, 0.5), { color: '#ffb09b', size: 11, offsetY: -13 });
      this._label(`r${this.lab.referenceIndex + 1}`, projection, { color: '#c5b7ee', size: 10, offsetY: 17 });
      return;
    }
    if (!this._isNormalized()) return;
    const active = this._activeReference();
    const lines = [];
    for (const point of points) {
      const reference = references[point.referenceIndex];
      if (!finite3(reference)) continue;
      const projection = this._world(this._projection(point.normalized, reference));
      const at = this._world(point.normalized);
      const selected = point.id === this.options.selectedId || (this.options.stage === 'niche' && point.id === this.snapshot.selection.nicheSteps[this.options.nicheIndex]?.trialId);
      if (selected) {
        this._dashed(at, projection, COLORS.coral, 0.95);
        this._cross(projection, this.extent * 0.014, COLORS.white, this.connectionGroup);
        this._label(`d⊥ ${fmt(point.distance)}`, at.clone().lerp(projection, 0.5), { color: '#ffb09b', size: 10, offsetY: -13 });
      } else if (this.options.stage === 'associate' || (this.options.stage === 'niche' && point.referenceIndex === active)) lines.push(at, projection);
    }
    this._segments(lines, COLORS.violet, 0.17, this.connectionGroup);
  }

  _makeSelectedLabel(points) {
    const chosenId = this.lab ? 'reference-lab' : this.options.selectedId;
    const point = points.find(p => p.id === chosenId);
    this.selectedPointId = point?.id ?? null;
    this.halo.visible = !!point;
    if (!point) return;
    const values = this._isNormalized() ? point.normalized : point.values;
    const title = this.lab ? 'probe' : `trial #${point.id}`;
    this.selectedLabel = this._label(`${title}  (${values.map(fmt).join(', ')})`, this._world(values), { color: '#fff2d4', size: 11, weight: '600', offsetY: -25, selected: true });
  }

  _segments(points, color, opacity, group) {
    if (!points.length) return;
    const line = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
    group.add(line);
    return line;
  }

  _tube(start, end, radius, color, opacity, group) {
    const delta = end.clone().sub(start);
    if (delta.length() < 1e-10) return;
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, delta.length(), 8), new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
    mesh.position.copy(start).add(end).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
    group.add(mesh);
  }

  _dashed(start, end, color, opacity) {
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([start, end]), new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: this.extent * 0.015, gapSize: this.extent * 0.009, depthTest: false, depthWrite: false }));
    line.computeLineDistances();
    line.renderOrder = 9;
    this.connectionGroup.add(line);
  }

  _cross(center, radius, color, group) {
    const points = [];
    for (let i = 0; i < 3; i++) {
      const delta = new THREE.Vector3().setComponent(i, radius);
      points.push(center.clone().sub(delta), center.clone().add(delta));
    }
    this._segments(points, color, 0.8, group);
  }

  _label(text, position, { color = '#b7b7a4', size = 11, weight = '400', offsetX = 0, offsetY = 0, selected = false } = {}) {
    const element = document.createElement('span');
    element.className = 'atlas-world-label';
    element.textContent = text;
    Object.assign(element.style, {
      position: 'absolute', left: '0', top: '0', whiteSpace: 'nowrap', color,
      font: `${weight} ${size}px/1.3 ui-monospace, SFMono-Regular, Consolas, monospace`,
      textShadow: '0 1px 6px #15170f, 0 0 12px #15170f',
      ...(selected ? { padding: '5px 7px', background: 'rgba(23,25,18,.86)', border: '1px solid rgba(236,237,202,.18)', borderRadius: '4px' } : {}),
    });
    this.labelLayer.append(element);
    const label = { element, position, offsetX, offsetY };
    this.labels.push(label);
    return label;
  }

  _updateLabels() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    for (const label of this.labels) {
      const p = label.position.clone().project(this.camera);
      const x = (p.x * 0.5 + 0.5) * width + label.offsetX;
      const y = (-p.y * 0.5 + 0.5) * height + label.offsetY;
      const visible = p.z > -1 && p.z < 1 && x > 9 && x < width - 9 && y > 8 && y < height - 37;
      label.element.style.display = visible ? 'block' : 'none';
      if (visible) label.element.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
    }
  }

  _pick(event) {
    const start = this.pointerStart;
    this.pointerStart = null;
    if (!start || start.button !== 0 || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 5 || !this.supported) return;
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const meshes = [...this.pointRecords.values()].filter(r => !r.removing && r.mesh.material.opacity > 0.25).map(r => r.mesh);
    const hit = this.raycaster.intersectObjects(meshes, false)[0];
    if (hit && hit.object.userData.pointId !== 'reference-lab') this.onSelect(hit.object.userData.pointId);
  }

  _invalidate() {
    if (!this.frame && !this.disposed && this.supported) this.frame = requestAnimationFrame(time => this._render(time));
  }

  _render(time) {
    this.frame = 0;
    if (this.disposed || !this.supported) return;
    let moving = false;
    for (const [id, r] of this.pointRecords) {
      const t = r.duration ? Math.min(1, (time - r.started) / r.duration) : 1;
      const k = ease(Math.max(0, t));
      r.mesh.position.lerpVectors(r.startPosition, r.targetPosition, k);
      r.mesh.material.opacity = THREE.MathUtils.lerp(r.startOpacity, r.targetOpacity, k);
      r.mesh.scale.setScalar(THREE.MathUtils.lerp(r.startRadius, r.targetRadius, k));
      if (t < 1) moving = true;
      else if (r.removing) {
        this.pointGroup.remove(r.mesh);
        r.mesh.material.dispose();
        this.pointRecords.delete(id);
      }
    }
    const selected = this.pointRecords.get(this.selectedPointId);
    if (selected && !selected.removing) {
      this.halo.position.copy(selected.mesh.position);
      this.halo.quaternion.copy(this.camera.quaternion);
      this.halo.scale.setScalar(this.pointRadius * 2.25);
      if (this.selectedLabel) this.selectedLabel.position.copy(selected.mesh.position);
    }
    const controlsChanged = this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this._updateLabels();
    if (moving || controlsChanged) this._invalidate();
  }

  _clearGroup(group) {
    group.traverse(object => {
      object.geometry?.dispose();
      if (Array.isArray(object.material)) object.material.forEach(m => m.dispose());
      else object.material?.dispose();
    });
    group.clear();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.resizeObserver?.disconnect();
    this.reducedMotion.removeEventListener('change', this.mediaChange);
    this.controls?.dispose();
    this.canvas?.removeEventListener('pointerdown', this.pointerDown);
    this.canvas?.removeEventListener('pointerup', this.pointerUp);
    this.canvas?.removeEventListener('webglcontextlost', this.contextLost);
    if (this.scene) {
      const geometries = new Set();
      const materials = new Set();
      this.scene.traverse(object => {
        if (object.geometry) geometries.add(object.geometry);
        if (Array.isArray(object.material)) object.material.forEach(m => materials.add(m));
        else if (object.material) materials.add(object.material);
      });
      geometries.forEach(g => g.dispose());
      materials.forEach(m => m.dispose());
    }
    this.renderer?.dispose();
    this.canvas?.remove();
    this.labelLayer?.remove();
    this.caption?.remove();
    this.backdrop?.remove();
    this.errorElement?.remove();
    this.pointRecords.clear();
  }
}
