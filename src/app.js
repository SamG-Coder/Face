import { GpuRuntime } from '../vendor/webshader/runtime/runtime.js';
import { createFaceRenderer, makeTopology } from './face-renderer.js';
import { drawConstruction } from './construction.js';

const RINGS = 96;
const LEVELS = 72;
const VERTEX_COUNT = RINGS * LEVELS;
const WORKGROUP = 128;

const controls = [
  { group: 'Cranium / silhouette', key: 'headWidth', label: 'Head width', min: .64, max: .92, step: .01, value: .78 },
  { group: 'Cranium / silhouette', key: 'headDepth', label: 'Cranium depth', min: .74, max: 1.08, step: .01, value: .90 },
  { group: 'Cranium / silhouette', key: 'headHeight', label: 'Head height', min: 1.00, max: 1.26, step: .01, value: 1.12 },
  { group: 'Cranium / silhouette', key: 'templeWidth', label: 'Temple width', min: .72, max: 1.00, step: .01, value: .88 },
  { group: 'Cranium / silhouette', key: 'cheekWidth', label: 'Cheek width', min: .92, max: 1.18, step: .01, value: 1.06 },
  { group: 'Cranium / silhouette', key: 'jawWidth', label: 'Jaw width', min: .52, max: .95, step: .01, value: .72 },

  { group: 'Eyes / brow', key: 'eyeSpacing', label: 'Eye spacing', min: .22, max: .36, step: .005, value: .29 },
  { group: 'Eyes / brow', key: 'eyeWidth', label: 'Eye width', min: .74, max: 1.28, step: .01, value: .92 },
  { group: 'Eyes / brow', key: 'socketDepth', label: 'Socket depth', min: .04, max: .18, step: .005, value: .11 },
  { group: 'Eyes / brow', key: 'browProjection', label: 'Brow projection', min: .45, max: 1.55, step: .01, value: 1.00 },

  { group: 'Nose', key: 'noseProjection', label: 'Nose projection', min: .62, max: 1.48, step: .01, value: 1.00 },
  { group: 'Nose', key: 'noseWidth', label: 'Nose width', min: .72, max: 1.38, step: .01, value: 1.00 },

  { group: 'Mid / lower face', key: 'cheekProjection', label: 'Cheek projection', min: .45, max: 1.55, step: .01, value: 1.00 },
  { group: 'Mid / lower face', key: 'muzzleProjection', label: 'Muzzle projection', min: .52, max: 1.52, step: .01, value: 1.00 },
  { group: 'Mid / lower face', key: 'mouthWidth', label: 'Mouth width', min: .72, max: 1.30, step: .01, value: 1.00 },
  { group: 'Mid / lower face', key: 'upperLip', label: 'Upper lip', min: .35, max: 1.75, step: .01, value: 1.00 },
  { group: 'Mid / lower face', key: 'lowerLip', label: 'Lower lip', min: .40, max: 1.75, step: .01, value: 1.00 },
  { group: 'Mid / lower face', key: 'chinProjection', label: 'Chin projection', min: .45, max: 1.58, step: .01, value: 1.00 }
];

const defaults = Object.fromEntries(controls.map(c => [c.key, c.value]));
const params = { ...defaults };
const inputs = new Map();

const status = document.querySelector('#status');
const statusText = status.querySelector('span:last-child');
const fatal = document.querySelector('#fatal');
const fatalText = document.querySelector('#fatalText');
const frontCanvas = document.querySelector('#frontConstruction');
const profileCanvas = document.querySelector('#profileConstruction');
const gpuCanvas = document.querySelector('#gpuCanvas');
const controlList = document.querySelector('#controlList');

document.querySelector('#vertexCount').textContent = VERTEX_COUNT.toLocaleString();
document.querySelector('#triangleCount').textContent = ((LEVELS - 1) * RINGS * 2).toLocaleString();

function setStatus(text, state = '') {
  status.className = 'status' + (state ? ' ' + state : '');
  statusText.textContent = text;
}

function buildControls() {
  let currentGroup = '';
  for (const control of controls) {
    if (control.group !== currentGroup) {
      currentGroup = control.group;
      const heading = document.createElement('div');
      heading.className = 'control-section-title';
      heading.textContent = currentGroup;
      controlList.appendChild(heading);
    }

    const row = document.createElement('div');
    row.className = 'control-group';

    const head = document.createElement('div');
    head.className = 'control-head';

    const label = document.createElement('label');
    label.textContent = control.label;

    const value = document.createElement('span');
    value.className = 'control-value';
    value.textContent = control.value.toFixed(control.step < .01 ? 3 : 2);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(control.min);
    input.max = String(control.max);
    input.step = String(control.step);
    input.value = String(control.value);
    input.setAttribute('aria-label', control.label);

    head.append(label, value);
    row.append(head, input);
    controlList.appendChild(row);

    inputs.set(control.key, { input, value, control });
  }
}

buildControls();
drawConstruction(frontCanvas, profileCanvas, params);

let runtime = null;
let positions = null;
let normals = null;
let renderer = null;
let invocation = null;
let disposed = false;
let rebuildQueued = false;

function scalarSnapshot() {
  return {
    rings: RINGS,
    levels: LEVELS,
    headWidth: params.headWidth,
    headDepth: params.headDepth,
    headHeight: params.headHeight,
    templeWidth: params.templeWidth,
    cheekWidth: params.cheekWidth,
    jawWidth: params.jawWidth,
    eyeSpacing: params.eyeSpacing,
    eyeWidth: params.eyeWidth,
    socketDepth: params.socketDepth,
    browProjection: params.browProjection,
    noseProjection: params.noseProjection,
    noseWidth: params.noseWidth,
    cheekProjection: params.cheekProjection,
    muzzleProjection: params.muzzleProjection,
    mouthWidth: params.mouthWidth,
    upperLip: params.upperLip,
    lowerLip: params.lowerLip,
    chinProjection: params.chinProjection
  };
}

function rebuildFace() {
  if (!invocation || !runtime || disposed) return;
  invocation.setScalars(scalarSnapshot());
  runtime.batch()
    .dispatch(invocation, [Math.ceil(VERTEX_COUNT / WORKGROUP), 1, 1])
    .submit();
}

function scheduleRebuild() {
  drawConstruction(frontCanvas, profileCanvas, params);
  if (rebuildQueued) return;
  rebuildQueued = true;
  requestAnimationFrame(() => {
    rebuildQueued = false;
    rebuildFace();
  });
}

for (const [key, item] of inputs) {
  item.input.addEventListener('input', () => {
    params[key] = Number(item.input.value);
    item.value.textContent = params[key].toFixed(item.control.step < .01 ? 3 : 2);
    scheduleRebuild();
  });
}

function setAll(values) {
  for (const control of controls) {
    params[control.key] = values[control.key];
    const item = inputs.get(control.key);
    item.input.value = String(params[control.key]);
    item.value.textContent = params[control.key].toFixed(control.step < .01 ? 3 : 2);
  }
  scheduleRebuild();
}

document.querySelector('#reset').addEventListener('click', () => setAll(defaults));

let randomSeed = 0x51facade;
function random01() {
  randomSeed ^= randomSeed << 13;
  randomSeed ^= randomSeed >>> 17;
  randomSeed ^= randomSeed << 5;
  return (randomSeed >>> 0) / 4294967296;
}

document.querySelector('#randomize').addEventListener('click', () => {
  const values = {};
  for (const c of controls) {
    const t = .12 + random01() * .76;
    values[c.key] = c.min + (c.max - c.min) * t;
  }
  setAll(values);
});

for (const button of document.querySelectorAll('.mode')) {
  button.addEventListener('click', () => {
    document.querySelectorAll('.mode').forEach(b => b.classList.toggle('active', b === button));
    renderer?.setMode(button.dataset.mode);
  });
}

function animationFrame() {
  if (disposed) return;
  renderer?.render();
  requestAnimationFrame(animationFrame);
}

async function start() {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable. Use a current WebGPU-capable browser on localhost or HTTPS.');

  setStatus('Creating WebGPU device…');
  runtime = await GpuRuntime.create({
    onError: error => console.error('WebShader GPU error', error)
  });

  setStatus('Loading face.cu…');
  const sourceResponse = await fetch('./kernels/face.cu');
  if (!sourceResponse.ok) throw new Error(`Failed to load kernels/face.cu: HTTP ${sourceResponse.status}`);
  const source = await sourceResponse.text();

  positions = runtime.createBuffer(new Float32Array(VERTEX_COUNT * 4), { label: 'Face CUDA positions' });
  normals = runtime.createBuffer(new Float32Array(VERTEX_COUNT * 4), { label: 'Face CUDA normals' });

  setStatus('Compiling CUDA → WGSL…');
  const kernel = await runtime.kernel(source, {
    entry: 'GenerateFace',
    workgroupSize: [WORKGROUP, 1, 1]
  });

  invocation = kernel.bind(
    { positions, normals },
    scalarSnapshot()
  );

  rebuildFace();

  setStatus('Creating raw WebGPU renderer…');
  const topology = makeTopology(RINGS, LEVELS);
  renderer = await createFaceRenderer(gpuCanvas, runtime, positions, normals, topology);

  const gpu = runtime.describe();
  const name = gpu.description || gpu.device || gpu.vendor || 'WEBGPU';
  setStatus(`READY · ${String(name).toUpperCase()}`, 'ready');

  requestAnimationFrame(animationFrame);
}

async function dispose() {
  if (disposed) return;
  disposed = true;
  try { await runtime?.idle(); } catch {}
  renderer?.dispose();
  if (runtime && positions && !positions.destroyed) runtime.destroyBuffer(positions);
  if (runtime && normals && !normals.destroyed) runtime.destroyBuffer(normals);
  runtime?.dispose();
}

window.addEventListener('beforeunload', dispose);

start().catch(error => {
  console.error(error);
  setStatus('STARTUP ERROR', 'error');
  fatal.hidden = false;
  fatalText.textContent = error?.stack || error?.message || String(error);
});
