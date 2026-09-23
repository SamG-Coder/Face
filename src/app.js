import { createFaceRenderer, makeTopology } from './face-renderer.js';
import { drawConstruction } from './construction.js';

const RINGS = 96;
const LEVELS = 72;
const VERTEX_COUNT = RINGS * LEVELS;
const WORKGROUP = 128;
const BLOCKS = [Math.ceil(VERTEX_COUNT / WORKGROUP), 1, 1];

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
let positionsA = null;
let positionsB = null;
let normals = null;
let renderer = null;

let baseInvocation = null;
let sculptInvocation = null;
let smoothInvocation = null;
let normalInvocation = null;

let disposed = false;
let rebuildRequested = false;
let plan = [];
let planIndex = 0;
let gpuName = 'WEBGPU';

function baseScalars() {
  return {
    rings: RINGS,
    levels: LEVELS,
    headWidth: params.headWidth,
    headDepth: params.headDepth,
    headHeight: params.headHeight
  };
}

function sculptScalars(stage, strength) {
  return {
    rings: RINGS,
    levels: LEVELS,
    stage,
    strength,
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

function smoothScalars(strength) {
  return {
    rings: RINGS,
    levels: LEVELS,
    strength
  };
}

function normalScalars() {
  return {
    rings: RINGS,
    levels: LEVELS
  };
}

function buildSculptPlan() {
  const result = [];

  const addStage = (stage, name, count, startStrength, endStrength, startSmooth, endSmooth) => {
    for (let i = 0; i < count; i++) {
      const t = count <= 1 ? 1 : i / (count - 1);
      result.push({
        stage,
        name,
        local: i + 1,
        count,
        strength: startStrength + (endStrength - startStrength) * t,
        smooth: startSmooth + (endSmooth - startSmooth) * t
      });
    }
  };

  addStage(0, 'BLOCKOUT', 12, 0.24, 0.10, 0.18, 0.11);
  addStage(1, 'SECONDARY', 14, 0.21, 0.09, 0.17, 0.10);
  addStage(2, 'FEATURES', 16, 0.17, 0.07, 0.14, 0.08);
  addStage(3, 'POLISH', 10, 0.085, 0.035, 0.12, 0.075);

  return result;
}

function resetSculpt() {
  if (!runtime || !baseInvocation || disposed) return;

  baseInvocation.setScalars(baseScalars());
  normalInvocation.setScalars(normalScalars());

  const batch = runtime.batch();
  batch.dispatch(baseInvocation, BLOCKS);
  batch.dispatch(normalInvocation, BLOCKS);
  batch.submit();

  plan = buildSculptPlan();
  planIndex = 0;
  rebuildRequested = false;
  setStatus('CLAY · BASE', 'ready');
}

function runSculptIteration() {
  if (!runtime || !sculptInvocation || planIndex >= plan.length || disposed) return;

  const step = plan[planIndex];

  sculptInvocation.setScalars(sculptScalars(step.stage, step.strength));
  smoothInvocation.setScalars(smoothScalars(step.smooth));
  normalInvocation.setScalars(normalScalars());

  const batch = runtime.batch();
  batch.dispatch(sculptInvocation, BLOCKS);
  batch.dispatch(smoothInvocation, BLOCKS);
  batch.dispatch(normalInvocation, BLOCKS);
  batch.submit();

  planIndex++;

  if (planIndex < plan.length) {
    setStatus(`CLAY · ${step.name} ${step.local}/${step.count}`, 'ready');
  } else {
    setStatus(`READY · ${gpuName}`, 'ready');
  }
}

function scheduleRebuild() {
  drawConstruction(frontCanvas, profileCanvas, params);
  rebuildRequested = true;
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

  if (rebuildRequested) {
    resetSculpt();
  } else if (planIndex < plan.length) {
    runSculptIteration();
  }

  renderer?.render();
  requestAnimationFrame(animationFrame);
}

async function loadGpuRuntime() {
  setStatus('Loading CUDA WebShader runtime…');

  try {
    return await import('../vendor/webshader/runtime/runtime.js');
  } catch (localError) {
    console.warn('Local WebShader runtime is unavailable; using the pinned GitHub Pages CDN copy.', localError);
  }

  return await import('https://cdn.jsdelivr.net/gh/SamG-Coder/cuda-webshader@f45a967480f0bcd8a2c9bd441e509745f94dd795/src/runtime/runtime.js');
}

async function start() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is unavailable in this browser. On Android, open the site in a current Chrome build with WebGPU support.');
  }

  const { GpuRuntime } = await loadGpuRuntime();

  setStatus('Creating WebGPU device…');
  runtime = await GpuRuntime.create({
    onError: error => console.error('WebShader GPU error', error)
  });

  setStatus('Loading clay kernels…');
  const sourceResponse = await fetch('./kernels/face.cu');

  if (!sourceResponse.ok) {
    throw new Error(`Failed to load kernels/face.cu: HTTP ${sourceResponse.status}`);
  }

  const source = await sourceResponse.text();

  positionsA = runtime.createBuffer(
    new Float32Array(VERTEX_COUNT * 4),
    { label: 'Clay positions A' }
  );
  positionsB = runtime.createBuffer(
    new Float32Array(VERTEX_COUNT * 4),
    { label: 'Clay positions B' }
  );
  normals = runtime.createBuffer(
    new Float32Array(VERTEX_COUNT * 4),
    { label: 'Clay normals' }
  );

  setStatus('Compiling clay stages…');

  const [baseKernel, sculptKernel, smoothKernel, normalKernel] = await Promise.all([
    runtime.kernel(source, {
      entry: 'GenerateBaseHead',
      workgroupSize: [WORKGROUP, 1, 1]
    }),
    runtime.kernel(source, {
      entry: 'SculptStage',
      workgroupSize: [WORKGROUP, 1, 1]
    }),
    runtime.kernel(source, {
      entry: 'SmoothClay',
      workgroupSize: [WORKGROUP, 1, 1]
    }),
    runtime.kernel(source, {
      entry: 'ComputeNormals',
      workgroupSize: [WORKGROUP, 1, 1]
    })
  ]);

  baseInvocation = baseKernel.bind(
    { positions: positionsA },
    baseScalars()
  );

  sculptInvocation = sculptKernel.bind(
    { source: positionsA, destination: positionsB },
    sculptScalars(0, .2)
  );

  smoothInvocation = smoothKernel.bind(
    { source: positionsB, destination: positionsA },
    smoothScalars(.15)
  );

  normalInvocation = normalKernel.bind(
    { positions: positionsA, normals },
    normalScalars()
  );

  setStatus('Creating raw WebGPU renderer…');
  const topology = makeTopology(RINGS, LEVELS);

  renderer = await createFaceRenderer(
    gpuCanvas,
    runtime,
    positionsA,
    normals,
    topology
  );

  const gpu = runtime.describe();
  gpuName = String(gpu.description || gpu.device || gpu.vendor || 'WEBGPU').toUpperCase();

  resetSculpt();
  requestAnimationFrame(animationFrame);
}

async function dispose() {
  if (disposed) return;
  disposed = true;

  try {
    await runtime?.idle();
  } catch {}

  renderer?.dispose();

  for (const buffer of [positionsA, positionsB, normals]) {
    if (runtime && buffer && !buffer.destroyed) {
      runtime.destroyBuffer(buffer);
    }
  }

  runtime?.dispose();
}

window.addEventListener('beforeunload', dispose);

start().catch(error => {
  console.error(error);
  setStatus('STARTUP ERROR', 'error');
  fatal.hidden = false;
  fatalText.textContent = error?.stack || error?.message || String(error);
});
