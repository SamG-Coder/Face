import { perspective, lookAt, multiply, orbitEye } from './math.js';

const shader = String.raw`
struct Uniforms {
  viewProj: mat4x4<f32>,
  camera: vec4<f32>,
  light: vec4<f32>,
  settings: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> u: Uniforms;

struct VertexOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) smoothNormal: vec3<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOut {
  let p = positions[index].xyz;
  var out: VertexOut;
  out.clip = u.viewProj * vec4<f32>(p, 1.0);
  out.world = p;
  out.smoothNormal = normals[index].xyz;
  return out;
}

@fragment
fn fragmentClay(input: VertexOut) -> @location(0) vec4<f32> {
  let smoothN = normalize(input.smoothNormal);
  var flatN = normalize(cross(dpdx(input.world), dpdy(input.world)));
  if (dot(flatN, smoothN) < 0.0) {
    flatN = -flatN;
  }

  let useFlat = step(0.5, u.settings.x);
  let n = normalize(mix(smoothN, flatN, useFlat));
  let l = normalize(u.light.xyz);
  let v = normalize(u.camera.xyz - input.world);
  let h = normalize(l + v);

  let ndl = max(dot(n, l), 0.0);
  let spec = pow(max(dot(n, h), 0.0), 52.0) * 0.18;
  let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.14;

  let clay = vec3<f32>(0.63, 0.46, 0.36);
  let warm = vec3<f32>(0.17, 0.09, 0.055) * max(n.y * 0.5 + 0.5, 0.0);
  let color = clay * (0.19 + ndl * 0.83) + warm + vec3<f32>(spec) + vec3<f32>(0.20, 0.43, 0.52) * rim;

  return vec4<f32>(color * u.settings.y, 1.0);
}

@fragment
fn fragmentWire(input: VertexOut) -> @location(0) vec4<f32> {
  let n = normalize(input.smoothNormal);
  let v = normalize(u.camera.xyz - input.world);
  let facing = 0.35 + 0.65 * abs(dot(n, v));
  return vec4<f32>(0.35, 0.92, 0.79, 0.34 + 0.52 * facing);
}
`;

function makeBuffer(device, data, usage, label) {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, data.byteLength),
    usage,
    mappedAtCreation: true
  });
  new data.constructor(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

export function makeTopology(rings, levels) {
  const triangles = new Uint32Array((levels - 1) * rings * 6);
  const lines = new Uint32Array(((levels - 1) * rings + levels * rings) * 2);
  let t = 0;
  for (let y = 0; y < levels - 1; y++) {
    for (let x = 0; x < rings; x++) {
      const nx = (x + 1) % rings;
      const a = y * rings + x;
      const b = y * rings + nx;
      const c = (y + 1) * rings + x;
      const d = (y + 1) * rings + nx;
      triangles[t++] = a; triangles[t++] = b; triangles[t++] = c;
      triangles[t++] = b; triangles[t++] = d; triangles[t++] = c;
    }
  }

  let l = 0;
  for (let y = 0; y < levels; y++) {
    for (let x = 0; x < rings; x++) {
      const nx = (x + 1) % rings;
      lines[l++] = y * rings + x;
      lines[l++] = y * rings + nx;
    }
  }
  for (let y = 0; y < levels - 1; y++) {
    for (let x = 0; x < rings; x++) {
      lines[l++] = y * rings + x;
      lines[l++] = (y + 1) * rings + x;
    }
  }

  return { triangles, lines };
}

export async function createFaceRenderer(canvas, runtime, positions, normals, topology) {
  const device = runtime.device;
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('Could not create a WebGPU canvas context.');

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const module = device.createShaderModule({ label: 'Face raw WebGPU shader', code: shader });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(m => m.type === 'error');
  if (errors.length) throw new Error(errors.map(e => e.message).join('\n'));

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'Face render resources',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
    ]
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  const depthStencil = {
    format: 'depth24plus',
    depthWriteEnabled: true,
    depthCompare: 'less'
  };

  const clayPipeline = await device.createRenderPipelineAsync({
    label: 'Face clay pipeline',
    layout: pipelineLayout,
    vertex: { module, entryPoint: 'vertexMain' },
    fragment: { module, entryPoint: 'fragmentClay', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
    depthStencil
  });

  const wirePipeline = await device.createRenderPipelineAsync({
    label: 'Face wire pipeline',
    layout: pipelineLayout,
    vertex: { module, entryPoint: 'vertexMain' },
    fragment: {
      module,
      entryPoint: 'fragmentWire',
      targets: [{
        format,
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
        }
      }]
    },
    primitive: { topology: 'line-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' }
  });

  const uniformBuffer = device.createBuffer({
    label: 'Face camera/material uniforms',
    size: 112,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: positions.gpuBuffer } },
      { binding: 1, resource: { buffer: normals.gpuBuffer } },
      { binding: 2, resource: { buffer: uniformBuffer } }
    ]
  });

  const triangleBuffer = makeBuffer(
    device,
    topology.triangles,
    GPUBufferUsage.INDEX,
    'Face triangle topology'
  );
  const lineBuffer = makeBuffer(
    device,
    topology.lines,
    GPUBufferUsage.INDEX,
    'Face wire topology'
  );

  let depthTexture = null;
  let width = 0, height = 0;
  let yaw = 0.38, pitch = 0.02, distance = 3.65;
  let mode = 'smooth';
  const target = [0, 0.03, 0];

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (w === width && h === height) return;
    width = w; height = h;
    canvas.width = w; canvas.height = h;
    depthTexture?.destroy();
    depthTexture = device.createTexture({
      label: 'Face depth',
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
  }

  function writeUniforms() {
    const eye = orbitEye(yaw, pitch, distance, target);
    const view = lookAt(eye, target);
    const projection = perspective(42 * Math.PI / 180, width / height, 0.05, 50);
    const viewProj = multiply(projection, view);
    const data = new Float32Array(28);
    data.set(viewProj, 0);
    data.set([eye[0], eye[1], eye[2], 1], 16);
    data.set([0.42, 0.72, 0.55, 0], 20);
    data.set([mode === 'planes' ? 1 : 0, 1.0, 0, 0], 24);
    device.queue.writeBuffer(uniformBuffer, 0, data);
  }

  function render() {
    resize();
    writeUniforms();

    const encoder = device.createCommandEncoder({ label: 'Face render commands' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.035, g: 0.045, b: 0.06, a: 1 }
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store'
      }
    });

    if (mode === 'wire') {
      pass.setPipeline(wirePipeline);
      pass.setBindGroup(0, bindGroup);
      pass.setIndexBuffer(lineBuffer, 'uint32');
      pass.drawIndexed(topology.lines.length);
    } else {
      pass.setPipeline(clayPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.setIndexBuffer(triangleBuffer, 'uint32');
      pass.drawIndexed(topology.triangles.length);
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  function setMode(next) {
    if (!['smooth', 'planes', 'wire'].includes(next)) return;
    mode = next;
  }

  function resetCamera() {
    yaw = 0.38; pitch = 0.02; distance = 3.65;
  }

  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', event => {
    dragging = true;
    lastX = event.clientX; lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX; lastY = event.clientY;
    yaw -= dx * 0.007;
    pitch = Math.max(-1.18, Math.min(1.18, pitch + dy * 0.006));
  });
  canvas.addEventListener('pointerup', event => {
    dragging = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointercancel', () => { dragging = false; });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    distance = Math.max(2.25, Math.min(7.0, distance * Math.exp(event.deltaY * 0.001)));
  }, { passive: false });
  canvas.addEventListener('dblclick', resetCamera);
  canvas.addEventListener('contextmenu', event => event.preventDefault());

  return {
    render,
    setMode,
    resetCamera,
    dispose() {
      depthTexture?.destroy();
      triangleBuffer.destroy();
      lineBuffer.destroy();
      uniformBuffer.destroy();
      context.unconfigure();
    }
  };
}
