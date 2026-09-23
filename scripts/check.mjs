import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { compile } from '../vendor/webshader/compiler/compiler.js';

const root = path.resolve(import.meta.dirname, '..');
const source = await readFile(path.join(root, 'kernels', 'face.cu'), 'utf8');

const entries = [
  ['GenerateBaseHead', ['positions:vec4<f32>']],
  ['SculptStage', ['source:vec4<f32>', 'destination:vec4<f32>']],
  ['SmoothClay', ['source:vec4<f32>', 'destination:vec4<f32>']],
  ['ComputeNormals', ['positions:vec4<f32>', 'normals:vec4<f32>']]
];

for (const [entry, expectedBindings] of entries) {
  const artifact = compile(source, {
    entry,
    workgroupSize: [128, 1, 1],
    optimize: 'dependencies'
  });

  const bindings = artifact.metadata.bindings.map(
    b => `${b.name}:${b.elementType}`
  );

  if (bindings.join('|') !== expectedBindings.join('|')) {
    throw new Error(
      `${entry} buffer ABI mismatch. Expected ${expectedBindings.join(', ')}, got ${bindings.join(', ')}`
    );
  }

  if (!artifact.wgsl?.includes('@compute')) {
    throw new Error(`${entry} did not produce a compute shader.`);
  }

  console.log(
    `${entry}: ${Buffer.byteLength(artifact.wgsl, 'utf8').toLocaleString()} WGSL bytes; ${bindings.join(', ')}`
  );
}

console.log('iterative clay pipeline compiled successfully');
