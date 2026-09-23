import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { compile } from '../vendor/webshader/compiler/compiler.js';

const root = path.resolve(import.meta.dirname, '..');
const source = await readFile(path.join(root, 'kernels', 'face.cu'), 'utf8');

const artifact = compile(source, {
  entry: 'GenerateFace',
  workgroupSize: [128, 1, 1],
  optimize: 'dependencies'
});

const bindings = artifact.metadata.bindings.map(b => `${b.name}:${b.elementType}`);
const scalars = artifact.metadata.scalars.map(s => `${s.name}:${s.type}`);

if (bindings.length !== 2 || !bindings[0].startsWith('positions:') || !bindings[1].startsWith('normals:')) {
  throw new Error(`Unexpected buffer ABI: ${bindings.join(', ')}`);
}
if (!artifact.wgsl?.includes('@compute')) {
  throw new Error('Compiler did not produce a compute shader.');
}

console.log('face.cu compiled successfully');
console.log(`WGSL bytes: ${Buffer.byteLength(artifact.wgsl, 'utf8').toLocaleString()}`);
console.log(`Bindings: ${bindings.join(', ')}`);
console.log(`Scalars (${scalars.length}): ${scalars.join(', ')}`);
