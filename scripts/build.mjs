import { cp, mkdir, rm } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const dist = path.join(root, 'dist');

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });

for (const entry of ['index.html', 'style.css', 'src', 'kernels', 'vendor']) {
  await cp(path.join(root, entry), path.join(dist, entry), { recursive: true });
}

console.log('Built static site in dist/.');
