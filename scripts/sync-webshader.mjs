import { cp, mkdir, rm, access } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

const root = path.resolve(import.meta.dirname, '..');
const cache = path.join(root, '.cache', 'cuda-webshader');
const vendor = path.join(root, 'vendor', 'webshader');

function run(command, args, cwd = root) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, stdio: 'inherit', shell: false });
    child.on('error', reject);
    child.on('exit', code => code === 0 ? resolve() : reject(new Error(`${command} exited with code ${code}`)));
  });
}

async function exists(target) {
  try { await access(target); return true; } catch { return false; }
}

await mkdir(path.dirname(cache), { recursive: true });

if (await exists(path.join(cache, '.git'))) {
  await run('git', ['fetch', '--depth', '1', 'origin', 'main'], cache);
  await run('git', ['reset', '--hard', 'FETCH_HEAD'], cache);
} else {
  await rm(cache, { recursive: true, force: true });
  await run('git', ['clone', '--depth', '1', '--branch', 'main', 'https://github.com/SamG-Coder/cuda-webshader.git', cache]);
}

await rm(vendor, { recursive: true, force: true });
await mkdir(vendor, { recursive: true });
await cp(path.join(cache, 'src', 'compiler'), path.join(vendor, 'compiler'), { recursive: true });
await cp(path.join(cache, 'src', 'runtime'), path.join(vendor, 'runtime'), {
  recursive: true,
  filter: source => !source.endsWith('three-bridge.js')
});

console.log('Synced CUDA WebShader compiler/runtime (Three bridge excluded).');
