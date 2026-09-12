// Reproduce the exact, local Three.js assets used by the algorithm atlas.
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const destination = resolve(root, 'wwwroot/vendor/three');
const version = '0.183.2';
const files = {
  'package/build/three.module.js': 'three.module.js',
  'package/build/three.core.js': 'three.core.js',
  'package/examples/jsm/controls/OrbitControls.js': 'addons/controls/OrbitControls.js',
  'package/LICENSE': 'LICENSE'
};
const hash = (data, algorithm = 'sha256', encoding = 'hex') => createHash(algorithm).update(data).digest(encoding);

if (process.argv.includes('--check')) {
  const manifest = JSON.parse(await readFile(resolve(destination, 'manifest.json'), 'utf8'));
  if (manifest.version !== version) throw new Error('Vendored Three.js version differs from the pinned version.');
  for (const file of manifest.files) {
    if (hash(await readFile(resolve(destination, file.path))) !== file.sha256) throw new Error(`Modified vendor file: ${file.path}`);
  }
  console.log(`PASS: Three.js ${version}, ${manifest.files.length} verified local assets.`);
} else {
  const metadataResponse = await fetch(`https://registry.npmjs.org/three/${version}`);
  if (!metadataResponse.ok) throw new Error(`npm metadata: HTTP ${metadataResponse.status}`);
  const metadata = await metadataResponse.json();
  const archiveResponse = await fetch(metadata.dist.tarball);
  if (!archiveResponse.ok) throw new Error(`npm archive: HTTP ${archiveResponse.status}`);
  const archive = Buffer.from(await archiveResponse.arrayBuffer());
  const integrity = `sha512-${hash(archive, 'sha512', 'base64')}`;
  if (integrity !== metadata.dist.integrity) throw new Error('npm archive integrity verification failed.');
  const tar = gunzipSync(archive);
  const written = [];
  for (let offset = 0; offset + 512 <= tar.length;) {
    const header = tar.subarray(offset, offset + 512);
    const name = header.subarray(0, 100).toString('utf8').replace(/\0.*$/, '');
    if (!name) break;
    const size = parseInt(header.subarray(124, 136).toString('ascii').replace(/\0.*$/, '').trim(), 8) || 0;
    const output = files[name];
    if (output) {
      const data = tar.subarray(offset + 512, offset + 512 + size);
      const path = resolve(destination, output);
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, data);
      written.push({ path: output, bytes: data.length, sha256: hash(data) });
    }
    offset += 512 + Math.ceil(size / 512) * 512;
  }
  if (written.length !== Object.keys(files).length) throw new Error('A required asset was missing from the npm archive.');
  await writeFile(resolve(destination, 'manifest.json'), JSON.stringify({
    package: 'three', version, license: 'MIT', source: metadata.dist.tarball,
    integrity, files: written
  }, null, 2) + '\n');
  console.log(`Vendored Three.js ${version}: ${written.map(x => x.path).join(', ')}`);
}
