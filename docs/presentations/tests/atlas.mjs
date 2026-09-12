import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { StudySession, sourceLines, findSourceMatches } from '../wwwroot/nsga3-atlas/model.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repo = resolve(root, '../..');
const bundle = JSON.parse(await readFile(resolve(root, 'wwwroot/nsga3-atlas/source-data.json'), 'utf8'));
let checked = 0;
for (const file of bundle.files) {
  const bytes = await readFile(resolve(repo, file.path));
  assert.equal(file.content, bytes.toString('utf8'), `${file.path}: complete source contents`);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), file.sha256);
  assert.equal(sourceLines(file.content).length, file.lineCount, `${file.path}: viewer line count`);
  checked += 3;
}
assert.equal(bundle.files.reduce((n, file) => n + file.lineCount, 0), bundle.totalLines);
for (const stage of bundle.stages) {
  const file = bundle.files.find(file => file.id === stage.fileId);
  const symbol = file.symbols.find(symbol => symbol.id === stage.symbolId);
  assert.equal(symbol.startLine, stage.startLine);
  assert.equal(symbol.endLine, stage.endLine);
  assert.match(sourceLines(file.content)[stage.startLine - 1], new RegExp(symbol.name));
  checked += 3;
}
assert.deepEqual(sourceLines('one\r\n\r\nthree\n'), ['one', '', 'three']);
assert.deepEqual(findSourceMatches(['return A;', '', 'RETURN b;'], 'return'), [1, 3]);

const snapshot = generation => ({
  runId: 'test-run', generation, config: { populationSize: 12 },
  candidates: [], survivors: Array.from({ length: 12 }, (_, id) => ({ id })),
  selection: {}, metrics: {}
});
function existing(request) {
  const session = new StudySession(request);
  session.id = 'test-run'; session.history = [snapshot(0)]; return session;
}

// A successful POST advances once; browsing recorded history never contacts the server.
let posts = 0;
let session = existing(async (url, body, method = 'POST') => { assert.equal(method, 'POST'); posts++; return snapshot(1); });
await session.advance();
assert.equal(posts, 1);
session.inspect(0);
await session.advance();
assert.equal(posts, 1);
assert.equal(session.snapshot.generation, 1);

// A response is lost AFTER the server advances. GET reconciliation recovers that same generation.
posts = 0;
session = existing(async (url, body, method = 'POST') => {
  if (method === 'GET') return snapshot(1);
  posts++; throw new Error('Lost response');
});
await session.advance();
assert.equal(posts, 1);
assert.equal(session.history.length, 2);
assert.equal(session.uncertain, false);

// If both response and reconciliation fail, the next attempt MUST read before mutating again.
posts = 0;
let online = false;
session = existing(async (url, body, method = 'POST') => {
  if (method !== 'GET') { posts++; throw new Error('Disconnected after POST'); }
  if (!online) throw new Error('Still offline');
  return snapshot(1);
});
await assert.rejects(session.advance(), /Disconnected/);
assert.equal(session.uncertain, true);
assert.equal(session.busy, false);
assert.equal(session.snapshot.generation, 0);
online = true;
await session.advance();
assert.equal(posts, 1);
assert.equal(session.snapshot.generation, 1);

// A failed mutation whose GET proves no advance may be retried without skipping a generation.
posts = 0;
session = existing(async (url, body, method = 'POST') => {
  if (method === 'GET') return snapshot(0);
  if (++posts === 1) throw new Error('Request did not arrive');
  return snapshot(1);
});
await assert.rejects(session.advance(), /did not arrive/);
await session.advance();
assert.equal(posts, 2);
assert.equal(session.snapshot.generation, 1);

// History from another study is rejected, and overlapping requests cannot advance twice.
session = existing(async () => ({ ...snapshot(1), runId: 'wrong-study' }));
await assert.rejects(session.advance(), /different study/);
assert.equal(session.history.length, 1);
let finish;
posts = 0;
session = existing(async () => { posts++; return await new Promise(resolve => { finish = resolve; }); });
const inFlight = session.advance();
assert.equal(await session.advance(), false);
assert.equal(posts, 1);
finish(snapshot(1));
await inFlight;

console.log(`PASS: ${checked + 3} source/line/mapping assertions; full ${bundle.totalLines}-line fidelity; six recovery/history/concurrency scenarios.`);

if (process.argv.includes('--http')) {
  const base = process.env.PRESENTATION_URL || 'http://127.0.0.1:5088';
  const assets = ['/nsga3-atlas/', '/nsga3-atlas/app.js', '/nsga3-atlas/model.js', '/nsga3-atlas/code-view.js', '/nsga3-atlas/scene.js', '/nsga3-atlas/atlas.css', '/nsga3-atlas/source-data.json', '/nsga3-atlas/sourcebook.html', '/nsga3-atlas/sourcebook.md', '/vendor/three/three.module.js', '/vendor/three/three.core.js', '/vendor/three/addons/controls/OrbitControls.js'];
  for (const path of assets) {
    const response = await fetch(base + path);
    assert.equal(response.status, 200, path);
    assert.ok((await response.text()).length > 100, `${path}: nonempty asset`);
  }
  console.log(`PASS: ${assets.length} complete atlas/document/Three.js assets served over HTTP.`);
}
