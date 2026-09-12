#!/usr/bin/env node
/**
 * Lossless, offline NSGA-III sourcebook. Node standard library only.
 * Run: node docs/presentations/build-sourcebook.mjs [--check]
 * SOURCE_DATE_EPOCH can pin the snapshot timestamp for reproducible builds.
 * A successful --check verifies source bytes, generated outputs, and stage links.
 */
import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
const atlas = path.join(here, 'wwwroot/nsga3-atlas');
const checking = process.argv.includes('--check');
if (process.argv.slice(2).some(arg => arg !== '--check')) throw new Error('Usage: node build-sourcebook.mjs [--check]');
const hash = value => createHash('sha256').update(value).digest('hex');
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const slug = value => value.replace(/([A-Z]+)([A-Z][a-z])/g, '$1-$2').replace(/([a-z0-9])([A-Z])/g, '$1-$2').replace(/[^\w]+/g, '-').replace(/^-|-$/g, '').toLowerCase();
const splitLines = content => { const lines = content.split(/\r\n|\n|\r/); if (lines.at(-1) === '') lines.pop(); return lines; };
const relative = name => `src/OptunaSharp.Core/${name}`;

// Whole files only. Recursion includes future operators added to the shared subtree.
const explicit = [
  ['sampler', 'Samplers/NSGAIIISampler.cs', 'NSGA-III core'],
  ['selection', 'Samplers/Nsgaiii/NSGAIIIElitePopulationSelectionStrategy.cs', 'NSGA-III core'],
  ['ga', 'Samplers/BaseGASampler.cs', 'Generation lifecycle'],
  ['base-sampler', 'Samplers/BaseSampler.cs', 'Generation lifecycle'],
  ['random', 'Samplers/RandomSampler.cs', 'Random sampling'],
  ['rng', 'Samplers/LazyRandomState.cs', 'Random sampling'],
  ['multiobjective', 'Internal/MultiObjective.cs', 'Ranking and constraints'],
  ['constraints', 'Internal/ConstrainedOptimization.cs', 'Ranking and constraints'],
  ['intersection', 'SearchSpace/IntersectionSearchSpace.cs', 'Search space'],
  ['conditional', 'SearchSpace/ConditionalParams.cs', 'Search space'],
  ['transform', 'Internal/SearchSpaceTransform.cs', 'Search space'],
  ['linalg', 'Internal/LinAlg.cs', 'Numerical support'],
  ['math', 'Internal/MathUtil.cs', 'Numerical support'],
  ['openblas', 'Internal/OpenBlasActivation.cs', 'Numerical support'],
  ['python-repr', 'Internal/PythonRepr.cs', 'Local support'],
  ['json', 'Internal/PyJson.cs', 'Local support'],
  ['warnings', 'Internal/OptunaWarnings.cs', 'Local support'],
];
const sharedIds = {
  NSGAIIChildGenerationStrategy: 'child', NSGAIIElitePopulationSelectionStrategy: 'sorting',
  NSGAIIAfterTrialStrategy: 'after-trial', NsgaiiConstraintsEvaluation: 'constraint-evaluation',
  NsgaiiCrossover: 'crossover', NsgaiiMutation: 'mutation',
};
async function csFiles(directory) {
  const result = [];
  for (const entry of await readdir(path.join(root, relative(directory)), { withFileTypes: true })) {
    const name = `${directory}/${entry.name}`;
    if (entry.isDirectory()) result.push(...await csFiles(name));
    else if (entry.isFile() && name.endsWith('.cs')) result.push(name);
  }
  return result.sort();
}
for (const name of await csFiles('Samplers/Nsgaii')) {
  const basename = path.basename(name, '.cs');
  const category = name.includes('/Crossovers/') ? 'Crossover operators' : name.includes('/Mutations/') ? 'Mutation operators' : 'Shared genetic machinery';
  explicit.push([sharedIds[basename] ?? slug(basename), name, category]);
}
for (const name of await csFiles('Distributions')) explicit.push([`distribution-${slug(path.basename(name, '.cs'))}`, name, 'Parameter distributions']);
for (const name of (await csFiles('Collections')).filter(name => path.basename(name).startsWith('OrderedDict'))) {
  explicit.push([slug(path.basename(name, '.cs')), name, 'Ordered collections']);
}

/**
 * Index ordinary four-space-indented type members in this codebase, not call sites.
 * Range boundaries use the repository's member indentation, avoiding braces inside
 * interpolated strings. A range begins at its declaration and includes its closing
 * brace/semicolon. The full file remains available even for unindexed nested members.
 */
function symbolsFor(fileId, content) {
  const lines = splitLines(content), symbols = [], occurrences = new Map();
  const declaration = /^    (?:public|private|internal|protected)\s+[^;={}]*?\b(?!(?:static|override|virtual|abstract|readonly|async)\b)([A-Za-z_]\w*)(?:<[^>\r\n]+>)?\s*\(/;
  for (let start = 0; start < lines.length; start++) {
    const match = declaration.exec(lines[start]);
    if (!match) continue;
    let end = start;
    let found = false;
    for (let j = start; j < lines.length; j++) {
      if (j > start && declaration.test(lines[j])) break;
      if (/^    \{\s*$/.test(lines[j])) {
        for (let k = j + 1; k < lines.length; k++) {
          if (/^    \}\s*(?:;)?\s*$/.test(lines[k])) { end = k; found = true; break; }
        }
        break;
      }
      if ((lines[j].includes('=>') || j === start && lines[j].trimEnd().endsWith(';'))) {
        for (let k = j; k < lines.length; k++) {
          if (lines[k].trimEnd().endsWith(';')) { end = k; found = true; break; }
        }
        break;
      }
    }
    if (!found) continue;
    const name = match[1], base = `${fileId}.${slug(name)}`;
    const ordinal = (occurrences.get(base) ?? 0) + 1;
    occurrences.set(base, ordinal);
    symbols.push({ id: ordinal === 1 ? base : `${base}-${ordinal}`, name, startLine: start + 1, endLine: end + 1 });
  }
  return symbols;
}

const files = [];
for (const [id, local, category] of explicit) {
  const sourcePath = relative(local), bytes = await readFile(path.join(root, sourcePath));
  // ignoreBOM:true preserves a BOM as U+FEFF. Re-encoding must reproduce every byte.
  const content = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
  if (!Buffer.from(content, 'utf8').equals(bytes)) throw new Error(`UTF-8 roundtrip failed: ${sourcePath}`);
  files.push({ id, path: sourcePath, name: path.basename(local), category, lineCount: splitLines(content).length,
    sha256: hash(bytes), content, symbols: symbolsFor(id, content) });
}
if (new Set(files.map(file => file.id)).size !== files.length) throw new Error('Duplicate source file IDs');
const byId = new Map(files.map(file => [file.id, file]));
const symbol = (fileId, symbolId) => {
  const found = byId.get(fileId)?.symbols.find(entry => entry.id === `${fileId}.${symbolId}`);
  if (!found) throw new Error(`Missing stage symbol: ${fileId}.${symbolId}`);
  return found;
};
function stage(id, title, summary, fileId, symbolId, formula, variables, next) {
  const method = symbol(fileId, symbolId);
  method.summary = summary;
  return { id, title, summary, fileId, symbolId: method.id, startLine: method.startLine, endLine: method.endLine, formula, variables, next };
}
const stages = [
  stage('initialize', 'Initialize a population', 'Initial parameters are sampled independently within their distributions. Study evaluates each suggested trial; the sampler does not evaluate the objective.',
    'sampler', 'sample-independent', 'xⱼ ∼ distributionⱼ; P₀ = {evaluate(x₁), …, evaluate(xₙ)}', ['N: population size', 'x: decision vector', 'P₀: evaluated initial population'], ['variation']),
  stage('variation', 'Breed and mutate', 'Dominance tournaments select parents. The configured crossover produces a child, or one parent is copied. Parameters mutate with the configured probability; absent proposals use independent sampling.',
    'child', 'generate', 'Qₜ = evaluate(mutate(crossover(Pₜ))); pₘ = configured value or 1 / max(1, n_params)', ['Pₜ: selected parents', 'Qₜ: evaluated offspring', 'pₘ: mutation probability'], ['combine']),
  stage('combine', 'Combine parents and offspring', 'SelectParent gathers the previous generation and its cached parent population, then invokes the NSGA-III elite strategy.',
    'sampler', 'select-parent', 'Rₜ = Qₜ ∪ Pₜ', ['Rₜ: combined candidate population', 'Qₜ: previous completed generation', 'Pₜ: previous cached parent population'], ['rank']),
  stage('rank', 'Sort by nondomination', 'Ranking converts maximization objectives to losses and applies nondominated sorting. When constrained, feasible trials precede infeasible trials ranked by total positive violation.',
    'sorting', 'rank-population', 'a ≺ b ⇔ (∀j: aⱼ ≤ bⱼ) ∧ (∃j: aⱼ < bⱼ); Rₜ → F₀, F₁, …', ['Fᵢ: nondomination front', 'a, b: loss vectors', 'penalty: Σ max(0, constraint)'], ['normalize']),
  stage('normalize', 'Normalize the selection set', 'After replacing infinities, subtract the ideal point, find ASF extreme points, and solve the extreme-point hyperplane. A degenerate matrix uses per-axis maxima instead. The selection set contains accepted fronts plus the borderline front.',
    'selection', 'normalize-objective-values', 'zⱼ = minᵢ fᵢⱼ; Eα = 1; yᵢⱼ = (fᵢⱼ − zⱼ) αⱼ', ['z: ideal point', 'E: ASF extreme-point matrix', 'α: inverse intercepts', 'y: normalized objective vector'], ['associate']),
  stage('associate', 'Associate with a reference direction', 'Associate every normalized individual with the reference line having minimum perpendicular distance. Exact ties keep the first reference row.',
    'selection', 'associate-individuals-with-reference-points', 'πᵣ(y) = ((y · r) / ‖r‖²) r; d(y,r) = ‖y − πᵣ(y)‖; r* = argminᵣ d(y,r)', ['r: reference vector', 'πᵣ: orthogonal projection onto span(r)', 'd: perpendicular distance'], ['niche']),
  stage('niche', 'Fill the least occupied niches', 'Only accepted fronts initially contribute occupancy. Among reference directions with borderline candidates, choose a least occupied direction. An empty niche takes its nearest candidate; a populated niche uses the sampler’s shuffled order.',
    'selection', 'preserve-niche-individuals', 'r* ∈ argminᵣ ρᵣ; ρᵣ = 0 ⇒ nearest candidate; ρᵣ > 0 ⇒ shuffled candidate; ρᵣ ← ρᵣ + 1', ['ρᵣ: selected occupancy of direction r', 'K: remaining survivor slots', 'borderline: front that reaches or exceeds N'], ['survive']),
  stage('survive', 'Keep the next parent population', 'Append the niche-selected members to all accepted earlier fronts and return the elite population. BaseGASampler caches these parents for subsequent child generation.',
    'selection', 'select', 'Pₜ₊₁ = accepted fronts ∪ niche-selected members; |Pₜ₊₁| = N', ['Pₜ₊₁: next parent population', 'N: target size, when enough completed candidates exist'], ['variation']),
];
for (const entry of stages) for (const next of entry.next) if (!stages.some(item => item.id === next)) throw new Error(`Invalid edge ${entry.id} → ${next}`);

// Curated local call relationships, resolved against the symbol index. Delegate
// edges describe the constructor's default configuration, not all possible overrides.
const callSpecs = [
  ['sampler.infer-relative-search-space', 'intersection.calculate-2', 'calculate the evolving search space'],
  ['sampler.select-parent', 'ga.get-population', 'load the completed generation'],
  ['sampler.select-parent', 'ga.get-parent-population', 'load its cached parents'],
  ['sampler.select-parent', 'selection.select', 'default elite-selection delegate'],
  ['sampler.sample-relative', 'ga.get-trial-generation', 'attribute this trial to a generation'],
  ['sampler.sample-relative', 'ga.get-parent-population', 'retrieve selected parents'],
  ['sampler.sample-relative', 'child.generate', 'default child-generation delegate'],
  ['sampler.sample-independent', 'random.sample-independent', 'independent parameter sampling'],
  ['sampler.after-trial', 'after-trial.call', 'default after-trial delegate'],
  ['ga.get-parent-population', 'sampler.select-parent', 'virtual dispatch to NSGAIIISampler on cache miss'],
  ['selection.select', 'constraints.is-constrained-optimization', 'detect constrained trials'],
  ['selection.select', 'constraint-evaluation.validate-constraints', 'validate constraint values'],
  ['selection.select', 'sorting.rank-population', 'shared nondomination ranking'],
  ['selection.select', 'selection.generate-default-reference-point', 'lazy default reference grid'],
  ['selection.select', 'selection.filter-inf', 'replace infinite objectives'],
  ['selection.select', 'selection.normalize-objective-values', 'normalize the selection set'],
  ['selection.select', 'selection.associate-individuals-with-reference-points', 'nearest reference direction'],
  ['selection.select', 'selection.preserve-niche-individuals', 'fill the final survivor slots'],
  ['selection.generate-default-reference-point', 'selection.combinations-with-replacement', 'enumerate the simplex lattice'],
  ['selection.normalize-objective-values', 'linalg.matrix-rank', 'test extreme-point degeneracy'],
  ['selection.normalize-objective-values', 'linalg.solve', 'fit inverse intercepts'],
  ['selection.preserve-niche-individuals', 'selection.get-bucket', 'occupancy buckets'],
  ['selection.preserve-niche-individuals', 'selection.shuffle-in-place', 'random tie order'],
  ['sorting.rank-population', 'constraint-evaluation.evaluate-penalty', 'positive constraint violation'],
  ['sorting.rank-population', 'multiobjective.fast-non-domination-rank', 'compute all fronts'],
  ['multiobjective.fast-non-domination-rank', 'multiobjective.calculate-nondomination-rank', 'rank feasible and infeasible sets'],
  ['multiobjective.calculate-nondomination-rank', 'multiobjective.unique-rows', 'lexicographically deduplicate'],
  ['multiobjective.calculate-nondomination-rank', 'multiobjective.is-pareto-front', 'peel each nondominated front'],
  ['multiobjective.is-pareto-front', 'multiobjective.is-pareto-front-for-unique-sorted', 'dimension-specialized frontier'],
  ['multiobjective.is-pareto-front-for-unique-sorted', 'multiobjective.is-pareto-front2d', 'two-objective scan'],
  ['multiobjective.is-pareto-front-for-unique-sorted', 'multiobjective.is-pareto-front-nd', 'many-objective frontier'],
  ['child.generate', 'constraints.is-constrained-optimization', 'choose dominance relation'],
  ['child.generate', 'crossover.perform-crossover', 'ordinary parameter crossover'],
  ['child.generate', 'crossover.perform-conditional-crossover', 'opt-in conditional crossover'],
  ['child.generate', 'conditional.prune', 'remove inactive parameters from copied parents'],
  ['child.generate', 'mutation.perform-mutation', 'apply configured mutation'],
  ['crossover.perform-crossover', 'crossover.select-parents', 'select distinct tournament winners'],
  ['crossover.perform-crossover', 'crossover.try-crossover', 'produce one candidate child'],
  ['crossover.perform-crossover', 'crossover.is-contained', 'reject out-of-bounds child'],
  ['crossover.perform-conditional-crossover', 'crossover.select-parents', 'same parent-selection rule'],
  ['crossover.perform-conditional-crossover', 'crossover.try-conditional-crossover', 'cross shared parameters and inherit missing ones'],
  ['crossover.perform-conditional-crossover', 'conditional.prune', 'discard child-inactive parameters'],
  ['crossover.perform-conditional-crossover', 'crossover.is-contained', 'validate active child values'],
  ['crossover.select-parents', 'crossover.select-parent', 'one binary tournament per parent'],
  ['crossover.select-parent', 'multiobjective.dominates', 'unconstrained dominance delegate'],
  ['crossover.select-parent', 'constraint-evaluation.constrained-dominates', 'constrained dominance delegate'],
  ['constraint-evaluation.constrained-dominates', 'multiobjective.dominates', 'objective comparison when both are feasible'],
  ['crossover.try-crossover', 'crossover.inlined-categorical-uniform-crossover', 'categorical uniform swap'],
  ['crossover.try-crossover', 'transform.transform', 'numerical parent encoding'],
  ['crossover.try-crossover', 'transform.untransform', 'numerical child decoding'],
  ['mutation.perform-mutation', 'transform.transform', 'encode before mutation'],
  ['mutation.perform-mutation', 'transform.untransform', 'decode mutated parameter'],
  ['random.sample-independent', 'transform.search-space-transform', 'distribution bounds'],
  ['random.sample-independent', 'transform.untransform', 'decode uniform draw'],
  ['after-trial.call', 'base-sampler.process-constraints-after-trial', 'legacy constraint callback'],
  ['intersection.calculate-2', 'intersection.calculate', 'ordinary intersection mode'],
  ['intersection.calculate-2', 'intersection.calculate-conditional', 'conditional extension mode'],
];
const symbolById = new Map(files.flatMap(file => file.symbols.map(method => [method.id, method])));
const calls = callSpecs.map(([from, to, label]) => {
  if (!symbolById.has(from) || !symbolById.has(to)) throw new Error(`Invalid local call reference ${from} → ${to}`);
  return { from, to, label };
});

const revision = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
const workingTreeModifiedFiles = execFileSync('git', ['diff', 'HEAD', '--name-only', '--', ...files.map(file => file.path)], { cwd: root, encoding: 'utf8' }).trim().split(/\r?\n/).filter(Boolean).sort();
const snapshotSha256 = hash(files.map(file => `${file.path}\0${file.sha256}\n`).join(''));
let previous;
try { previous = JSON.parse(await readFile(path.join(atlas, 'source-data.json'), 'utf8')); } catch (error) { if (checking) throw new Error(`Generate sourcebook first: ${error.message}`); }
const generatedAt = process.env.SOURCE_DATE_EPOCH
  ? new Date(Number(process.env.SOURCE_DATE_EPOCH) * 1000).toISOString()
  : previous?.revision === revision && previous?.snapshotSha256 === snapshotSha256 ? previous.generatedAt : new Date().toISOString();
const boundary = {
  included: 'Every line of NSGAIIISampler, NSGAIIIElitePopulationSelectionStrategy, BaseGASampler, BaseSampler (including constraint post-processing), RandomSampler, LazyRandomState, the entire shared Nsgaii subtree, nondominated sorting, constraint evaluation, intersection and conditional search spaces, transforms, distributions, ordered dictionaries, and the listed numerical/formatting support files.',
  excluded: [
    'Study, Trial, FrozenTrial, Optuna entry points, trial states/directions, objective evaluation, optimization orchestration, storage implementations, logging, project/assembly setup, and other OptunaSharp framework services.',
    'NumSharp NDArray, NumPyRandom/MT19937 implementation, NumSharp exceptions, NumSharp.Interop.OpenBLAS, native BLAS/LAPACK, .NET runtime/BCL, and their transitive dependencies.',
    'Other samplers (including NSGAIISampler itself), tests, the upstream Python implementation, and this presentation host/JavaScript; these are not the NSGA-III algorithm source bundle.',
  ],
  completeDependencyClosure: false,
  notes: [
    'Whole files are included even when only one method is used by NSGA-III. In particular NSGA-II crowding-distance selection is present because the same file supplies RankPopulation; NSGA-III does not invoke that crowding-distance path.',
    'The default sampler creates UniformCrossover and mutates by independent resampling. The presentation host configures SBXCrossover and PolynomialMutation; both full operator implementations are included.',
    'ConditionalParams is an opt-in OptunaSharp extension. It preserves eligible conditional parameters for evolution; its disabled path retains the standard intersection-space behavior.',
    'Source comments are reproduced as written, including historical parity statements. This source archive is not a new claim that every comment has been independently verified.',
    'The presentation replays a real completed backend generation. Diagnostic stages and highlighted methods explain those results; they are not instruction-level debugger stepping or a live CPU program counter.',
    'The eight stages form a teaching-level flow graph. Objective evaluation happens in Study/the user objective outside this archive; member ranges link the actual implementation responsible for each step.',
  ],
};
const data = { title: 'NSGA-III — the complete algorithm sourcebook', generatedAt, revision, snapshotSha256,
  totalLines: files.reduce((sum, file) => sum + file.lineCount, 0), fileCount: files.length, workingTreeModifiedFiles,
  sourceEncoding: 'UTF-8; content preserves original BOM, line endings and trailing newline',
  lineCountConvention: 'Physical lines; a terminal line break does not add an extra empty line. Blank lines within a file count.',
  symbolIndexConvention: 'Top-level member declarations in the current repository formatting; nested members may be unindexed. All source remains present.',
  callGraphConvention: 'Curated local calls, with default delegate and NSGAIIISampler virtual dispatch explicitly labeled. External runtime calls and custom strategy overrides are outside this graph.',
  boundary, files, stages, calls };

const intro = `This is a verbatim snapshot of the local OptunaSharp NSGA-III implementation and its explicitly listed algorithm/support files. **${data.fileCount} complete files · ${data.totalLines.toLocaleString('en-US')} physical lines.** Every included file retains its comments, blank lines, imports, declarations, and implementation. Nothing inside a file is elided or rewritten.\n\nGit HEAD: \`${revision}\`. Snapshot: \`${generatedAt}\`. Aggregate source SHA-256: \`${snapshotSha256}\`.\n\nThe bytes come from the current working tree, not a Git checkout reconstructed at HEAD. ${workingTreeModifiedFiles.length ? `${workingTreeModifiedFiles.length} included files differ from HEAD; see the manifest metadata in source-data.json.` : 'All included files match HEAD.'} Per-file SHA-256 values hash the original bytes, and UTF-8 decoding is checked by byte-for-byte re-encoding. Line endings are preserved in the machine-readable bundle; HTML renders them as numbered physical lines.\n`;
const instructions = `Run from the repository root with Node.js:\n\n\`\`\`sh\nnode docs/presentations/build-sourcebook.mjs\nnode docs/presentations/build-sourcebook.mjs --check\n\`\`\`\n\nThe generator uses only Node standard-library modules. It reads full source files, discovers every .cs file beneath the shared Nsgaii directory, computes hashes and symbol ranges, and regenerates this Markdown document, the printable HTML sourcebook, its CSS, the served Markdown copy, and source-data.json. The check mode writes nothing and fails if any source or generated output has drifted. Unchanged source/revision retains its snapshot timestamp; set SOURCE_DATE_EPOCH to pin that timestamp for a reproducible build.\n\nsource-data.json is the lossless download: each files[].content string re-encodes to the original UTF-8 bytes and SHA-256. HTML line anchors are \`#<file-id>-L<number>\`. Symbol IDs are stable semantic names; overloaded members have ordinal suffixes. Stage links are regenerated from actual member boundaries, not hard-coded line numbers.\n`;
let markdown = `# ${data.title}\n\n${intro}\n## Scope and dependency boundary\n\n${boundary.included}\n\nThis is **not the entire .NET dependency closure** and is not a standalone compilable project. The following boundary remains outside the source bundle:\n\n${boundary.excluded.map(text => `- ${text}`).join('\n')}\n\n${boundary.notes.map(text => `- ${text}`).join('\n')}\n\n## Regenerate and verify\n\n${instructions}\n## Algorithm map\n\n`;
markdown += stages.map(entry => `### ${entry.title}\n\n${entry.summary}\n\n\`${entry.formula}\`\n\nImplementation: \`${byId.get(entry.fileId).path}\`, lines ${entry.startLine}–${entry.endLine} (\`${entry.symbolId}\`).\n\n${entry.variables.map(text => `- ${text}`).join('\n')}\n`).join('\n');
markdown += '\n## Source manifest\n\n| File | Lines | SHA-256 of original bytes |\n| --- | ---: | --- |\n';
markdown += files.map(file => `| [${file.path}](#${file.id}) | ${file.lineCount} | \`${file.sha256}\` |`).join('\n');
for (const file of files) {
  const fence = '`'.repeat(Math.max(3, ...[...file.content.matchAll(/`+/g)].map(match => match[0].length + 1)));
  markdown += `\n\n<a id="${file.id}"></a>\n\n## ${file.path}\n\n${file.category} · ${file.lineCount} physical lines · SHA-256: \`${file.sha256}\`\n\n${fence}csharp\n${file.content}${/\r?\n$/.test(file.content) ? '' : '\n'}${fence}\n`;
}

const css = `:root{color-scheme:dark;--bg:#0b111b;--ink:#e9eef6;--muted:#a0adbf;--edge:#233043;--cyan:#72e4d1;--gold:#edc486;font-family:Inter,Segoe UI,Arial,sans-serif}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);line-height:1.65}a{color:var(--cyan);text-underline-offset:.2em}a:hover{color:#fff}.layout{display:grid;grid-template-columns:290px minmax(0,1fr);max-width:1760px;margin:auto}.sidebar{position:sticky;top:0;height:100vh;overflow:auto;padding:30px 22px;border-right:1px solid var(--edge);font-size:12px}.brand{font-size:12px;letter-spacing:.18em;color:var(--gold)}.sidebar nav>a{display:block;padding:8px 0;text-decoration:none}.sidebar .files{border-top:1px solid var(--edge);margin-top:22px;padding-top:14px}.sidebar small{display:block;color:var(--muted)}main{min-width:0;padding:54px clamp(20px,5vw,80px) 100px}.eyebrow{text-transform:uppercase;color:var(--cyan);letter-spacing:.22em;font-size:11px}h1{font-family:Georgia,serif;font-weight:400;font-size:clamp(36px,5vw,70px);line-height:1.06;letter-spacing:-.025em;max-width:800px;margin:18px 0 28px}h2{font-size:25px;font-weight:500;margin-top:48px}h3{font-size:18px;color:var(--gold)}p,li{max-width:960px}.lead{font-size:20px;color:var(--muted);max-width:850px}.stats{display:flex;flex-wrap:wrap;gap:36px;padding:25px 0;border-top:1px solid var(--edge);border-bottom:1px solid var(--edge);margin:32px 0}.stats strong{display:block;font-size:29px;font-weight:400;color:var(--gold)}.stats span,.meta{font-size:12px;color:var(--muted)}.actions{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.actions a,.actions button{border:1px solid var(--edge);background:#142231;color:var(--ink);padding:10px 15px;border-radius:5px;font:inherit;font-size:13px;text-decoration:none;cursor:pointer}.actions a:hover,.actions button:hover{border-color:var(--cyan)}code{font-family:Cascadia Code,Consolas,monospace;font-size:.9em}.meta code{overflow-wrap:anywhere}.scope{border-left:2px solid var(--gold);padding:3px 22px;background:#111b28}.manifest{width:100%;border-collapse:collapse;font-size:12px}.manifest th,.manifest td{text-align:left;padding:12px 8px;border-bottom:1px solid var(--edge);vertical-align:top}.manifest code{font-size:10px;overflow-wrap:anywhere}.manifest .file-path{overflow-wrap:anywhere}.manifest .digest{max-width:220px}.stage-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.stage{padding:20px;border:1px solid var(--edge);border-radius:7px}.stage h3{margin:0 0 10px}.stage p{font-size:13px;margin:12px 0}.formula{color:var(--cyan);font-family:Georgia,serif}.source-file{scroll-margin-top:20px;margin-top:64px}.source-file h2{font-size:22px;overflow-wrap:anywhere}.source-file .path{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.source{margin-top:20px;border:1px solid var(--edge);border-radius:5px;background:#0e1724;overflow:auto;padding:15px 0;font-family:Cascadia Code,Consolas,monospace;font-size:12px;line-height:1.65;tab-size:4}.source-line{display:block;min-height:1.65em;scroll-margin-top:80px;white-space:pre}.source-line:target{background:#3c3521;box-shadow:inset 3px 0 var(--gold)}.line-no{display:inline-block;vertical-align:top;width:5.5em;padding-right:1.5em;text-align:right;color:#6f829b;text-decoration:none;user-select:none}.line-content{white-space:pre}.source-line:hover{background:#18283b}.file-top{font-size:11px}.symbol-list{font-size:12px;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 18px}.run{white-space:pre-wrap;padding:18px;background:#101e2c;border-radius:4px}.footer{margin-top:60px;border-top:1px solid var(--edge);padding-top:25px;color:var(--muted);font-size:12px}@media(max-width:900px){.layout{grid-template-columns:1fr}.sidebar{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--edge);padding:20px}.sidebar .files{display:none}.sidebar nav{display:flex;gap:16px;flex-wrap:wrap}.stage-grid{grid-template-columns:1fr}main{padding:30px 16px 80px}.source{font-size:11px}.manifest .digest{max-width:120px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{:root{color-scheme:light;--bg:#fff;--ink:#111;--muted:#555;--edge:#ddd;--cyan:#164b56;--gold:#76521e}body{font-size:10pt}.layout{display:block;max-width:none}.sidebar,.actions,.file-top,.symbol-list{display:none}main{padding:0}.stage-grid{display:block}.stage{break-inside:avoid;margin:10pt 0}h1{font-size:35pt}.source-file{break-before:page}.source{background:white;border:0;overflow:visible;font-size:7.1pt;line-height:1.4;padding:0}.source-line{white-space:pre-wrap;overflow-wrap:anywhere;min-height:1.4em;break-inside:avoid;display:flex}.line-no{flex:0 0 4.5em}.line-content{white-space:pre-wrap;overflow-wrap:anywhere;min-width:0;flex:1}.source-line:target{background:white;box-shadow:none}.meta{font-size:8pt}.manifest{font-size:8pt}.manifest code{font-size:7pt}.manifest thead{display:table-header-group}.manifest tr{break-inside:avoid}a{color:inherit;text-decoration:none}.scope,.run{background:#f7f7f7}.stats{margin:20pt 0}.footer{font-size:8pt}@page{size:A4 landscape;margin:14mm 12mm}`;

const fileNav = files.map(file => `<a href="#${file.id}">${escape(file.name)}<small>${file.category} · ${file.lineCount} lines</small></a>`).join('');
const stageCards = stages.map((entry, index) => `<article class="stage"><h3>${String(index + 1).padStart(2, '0')} · ${entry.title}</h3><p>${escape(entry.summary)}</p><p class="formula">${escape(entry.formula)}</p><p><a href="#${entry.fileId}-L${entry.startLine}">${escape(entry.symbolId)} · L${entry.startLine}–${entry.endLine}</a></p></article>`).join('');
const manifest = files.map(file => `<tr><td class="file-path"><a href="#${file.id}">${escape(file.path)}</a></td><td>${file.lineCount}</td><td class="digest"><code>${file.sha256}</code></td></tr>`).join('');
const sourceSections = files.map(file => `<section class="source-file" id="${file.id}"><a class="file-top" href="#manifest">↑ Source manifest</a><h2>${escape(file.name)}</h2><p class="path">${escape(file.path)}</p><p class="meta">${file.category} · ${file.lineCount} physical lines<br>SHA-256 <code>${file.sha256}</code></p><div class="symbol-list">${file.symbols.map(method => `<a href="#${file.id}-L${method.startLine}">${escape(method.name)}:${method.startLine}</a>`).join('')}</div><pre class="source" aria-label="Complete source for ${escape(file.name)}"><code>${splitLines(file.content).map((line, index) => `<span class="source-line" id="${file.id}-L${index + 1}"><a class="line-no" href="#${file.id}-L${index + 1}" aria-label="Line ${index + 1}">${index + 1}</a><span class="line-content">${escape(line)}</span></span>`).join('')}</code></pre></section>`).join('');
const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Every line of OptunaSharp NSGA-III and its declared algorithm support: printable sourcebook, verified source hashes, and navigable methods."><title>${data.title}</title><style>${css}</style></head>
<body id="top"><div class="layout"><aside class="sidebar"><div class="brand">OPTUNASHARP / SOURCEBOOK</div><nav><a href="./">← Interactive atlas</a><a href="#scope">Scope &amp; provenance</a><a href="#algorithm">Algorithm map</a><a href="#manifest">Source manifest</a><a href="#regenerate">Regenerate &amp; verify</a></nav><nav class="files" aria-label="Source files">${fileNav}</nav></aside><main><header><p class="eyebrow">NSGA-III / every line, in context</p><h1>The geometry.<br>The machinery.<br>The complete source.</h1><p class="lead">A readable, verifiable document of the actual local C# algorithm. Full files, original comments, numbered lines, and a path from each idea to its implementation.</p><div class="stats"><div><strong>${data.fileCount}</strong><span>complete files</span></div><div><strong>${data.totalLines.toLocaleString('en-US')}</strong><span>physical source lines</span></div><div><strong>8</strong><span>linked algorithm stages</span></div></div><div class="actions"><a href="sourcebook.md" download="NSGA-III-SOURCEBOOK.md">Download Markdown</a><a href="source-data.json" download>Download lossless source JSON</a><button type="button" onclick="window.print()">Print / save PDF</button><a href="./">Open interactive atlas</a></div></header>
<section id="scope"><h2>Scope &amp; provenance</h2><p>${escape(boundary.included)}</p><div class="scope"><p>This includes every line of every file listed below. It is <strong>not the entire .NET dependency closure</strong> or a standalone compilable project.</p><ul>${boundary.excluded.map(note => `<li>${escape(note)}</li>`).join('')}</ul></div><p class="meta">Git HEAD <code>${revision}</code><br>Source snapshot ${generatedAt}<br>Aggregate source SHA-256 <code>${snapshotSha256}</code><br>${workingTreeModifiedFiles.length ? `${workingTreeModifiedFiles.length} included files differ from HEAD; the bundle records current working-tree bytes.` : 'Every included file matches HEAD; the bundle records current working-tree bytes.'}</p><p>Hashes use raw original bytes. UTF-8 roundtrip checks guarantee that every JSON content string reproduces those bytes, including original line endings and any BOM. Physical line counts include internal blank lines; a terminal newline does not add an extra line.</p><ul>${boundary.notes.map(note => `<li>${escape(note)}</li>`).join('')}</ul></section>
<section id="algorithm"><h2>From equation to implementation</h2><p>Follow each link into a complete method, then continue through its neighboring code. This is a teaching flow graph; the source remains the authority for branching and implementation details.</p><div class="stage-grid">${stageCards}</div></section>
<section id="manifest"><h2>Source manifest</h2><table class="manifest"><thead><tr><th>Exact repository path</th><th>Lines</th><th>SHA-256 of original bytes</th></tr></thead><tbody>${manifest}</tbody></table></section>
<section id="regenerate"><h2>Regenerate &amp; verify</h2><p>Run from the repository root. Only Node standard-library modules are used.</p><pre class="run"><code>node docs/presentations/build-sourcebook.mjs
node docs/presentations/build-sourcebook.mjs --check</code></pre><p>The check command writes nothing and fails on stale source, metadata, line links, or generated output. The generator includes new .cs files added to the shared Nsgaii subtree. An unchanged source/revision retains its snapshot timestamp; set <code>SOURCE_DATE_EPOCH</code> to pin it for a reproducible build.</p><p><code>source-data.json</code> contains the original UTF-8 text. HTML anchors use <code>#file-id-L123</code>; method ranges begin at declarations and end at their closing braces or semicolons. Nested members may be absent from the convenience symbol index; they are fully present in the file.</p></section>
${sourceSections}<footer class="footer">Generated from the local OptunaSharp working tree by docs/presentations/build-sourcebook.mjs. ${data.fileCount} files · ${data.totalLines.toLocaleString('en-US')} lines. No source elisions.</footer></main></div></body></html>
`;

const outputs = new Map([
  [path.join(atlas, 'source-data.json'), `${JSON.stringify(data, null, 2)}\n`],
  [path.join(here, 'NSGA-III-SOURCEBOOK.md'), markdown],
  [path.join(atlas, 'sourcebook.md'), markdown],
  [path.join(atlas, 'sourcebook.html'), html],
  [path.join(atlas, 'sourcebook.css'), `${css}\n`],
]);
// Validate generated artifacts directly. The HTML is fully populated without JS.
for (const file of files) {
  if (hash(Buffer.from(file.content, 'utf8')) !== file.sha256) throw new Error(`Hash agreement failed: ${file.path}`);
  if (!markdown.includes(file.content)) throw new Error(`Markdown source omission: ${file.path}`);
  const rendered = [...html.matchAll(new RegExp(`<span class="source-line" id="${file.id}-L\\d+">.*?<span class="line-content">(.*?)<\\/span><\\/span>`, 'g'))].map(match => match[1]);
  if (rendered.length !== file.lineCount || rendered.some((line, index) => line !== escape(splitLines(file.content)[index]))) throw new Error(`HTML source omission: ${file.path}`);
  for (const method of file.symbols) if (method.startLine < 1 || method.endLine > file.lineCount || method.startLine > method.endLine) throw new Error(`Invalid method range: ${method.id}`);
}
if (checking) {
  const stale = [];
  for (const [filePath, expected] of outputs) {
    let actual;
    try { actual = await readFile(filePath, 'utf8'); } catch { stale.push(path.relative(root, filePath)); continue; }
    if (actual !== expected) stale.push(path.relative(root, filePath));
  }
  if (stale.length) throw new Error(`Sourcebook is stale. Run node docs/presentations/build-sourcebook.mjs\n${stale.join('\n')}`);
  console.log(`Sourcebook verified: ${data.fileCount} full files, ${data.totalLines} lines, ${files.reduce((count, file) => count + file.symbols.length, 0)} indexed members, ${stages.length} stage links. Every UTF-8 byte/hash and rendered line agrees.`);
} else {
  await mkdir(atlas, { recursive: true });
  for (const [filePath, content] of outputs) await writeFile(filePath, content, 'utf8');
  console.log(`Sourcebook generated: ${data.fileCount} full files, ${data.totalLines} lines, revision ${revision}.`);
  console.log('Verify with node docs/presentations/build-sourcebook.mjs --check');
}
