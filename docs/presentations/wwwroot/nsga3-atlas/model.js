/** API state is independent of rendering, allowing deterministic recovery checks. */
export async function requestJson(url, body, method = 'POST') {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(url, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal
    });
    if (!response.ok) {
      let message = `The host returned HTTP ${response.status}.`;
      try { const error = await response.json(); message = error.error || error.detail || message; } catch { /* Use status for non-JSON failures. */ }
      throw new Error(message);
    }
    return response.status === 204 ? null : await response.json();
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The host did not respond within 30 seconds. Retry to reconcile the run.');
    throw error;
  } finally { clearTimeout(timeout); }
}

function validateSnapshot(snapshot) {
  if (!snapshot || typeof snapshot.runId !== 'string' || !Number.isInteger(snapshot.generation)
      || !snapshot.config || !snapshot.selection || !snapshot.metrics
      || !Array.isArray(snapshot.candidates) || !Array.isArray(snapshot.survivors)
      || snapshot.survivors.length !== snapshot.config.populationSize) {
    throw new Error('The host returned an incomplete generation snapshot.');
  }
  return snapshot;
}

export class StudySession {
  constructor(request = requestJson) {
    this.request = request;
    this.id = null;
    this.history = [];
    this.index = 0;
    this.busy = false;
    this.uncertain = false;
  }
  get snapshot() { return this.history[this.index] || null; }
  get latest() { return this.history.at(-1) || null; }

  async create(config) {
    if (this.busy) return false;
    this.busy = true;
    try {
      const snapshot = validateSnapshot(await this.request('/api/nsga3/runs', config));
      const previous = this.id;
      this.id = snapshot.runId;
      this.history = [snapshot];
      this.index = 0;
      this.uncertain = false;
      if (previous && previous !== this.id) {
        this.request(`/api/nsga3/runs/${encodeURIComponent(previous)}`, undefined, 'DELETE').catch(() => {});
      }
      return snapshot;
    } finally { this.busy = false; }
  }

  accept(snapshot) {
    validateSnapshot(snapshot);
    if (snapshot.runId !== this.id) throw new Error('The snapshot belongs to a different study. Start a new run.');
    if (snapshot.generation === this.latest.generation) return false;
    if (snapshot.generation !== this.latest.generation + 1) {
      throw new Error('The server generation is outside this recorded history. Start a new run to restore synchronization.');
    }
    this.history.push(snapshot);
    this.index = this.history.length - 1;
    return snapshot;
  }

  async advance() {
    if (this.busy || !this.id) return false;
    // Scrubbing changes playback only. Replay recorded snapshots before advancing C#.
    if (this.index < this.history.length - 1) return this.history[++this.index];
    this.busy = true;
    const url = `/api/nsga3/runs/${encodeURIComponent(this.id)}`;
    try {
      if (this.uncertain) {
        const recovered = await this.request(url, undefined, 'GET');
        const accepted = this.accept(recovered);
        this.uncertain = false;
        if (accepted) return accepted;
      }
      this.uncertain = true;
      const response = await this.request(`${url}/step`, {});
      const accepted = this.accept(response);
      this.uncertain = false;
      return accepted;
    } catch (originalError) {
      // A lost HTTP response can conceal a completed synchronous C# generation.
      // Resolve it by reading before allowing any further mutating request.
      if (this.uncertain) {
        try {
          const accepted = this.accept(await this.request(url, undefined, 'GET'));
          this.uncertain = false;
          if (accepted) return accepted;
        } catch { /* Remain uncertain: the next attempt must GET before POST. */ }
      }
      throw originalError;
    } finally { this.busy = false; }
  }

  inspect(index) {
    if (!this.busy && Number.isInteger(index) && index >= 0 && index < this.history.length) this.index = index;
    return this.snapshot;
  }
}

export const STAGES = [
  { id: 'initialize', title: 'Initialize', formula: 'xᵢ ∼ U(0, 1),  i = 1…N', description: 'Evaluate the first random population in decision space.' },
  { id: 'variation', title: 'Vary + evaluate', formula: 'Qₜ = evaluate(mutate(crossover(Pₜ)))', description: 'Recombine decision variables with SBX, apply polynomial mutation, and evaluate DTLZ2.' },
  { id: 'combine', title: 'Combine', formula: 'Rₜ = Pₜ ∪ Qₜ', description: 'Previous survivors and new offspring enter one selection pool.' },
  { id: 'rank', title: 'Non-dominated sort', formula: 'a ≺ b ⇔ ∀j: aⱼ ≤ bⱼ ∧ ∃j: aⱼ < bⱼ', description: 'Peel away non-dominated fronts. Whole earlier fronts survive before the borderline front is considered.' },
  { id: 'normalize', title: 'Normalize', formula: 'f̄ⱼ(x) = (fⱼ(x) − zⱼ*) / aⱼ', description: 'Translate by the ideal point and scale by hyperplane intercepts, with the core’s fallback for degenerate extremes.' },
  { id: 'associate', title: 'Associate', formula: 'd⊥(f̄,r) = ‖f̄ − r(f̄·r)/(r·r)‖₂', description: 'Assign each eligible solution to its nearest reference ray by perpendicular distance.' },
  { id: 'niche', title: 'Fill niches', formula: 'r* ∈ argmin ρ(r)', description: 'Choose a least occupied eligible direction; use nearest distance for an empty niche and the seeded shuffle otherwise.' },
  { id: 'survive', title: 'Survive', formula: '|Pₜ₊₁| = N', description: 'Keep the actual selected population and use it as the next generation’s parents.' }
];

export function sourceLines(content) {
  const lines = content.split(/\r\n|\n|\r/);
  if (lines.at(-1) === '' && /(?:\r\n|\n|\r)$/.test(content)) lines.pop();
  return lines;
}

export function findSourceMatches(lines, query) {
  if (!query.trim()) return [];
  const search = query.toLocaleLowerCase();
  return lines.flatMap((line, i) => line.toLocaleLowerCase().includes(search) ? [i + 1] : []);
}
