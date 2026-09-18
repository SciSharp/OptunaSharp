<div class="ns-home">
  <section class="ns-home-intro">
    <p class="ns-home-kicker">Hyperparameter optimization for .NET</p>
    <div class="ns-home-title-row">
      <img class="ns-home-logo" src="images/optunasharp-logo.png" alt="" width="70" height="48" style="max-width: 70px; max-height: 48px; object-fit: contain;">
      <h1>OptunaSharp</h1>
    </div>
    <p class="ns-home-lede">
      A .NET port of <a href="https://github.com/optuna/optuna">Optuna</a> — the define-by-run
      hyperparameter optimization framework — built on top of
      <a href="https://github.com/SciSharp/NumSharp">NumSharp</a>. Optuna is the behavioral source
      of truth; OptunaSharp aligns to it.
    </p>
    <div class="ns-home-actions" aria-label="Primary documentation links">
      <a class="ns-home-button ns-home-button-primary" href="docs/intro.md">Introduction</a>
      <a class="ns-home-button" href="docs/getting-started.md">Getting started</a>
      <a class="ns-home-button" href="integrations/index.md">Integrations</a>
      <a class="ns-home-button" href="api/index.md">API reference</a>
    </div>
  </section>

  <section class="ns-home-code" aria-label="Status">
    <div class="ns-home-code-head">
      <h2>Status: core surface ported</h2>
      <p>The core algorithm surface is in place — samplers, pruners, storages, importance, and visualization — with more landing module-by-module.</p>
    </div>
    <div class="ns-code-grid">
      <pre><code class="lang-bash">dotnet add package OptunaSharp</code></pre>
      <pre><code class="lang-python"># The Optuna API OptunaSharp targets (Python)
import optuna
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2
study = optuna.create_study()
study.optimize(objective, n_trials=100)</code></pre>
    </div>
  </section>

  <section class="ns-home-section">
    <div class="ns-home-section-head">
      <h2>Documentation Map</h2>
      <p>Pick where you want to start.</p>
    </div>
    <div class="ns-card-grid">
      <a class="ns-doc-card" href="docs/intro.md">
        <span>Overview</span>
        <strong>Introduction</strong>
        <p>What OptunaSharp is, the study/trial model, and how it relates to Optuna and NumSharp.</p>
      </a>
      <a class="ns-doc-card" href="docs/getting-started.md">
        <span>Setup</span>
        <strong>Getting started</strong>
        <p>Install, build, test, and pack. How the NumSharp reference resolves.</p>
      </a>
      <a class="ns-doc-card" href="api/index.md">
        <span>Reference</span>
        <strong>API reference</strong>
        <p>Generated from the OptunaSharp assemblies by DocFX.</p>
      </a>
    </div>
  </section>

  <section class="ns-home-links" aria-label="Community links">
    <a href="integrations/index.md">Explore all integrations</a>
    <a href="https://github.com/SciSharp/OptunaSharp">GitHub Repository</a>
    <a href="https://www.nuget.org/packages/OptunaSharp">NuGet Package</a>
    <a href="https://github.com/SciSharp/NumSharp">NumSharp</a>
  </section>
</div>
