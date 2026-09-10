#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Conditional neural-architecture search — a variable-depth network where each hidden layer's width only
// exists once the network is that deep. This is the define-by-run "loop that suggests one width per
// layer" from Optuna's tutorial, made evolvable on the GA samplers.
//
// Read `Rules` against `Objective`: the rules gate each width by a NUMERIC PREDICATE on `n_layers`
// (`units_2` exists only when n_layers >= 2, and so on), and gate a training schedule through a two-level
// CHAIN (`use_scheduler` -> `scheduler` -> `warmup_frac`). The objective's `if (nLayers >= i)` loop and
// its nested scheduler block suggest exactly those gated parameters — the code mirrors the rules.
//
// Without the rules (OptunaSharp extension; no Optuna analog for the GA samplers) a deeper layer's width
// would drop out of the intersection search space the first time a shallower network was tried, and be
// resampled at random forever. With them, each width is crossed over and mutated whenever its layer is
// present, and is simply absent when it is not — so the Pareto front below shows every architecture
// carrying exactly its own widths, no more, no fewer.
//
// source (define-by-run depth loop):   https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
// source (NAS example this mirrors):    https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
//
// Deterministic surrogate (deeper reaches a lower error floor; wider costs size), so it runs anywhere.
//
//   Run:  dotnet run examples/ConditionalNeuralArchitecture.cs

using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.SearchSpace;
using OptunaSharp.Storages;

// read against Objective below: a predicate gates each layer's width, and a chain gates the schedule knobs.
var rules = new ConditionalParams()
    .When("n_layers", depth => (long)depth! >= 2).Activates("units_2")   // numeric predicate on an int parameter
    .When("n_layers", depth => (long)depth! >= 3).Activates("units_3")
    .When("n_layers", depth => (long)depth! >= 4).Activates("units_4")
    .When("use_scheduler", true).Activates("scheduler")                  // a bool activator ...
    .When("scheduler", "cosine").Activates("warmup_frac");               // ... feeding a chain (only cosine warms up)

OptunaLogging.SetVerbosity(LogLevel.Warning);
var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new NSGAIISampler(populationSize: 24, seed: 1, conditionalParams: rules),   // <- wires the rules in
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });             // minimize (error, size)
study.Optimize(Objective, nTrials: 480);

// one line per depth (the lowest-error architecture the front holds at that depth). the point to see:
// each row lists exactly its depth's widths — L1 alone, then L1-L2, ... — and warmup_frac shows up only
// where the schedule chain reached "cosine". the rules, not the print, decided which keys exist.
Console.WriteLine("best architecture at each depth on the Pareto front:");
foreach (var t in study.BestTrials
             .GroupBy(t => Convert.ToInt64(t.Params["n_layers"]))
             .OrderBy(g => g.Key)
             .Select(g => g.MinBy(t => t.Values![0])!))
{
    string widths = string.Join(" ", Enumerable.Range(1, 4)
        .Where(i => t.Params.ContainsKey($"units_{i}"))
        .Select(i => $"L{i}={Convert.ToInt64(t.Params[$"units_{i}"]),3}"));
    string schedule = t.Params.TryGetValue("warmup_frac", out var w) ? $"cosine warmup {(double)w!:F2}"
        : (bool)t.Params["use_scheduler"]! ? $"{t.Params["scheduler"]}" : "no schedule";
    Console.WriteLine($"  depth {t.Params["n_layers"]}   err {t.Values![0]:F3}  size {t.Values![1],4:F0}   {widths,-26} {schedule}");
}

// error: a floor that drops with depth, plus a per-layer width penalty. size: total width. the two compete,
// so the front spans shallow-and-small to deep-and-wide. the `if (nLayers >= i)` guards ARE the rules above.
static IReadOnlyList<double> Objective(Trial t)
{
    long nLayers = t.SuggestInt("n_layers", 1, 4);                       // the depth activator
    double error = 0.30 - 0.05 * nLayers;                               // deeper reaches a lower floor
    double size = 0;

    size += AddLayer(t, "units_1", ref error);                         // layer 1 always present
    if (nLayers >= 2) size += AddLayer(t, "units_2", ref error);       // deeper layers only when deep enough ...
    if (nLayers >= 3) size += AddLayer(t, "units_3", ref error);       // ... exactly the predicates in `rules`
    if (nLayers >= 4) size += AddLayer(t, "units_4", ref error);

    if (t.SuggestCategorical("use_scheduler", new[] { true, false }))  // the chain's root
    {
        string scheduler = t.SuggestCategorical("scheduler", new[] { "cosine", "step" });
        if (scheduler == "cosine")                                     // only cosine has a warmup knob
            error -= 0.02 * (1.0 - Math.Abs(t.SuggestFloat("warmup_frac", 0.0, 0.5) - 0.1) / 0.5);
    }
    return new[] { Math.Max(error, 0.0), size };
}

// suggest one layer's width and fold it into (error, size). optimum width 96; wider costs size.
static long AddLayer(Trial t, string name, ref double error)
{
    long units = t.SuggestInt(name, 16, 256, log: true);
    error += Math.Pow((units - 96) / 128.0, 2.0) * 0.1;               // Math.Pow, not x*x, for Python x**2 parity
    return units;
}
