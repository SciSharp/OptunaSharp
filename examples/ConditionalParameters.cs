#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Conditional parameters — `ConditionalParams` activation rules that let NSGA-II/III evolve define-by-run
// parameters which only exist under some values of another parameter (an OptunaSharp extension).
//
// source: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
//         (Optuna's pythonic search spaces — the branch-defined parameters this example is about)
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.search_space.IntersectionSearchSpace.html
//         (the intersection space Optuna's GA samplers evolve — why such parameters otherwise go random)
//
// Deterministic surrogate: two "tasks" that share `lr`; `beta1` exists only under optimizer=adam and
// `momentum` only under optimizer=sgd, each with its own optimum. Two objectives keep both branches on the
// Pareto front, so both branch-only parameters keep being selected for. The same seeded NSGA-III study runs
// twice — with the activation rules and without (Optuna's plain intersection behavior) — and each
// parameter's distance from its optimum is reported side by side: the median over the trials of a
// generation (robust to the mutation re-draws) and the share of trials within 0.05 of the optimum. With
// rules the branch-only parameters converge (crossed over, mutated and inherited whenever active); without,
// they sit at the random-draw level, because they fell out of the intersection search space after the first
// trial that lacked them and are re-drawn uniformly ever since. `lr`, unconditional, converges either way.
// Generation 0 is identical in both runs (same seed, pure random sampling) — the divergence is the rules.
//
//   Run:  dotnet run examples/ConditionalParameters.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.SearchSpace;
using OptunaSharp.Storages;

OptunaLogging.SetVerbosity(LogLevel.Warning);               // silence the per-trial INFO lines

const int PopulationSize = 24;
const int Generations = 16;
const int LateFrom = Generations - 4;                       // "late" = the last four generations
const double LrOpt = 0.3, Beta1Opt = 0.9, MomentumOpt = 0.7;

// the activation rules: `beta1` exists while optimizer == "adam", `momentum` while optimizer == "sgd".
// rules on the same activator OR together; a parameter declared under several activators needs all of them.
var rules = new ConditionalParams()
    .When("optimizer", "adam").Activates("beta1")
    .When("optimizer", "sgd").Activates("momentum");
Console.WriteLine(rules);
Console.WriteLine();

// ---- 1. one seed, generation by generation (default mutation: each parameter re-drawn with prob 1/n) -----
var withRules = Run(seed: 0, rules, mutationProb: null);
var plain = Run(seed: 0, rules: null, mutationProb: null);   // Optuna's behavior: intersection space only

Console.WriteLine("seed 0, default mutation — per generation: median |parameter - optimum| and share within 0.05 (rules / plain)");
Console.WriteLine();
Console.WriteLine("gen | adam n      beta1 median    beta1 hit     | sgd n       momentum median momentum hit  | lr median");
Console.WriteLine("    | rules plain rules   plain   rules  plain  | rules plain rules   plain   rules  plain  | rules  plain");
Console.WriteLine("----+--------------------------------------------+--------------------------------------------+--------------");
for (int g = 0; g < Generations; g++)
{
    var r = Gen(withRules, g);
    var p = Gen(plain, g);
    Console.WriteLine(
        $"{g,3} | {Count(r, "beta1"),5} {Count(p, "beta1"),5} {F(Median(r, "beta1", Beta1Opt))}  {F(Median(p, "beta1", Beta1Opt))}  {F(Hit(r, "beta1", Beta1Opt))} {F(Hit(p, "beta1", Beta1Opt))}  " +
        $"| {Count(r, "momentum"),5} {Count(p, "momentum"),5} {F(Median(r, "momentum", MomentumOpt))}  {F(Median(p, "momentum", MomentumOpt))}  {F(Hit(r, "momentum", MomentumOpt))} {F(Hit(p, "momentum", MomentumOpt))}  " +
        $"| {F(Median(r, "lr", LrOpt))} {F(Median(p, "lr", LrOpt))}");
}
Console.WriteLine();
Console.WriteLine("random-draw levels: median |U(0.5,0.999)-0.90| = 0.150 (hit 0.20)   median |U(0,0.99)-0.70| = 0.245 (hit 0.10)   median |U(0,1)-0.30| = 0.250");
Console.WriteLine();

// ---- 2. controllability: what exists in a trial is exactly what its activator allows ----------------------
Console.WriteLine("last generation with rules — each trial's parameter set follows its activator:");
foreach (var t in Gen(withRules, Generations - 1).Take(6))
    Console.WriteLine($"  trial {t.Number,3}: optimizer={t.Params["optimizer"],-4} -> {{ {string.Join(", ", t.Params.Keys.Where(k => k != "optimizer"))} }}");
Console.WriteLine();

// ---- 3. five seeds pooled, late generations, at two mutation rates -------------------------------------
Console.WriteLine($"seeds 0-4 pooled, generations {LateFrom}-{Generations - 1}: median |parameter - optimum| and share within 0.05 (rules / plain)");
Console.WriteLine();
Console.WriteLine("mutation         | beta1 median    beta1 hit     | momentum median momentum hit  | lr median");
Console.WriteLine("                 | rules   plain   rules  plain  | rules   plain   rules  plain  | rules  plain");
Console.WriteLine("-----------------+-------------------------------+-------------------------------+--------------");
foreach (var (label, mutationProb) in new[] { ("default (1/n)", (double?)null), ("mutationProb 0.1", 0.1) })
{
    var r = new List<FrozenTrial>();
    var p = new List<FrozenTrial>();
    for (int seed = 0; seed < 5; seed++)
    {
        r.AddRange(Late(seed == 0 && mutationProb is null ? withRules : Run(seed, rules, mutationProb)));
        p.AddRange(Late(seed == 0 && mutationProb is null ? plain : Run(seed, rules: null, mutationProb)));
    }
    Console.WriteLine(
        $"{label,-16} | {F(Median(r, "beta1", Beta1Opt))}  {F(Median(p, "beta1", Beta1Opt))}  {F(Hit(r, "beta1", Beta1Opt))} {F(Hit(p, "beta1", Beta1Opt))}  " +
        $"| {F(Median(r, "momentum", MomentumOpt))}  {F(Median(p, "momentum", MomentumOpt))}  {F(Hit(r, "momentum", MomentumOpt))} {F(Hit(p, "momentum", MomentumOpt))}  " +
        $"| {F(Median(r, "lr", LrOpt))} {F(Median(p, "lr", LrOpt))}");
}
Console.WriteLine();
Console.WriteLine("with rules the branch-only parameters are learned; without them they stay at the random-draw level.");

// the same seeded NSGA-III study, with or without the activation rules
static Study Run(int seed, ConditionalParams? rules, double? mutationProb)
{
    var study = Optuna.CreateStudy(
        storage: new InMemoryStorage(),                        // in-memory — nothing persisted
        sampler: new NSGAIIISampler(
            populationSize: PopulationSize, seed: seed,        // seeded, so the two runs share generation 0
            mutationProb: mutationProb,                        // null = Optuna's default 1 / n_params
            conditionalParams: rules),                         // null = Optuna's intersection-space behavior
        directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
    study.Optimize(Objective, nTrials: PopulationSize * Generations);
    return study;
}

// two tasks that share `lr`. adam trials can only lower f1, sgd trials only f2, so no adam trial ever
// dominates an sgd trial or vice versa: both branches survive elite selection and keep evolving.
static IReadOnlyList<double> Objective(Trial t)
{
    string optimizer = t.SuggestCategorical("optimizer", new[] { "adam", "sgd" });   // the activator
    double lr = t.SuggestFloat("lr", 0.0, 1.0);                                       // shared, optimum 0.3
    double shared = Math.Pow(lr - LrOpt, 2.0);                                        // Math.Pow, not x*x (Python x**2 parity)
    if (optimizer == "adam")
    {
        double beta1 = t.SuggestFloat("beta1", 0.5, 0.999);                            // exists only under adam, optimum 0.9
        return new[] { shared + Math.Pow(beta1 - Beta1Opt, 2.0), 1.0 + shared };
    }
    double momentum = t.SuggestFloat("momentum", 0.0, 0.99);                           // exists only under sgd, optimum 0.7
    return new[] { 1.0 + shared, shared + Math.Pow(momentum - MomentumOpt, 2.0) };
}

static int Generation(FrozenTrial t) => Convert.ToInt32(t.SystemAttrs["NSGAIIISampler:generation"]);   // set by the sampler
static List<FrozenTrial> Gen(Study s, int g) => s.Trials.Where(t => Generation(t) == g).ToList();
static List<FrozenTrial> Late(Study s) => s.Trials.Where(t => Generation(t) >= LateFrom).ToList();
static int Count(IEnumerable<FrozenTrial> ts, string name) => ts.Count(t => t.Params.ContainsKey(name));

// |value - optimum| over the trials that carry the parameter
static List<double> Distances(IEnumerable<FrozenTrial> ts, string name, double optimum)
    => ts.Where(t => t.Params.ContainsKey(name)).Select(t => Math.Abs((double)t.Params[name]! - optimum)).ToList();

static double Median(IEnumerable<FrozenTrial> ts, string name, double optimum)
{
    var sorted = Distances(ts, name, optimum).OrderBy(d => d).ToList();
    return sorted.Count == 0 ? double.NaN : sorted[sorted.Count / 2];
}

// the share of carrying trials within 0.05 of the optimum
static double Hit(IEnumerable<FrozenTrial> ts, string name, double optimum)
{
    var d = Distances(ts, name, optimum);
    return d.Count == 0 ? double.NaN : d.Count(x => x <= 0.05) / (double)d.Count;
}

static string F(double v) => double.IsNaN(v) ? "   -  " : v.ToString("0.000", CultureInfo.InvariantCulture).PadLeft(6);
