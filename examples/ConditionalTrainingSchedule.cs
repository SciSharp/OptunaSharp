#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Conditional training schedule — a warmup that only makes sense under two conditions at once: a
// non-constant scheduler AND a long-enough run. This is the example to read for how `ConditionalParams`
// combines rules.
//
// The combining rules (read them against `Objective`):
//   * TWO `.When(...)` for the SAME target AND together — `warmup_epochs` is declared under both
//     `scheduler in {cosine, step}` and `n_epochs >= 30`, so it is active only when BOTH hold.
//   * a numeric PREDICATE activator — `n_epochs >= 30` — sits next to a categorical one.
//   * a plain single-activator rule — `step_gamma` exists only for the step scheduler.
// (For the OR case — several values of one activator — see ConditionalModelSelection; for chains, see
// ConditionalNeuralArchitecture.)
//
// Why declare them (an OptunaSharp extension; no Optuna analog for the GA samplers): a trial that used a
// constant scheduler, or trained briefly, never suggests `warmup_epochs`, so to Optuna's plain
// intersection it looks gone and NSGA-II would resample it at random forever. Declared conditional, it is
// evolved whenever both conditions hold and dormant otherwise.
//
// source (FAQ, conditional search spaces):  https://optuna.readthedocs.io/en/stable/faq.html
// source (schedule tuning in the wild):      https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
//
// Deterministic surrogate (a known-best schedule), so it runs anywhere.
//
//   Run:  dotnet run examples/ConditionalTrainingSchedule.cs

using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.NsgaiiiCpt;
using OptunaSharp.SearchSpace;
using OptunaSharp.Storages;

// `warmup_epochs` appears twice: the two rules AND together (both conditions must hold). the others are a
// single categorical activator each. read this next to the `if` structure in Objective below.
var rules = new ConditionalParams()
    .When("scheduler", "cosine", "step").Activates("warmup_epochs")     // condition A: the schedule can warm up ...
    .When("n_epochs", epochs => (long)epochs! >= 30).Activates("warmup_epochs")  // ... AND B: the run is long enough
    .When("scheduler", "step").Activates("step_gamma");                 // step's own decay factor

OptunaLogging.SetVerbosity(LogLevel.Warning);
var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new NSGAIIICptSampler(populationSize: 20, seed: 2, conditionalParams: rules),   // <- wires the rules in
    direction: StudyDirection.Minimize);
study.Optimize(Objective, nTrials: 400);

// the best trial holds warmup_epochs only if its schedule warms up AND it trained long enough — the AND is
// visible right here in which keys are present.
var best = study.BestTrial;
Console.WriteLine($"best schedule: {best.Params["scheduler"]}, {best.Params["n_epochs"]} epochs  (loss {best.Value:F4})");
Console.WriteLine($"    warmup_epochs present: {best.Params.ContainsKey("warmup_epochs")}   step_gamma present: {best.Params.ContainsKey("step_gamma")}");
foreach (var (name, value) in best.Params.Where(p => p.Key is "warmup_epochs" or "step_gamma"))
    Console.WriteLine($"    {name} = {value}");

// the guards here are exactly the rules above: warmup is suggested only when both conditions hold; step_gamma
// only for the step scheduler. the surrogate loss rewards a warmed-up cosine schedule run for ~60 epochs.
static double Objective(Trial t)
{
    string scheduler = t.SuggestCategorical("scheduler", new[] { "constant", "cosine", "step" });   // activator A
    long nEpochs = t.SuggestInt("n_epochs", 5, 100);                                                 // activator B
    double loss = 0.5 - 0.30 * (scheduler == "cosine" ? 1.0 : scheduler == "step" ? 0.6 : 0.0);      // cosine best

    if (scheduler != "constant" && nEpochs >= 30)                    // both conditions -> warmup exists (the AND)
    {
        long warmup = t.SuggestInt("warmup_epochs", 0, 20);         // optimum ~5
        loss += Math.Pow((warmup - 5) / 20.0, 2.0);                // Math.Pow, not x*x, for Python x**2 parity
    }
    if (scheduler == "step")                                        // step's own knob
    {
        double gamma = t.SuggestFloat("step_gamma", 0.1, 0.9);     // optimum ~0.5
        loss += Math.Pow(gamma - 0.5, 2.0) * 0.2;
    }
    loss += Math.Pow((nEpochs - 60) / 100.0, 2.0);                 // ~60 epochs is enough
    return loss;
}
