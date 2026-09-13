#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Conditional model selection — the classic "pick an algorithm, then tune only ITS hyperparameters"
// search. Read the two halves side by side: `Rules` declares which knob belongs to which model, and
// `Objective` suggests exactly those knobs inside the matching branch. That pairing is the whole idea.
//
// Why the rules matter (an OptunaSharp extension; no Optuna analog for the GA samplers): a completed
// trial only ever holds the knobs of the model it chose, so to Optuna's plain intersection search space
// every model-specific knob looks like it "disappeared" after the first trial that chose another model —
// and NSGA-II would resample it at random forever, never able to tune any model. Declaring the knobs
// conditional tells the sampler their absence is deliberate, so it evolves each model's knobs whenever
// that model is active and leaves them dormant otherwise.
//
// source (FAQ, branching search spaces):  https://optuna.readthedocs.io/en/stable/faq.html
// source (the pattern in the wild):        https://github.com/optuna/optuna-examples/blob/main/sklearn/sklearn_simple.py
//
// The objective is a deterministic surrogate (each model has a closed-form error with a known optimum, mlp
// lowest), so the example runs anywhere; a real sklearn objective would slot straight into `Objective`.
//
//   Run:  dotnet run examples/ConditionalModelSelection.cs

using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.NsgaiiiCpt;
using OptunaSharp.SearchSpace;
using OptunaSharp.Storages;

// each knob is active only while its rule holds. read this next to Objective below: every conditional
// suggest there is declared here, and vice versa.
var rules = new ConditionalParams()
    .When("model", "svm").Activates("C", "kernel")             // svm's own knobs
    .When("kernel", "rbf", "poly").Activates("gamma")           // a chain: gamma lives under svm's rbf/poly kernel
    .When("model", "random_forest").Activates("n_estimators", "max_depth")
    .When("model", "mlp").Activates("mlp_units", "mlp_lr", "mlp_beta1");

OptunaLogging.SetVerbosity(LogLevel.Warning);                   // no per-trial INFO spam
var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new NSGAIIICptSampler(populationSize: 24, seed: 0, conditionalParams: rules),   // <- the one line that wires it in
    direction: StudyDirection.Minimize);
study.Optimize(Objective, nTrials: 480);

// the best trial naturally contains only the winning model's knobs — that is the feature, visible in one line.
var best = study.BestTrial;
Console.WriteLine($"best model: {best.Params["model"]}  (validation error {best.Value:F4})");
foreach (var (name, value) in best.Params.Where(p => p.Key != "model"))
    Console.WriteLine($"    {name} = {value}");

// each model's branch suggests exactly the knobs declared conditional on that model above. the surrogate
// value stands in for "train this model and return its validation error".
static double Objective(Trial t)
{
    string model = t.SuggestCategorical("model", new[] { "svm", "random_forest", "mlp" });   // the activator
    return model switch
    {
        "svm" => Svm(t),
        "random_forest" => RandomForest(t),
        _ => Mlp(t),
    };
}

static double Svm(Trial t)
{
    double c = t.SuggestFloat("C", 1e-3, 1e3, log: true);                                     // optimum 1.0
    string kernel = t.SuggestCategorical("kernel", new[] { "linear", "rbf", "poly" });        // optimum rbf
    double gammaPenalty = 0.0;
    if (kernel is "rbf" or "poly")                                                            // gamma matters only for rbf/poly
        gammaPenalty = Sq(Math.Log10(t.SuggestFloat("gamma", 1e-4, 1e0, log: true)) + 2) * 0.01;  // optimum 1e-2
    return 0.10 + Sq(Math.Log10(c)) * 0.01 + (kernel == "rbf" ? 0.0 : 0.05) + gammaPenalty;  // floor 0.10
}

static double RandomForest(Trial t)
{
    long nEstimators = t.SuggestInt("n_estimators", 10, 500);                                 // optimum 200
    long maxDepth = t.SuggestInt("max_depth", 2, 32);                                         // optimum 12
    return 0.06 + Sq((nEstimators - 200) / 500.0) + Sq((maxDepth - 12) / 32.0);              // floor 0.06
}

static double Mlp(Trial t)   // the best model — the sampler should discover this and tune it
{
    long units = t.SuggestInt("mlp_units", 16, 512, log: true);                              // optimum 128
    double lr = t.SuggestFloat("mlp_lr", 1e-5, 1e-1, log: true);                             // optimum 1e-3
    double beta1 = t.SuggestFloat("mlp_beta1", 0.5, 0.999);                                  // optimum 0.90
    return 0.02 + Sq((Math.Log2(units) - 7) / 5) + Sq(Math.Log10(lr) + 3) * 0.02 + Sq(beta1 - 0.90);  // floor 0.02
}

static double Sq(double x) => Math.Pow(x, 2.0);   // Math.Pow, not x*x, for Python x**2 parity
