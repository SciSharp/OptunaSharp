#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Reaction yield, Bayesian optimization — GPSampler plans the next batch-reactor experiment (temperature,
// time, catalyst loading) from a Gaussian-process model of every run so far, starting from the lab's own
// historical notebook entries and respecting an impurity specification.
//
// source: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html
//         (the GP sampler, `deterministic_objective`, constraints)
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.create_trial.html
//         (`create_trial` + `study.add_trial`: seeding a study with results obtained outside it)
//
// Real objective, physical model: a batch reactor runs A →k1→ B →k2→ C with a parallel side reaction
// A →k3→ D, all first order with Arrhenius rate constants (the catalyst only accelerates k1). The analytic
// concentrations at the end of the batch give the yield of the product B and the impurities C + D. The
// study maximizes an "economic yield" — % yield minus catalyst and reactor-time charges — under the purity
// specification impurities ≤ 2 % (recorded with `SetConstraint`). Each real experiment costs a day, which is
// why a sample-efficient GP is the right tool: 30 suggested experiments on top of 8 historical ones.
//
//   Run:  dotnet run examples/0_ReactionYieldBayesOpt.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var space = new OrderedDict<string, BaseDistribution>
{
    ["temp_c"] = new FloatDistribution(40, 160),                  // jacket temperature, °C
    ["time_h"] = new FloatDistribution(0.1, 10, log: true),       // batch time, hours
    ["catalyst_mol_pct"] = new FloatDistribution(0.1, 5, log: true),
};

var study = Optuna.CreateStudy(
    sampler: new GPSampler(seed: 1, nStartupTrials: 10, deterministicObjective: true),   // no assay noise
    direction: StudyDirection.Maximize,
    studyName: "suzuki-batch");

// the lab notebook: eight runs done by hand before anyone thought of Bayesian optimization. They enter the
// study as completed trials, so the GP learns from them and they count toward `n_startup_trials`.
(double T, double H, double Cat)[] notebook =
{
    (60, 4, 1), (80, 2, 1), (80, 6, 0.5), (100, 1, 1), (100, 3, 2), (120, 0.5, 1), (120, 2, 0.5), (140, 0.3, 1),
};
foreach (var (temp, hours, cat) in notebook)
{
    var run = Reactor.Run(temp, hours, cat);
    study.AddTrial(FrozenTrial.CreateTrial(
        value: run.EconomicYield,
        @params: new OrderedDict<string, object?> { ["temp_c"] = temp, ["time_h"] = hours, ["catalyst_mol_pct"] = cat },
        distributions: space,
        userAttrs: new OrderedDict<string, object?> { ["source"] = "notebook" },
        constraints: new OrderedDict<string, double> { ["impurity_pct"] = run.ImpurityPct - 2.0 }));
}

study.Optimize(Objective, nTrials: 30);

var notebookBest = study.Trials.Where(t => (string?)t.UserAttrs["source"] == "notebook" && t.Constraints.Values.All(v => v <= 0))
    .OrderByDescending(t => t.Value).First();
var best = study.BestTrial;                      // best FEASIBLE trial (impurities within spec)
Console.WriteLine($"experiments: {notebook.Length} from the notebook + {study.Trials.Count - notebook.Length} planned by the GP");
Console.WriteLine();
Console.WriteLine("                       T °C   time h   cat mol%   yield %  impurity %   economic");
Row("best notebook run", notebookBest);
Row("best GP-planned run", best);
Console.WriteLine();
Console.WriteLine("the GP's experiment sequence (economic yield, * = out of spec):");
Console.WriteLine("  " + string.Join(" ", study.Trials.Skip(notebook.Length).Select(t =>
    string.Format(CultureInfo.InvariantCulture, "{0:F1}{1}", t.Value, t.Constraints.Values.All(v => v <= 0) ? "" : "*"))));

/// <summary>Runs one (simulated) batch experiment.</summary>
/// <param name="t">The trial proposing the reactor conditions.</param>
/// <returns>The economic yield (higher is better); the impurity spec is recorded as a constraint.</returns>
static double Objective(Trial t)
{
    double temp = t.SuggestFloat("temp_c", 40, 160);
    double hours = t.SuggestFloat("time_h", 0.1, 10, log: true);
    double cat = t.SuggestFloat("catalyst_mol_pct", 0.1, 5, log: true);
    var run = Reactor.Run(temp, hours, cat);
    t.SetUserAttr("source", "gp");
    t.SetUserAttr("yield_pct", run.YieldPct);
    t.SetConstraint("impurity_pct", run.ImpurityPct - 2.0);       // ≤ 2 % impurities, else out of spec
    return run.EconomicYield;
}

/// <summary>Prints one experiment as a table row.</summary>
/// <param name="label">Row label.</param>
/// <param name="t">The experiment (trial).</param>
static void Row(string label, FrozenTrial t)
{
    var r = Reactor.Run((double)t.Params["temp_c"]!, (double)t.Params["time_h"]!, (double)t.Params["catalyst_mol_pct"]!);
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,-20} {1,7:F1} {2,8:F2} {3,10:F2} {4,9:F2} {5,11:F2} {6,10:F2}",
        label, t.Params["temp_c"], t.Params["time_h"], t.Params["catalyst_mol_pct"], r.YieldPct, r.ImpurityPct, t.Value));
}

/// <summary>The batch-reactor kinetics: first-order A→B→C plus the side reaction A→D.</summary>
static class Reactor
{
    private const double R = 8.314;   // J/(mol·K)

    /// <summary>Solves the kinetics in closed form for one batch.</summary>
    /// <param name="tempC">Reactor temperature in °C.</param>
    /// <param name="hours">Batch time in hours.</param>
    /// <param name="catMolPct">Catalyst loading in mol % (scales only the desired step's rate).</param>
    /// <returns>The yield of B, the impurities C + D (both in % of the starting A), and the economic yield.</returns>
    public static (double YieldPct, double ImpurityPct, double EconomicYield) Run(double tempC, double hours, double catMolPct)
    {
        double tk = tempC + 273.15;
        double k1 = catMolPct * 2e8 * Math.Exp(-55_000 / (R * tk));   // A → B, catalysed (1/h)
        double k2 = 5e8 * Math.Exp(-75_000 / (R * tk));                // B → C, over-reaction
        double k3 = 1e9 * Math.Exp(-80_000 / (R * tk));                // A → D, thermal side reaction
        double ka = k1 + k3;
        double a = Math.Exp(-ka * hours);
        // B(t) for consecutive first-order steps; the k2 → ka limit is B = k1·t·e^(−ka·t).
        double b = Math.Abs(k2 - ka) < 1e-12
            ? k1 * hours * a
            : k1 / (k2 - ka) * (a - Math.Exp(-k2 * hours));
        double d = k3 / ka * (1 - a);
        double c = 1 - a - b - d;                                      // mass balance
        double yieldPct = 100 * b, impurityPct = 100 * (c + d);
        // what the plant manager actually maximizes: product, minus catalyst (expensive Pd) and reactor time.
        double economic = yieldPct - 4.0 * catMolPct - 1.5 * hours;
        return (yieldPct, impurityPct, economic);
    }
}
