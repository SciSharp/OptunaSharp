#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Concrete mix design with Bayesian optimization over a mixed space — GPSampler picks the cement content
// (5 kg steps), water–cement ratio (0.01 steps), slag and fly-ash replacement, superplasticizer dose and the
// aggregate type to maximize 28-day compressive strength, under a cost cap and a minimum-workability rule:
// each trial batch is a lab pour, so few trials matter.
//
// source: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html
//         (GP sampling over float — including stepped — int and categorical parameters, with constraints)
//         https://en.wikipedia.org/wiki/Water%E2%80%93cement_ratio (Abrams' law: strength falls with w/c)
//
// Surrogate of the lab, built on the textbook relations: Abrams' law f = A / B^(w/c) for the paste, a
// cement-content factor, slag that costs early strength but adds to it by 28 days while fly ash mostly
// dilutes it, superplasticizer that buys workability (so a lower w/c is placeable), and an aggregate factor
// (crushed granite > river gravel > recycled). Constraints, via `SetConstraint`: material cost ≤ $95/m³ and
// slump ≥ 75 mm. 12 random pours, then 28 GP-planned ones.
//
//   Run:  dotnet run examples/0_ConcreteMixDesignGp.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var study = Optuna.CreateStudy(sampler: new GPSampler(seed: 50, nStartupTrials: 12), direction: StudyDirection.Maximize, studyName: "c40-mix");
study.Optimize(Objective, nTrials: 40);

var best = study.BestTrial;                        // best FEASIBLE mix
int feasible = study.Trials.Count(t => t.Constraints.Values.All(c => c <= 0));
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0} pours, {1} within cost and slump; best 28-day strength {2:F1} MPa (pour #{3})",
    study.Trials.Count, feasible, best.Value, best.Number));
foreach (var kv in best.Params)
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "    {0,-20} {1}", kv.Key, kv.Value));
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "    cost ${0:F1}/m³, slump {1:F0} mm", best.UserAttrs["cost"], best.UserAttrs["slump_mm"]));
var randomBest = study.Trials.Take(12).Where(t => t.Constraints.Values.All(c => c <= 0)).Select(t => t.Value ?? 0).DefaultIfEmpty(0).Max();
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best of the 12 random pours: {0:F1} MPa", randomBest));

/// <summary>"Pours" one mix: predicts its 28-day strength, cost and slump.</summary>
/// <param name="t">The trial proposing the mix.</param>
/// <returns>28-day compressive strength in MPa; cost and slump limits are recorded as constraints.</returns>
static double Objective(Trial t)
{
    double cement = t.SuggestFloat("cement_kg", 250, 500, step: 5);           // binder per m³, 5 kg steps
    double wc = t.SuggestFloat("water_cement", 0.30, 0.65, step: 0.01);       // water–cement ratio
    long slagPct = t.SuggestInt("slag_pct", 0, 50, step: 5);                  // GGBS replacing cement
    long ashPct = t.SuggestInt("fly_ash_pct", 0, 30, step: 5);                // fly ash replacing cement
    double sp = t.SuggestFloat("superplasticizer_kg", 0, 8);                   // admixture dose
    string aggregate = t.SuggestCategorical("aggregate", new[] { "crushed_granite", "river_gravel", "recycled" });

    double clinkerShare = Math.Max(0.2, 1 - (slagPct + ashPct) / 100.0);
    double abrams = 96.5 / Math.Pow(7.0, wc);                                  // MPa, Abrams' law
    double cementFactor = 0.75 + 0.25 * Math.Min(1, cement / 400);
    double scm = clinkerShare + 0.85 * slagPct / 100.0 + 0.45 * ashPct / 100.0; // 28-day contribution
    double aggFactor = aggregate switch { "crushed_granite" => 1.06, "river_gravel" => 1.0, _ => 0.88 };
    double strength = abrams * cementFactor * scm * aggFactor;

    double water = wc * cement;
    double slump = 20 + 0.55 * (water - 140) + 22 * sp + 0.6 * ashPct - (aggregate == "crushed_granite" ? 15 : 0);
    double cost = cement * clinkerShare * 0.14 + cement * slagPct / 100.0 * 0.09 + cement * ashPct / 100.0 * 0.05
                  + sp * 2.8 + (aggregate switch { "crushed_granite" => 24, "river_gravel" => 20, _ => 14 });
    t.SetUserAttr("cost", cost);
    t.SetUserAttr("slump_mm", slump);
    t.SetConstraint("cost_cap", cost - 95.0);                                  // ≤ $95 per m³
    t.SetConstraint("workability", 75.0 - slump);                              // ≥ 75 mm slump
    return strength;
}
