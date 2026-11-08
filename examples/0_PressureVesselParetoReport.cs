#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:project ../src/OptunaSharp.Visualization.Plotly/OptunaSharp.Visualization.Plotly.csproj
#:property PublishAot=false

// Pressure-vessel design, cost vs. capacity — NSGA-II trades the fabrication cost of a cylindrical pressure
// vessel against its volume under the ASME wall-thickness rules (recorded with `Trial.set_constraint`), and
// the design review gets the Pareto-front chart (infeasible designs greyed out) and the hypervolume-history
// chart as HTML.
//
// source: https://github.com/ryojitanabe/reproblems (problem RE23, "pressure vessel design" — the classic
//         Sandgren / Kannan & Kramer problem — in Tanabe & Ishibuchi's real-world suite)
//         https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_pareto_front.html
//         https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_hypervolume_history.html
//
// Real objective, the textbook cost model: shell and head plates come in 1/16-inch steps (integers × 0.0625
// in), the inner radius R and the cylinder length L are continuous; cost = 0.6224·Ts·R·L + 1.7781·Th·R² +
// 3.1661·Ts²·L + 19.84·Ts²·R (material, forming and welding) and volume = πR²L + 4/3·πR³. The constraints are
// the code's minimum thicknesses Ts ≥ 0.0193·R and Th ≥ 0.00954·R. Two objectives: cost (minimize) and
// volume in thousands of cubic inches (maximize) — mixed directions, which the plots handle.
//
//   Run:  dotnet run examples/0_PressureVesselParetoReport.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET.CSharp;   // .SaveHtml

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var study = Optuna.CreateStudy(
    sampler: new NSGAIISampler(populationSize: 40, seed: 17),
    directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize },   // (cost $, volume k·in³)
    studyName: "pressure-vessel");
study.SetMetricNames(new[] { "cost_usd", "volume_kin3" });
study.Optimize(Objective, nTrials: 1200);

var front = study.BestTrials.OrderBy(t => t.Values![0]).ToList();
int feasible = study.Trials.Count(t => t.Constraints.Values.All(c => c <= 0));
Console.WriteLine($"{study.Trials.Count} designs, {feasible} meet the thickness code; Pareto front: {front.Count} designs (all feasible: {front.All(t => t.Constraints.Values.All(c => c <= 0))})");
foreach (var t in front.Where((_, i) => i % Math.Max(1, front.Count / 6) == 0))
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "  cost ${0,9:N0}   volume {1,8:N0} k·in³   Ts={2} in  Th={3} in  R={4:F1} in  L={5:F1} in",
        t.Values![0], t.Values[1], 0.0625 * (long)t.Params["ts_16ths"]!, 0.0625 * (long)t.Params["th_16ths"]!, t.Params["radius_in"], t.Params["length_in"]));

string dir = Path.Combine(Path.GetTempPath(), "optunasharp-pressure-vessel");
Directory.CreateDirectory(dir);
string pareto = Path.Combine(dir, "pareto_front.html"), hv = Path.Combine(dir, "hypervolume_history.html");
PlotlyVisualization.PlotParetoFront(study).SaveHtml(pareto);                        // infeasible trials greyed
PlotlyVisualization.PlotHypervolumeHistory(study, referencePoint: new[] { 400_000.0, 0.0 }).SaveHtml(hv);
Console.WriteLine($"charts: {pareto} ({new FileInfo(pareto).Length / 1024} KiB), {hv} ({new FileInfo(hv).Length / 1024} KiB)");

/// <summary>Prices one vessel and checks it against the thickness code.</summary>
/// <param name="t">The trial proposing plate thicknesses (in 1/16 in), radius and length.</param>
/// <returns>Fabrication cost in USD and volume in thousands of cubic inches.</returns>
static IReadOnlyList<double> Objective(Trial t)
{
    double ts = 0.0625 * t.SuggestInt("ts_16ths", 1, 99);    // shell plate: stocked in 1/16-inch steps
    double th = 0.0625 * t.SuggestInt("th_16ths", 1, 99);    // head plate
    double r = t.SuggestFloat("radius_in", 10, 200);
    double l = t.SuggestFloat("length_in", 10, 240);
    double cost = 0.6224 * ts * r * l + 1.7781 * th * r * r + 3.1661 * ts * ts * l + 19.84 * ts * ts * r;
    double volume = Math.PI * r * r * l + 4.0 / 3.0 * Math.PI * r * r * r;
    t.SetConstraint("shell_thickness", 0.0193 * r - ts);      // ≤ 0: thick enough for the hoop stress
    t.SetConstraint("head_thickness", 0.00954 * r - th);
    return new[] { cost, volume / 1000.0 };
}
