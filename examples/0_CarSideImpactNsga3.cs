#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Car side-impact design — NSGA-III trades off the weight of a car door's structure against the occupant's
// pubic-symphysis force and the B-pillar/front-door velocities in a side crash, under the ten safety limits
// of the European side-impact procedure, recorded with `Trial.set_constraint`.
//
// source: https://github.com/ryojitanabe/reproblems (problem RE41, "car side impact design", of the
//         real-world multi-objective problem suite by Tanabe & Ishibuchi, 2020)
//         https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIIISampler.html
//
// Real objective: the response-surface model of the side-impact FE simulations (Gu et al., used by Jain &
// Deb's NSGA-III Part II paper). Seven design variables are sheet thicknesses/materials of the B-pillar,
// floor, door beam and roof rail; the objectives are (1) weight in kg, (2) pubic force in kN, (3) the mean
// of the B-pillar and front-door velocities in mm/ms; all minimized. The ten constraints — abdomen load ≤ 1
// kN, the three rib deflections ≤ 32 mm, the three viscous criteria ≤ 0.32 m/s, pubic force ≤ 4 kN, the
// two velocities ≤ 9.9 and 15.7 mm/ms — are stored as "value − limit" (≤ 0 is legal). NSGA-III's
// constrained domination ranks every legal design above every illegal one, so the front printed at the end
// is entirely street-legal. The objective is deterministic: same seed, same front.
//
//   Run:  dotnet run examples/0_CarSideImpactNsga3.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

const int Population = 60, Generations = 40;
var study = Optuna.CreateStudy(
    sampler: new NSGAIIISampler(populationSize: Population, seed: 1),     // Das–Dennis reference points (p = 3)
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize, StudyDirection.Minimize },
    studyName: "car-side-impact");
study.SetMetricNames(new[] { "weight_kg", "pubic_force_kN", "velocity_mm_per_ms" });
study.Optimize(Objective, nTrials: Population * Generations);

var trials = study.Trials;
Console.WriteLine("generation  feasible designs (of 60)");
for (int g = 0; g < Generations; g += 8)
    Console.WriteLine($"{g,10}  {trials.Skip(g * Population).Take(Population).Count(IsLegal),3}");

var front = study.BestTrials;
Console.WriteLine();
Console.WriteLine($"{trials.Count} designs evaluated, {trials.Count(IsLegal)} legal; Pareto front: {front.Count} designs, all legal: {front.All(IsLegal)}");
Console.WriteLine("the front's extremes:");
foreach (var (label, idx) in new[] { ("lightest", 0), ("lowest pubic force", 1), ("lowest velocity", 2) })
{
    var t = front.OrderBy(x => x.Values![idx]).First();
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
        "  {0,-19} weight {1,6:F2} kg   pubic force {2,5:F3} kN   velocity {3,6:F3} mm/ms   (trial #{4})",
        label, t.Values![0], t.Values[1], t.Values[2], t.Number));
}

/// <summary>Evaluates one door/pillar design with the side-impact response surface.</summary>
/// <param name="t">The trial proposing the seven design variables.</param>
/// <returns>Weight, pubic force and mean velocity; the ten safety limits are recorded as constraints.</returns>
static IReadOnlyList<double> Objective(Trial t)
{
    double x1 = t.SuggestFloat("b_pillar_inner", 0.5, 1.5);      // thicknesses in mm
    double x2 = t.SuggestFloat("b_pillar_reinf", 0.45, 1.35);
    double x3 = t.SuggestFloat("floor_side_inner", 0.5, 1.5);
    double x4 = t.SuggestFloat("cross_member", 0.5, 1.5);
    double x5 = t.SuggestFloat("door_beam", 0.875, 2.625);
    double x6 = t.SuggestFloat("door_belt_line", 0.4, 1.2);
    double x7 = t.SuggestFloat("roof_rail", 0.4, 1.2);

    double weight = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7;
    double pubicForce = 4.72 - 0.5 * x4 - 0.19 * x2 * x3;
    double vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2;       // B-pillar middle-point velocity
    double vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6;    // front-door velocity
    double velocity = 0.5 * (vmbp + vfd);

    // the safety limits, each as (response − limit): ≤ 0 passes the regulation.
    t.SetConstraint("abdomen_load", (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3) - 1);
    t.SetConstraint("vc_upper", (0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6) - 0.32);
    t.SetConstraint("vc_middle", (0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2) - 0.32);
    t.SetConstraint("vc_lower", (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2) - 0.32);
    t.SetConstraint("rib_upper", (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7) - 32);
    t.SetConstraint("rib_middle", (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728) - 32);
    t.SetConstraint("rib_lower", (46.36 - 9.9 * x2 - 4.4505 * x1) - 32);
    t.SetConstraint("pubic_force", pubicForce - 4);
    t.SetConstraint("v_b_pillar", vmbp - 9.9);
    t.SetConstraint("v_front_door", vfd - 15.7);
    return new[] { weight, pubicForce, velocity };
}

/// <summary>Whether a design passes all ten safety limits.</summary>
/// <param name="t">A finished trial.</param>
/// <returns><c>true</c> when every recorded constraint is ≤ 0.</returns>
static bool IsLegal(FrozenTrial t) => t.Constraints.Count == 10 && t.Constraints.Values.All(c => c <= 0);
