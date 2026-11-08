#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Four-bar truss design, crossover shoot-out — NSGA-II sizes the four member cross-sections of a planar
// truss to trade structural volume (material, cost) against the joint's displacement under load, once per
// crossover operator Optuna ships, and compares the fronts by hypervolume.
//
// source: https://github.com/ryojitanabe/reproblems (problem RE21, "four bar truss design", of the real-world
//         multi-objective problem suite by Tanabe & Ishibuchi, 2020)
//         https://optuna.readthedocs.io/en/stable/reference/samplers/nsgaii.html
//         (optuna.samplers.nsgaii: UniformCrossover, SBXCrossover, BLXAlphaCrossover, UNDXCrossover,
//         SPXCrossover, VSBXCrossover)
//
// Real objective, closed-form structural mechanics: F = 10 kN, σ = 10 kN/cm², E = 2·10⁵ kN/cm², L = 200 cm;
// volume = L(2x₁ + √2·x₂ + √x₃ + x₄) and displacement = (F·L/E)(2/x₁ + 2√2/x₂ − 2√2/x₃ + 2/x₄), with each
// area bounded by the stress limit (F/σ … 3F/σ). Same seed, same budget (50 × 20 generations) for every
// operator; the front's quality is its 2-D hypervolume against the reference point (3,000 cm³, 0.045 cm)
// — larger is better. The objective is deterministic, so the table reproduces exactly.
//
//   Run:  dotnet run examples/0_TrussCrossoverShootout.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var operators = new (string Name, Func<BaseCrossover?> Make)[]
{
    ("uniform (default)", () => null),
    ("SBX", () => new SBXCrossover()),
    ("BLX-α (α = 0.5)", () => new BLXAlphaCrossover(alpha: 0.5)),
    ("UNDX", () => new UNDXCrossover()),
    ("SPX", () => new SPXCrossover()),
    ("VSBX", () => new VSBXCrossover()),
};

Console.WriteLine("crossover            hypervolume   front size   lightest design on the front");
foreach (var (name, make) in operators)
{
    var study = Optuna.CreateStudy(
        sampler: new NSGAIISampler(populationSize: 50, crossover: make(), seed: 8),
        directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
        studyName: "truss-" + name);
    study.Optimize(Objective, nTrials: 1000);
    var front = study.BestTrials;
    var lightest = front.OrderBy(t => t.Values![0]).First();
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,-20} {1,11:F4} {2,12}   {3:F1} cm³, displacement {4:F4} cm",
        name, Hypervolume2D(front.Select(t => (t.Values![0], t.Values[1])), (3000, 0.045)), front.Count, lightest.Values![0], lightest.Values[1]));
}

/// <summary>Evaluates one truss: four cross-sectional areas → volume and displacement.</summary>
/// <param name="t">The trial proposing the four areas (cm²).</param>
/// <returns>Structural volume (cm³) and joint displacement (cm), both minimized.</returns>
static IReadOnlyList<double> Objective(Trial t)
{
    const double F = 10.0, Sigma = 10.0, E = 2.0 * 1e5, L = 200.0;
    double a = F / Sigma;                                                   // the stress-limit unit area
    double x1 = t.SuggestFloat("x1", a, 3 * a);
    double x2 = t.SuggestFloat("x2", Math.Sqrt(2.0) * a, 3 * a);
    double x3 = t.SuggestFloat("x3", Math.Sqrt(2.0) * a, 3 * a);
    double x4 = t.SuggestFloat("x4", a, 3 * a);
    double volume = L * ((2 * x1) + Math.Sqrt(2.0) * x2 + Math.Sqrt(x3) + x4);
    double displacement = ((F * L) / E) * ((2.0 / x1) + (2.0 * Math.Sqrt(2.0) / x2) - (2.0 * Math.Sqrt(2.0) / x3) + (2.0 / x4));
    return new[] { volume, displacement };
}

/// <summary>Hypervolume of a two-objective minimization front against a reference point.</summary>
/// <param name="points">The front's objective vectors (dominated points are tolerated and add nothing).</param>
/// <param name="reference">The reference (worst acceptable) point.</param>
/// <returns>The area dominated by the front and bounded by the reference point, normalized by the reference box.</returns>
static double Hypervolume2D(IEnumerable<(double A, double B)> points, (double A, double B) reference)
{
    double area = 0, lastB = reference.B;
    foreach (var (a, b) in points.Where(p => p.A < reference.A && p.B < reference.B).OrderBy(p => p.A))
    {
        if (b >= lastB)
            continue;                                                       // dominated by an earlier point
        area += (reference.A - a) * (lastB - b);
        lastB = b;
    }
    return area / (reference.A * reference.B);
}
