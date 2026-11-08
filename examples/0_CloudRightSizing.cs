#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Cloud right-sizing — TPE with `Trial.set_constraint` picks the cheapest fleet (instance family × size × count)
// that still meets a p99-latency SLO, a per-instance memory floor, and survives the loss of one availability
// zone.
//
// source: https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-optimize-a-model-with-some-constraints
//         (Optuna's constrained-optimization recipe; Optuna 5 moved it from `constraints_func` to
//         `Trial.set_constraint`, which this example uses)
//         https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
//
// Real objective, analytic model: the monthly bill is count × hourly price × 730 h. Latency comes from an
// M/M/c queue (Erlang C) at the peak arrival rate — each vCPU is a server whose service rate depends on the
// instance family's per-core speed — so p99 = queueing-wait p99 + service-time p99. The three constraints
// are the kind a capacity review asks for: p99 ≤ 120 ms, ≥ 7 GiB per instance (a 6 GiB JVM heap plus the
// OS), and utilization < 95 % after one of three AZs goes dark. A constraint value ≤ 0 means "satisfied";
// the objective records each with `SetConstraint`, TPE uses them to prefer feasible regions, and
// `study.best_trial` returns the best FEASIBLE trial.
//
//   Run:  dotnet run examples/0_CloudRightSizing.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var sampler = new TPESampler(
    nStartupTrials: 25,                          // random exploration before the Parzen estimators kick in
    multivariate: true,                          // model family/size/count jointly (they interact)
    seed: 2024);                                 // no constraints_func: Optuna 5 reads Trial.SetConstraint

var study = Optuna.CreateStudy(sampler: sampler, direction: StudyDirection.Minimize, studyName: "right-sizing");
study.Optimize(Objective, nTrials: 300);

var trials = study.Trials;
int feasible = trials.Count(t => IsFeasible(t));
var cheapestOverall = trials.OrderBy(t => t.Value).First();
var best = study.BestTrial;                       // Optuna 5: the best among feasible trials

Console.WriteLine($"{trials.Count} trials, {feasible} feasible");
Console.WriteLine();
Describe("cheapest of all (ignores the SLO)", cheapestOverall);
Describe("best feasible (study.BestTrial)", best);

/// <summary>Prices one fleet and records the three constraint values with <c>SetConstraint</c>.</summary>
/// <param name="t">The trial proposing the family, size and instance count.</param>
/// <returns>The monthly cost in USD (what the study minimizes).</returns>
static double Objective(Trial t)
{
    string family = t.SuggestCategorical("family", new[] { "general", "compute", "memory", "arm" });
    string size = t.SuggestCategorical("size", new[] { "large", "xlarge", "2xlarge", "4xlarge" });
    long count = t.SuggestInt("count", 1, 60);                  // instances behind the load balancer

    var (vcpu, memGib, pricePerHour, coreSpeed) = Catalog(family, size);
    const double PeakRps = 2400;                                // the Black-Friday peak arrival rate
    const double BaseServiceRate = 111;                         // requests/s one baseline vCPU completes
    double mu = BaseServiceRate * coreSpeed;
    int servers = (int)(vcpu * count);

    double p99Ms = P99Millis(PeakRps, mu, servers);
    // one AZ of three is lost: the survivors are the instances in the other two AZs (round-robin placement).
    long survivors = count - (count + 2) / 3;
    double rhoAfterLoss = survivors == 0 ? double.PositiveInfinity : PeakRps / (mu * vcpu * survivors);

    t.SetUserAttr("p99_ms", Math.Min(p99Ms, 1e6));                 // kept for the report (capped: JSON-safe)
    // each constraint is "≤ 0 means satisfied"; a positive value says by how much it is violated.
    t.SetConstraint("latency_ms", Math.Min(p99Ms, 1e6) - 120.0);   // p99 ≤ 120 ms (capped: no infinities)
    t.SetConstraint("memory_gib", 7.0 - memGib);                   // ≥ 7 GiB per instance
    t.SetConstraint("az_loss_rho", Math.Min(rhoAfterLoss, 1e6) - 0.95);
    return count * pricePerHour * 730.0;
}

/// <summary>Looks up an instance type in a small on-demand price list (us-east-1-like, illustrative).</summary>
/// <param name="family">general, compute, memory or arm.</param>
/// <param name="size">large, xlarge, 2xlarge or 4xlarge (each step doubles vCPU, memory and price).</param>
/// <returns>vCPUs, memory in GiB, USD per hour, and the per-core speed relative to the general family.</returns>
/// <exception cref="ArgumentException">An unknown family or size (the search space never produces one).</exception>
static (int Vcpu, double MemGib, double PricePerHour, double CoreSpeed) Catalog(string family, string size)
{
    int mult = size switch
    {
        "large" => 1, "xlarge" => 2, "2xlarge" => 4, "4xlarge" => 8,
        _ => throw new ArgumentException($"unknown size {size}"),
    };
    // (memory GiB per vCPU, price per vCPU-hour, per-core speed)
    var (gibPerVcpu, pricePerVcpu, speed) = family switch
    {
        "general" => (4.0, 0.0504, 1.00),
        "compute" => (2.0, 0.04465, 1.15),
        "memory" => (8.0, 0.06615, 1.00),
        "arm" => (4.0, 0.0408, 0.92),
        _ => throw new ArgumentException($"unknown family {family}"),
    };
    int vcpu = 2 * mult;
    return (vcpu, gibPerVcpu * vcpu, pricePerVcpu * vcpu, speed);
}

/// <summary>The 99th-percentile response time of an M/M/c queue, in milliseconds.</summary>
/// <param name="lambda">Arrival rate (requests per second).</param>
/// <param name="mu">Service rate of one server (requests per second).</param>
/// <param name="c">Number of servers (vCPUs across the fleet).</param>
/// <returns>
///     Wait-time p99 (from the Erlang-C tail, P(W &gt; t) = C·e^{−(cμ−λ)t}) plus the exponential service-time
///     p99; positive infinity when the queue is unstable (λ ≥ cμ).
/// </returns>
static double P99Millis(double lambda, double mu, int c)
{
    double a = lambda / mu;                                     // offered load in Erlangs
    if (a >= c)
        return double.PositiveInfinity;
    // Erlang B by the stable recursion B(k) = a·B(k−1) / (k + a·B(k−1)), then Erlang C from it.
    double b = 1.0;
    for (int k = 1; k <= c; k++)
        b = a * b / (k + a * b);
    double erlangC = c * b / (c - a * (1 - b));
    double waitP99 = erlangC <= 0.01 ? 0.0 : Math.Log(erlangC / 0.01) / (c * mu - lambda);
    double serviceP99 = Math.Log(100) / mu;
    return 1000.0 * (waitP99 + serviceP99);
}

/// <summary>Whether every recorded constraint of a trial is satisfied (≤ 0).</summary>
/// <param name="t">A finished trial carrying the three constraints set by the objective.</param>
/// <returns><c>true</c> when the fleet meets the SLO, the memory floor and the AZ-loss headroom.</returns>
static bool IsFeasible(FrozenTrial t) => t.Constraints.Count == 3 && t.Constraints.Values.All(v => v <= 0);

/// <summary>Prints a trial's fleet, its bill and its constraint status.</summary>
/// <param name="label">What the trial represents.</param>
/// <param name="t">The trial to describe.</param>
static void Describe(string label, FrozenTrial t) =>
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
        "{0,-36} #{1,-4} {2,2} × {3}.{4,-8} ${5,9:N0}/month   p99 {6,8:F1} ms   feasible: {7}",
        label, t.Number, t.Params["count"], t.Params["family"], t.Params["size"], t.Value,
        Math.Min((double)t.UserAttrs["p99_ms"]!, 99999), IsFeasible(t)));
