#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// EV fast-charging protocol design — multi-objective Bayesian optimization (GPSampler with two objectives
// and a constraint) designs a four-step constant-current charging protocol that charges a cell from 0 to
// 80 % as fast as possible while keeping its cycle life long and its temperature under the safety limit.
//
// source: https://www.nature.com/articles/s41586-020-1994-5
//         (Attia et al., "Closed-loop optimization of fast-charging protocols for batteries with machine
//         learning", Nature 2020 — four CC steps over 0–80 % SOC, optimized by Bayesian optimization)
//         https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html
//         (multi-objective + constrained GP sampling)
//
// Surrogate physics (the real problem needs months of cycling per protocol, which is exactly why a
// sample-efficient GP is used): a protocol is four C-rates, one per 20 %-SOC window. Charge time is the sum
// of the window times. Degradation per cycle follows the mechanisms the paper discusses — lithium-plating
// stress that grows with C-rate and steeply with state of charge, plus an Arrhenius-accelerated SEI-growth
// term driven by the cell temperature, which is integrated through a lumped thermal model (I²R heating,
// convective cooling). Cycle life = cycles until 20 % capacity fade. The constraint: peak cell temperature
// ≤ 45 °C. 12 random protocols, then 28 GP-chosen ones; the Pareto front is the charge-time / life trade.
//
//   Run:  dotnet run examples/0_FastChargingProtocol.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var study = Optuna.CreateStudy(
    sampler: new GPSampler(seed: 3, nStartupTrials: 12, deterministicObjective: true),
    directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize },   // (minutes to 80 %, cycle life)
    studyName: "fast-charge");
study.SetMetricNames(new[] { "minutes_to_80pct", "cycle_life" });
study.Optimize(Objective, nTrials: 40);

var front = study.BestTrials.OrderBy(t => t.Values![0]).ToList();
Console.WriteLine($"protocols tested: {study.Trials.Count}; Pareto-optimal and within 45 °C: {front.Count}");
Console.WriteLine();
Console.WriteLine("  minutes  cycle life  peak °C   C-rates per SOC window (0-20, 20-40, 40-60, 60-80 %)");
foreach (var t in front)
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,9:F1} {1,11:F0} {2,8:F1}   {3:F2} / {4:F2} / {5:F2} / {6:F2}",
        t.Values![0], t.Values[1], t.UserAttrs["peak_temp_c"], t.Params["c1"], t.Params["c2"], t.Params["c3"], t.Params["c4"]));
var fastestLegal = front.First();
var flat = Cell.Charge(new[] { 2.0, 2.0, 2.0, 2.0 });
Console.WriteLine();
Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
    "reference: a flat 2C protocol charges in {0:F1} min and lasts {1:F0} cycles; the front's fastest legal protocol takes {2:F1} min",
    flat.Minutes, flat.CycleLife, fastestLegal.Values![0]));

/// <summary>Simulates one charging protocol.</summary>
/// <param name="t">The trial proposing the four C-rates.</param>
/// <returns>Minutes to 80 % SOC (minimize) and cycle life (maximize); the thermal limit is a constraint.</returns>
static IReadOnlyList<double> Objective(Trial t)
{
    var rates = new[]
    {
        t.SuggestFloat("c1", 0.5, 6.0),   // C-rate while SOC is 0–20 %
        t.SuggestFloat("c2", 0.5, 6.0),   // 20–40 %
        t.SuggestFloat("c3", 0.5, 6.0),   // 40–60 %
        t.SuggestFloat("c4", 0.5, 6.0),   // 60–80 %
    };
    var r = Cell.Charge(rates);
    t.SetUserAttr("peak_temp_c", r.PeakTempC);
    t.SetConstraint("peak_temp_c", r.PeakTempC - 45.0);   // ≤ 45 °C or the protocol is unsafe
    return new[] { r.Minutes, r.CycleLife };
}

/// <summary>A lumped electro-thermal degradation model of a 2.4 Ah cylindrical cell.</summary>
static class Cell
{
    /// <summary>Charges from 0 to 80 % SOC with one C-rate per 20 % window.</summary>
    /// <param name="rates">Four C-rates.</param>
    /// <returns>Charge time in minutes, the resulting cycle life, and the peak cell temperature in °C.</returns>
    public static (double Minutes, double CycleLife, double PeakTempC) Charge(double[] rates)
    {
        const double Capacity = 2.4, R = 0.035, Ambient = 25, HConv = 0.35, ThermalMass = 45;   // Ah, Ω, °C, W/K, J/K
        double temp = Ambient, peak = Ambient, minutes = 0, fade = 0;
        for (int w = 0; w < 4; w++)
        {
            double current = rates[w] * Capacity;                       // A
            double seconds = 0.2 / rates[w] * 3600;                     // 20 % of capacity at this rate
            double socMid = 0.1 + 0.2 * w;
            // integrate in 1 s steps: temperature (I²R heating vs. convection) and the two fade mechanisms.
            for (double s = 0; s < seconds; s += 1)
            {
                double dt = Math.Min(1, seconds - s);
                temp += dt * (current * current * R - HConv * (temp - Ambient)) / ThermalMass;
                peak = Math.Max(peak, temp);
                double plating = 2.2e-9 * Math.Pow(rates[w], 2.6) * Math.Exp(4.0 * socMid) * Math.Exp(-0.02 * (temp - 25));
                double sei = 1.1e-9 * Math.Exp(0.069 * (temp - 25));
                fade += dt * (plating + sei);
            }
            minutes += seconds / 60;
        }
        fade += 2.0e-5;                                                  // calendar + discharge share per cycle
        return (minutes, 0.2 / fade, peak);
    }
}
