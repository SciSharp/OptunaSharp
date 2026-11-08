#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Off-grid microgrid sizing — CMA-ES with margin (mixed-integer CMA-ES) sizes a solar + battery + diesel
// system for a remote site: how many PV panels and battery modules, how big an inverter and generator, and
// how much battery to hold in reserve — minimizing the levelized cost of energy over a simulated year.
//
// source: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html
//         (`with_margin=True`: CMA-ES with margin keeps integer parameters from collapsing to one value)
//         https://en.wikipedia.org/wiki/Levelized_cost_of_electricity
//
// Real objective, energy-balance simulation (HOMER-style): 8,760 hourly steps of a site load (40 kW base,
// morning/evening peaks, a summer cooling bump) against PV output (a clear-sky bell × season × seeded
// cloudiness). Each hour solar serves the load through the inverter, surplus charges the batteries (95 %
// round-trip each way), a deficit is covered by the batteries down to the reserve, then by the diesel
// generator; whatever is left is unserved. Cost per year = annualized capital (7 % discount rate, each
// component's lifetime) + diesel fuel and O&M + $10 per unserved kWh (the value of lost load), divided by the
// energy served. Panel, module and generator counts are integers — exactly the case CMA-ES with margin
// exists for, since plain CMA-ES would let an integer coordinate's variance shrink to nothing.
//
//   Run:  dotnet run examples/0_MicrogridSizingCmaMargin.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var site = Site.Year(seed: 21);
var study = Optuna.CreateStudy(
    sampler: new CmaEsSampler(seed: 4, withMargin: true),
    direction: StudyDirection.Minimize,
    studyName: "microgrid");
study.Optimize(t => Objective(t, site), nTrials: 300);

var best = study.BestTrial;
var r = Microgrid.Simulate(site, (long)best.Params["pv_panels"]!, (long)best.Params["battery_modules"]!,
    (double)best.Params["inverter_kw"]!, (long)best.Params["diesel_kw"]!, (double)best.Params["soc_reserve"]!);
var dieselOnly = Microgrid.Simulate(site, 0, 0, 20, 120, 0.2);
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "annual load: {0:N0} kWh", site.Load.Sum()));
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "diesel only : LCOE ${0:F3}/kWh", dieselOnly.Lcoe));
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best (#{0,3}): LCOE ${1:F3}/kWh  solar share {2:P0}  diesel {3:N0} kWh  unserved {4:N0} kWh",
    best.Number, r.Lcoe, r.SolarShare, r.DieselKwh, r.UnservedKwh));
foreach (var kv in best.Params)
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "    {0,-16} {1}", kv.Key, kv.Value));
var distinctPanels = study.Trials.Skip(200).Select(t => t.Params["pv_panels"]).Distinct().Count();
Console.WriteLine($"distinct pv_panels values in the last 100 trials (the margin keeps integers exploring): {distinctPanels}");

/// <summary>Simulates one system design for a year.</summary>
/// <param name="t">The trial proposing the component sizes and the reserve.</param>
/// <param name="s">The site's hourly load and solar profiles.</param>
/// <returns>The levelized cost of energy in USD per kWh served (lost load priced in).</returns>
static double Objective(Trial t, Site s)
{
    long panels = t.SuggestInt("pv_panels", 0, 1500, step: 10);      // 400 W modules
    long modules = t.SuggestInt("battery_modules", 0, 60);           // 13.5 kWh each
    double inverter = t.SuggestFloat("inverter_kw", 10, 300);
    long diesel = t.SuggestInt("diesel_kw", 0, 150, step: 10);
    double reserve = t.SuggestFloat("soc_reserve", 0.05, 0.6);       // battery kept back for the night
    return Microgrid.Simulate(s, panels, modules, inverter, diesel, reserve).Lcoe;
}

/// <summary>A year of hourly site data.</summary>
/// <param name="Load">Load in kW for each hour.</param>
/// <param name="SolarPerPanel">PV output of one 400 W panel in kW for each hour.</param>
sealed record Site(double[] Load, double[] SolarPerPanel)
{
    /// <summary>Synthesizes the profiles; the same seed gives the same year.</summary>
    /// <param name="seed">Weather and load-noise seed.</param>
    /// <returns>The site.</returns>
    public static Site Year(int seed)
    {
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        var load = new double[8760];
        var solar = new double[8760];
        double cloud = 0.8;
        for (int h = 0; h < 8760; h++)
        {
            int day = h / 24, hour = h % 24;
            if (hour == 0)
                cloud = Math.Clamp(0.55 * cloud + 0.45 * rng.NextDouble() + 0.1, 0.1, 1.0);   // persistent weather
            double season = Math.Cos(2 * Math.PI * (day - 172) / 365.0);                       // +1 at midsummer
            double peaks = 12 * Math.Exp(-Math.Pow(hour - 8, 2) / 4) + 18 * Math.Exp(-Math.Pow(hour - 19, 2) / 6);
            load[h] = 40 + peaks + 10 * Math.Max(0, season) * Math.Max(0, Math.Sin(Math.PI * (hour - 9) / 12)) + 3 * (rng.NextDouble() - 0.5);
            double dayLength = 12 + 3 * season;
            double sunrise = 12 - dayLength / 2;
            double x = (hour + 0.5 - sunrise) / dayLength;
            double bell = x is > 0 and < 1 ? Math.Sin(Math.PI * x) : 0;
            solar[h] = 0.4 * bell * (0.75 + 0.25 * season) * cloud;
        }
        return new Site(load, solar);
    }
}

/// <summary>The hourly dispatch simulation and the cost model.</summary>
static class Microgrid
{
    /// <summary>Simulates a year and prices it.</summary>
    /// <param name="s">The site.</param>
    /// <param name="panels">PV panel count.</param>
    /// <param name="modules">Battery module count.</param>
    /// <param name="inverterKw">Inverter AC rating.</param>
    /// <param name="dieselKw">Generator rating.</param>
    /// <param name="reserve">State-of-charge fraction the battery will not discharge below.</param>
    /// <returns>LCOE, solar share of the load served, diesel energy and unserved energy.</returns>
    public static (double Lcoe, double SolarShare, double DieselKwh, double UnservedKwh) Simulate(
        Site s, long panels, long modules, double inverterKw, long dieselKw, double reserve)
    {
        double capacity = modules * 13.5, soc = capacity * 0.5;
        double solarUsed = 0, dieselKwh = 0, unserved = 0, served = 0, dieselHours = 0;
        for (int h = 0; h < 8760; h++)
        {
            double pv = Math.Min(panels * s.SolarPerPanel[h], inverterKw);
            double need = s.Load[h];
            double direct = Math.Min(pv, need);
            solarUsed += direct;
            need -= direct;
            soc = Math.Min(capacity, soc + (pv - direct) * 0.95);                 // charge with the surplus
            double fromBattery = Math.Min(need, Math.Max(0, soc - reserve * capacity) * 0.95);
            soc -= fromBattery / 0.95;
            need -= fromBattery;
            double fromDiesel = Math.Min(need, dieselKw);
            if (fromDiesel > 0)
                dieselHours++;
            dieselKwh += fromDiesel;
            need -= fromDiesel;
            unserved += need;
            served += s.Load[h] - need;
        }
        // capital recovery factor: annual payment per $ of capital over n years at 7 %.
        static double Crf(double years) => 0.07 / (1 - Math.Pow(1.07, -years));
        double capital = panels * 300 * Crf(25) + modules * 9000 * Crf(10) + inverterKw * 150 * Crf(12) + dieselKw * 500 * Crf(15);
        double operating = dieselKwh * 0.35 + dieselHours * 2.0;                 // fuel + per-run-hour O&M
        double lostLoad = unserved * 10.0;
        double lcoe = (capital + operating + lostLoad) / Math.Max(served, 1);
        return (lcoe, (solarUsed + Math.Max(0, served - solarUsed - dieselKwh)) / Math.Max(served, 1), dieselKwh, unserved);
    }
}
