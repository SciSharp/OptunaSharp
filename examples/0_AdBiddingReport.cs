#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:project ../src/OptunaSharp.Visualization.Plotly/OptunaSharp.Visualization.Plotly.csproj
#:property PublishAot=false

// Ad-bidding strategy, with the stakeholder report — TPE tunes a programmatic-advertising bidder (bid
// multiplier, quality threshold, budget pacing, frequency cap) on a 14-day auction simulation, MedianPruner
// stops money-losing strategies after a few simulated days, and the study is then rendered as the nine
// Plotly HTML charts a marketing team actually looks at.
//
// source: https://optuna.readthedocs.io/en/stable/reference/visualization/index.html
//         (optuna.visualization: optimization history, importances, parallel coordinate, contour, slice,
//         EDF, rank, timeline, intermediate values)
//         https://en.wikipedia.org/wiki/Real-time_bidding (second-price auctions, pacing, frequency capping)
//
// Real objective, Monte Carlo: each simulated day offers 20,000 impressions; each has a predicted CTR
// (log-normal around 1 %), a 5 % conversion rate after a click worth $40, and a highest competing bid. The
// bidder bids multiplier × expected value, skips impressions under its CTR floor, caps impressions per user
// per day, and paces a $500 daily budget ("none" spends until empty, "even" spreads it over the day,
// "throttle" randomly sits out auctions). Second-price auctions: a win costs the competing bid. The daily
// cumulative profit is reported, so MedianPruner can cut losing strategies from day 4 on. The charts are
// written to a temp folder; open them in a browser.
//
//   Run:  dotnet run examples/0_AdBiddingReport.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET.CSharp;   // .SaveHtml (importing Plotly.NET too would make SaveHtml ambiguous)

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var study = Optuna.CreateStudy(
    sampler: new TPESampler(seed: 15, nStartupTrials: 20),
    pruner: new MedianPruner(nStartupTrials: 10, nWarmupSteps: 3),   // judge a strategy from day 4
    direction: StudyDirection.Maximize,
    studyName: "rtb-bidder");
study.Optimize(Objective, nTrials: 120);

var best = study.BestTrial;
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best 14-day profit: ${0:N0} (trial #{1}); pruned strategies: {2} of {3}",
    best.Value, best.Number, study.Trials.Count(t => t.State == TrialState.Pruned), study.Trials.Count));
foreach (var kv in best.Params)
    Console.WriteLine($"    {kv.Key,-18} {kv.Value}");

// the report: one HTML file per chart.
string dir = Path.Combine(Path.GetTempPath(), "optunasharp-rtb-report");
Directory.CreateDirectory(dir);
var charts = new (string Name, Func<Plotly.NET.GenericChart> Make)[]
{
    ("optimization_history", () => PlotlyVisualization.PlotOptimizationHistory(study)),
    ("intermediate_values", () => PlotlyVisualization.PlotIntermediateValues(study)),
    ("param_importances", () => PlotlyVisualization.PlotParamImportances(study)),
    ("parallel_coordinate", () => PlotlyVisualization.PlotParallelCoordinate(study)),
    ("contour_bid_vs_ctr", () => PlotlyVisualization.PlotContour(study, new[] { "bid_multiplier", "min_ctr" })),
    ("slice", () => PlotlyVisualization.PlotSlice(study)),
    ("edf", () => PlotlyVisualization.PlotEdf(study)),
    ("rank", () => PlotlyVisualization.PlotRank(study)),
    ("timeline", () => PlotlyVisualization.PlotTimeline(study)),
};
Console.WriteLine();
foreach (var (name, make) in charts)
{
    string path = Path.Combine(dir, name + ".html");
    make().SaveHtml(path);
    Console.WriteLine($"  {name,-22} {new FileInfo(path).Length / 1024,5} KiB  {path}");
}

/// <summary>Simulates 14 days of the trial's bidding strategy, reporting cumulative profit daily.</summary>
/// <param name="t">The trial proposing the strategy.</param>
/// <returns>Total profit in USD (conversion value minus spend).</returns>
/// <exception cref="TrialPruned">The strategy's profit trails the median of earlier strategies.</exception>
static double Objective(Trial t)
{
    double multiplier = t.SuggestFloat("bid_multiplier", 0.2, 5.0, log: true);
    double minCtr = t.SuggestFloat("min_ctr", 1e-4, 0.05, log: true);           // skip low-quality inventory
    string pacing = t.SuggestCategorical("pacing", new[] { "none", "even", "throttle" });
    double throttle = t.SuggestFloat("throttle_rate", 0.0, 0.9);                // only used by "throttle"
    long freqCap = t.SuggestInt("frequency_cap", 1, 10);                        // impressions per user per day

    double profit = 0;
    for (int day = 0; day < 14; day++)
    {
        profit += Auction.SimulateDay(multiplier, minCtr, pacing, throttle, freqCap, new Random(1000 + day));
        t.Report(profit, day);
        if (t.ShouldPrune())
            throw new TrialPruned();
    }
    return profit;
}

/// <summary>One day of second-price auctions for a budget-constrained bidder.</summary>
static class Auction
{
    /// <summary>Runs one day.</summary>
    /// <param name="multiplier">Bid = multiplier × expected value of the impression.</param>
    /// <param name="minCtr">Impressions with a lower predicted CTR are skipped.</param>
    /// <param name="pacing">"none", "even" or "throttle".</param>
    /// <param name="throttle">Probability of sitting out an auction under "throttle" pacing.</param>
    /// <param name="freqCap">Maximum impressions shown to one user per day.</param>
    /// <param name="rng">The day's random stream (shared by all strategies: common random numbers).</param>
    /// <returns>The day's profit.</returns>
    public static double SimulateDay(double multiplier, double minCtr, string pacing, double throttle, long freqCap, Random rng)
    {
        const int Opportunities = 20_000, Users = 4_000;
        const double Budget = 500, Cvr = 0.05, ConversionValue = 40;
        var shown = new int[Users];
        double spend = 0, value = 0;
        for (int k = 0; k < Opportunities; k++)
        {
            double ctr = Math.Exp(Math.Log(0.01) + 0.9 * Normal(rng));
            double competitor = Math.Exp(Math.Log(0.012) + 0.7 * Normal(rng));   // highest other bid, $ per impression
            int user = rng.Next(Users);
            bool sitOut = pacing == "throttle" && rng.NextDouble() < throttle;
            bool overPace = pacing == "even" && spend > Budget * (k + 1) / Opportunities;
            if (spend >= Budget || sitOut || overPace || ctr < minCtr || shown[user] >= freqCap)
                continue;
            // fatigue: each repeat impression to the same user converts less.
            double expected = ctr * Cvr * ConversionValue * Math.Pow(0.7, shown[user]);
            double bid = multiplier * expected;
            if (bid <= competitor)
                continue;
            spend += competitor;                                              // second price
            shown[user]++;
            if (rng.NextDouble() < ctr && rng.NextDouble() < Cvr * Math.Pow(0.7, shown[user] - 1))
                value += ConversionValue;
        }
        return value - spend;
    }

    /// <summary>A standard normal draw (Box–Muller).</summary>
    /// <param name="rng">The random stream.</param>
    /// <returns>A N(0, 1) variate.</returns>
    private static double Normal(Random rng) => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
}
