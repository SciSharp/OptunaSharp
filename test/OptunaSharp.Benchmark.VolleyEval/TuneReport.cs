using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace OptunaSharp.Benchmark.VolleyEval;

/// <summary>Renders the nested-hyperopt results: default vs tuned (held-out) per profile, plus the recipe.</summary>
public static class TuneReport
{
    public static string Render(IReadOnlyList<TunedProfile> profiles)
    {
        var sb = new StringBuilder();
        int nameWidth = Math.Max(24, profiles.Max(p => (p.Challenge + "/" + p.Contender).Length));

        sb.Append("profile".PadRight(nameWidth));
        sb.Append("  ").Append("default".PadLeft(8));
        sb.Append("  ").Append("tuned".PadLeft(8));
        sb.Append("  ").Append("delta".PadLeft(8));
        sb.Append("  ").Append("search".PadLeft(8));
        sb.Append("  ").Append("recipe");
        sb.AppendLine();
        sb.AppendLine(new string('-', nameWidth + 40));

        foreach (var p in profiles.OrderBy(p => p.Challenge).ThenBy(p => p.Contender))
        {
            string label = $"{p.Challenge}/{p.Contender}";
            sb.Append(label.PadRight(nameWidth));
            if (p.Dnf)
            {
                sb.Append("  ").Append("DNF".PadLeft(8)).AppendLine();
                continue;
            }
            if (!p.Tunable)
            {
                sb.Append("  ").Append(F(p.DefaultScore)).Append("  ").Append("-".PadLeft(8))
                  .Append("  ").Append("-".PadLeft(8)).Append("  ").Append("-".PadLeft(8))
                  .Append("  (no tunable knobs)").AppendLine();
                continue;
            }
            sb.Append("  ").Append(F(p.DefaultScore));
            sb.Append("  ").Append(F(p.TunedScore));
            sb.Append("  ").Append(Signed(p.Delta));
            sb.Append("  ").Append(F(p.SearchScore));
            sb.Append("  ").Append(Recipe(p.Recipe));
            if (p.Overfit) sb.Append("  [overfit: search >> held-out]");
            sb.AppendLine();
        }

        var tunable = profiles.Where(p => p.Tunable && !p.Dnf).ToList();
        sb.AppendLine();
        if (tunable.Count > 0)
        {
            int improved = tunable.Count(p => p.Improved);
            int overfit = tunable.Count(p => p.Overfit);
            double meanDelta = tunable.Average(p => p.Delta);
            sb.AppendLine($"summary: tuned {tunable.Count} profiles | improved {improved} | " +
                          $"regressed {tunable.Count(p => p.Delta < -1e-6)} | overfit-flagged {overfit} | " +
                          $"mean held-out delta {Signed(meanDelta).Trim()}");
        }
        sb.AppendLine("note: 'default'/'tuned'/'search' are median normalized scores; 'delta' = tuned − default on " +
                      "HELD-OUT seeds (the honest figure); 'search' = the outer optimum on the search seeds.");
        return sb.ToString();
    }

    private static string F(double v) => double.IsNaN(v) ? "-".PadLeft(8) : v.ToString("F4", CultureInfo.InvariantCulture).PadLeft(8);

    private static string Signed(double v)
        => double.IsNaN(v) ? "-".PadLeft(8) : ((v >= 0 ? "+" : "") + v.ToString("F4", CultureInfo.InvariantCulture)).PadLeft(8);

    private static string Recipe(IReadOnlyDictionary<string, object?> recipe)
        => recipe.Count == 0 ? "-" : string.Join(", ", recipe.Select(kv => $"{kv.Key}={FormatVal(kv.Value)}"));

    private static string FormatVal(object? v) => v switch
    {
        null => "null",
        bool b => b ? "true" : "false",
        double d => d.ToString("0.###", CultureInfo.InvariantCulture),
        float f => f.ToString("0.###", CultureInfo.InvariantCulture),
        _ => Convert.ToString(v, CultureInfo.InvariantCulture) ?? "?",
    };
}
