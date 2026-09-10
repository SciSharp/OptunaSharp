using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace OptunaSharp.Benchmark.VolleyEval;

/// <summary>
///     The profile matrix (contender × challenge → normalized score) — VolleyEval's product, and the
///     deliberate alternative to a single misleading leaderboard. Backed by the flat
///     <see cref="CellResult" /> list from the <see cref="Evaluator" />, it answers the questions the DoD
///     poses: who wins each column, are the column winners distinct, and does each challenge's home beat
///     random.
/// </summary>
public sealed class ProfileMatrix
{
    private readonly Dictionary<(string challenge, string contender), CellResult> _cells;

    public ProfileMatrix(
        IReadOnlyList<IChallenge> challenges,
        IReadOnlyList<Contender> contenders,
        IReadOnlyList<CellResult> cells)
    {
        Challenges = challenges;
        Contenders = contenders;
        Cells = cells;
        _cells = cells.ToDictionary(c => (c.Challenge, c.Contender));
    }

    public IReadOnlyList<IChallenge> Challenges { get; }
    public IReadOnlyList<Contender> Contenders { get; }
    public IReadOnlyList<CellResult> Cells { get; }

    /// <summary>The result cell for a pair.</summary>
    public CellResult Cell(string challenge, string contender) => _cells[(challenge, contender)];

    /// <summary>The median normalized score for a pair (NaN if it did not finish).</summary>
    public double Score(string challenge, string contender) => _cells[(challenge, contender)].Value;

    /// <summary>The contender with the highest median score in a challenge column (DNF/NaN excluded).</summary>
    public string ColumnWinner(string challenge)
    {
        CellResult? best = null;
        foreach (var contender in Contenders)
        {
            var cell = _cells[(challenge, contender.Name)];
            if (cell.Dnf || double.IsNaN(cell.Value)) continue;
            if (best is null || cell.Value > best.Value) best = cell;
        }
        return best?.Contender ?? throw new InvalidOperationException($"No finishing contender for '{challenge}'.");
    }

    /// <summary>The winner of every challenge column, in challenge order.</summary>
    public IReadOnlyList<string> ColumnWinners() => Challenges.Select(c => ColumnWinner(c.Name)).ToList();

    /// <summary>True when every challenge column has a different winner — the central three-radical claim.</summary>
    public bool WinnersDistinct()
    {
        var winners = ColumnWinners();
        return winners.Distinct(StringComparer.Ordinal).Count() == winners.Count;
    }

    /// <summary>True when at least one of <paramref name="homes" /> beats the <c>random</c> contender in the column.</summary>
    public bool HomeBeatsRandom(string challenge, params string[] homes)
    {
        double random = Score(challenge, "random");
        return homes.Any(h => Score(challenge, h) > random);
    }

    /// <summary>True when the worst of <paramref name="homes" /> beats the best of <paramref name="aways" /> (challenges 1 &amp; 3 DoD).</summary>
    public bool HomeBeatsAway(string challenge, string[] homes, string[] aways)
    {
        double worstHome = homes.Min(h => Score(challenge, h));
        double bestAway = aways
            .Select(a => Score(challenge, a))
            .Where(v => !double.IsNaN(v))
            .DefaultIfEmpty(double.NegativeInfinity)
            .Max();
        return worstHome > bestAway;
    }

    /// <summary>Render the matrix as an aligned text table with per-cell markers (★ winner, inert, DNF).</summary>
    public string Render()
    {
        var sb = new StringBuilder();
        int nameWidth = Math.Max(9, Contenders.Max(c => c.Name.Length));
        int colWidth = Math.Max(16, Challenges.Max(c => c.Name.Length));

        // Header.
        sb.Append("contender".PadRight(nameWidth));
        foreach (var ch in Challenges) sb.Append("  ").Append(ch.Name.PadLeft(colWidth));
        sb.AppendLine();
        sb.Append(new string('-', nameWidth));
        foreach (var _ in Challenges) sb.Append("  ").Append(new string('-', colWidth));
        sb.AppendLine();

        var winners = Challenges.ToDictionary(c => c.Name, c => ColumnWinner(c.Name));

        foreach (var contender in Contenders)
        {
            sb.Append(contender.Name.PadRight(nameWidth));
            foreach (var ch in Challenges)
            {
                var cell = _cells[(ch.Name, contender.Name)];
                string text;
                if (cell.Dnf)
                    text = $"DNF({cell.Error})";
                else
                {
                    text = cell.Median.ToString("F4");
                    if (winners[ch.Name] == contender.Name) text += "*";     // * column winner
                    if (cell.Inert) text += " inert";
                    if (cell.EffectiveBudget < ch.Budget) text += $" [{cell.EffectiveBudget}]"; // trial-capped
                }
                sb.Append("  ").Append(text.PadLeft(colWidth));
            }
            sb.AppendLine();
        }

        // Winners + legend.
        sb.AppendLine();
        sb.Append("winners: ");
        sb.AppendLine(string.Join("  ",
            Challenges.Select(c => $"{c.Name}={ColumnWinner(c.Name)}")));
        sb.AppendLine("legend: value = median normalized score over seeds (1 = optimum, 0 = random baseline); " +
                      "* = column winner; inert = trajectory == RandomSampler; [n] = trial-capped to n; DNF = could not run.");
        return sb.ToString();
    }

    public override string ToString() => Render();
}
