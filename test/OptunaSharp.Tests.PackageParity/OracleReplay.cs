using System.Text.Json;
using System.Text.Json.Serialization;
using AwesomeAssertions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.PackageParity;

// ------------------------------------------------------------------------------------------------
// Oracle model — the trial trajectory a REAL Optuna + REAL ML package (xgboost / lightgbm / sklearn)
// produced, emitted by gen_oracle.py. The C# side replays it against the PACKAGED OptunaSharp.
// ------------------------------------------------------------------------------------------------

public sealed class Oracle
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("sampler")] public string Sampler { get; set; } = "";   // "random" | "tpe"
    [JsonPropertyName("seed")] public int Seed { get; set; }
    [JsonPropertyName("direction")] public string Direction { get; set; } = "minimize";
    [JsonPropertyName("space")] public List<SpaceEntry> Space { get; set; } = new();
    [JsonPropertyName("trials")] public List<TrialRec> Trials { get; set; } = new();
}

public sealed class SpaceEntry
{
    [JsonPropertyName("kind")] public string Kind { get; set; } = "float";     // "float" | "int"
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("low")] public double Low { get; set; }
    [JsonPropertyName("high")] public double High { get; set; }
    [JsonPropertyName("log")] public bool Log { get; set; }
}

public sealed class TrialRec
{
    [JsonPropertyName("number")] public int Number { get; set; }
    [JsonPropertyName("state")] public int State { get; set; }
    [JsonPropertyName("params")] public Dictionary<string, JsonElement> Params { get; set; } = new();
    [JsonPropertyName("value")] public double? Value { get; set; }
}

// ------------------------------------------------------------------------------------------------

public static class OracleStore
{
    /// <summary>Directory holding the generated oracle JSONs (env, else ./oracle if present).</summary>
    public static string? Dir =>
        Environment.GetEnvironmentVariable("PACKAGE_PARITY_ORACLE_DIR") is { Length: > 0 } d ? d
        : Directory.Exists("oracle") ? "oracle" : null;

    /// <summary>When truthy, a missing oracle is a hard FAIL instead of Inconclusive (CI sets it).</summary>
    public static bool Required =>
        Environment.GetEnvironmentVariable("PACKAGE_PARITY_REQUIRE") is { Length: > 0 } v
        && v != "0" && !v.Equals("false", StringComparison.OrdinalIgnoreCase);

    /// <summary>Load a named oracle, or self-skip (Inconclusive) when none was generated.</summary>
    public static Oracle LoadOrInconclusive(string name)
    {
        string? dir = Dir;
        string? path = dir is null ? null : Path.Combine(dir, name + ".json");
        if (path is null || !File.Exists(path))
        {
            string msg = $"oracle '{name}.json' not found (PACKAGE_PARITY_ORACLE_DIR={dir ?? "<unset>"}); " +
                         "run gen_oracle.py (real optuna + xgboost/lightgbm/sklearn) first.";
            if (Required)
                Assert.Fail(msg);
            Assert.Inconclusive(msg);
        }

        return JsonSerializer.Deserialize<Oracle>(File.ReadAllText(path!))
               ?? throw new InvalidOperationException($"could not parse oracle at {path}");
    }
}

// ------------------------------------------------------------------------------------------------

public static class Replay
{
    private const double FloatTol = 1e-9;

    /// <summary>
    ///     Re-drive the PACKAGED OptunaSharp through the oracle's trajectory: for each recorded trial,
    ///     Ask a trial, make the identical suggest_* calls (which consume the sampler RNG and produce
    ///     OptunaSharp's own suggestion), assert that suggestion matches what real Optuna suggested,
    ///     then Tell the REAL value the ML package produced so the next Ask sees the same history.
    ///     RandomSampler is asserted bit-exact; TPE to a small tolerance on continuous params.
    /// </summary>
    public static void AssertParity(Oracle oracle)
    {
        bool bitExact = oracle.Sampler == "random";
        BaseSampler sampler = oracle.Sampler switch
        {
            "random" => new RandomSampler(oracle.Seed),
            "tpe" => new TPESampler(seed: oracle.Seed),
            _ => throw new InvalidOperationException($"unknown sampler '{oracle.Sampler}'"),
        };
        StudyDirection direction = oracle.Direction == "maximize"
            ? StudyDirection.Maximize
            : StudyDirection.Minimize;

        var study = Optuna.CreateStudy(sampler: sampler, direction: direction, studyName: oracle.Name);

        foreach (var rec in oracle.Trials)
        {
            var trial = study.Ask();
            foreach (var entry in oracle.Space)
            {
                if (entry.Kind == "int")
                {
                    long got = trial.SuggestInt(entry.Name, (long)entry.Low, (long)entry.High);
                    long expected = rec.Params[entry.Name].GetInt64();
                    got.Should().Be(expected, $"{oracle.Name} trial {rec.Number} int '{entry.Name}'");
                }
                else
                {
                    double got = trial.SuggestFloat(entry.Name, entry.Low, entry.High, log: entry.Log);
                    double expected = rec.Params[entry.Name].GetDouble();
                    if (bitExact)
                        BitEqual(got, expected).Should().BeTrue(
                            $"{oracle.Name} trial {rec.Number} float '{entry.Name}': " +
                            $"got {got:R} (0x{BitConverter.DoubleToInt64Bits(got):x16}) != " +
                            $"expected {expected:R} (0x{BitConverter.DoubleToInt64Bits(expected):x16})");
                    else
                        got.Should().BeApproximately(expected, Math.Max(FloatTol, FloatTol * Math.Abs(expected)),
                            $"{oracle.Name} trial {rec.Number} float '{entry.Name}'");
                }
            }

            study.Tell(trial, rec.Value
                ?? throw new InvalidOperationException($"trial {rec.Number} has no value to replay"));
        }
    }

    private static bool BitEqual(double a, double b)
        => BitConverter.DoubleToInt64Bits(a) == BitConverter.DoubleToInt64Bits(b)
           || (double.IsNaN(a) && double.IsNaN(b));
}
