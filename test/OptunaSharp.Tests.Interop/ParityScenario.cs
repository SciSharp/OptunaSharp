using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     A deterministic differential-fuzz scenario: a randomized-but-seeded search space plus a
///     per-trial operation script (reports incl. NaN/±inf, user attrs incl. unicode and nested
///     containers, constraints, complete/fail/prune outcomes, enqueued fixed params, waiting
///     leftovers), executable IDENTICALLY on OptunaSharp and on live Python Optuna. The spec is
///     carried to Python as JSON (via scope binding — no source escaping), and both executors
///     derive objective values from the suggested params with the same IEEE operation order, so
///     a full-state <see cref="StudySnapshot.AssertParity" /> at the end is a bit-exact claim
///     over the entire run.
/// </summary>
public sealed class ParityScenario
{
    public int Seed;
    public int SamplerSeed;
    public int NTrials;
    public bool MultiObjective;
    public List<SpaceEntry> Space = new();
    public List<TrialOps> Ops = new();
    public List<OrderedDict<string, object?>> Enqueues = new();
    public List<OrderedDict<string, object?>> EnqueuesAfter = new();

    public sealed class SpaceEntry
    {
        public string Kind = "float";  // float | int | categorical
        public string Name = string.Empty;
        public double Low;
        public double High;
        public double? Step;
        public bool Log;
        public List<object?> Choices = new();
    }

    public sealed class TrialOps
    {
        public List<(long Step, double Value)> Reports = new();
        public List<(string Key, object? Value)> UserAttrs = new();
        public List<(string Key, double Value)> Constraints = new();
        public string Outcome = "complete";  // complete | fail | prune
        public double ValueOffset;
    }

    // ------------------------------------------------------------------ generation

    public static ParityScenario Generate(int seed, bool multiObjective = false)
    {
        var rng = new Random(seed);
        var scenario = new ParityScenario
        {
            Seed = seed,
            SamplerSeed = rng.Next(0, 100000),
            NTrials = rng.Next(8, 14),
            MultiObjective = multiObjective,
        };

        // ---- search space: 3-6 entries drawn from a pool that covers every distribution shape.
        int nParams = rng.Next(3, 7);
        var pool = new Func<int, SpaceEntry>[]
        {
            i => new SpaceEntry { Kind = "float", Name = $"f{i}", Low = -10, High = 10 },
            i => new SpaceEntry { Kind = "float", Name = $"flog{i}", Low = 1e-5, High = 1e3, Log = true },
            i => new SpaceEntry { Kind = "float", Name = $"fstep{i}", Low = 0, High = 1, Step = 0.1 },
            i => new SpaceEntry { Kind = "float", Name = $"fadj{i}", Low = 0, High = 1, Step = 0.3 },  // high-adjust
            i => new SpaceEntry { Kind = "float", Name = $"fsingle{i}", Low = 2.5, High = 2.5 },
            i => new SpaceEntry { Kind = "float", Name = $"fhuge{i}", Low = -1e300, High = 1e300 },
            i => new SpaceEntry { Kind = "int", Name = $"i{i}", Low = -50, High = 50 },
            i => new SpaceEntry { Kind = "int", Name = $"ilog{i}", Low = 1, High = 1024, Log = true },
            i => new SpaceEntry { Kind = "int", Name = $"istep{i}", Low = 0, High = 100, Step = 7 },  // high-adjust to 98
            i => new SpaceEntry { Kind = "int", Name = $"isingle{i}", Low = 5, High = 5 },
            i => new SpaceEntry
            {
                Kind = "categorical", Name = $"c{i}",
                Choices = new List<object?> { "adam", "sgd", "rmsprop" },
            },
            i => new SpaceEntry
            {
                Kind = "categorical", Name = $"cmixed{i}",
                Choices = new List<object?> { null, true, false, 1L, 2.5, "x" },
            },
            i => new SpaceEntry
            {
                Kind = "categorical", Name = $"cuni{i}",
                Choices = new List<object?> { "日本語", "ünïcode", "emoji😀", "plain" },
            },
        };
        for (int i = 0; i < nParams; i++)
            scenario.Space.Add(pool[rng.Next(pool.Length)](i));

        // ---- enqueued fixed-param trials consumed by the run (0-2) and left WAITING after (0-2).
        int nEnqueue = rng.Next(0, 3);
        for (int i = 0; i < nEnqueue; i++)
            scenario.Enqueues.Add(scenario.PickGridParams(rng));
        int nEnqueueAfter = rng.Next(0, 3);
        for (int i = 0; i < nEnqueueAfter; i++)
            scenario.EnqueuesAfter.Add(scenario.PickGridParams(rng));

        // ---- per-trial operation scripts.
        var attrPool = new Func<Random, object?>[]
        {
            r => "текст-😀-" + r.Next(100),
            r => (long)r.Next(-1000, 1000),
            r => r.NextDouble() * 10 - 5,
            r => r.Next(2) == 0,
            r => null,
            r => new List<object?> { 1L, "a", null, 2.5 },
            r => new OrderedDict<string, object?>
            {
                ["nested"] = new List<object?> { (long)r.Next(10), "n" },
                ["λ"] = 0.5,
            },
        };

        for (int number = 0; number < scenario.NTrials; number++)
        {
            var ops = new TrialOps { ValueOffset = rng.Next(-24, 25) * 0.125 };

            if (!multiObjective)
            {
                int nReports = rng.Next(0, 6);
                long step = rng.Next(0, 3);
                for (int r = 0; r < nReports; r++)
                {
                    double value = rng.Next(12) switch
                    {
                        0 => double.NaN,
                        1 => double.PositiveInfinity,
                        2 => -1e300,
                        _ => Math.Round(rng.NextDouble() * 20 - 10, 6),
                    };
                    ops.Reports.Add((step, value));
                    if (rng.Next(8) == 0)
                        ops.Reports.Add((step, 12345.0));  // Duplicate step: ignored identically.
                    step += rng.Next(1, 4);
                }
            }

            int nAttrs = rng.Next(0, 4);
            for (int a = 0; a < nAttrs; a++)
            {
                string key = rng.Next(5) == 0 ? $"ключ-{a}" : $"attr_{a}";
                ops.UserAttrs.Add((key, attrPool[rng.Next(attrPool.Length)](rng)));
            }

            if (rng.Next(4) == 0)
                ops.Constraints.Add(($"c{rng.Next(2)}", Math.Round(rng.NextDouble() * 4 - 2, 6)));

            ops.Outcome = rng.Next(10) switch
            {
                0 => "fail",
                1 when !multiObjective => "prune",
                _ => "complete",
            };

            scenario.Ops.Add(ops);
        }

        return scenario;
    }

    /// <summary>A valid, on-grid fixed-params dict for enqueue (identical selection both sides).</summary>
    private OrderedDict<string, object?> PickGridParams(Random rng)
    {
        var @params = new OrderedDict<string, object?>();
        foreach (var entry in Space)
        {
            switch (entry.Kind)
            {
                case "float" when entry.Step is not null:
                {
                    int k = rng.Next(0, (int)Math.Floor((entry.High - entry.Low) / entry.Step.Value) + 1);
                    @params[entry.Name] = entry.Low + k * entry.Step.Value;
                    break;
                }
                case "float":
                    @params[entry.Name] = entry.Low == entry.High
                        ? entry.Low
                        : Math.Round(entry.Low + rng.NextDouble() * (Math.Min(entry.High, 1e6) - Math.Max(entry.Low, -1e6)), 6);
                    break;
                case "int" when entry.Step is not null:
                {
                    int k = rng.Next(0, (int)((entry.High - entry.Low) / entry.Step.Value) + 1);
                    @params[entry.Name] = (long)(entry.Low + k * entry.Step.Value);
                    break;
                }
                case "int":
                    @params[entry.Name] = (long)rng.Next((int)entry.Low, (int)entry.High + 1);
                    break;
                default:
                    @params[entry.Name] = Choices(entry)[rng.Next(entry.Choices.Count)].Value;
                    break;
            }
        }
        return @params;
    }

    private static CategoricalChoice[] Choices(SpaceEntry entry)
        => entry.Choices.Select(CategoricalChoice.From).ToArray();

    // ------------------------------------------------------------------ spec JSON (→ Python)

    public string ToSpecJson()
    {
        var space = new List<object?>();
        foreach (var entry in Space)
        {
            var d = new OrderedDict<string, object?> { ["kind"] = entry.Kind, ["name"] = entry.Name };
            if (entry.Kind == "categorical")
            {
                d["choices"] = entry.Choices;
            }
            else
            {
                if (entry.Kind == "int")
                {
                    d["low"] = (long)entry.Low;
                    d["high"] = (long)entry.High;
                    if (entry.Step is not null)
                        d["step"] = (long)entry.Step.Value;
                }
                else
                {
                    d["low"] = entry.Low;
                    d["high"] = entry.High;
                    if (entry.Step is not null)
                        d["step"] = entry.Step.Value;
                }
                d["log"] = entry.Log;
            }
            space.Add(d);
        }

        var trialOps = new List<object?>();
        foreach (var ops in Ops)
        {
            trialOps.Add(new OrderedDict<string, object?>
            {
                ["reports"] = ops.Reports
                    .Select(r => (object?)new List<object?> { r.Step, r.Value }).ToList(),
                ["user_attrs"] = ops.UserAttrs
                    .Select(a => (object?)new List<object?> { a.Key, a.Value }).ToList(),
                ["constraints"] = ops.Constraints
                    .Select(c => (object?)new List<object?> { c.Key, c.Value }).ToList(),
                ["outcome"] = ops.Outcome,
                ["value_offset"] = ops.ValueOffset,
            });
        }

        var payload = new OrderedDict<string, object?>
        {
            ["sampler_seed"] = (long)SamplerSeed,
            ["n_trials"] = (long)NTrials,
            ["multi_objective"] = MultiObjective,
            ["space"] = space,
            ["trial_ops"] = trialOps,
            ["enqueues"] = Enqueues.Cast<object?>().ToList(),
            ["enqueues_after"] = EnqueuesAfter.Cast<object?>().ToList(),
        };
        return PyJson.DumpsCompact(payload);
    }

    /// <summary>
    ///     The generic Python runner: reads <c>spec_json</c> from the scope, builds the study on
    ///     <c>storage</c> (bound in the scope) and runs the identical workload.
    /// </summary>
    public const string PythonRunnerSource =
        "spec = json.loads(spec_json)\n" +
        "def _oss_numeric(v):\n" +
        "    if v is None: return 0.25\n" +
        "    if isinstance(v, bool): return 1.5 if v else 0.5\n" +
        "    if isinstance(v, str): return float(len(v))\n" +
        "    return float(v)\n" +
        "def _oss_suggest_all(t):\n" +
        "    params = []\n" +
        "    for d in spec['space']:\n" +
        "        k = d['name']\n" +
        "        if d['kind'] == 'float':\n" +
        "            params.append(t.suggest_float(k, d['low'], d['high'], step=d.get('step'), log=d['log']))\n" +
        "        elif d['kind'] == 'int':\n" +
        "            params.append(t.suggest_int(k, d['low'], d['high'], step=d.get('step', 1), log=d['log']))\n" +
        "        else:\n" +
        "            params.append(t.suggest_categorical(k, d['choices']))\n" +
        "    return params\n" +
        "def _oss_objective(t):\n" +
        "    params = _oss_suggest_all(t)\n" +
        "    ops = spec['trial_ops'][t.number]\n" +
        "    for r in ops['reports']:\n" +
        "        t.report(r[1], int(r[0]))\n" +
        "    for a in ops['user_attrs']:\n" +
        "        t.set_user_attr(a[0], a[1])\n" +
        "    for c in ops['constraints']:\n" +
        "        t.study._storage.set_trial_system_attr(t._trial_id, 'constraints:' + c[0], c[1])\n" +
        "    if ops['outcome'] == 'fail':\n" +
        "        raise RuntimeError('scenario-fail')\n" +
        "    if ops['outcome'] == 'prune':\n" +
        "        raise optuna.TrialPruned()\n" +
        "    acc = 0.0\n" +
        "    for v in params:\n" +
        "        acc = acc + _oss_numeric(v)\n" +
        "    if spec['multi_objective']:\n" +
        "        return [0.5 * acc + ops['value_offset'], 3.0 - 0.25 * acc + ops['value_offset']]\n" +
        "    return 0.5 * acc + ops['value_offset']\n" +
        "def _oss_run(study):\n" +
        "    for p in spec['enqueues']:\n" +
        "        study.enqueue_trial(dict(p))\n" +
        "    study.optimize(_oss_objective, n_trials=spec['n_trials'], catch=(RuntimeError,))\n" +
        "    for p in spec['enqueues_after']:\n" +
        "        study.enqueue_trial(dict(p))\n";

    // ------------------------------------------------------------------ C# executor

    /// <summary>Run the identical workload on an OptunaSharp study.</summary>
    public void RunCSharp(Study study)
    {
        foreach (var @params in Enqueues)
            study.EnqueueTrial(@params.ShallowClone());

        if (MultiObjective)
        {
            study.Optimize(
                (Func<Trial, IReadOnlyList<double>>)(t => (IReadOnlyList<double>)ObjectiveCore(t)!),
                nTrials: NTrials,
                @catch: new[] { typeof(InvalidOperationException) });
        }
        else
        {
            study.Optimize(
                (Func<Trial, double>)(t => (double)ObjectiveCore(t)!),
                nTrials: NTrials,
                @catch: new[] { typeof(InvalidOperationException) });
        }

        foreach (var @params in EnqueuesAfter)
            study.EnqueueTrial(@params.ShallowClone());
    }

    private object? ObjectiveCore(Trial t)
    {
        var paramValues = new List<object?>();
        foreach (var entry in Space)
        {
            switch (entry.Kind)
            {
                case "float":
                    paramValues.Add(t.SuggestFloat(entry.Name, entry.Low, entry.High, step: entry.Step, log: entry.Log));
                    break;
                case "int":
                    paramValues.Add(t.SuggestInt(
                        entry.Name, (long)entry.Low, (long)entry.High,
                        step: entry.Step is null ? 1L : (long)entry.Step.Value, log: entry.Log));
                    break;
                default:
                    paramValues.Add(t.SuggestCategorical(entry.Name, Choices(entry)).Value);
                    break;
            }
        }

        var ops = Ops[t.Number];
        foreach (var (step, value) in ops.Reports)
            t.Report(value, (int)step);
        foreach (var (key, value) in ops.UserAttrs)
            t.SetUserAttr(key, value);
        foreach (var (key, value) in ops.Constraints)
            t.SetConstraint(key, value);

        if (ops.Outcome == "fail")
            throw new InvalidOperationException("scenario-fail");
        if (ops.Outcome == "prune")
            throw new TrialPruned();

        double acc = 0.0;
        foreach (var v in paramValues)
            acc += Numeric(v);
        if (MultiObjective)
            return new List<double> { 0.5 * acc + ops.ValueOffset, 3.0 - 0.25 * acc + ops.ValueOffset };
        return 0.5 * acc + ops.ValueOffset;
    }

    /// <summary>
    ///     The shared params→number map. Strings count CODE POINTS (Python <c>len(str)</c>), not
    ///     UTF-16 units — "emoji😀" is 6 in Python and must be 6 here.
    /// </summary>
    private static double Numeric(object? v) => v switch
    {
        null => 0.25,
        bool b => b ? 1.5 : 0.5,
        string s => CodePointCount(s),
        long l => l,
        double d => d,
        _ => Convert.ToDouble(v, System.Globalization.CultureInfo.InvariantCulture),
    };

    private static int CodePointCount(string s)
    {
        int count = 0;
        for (int i = 0; i < s.Length; i++)
        {
            count++;
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                i++;
        }
        return count;
    }

    public RandomSampler NewSampler() => new(SamplerSeed);

    public override string ToString()
        => $"scenario(seed={Seed}, samplerSeed={SamplerSeed}, nTrials={NTrials}, " +
           $"mo={MultiObjective}, space=[{string.Join(",", Space.Select(s => s.Name))}], " +
           $"enq={Enqueues.Count}+{EnqueuesAfter.Count})";
}
