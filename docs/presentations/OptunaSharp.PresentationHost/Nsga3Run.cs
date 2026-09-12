using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.PresentationHost;

/// <summary>
/// A real in-memory OptunaSharp study. Each call evaluates one whole generation via Ask/Tell.
/// The next generation's first trial is suggested (not evaluated) at each boundary, which invokes
/// selection naturally. Retaining this pending trial preserves the normal Ask/Tell RNG trajectory.
/// Reading the presentation never invokes selection again.
/// </summary>
public sealed class Nsga3Run
{
    public const int MaximumGeneration = 250;
    private readonly ObservedSampler _sampler;
    private readonly Study _study;
    private readonly RunConfig _config;
    private readonly string _id = Guid.NewGuid().ToString("N");
    private RunSnapshot _snapshot = null!;
    private (Trial Trial, double[] Parameters)? _pending;

    public Nsga3Run(RunConfig config)
    {
        config.Validate();
        _config = config;
        _sampler = new ObservedSampler(config);
        _study = Optuna.CreateStudy(sampler: _sampler,
            directions: [StudyDirection.Minimize, StudyDirection.Minimize, StudyDirection.Minimize]);
        EvaluateGeneration(0);
    }

    public RunSnapshot Snapshot => _snapshot;
    internal int SelectionCalls => _sampler.SelectionCalls;

    // The store holds a per-run lock around this method and snapshot reads.
    public RunSnapshot Step()
    {
        if (_snapshot.Generation >= MaximumGeneration)
            throw new InvalidOperationException($"This run reached {MaximumGeneration} generations. Start a new run.");
        EvaluateGeneration(_snapshot.Generation + 1);
        return _snapshot;
    }

    private void EvaluateGeneration(int generation)
    {
        var parents = _sampler.GetParentPopulation(_study, generation);
        for (var i = 0; i < _config.PopulationSize; i++)
        {
            var (trial, x) = _pending ?? SuggestTrial();
            _pending = null;
            _study.Tell(trial, EvaluateDtlz2(x));
        }

        var offspring = _sampler.GetPopulation(_study, generation);
        // First next-generation Suggest invokes base.SelectParent once, stores its result in the
        // core cache, and uses the original selection order for this first child. Later children
        // use cached order, exactly as normal Ask/Tell. Keep this child pending until Step.
        _pending = SuggestTrial();
        var selected = _sampler.LastSelected;
        _snapshot = SelectionGeometry.Snapshot(_id, generation, _config,
            parents, offspring, selected);
    }

    private (Trial Trial, double[] Parameters) SuggestTrial()
    {
        var trial = _study.Ask();
        var x = new double[12];
        for (var j = 0; j < x.Length; j++)
            x[j] = trial.SuggestFloat($"x{j:00}", 0.0, 1.0);
        return (trial, x);
    }

    /// <summary>DTLZ2 with M=3, n=12, k=n-M+1=10; the Pareto front has ||f||₂=1.</summary>
    public static double[] EvaluateDtlz2(IReadOnlyList<double> x)
    {
        var g = 0.0;
        for (var i = 2; i < x.Count; i++)
            g += (x[i] - 0.5) * (x[i] - 0.5);
        var a = x[0] * Math.PI / 2.0;
        var b = x[1] * Math.PI / 2.0;
        return [(1.0 + g) * Math.Cos(a) * Math.Cos(b),
            (1.0 + g) * Math.Cos(a) * Math.Sin(b), (1.0 + g) * Math.Sin(a)];
    }

    private sealed class ObservedSampler(RunConfig config) : NSGAIIISampler(
        populationSize: config.PopulationSize,
        mutation: new PolynomialMutation(20.0), mutationProb: config.MutationProbability,
        crossover: new SBXCrossover(eta: 20.0), crossoverProb: 0.9,
        seed: config.Seed, dividingParameter: config.Divisions)
    {
        public List<FrozenTrial> LastSelected { get; private set; } = [];
        public int SelectionCalls { get; private set; }

        public override List<FrozenTrial> SelectParent(Study study, int generation)
        {
            var result = base.SelectParent(study, generation);
            LastSelected = result.ToList();
            SelectionCalls++;
            return result;
        }
    }
}
