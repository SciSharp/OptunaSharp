using System;

namespace OptunaSharp.Interop.Pymoo;

/// <summary>
///     Selects which of pymoo's multi-objective evolutionary algorithms
///     (<c>pymoo.algorithms.moo</c>) a <see cref="PymooSampler" /> drives. Each value maps to a real
///     pymoo class, constructed and driven through its native <c>ask()</c>/<c>tell()</c> — see
///     <see cref="PymooSampler" />.
///
///     <para><b>Two of pymoo's MOO algorithms are intentionally absent</b>, because their ask/tell
///     contract is incompatible with a static, externally-evaluated bridge:</para>
///     <list type="bullet">
///         <item><b>MOEA/D.</b> In pymoo 0.6.x <c>MOEAD</c> exposes a <em>steady-state</em> ask/tell
///         generator — its first <c>ask()</c> returns the whole population (initialization) but every
///         subsequent <c>ask()</c> yields a single <c>Individual</c>, advancing a generation only every
///         <c>pop_size</c> steady tells. That cadence is incompatible with the uniform "ask a full
///         population, tell a full population" batch replay every algorithm here shares. pymoo's
///         decomposition regime (Tchebycheff / PBI) is instead the target of a future <em>native</em>
///         Layer-1 MOEA/D port.</item>
///         <item><b>DNSGA-II.</b> "Dynamic" NSGA-II is built for <em>time-varying</em> objectives: on each
///         <c>tell</c> it detects landscape changes and responds by re-sampling part of the population and
///         re-evaluating it <em>through the problem</em> (<c>self.initialization.sampling(self.problem,
///         n)</c> + an evaluator call). This bridge evaluates the objective on the OptunaSharp side (the
///         pymoo <c>Problem</c> is a bare box with no <c>_evaluate</c>), so DNSGA-II's re-evaluation has
///         nothing to call — the wrong tool for static multi-objective optimization.</item>
///     </list>
///     <para>All eleven algorithms here return a full <c>pop_size</c> population on every <c>ask()</c> and
///     never re-evaluate through the problem.</para>
/// </summary>
public enum PymooAlgorithm
{
    /// <summary><c>pymoo.algorithms.moo.nsga2.NSGA2</c> — the canonical fast non-dominated-sort GA.</summary>
    Nsga2,

    /// <summary><c>pymoo.algorithms.moo.nsga3.NSGA3</c> — reference-direction NSGA for many objectives
    /// (needs <see cref="PymooAlgorithmOptions" /> reference directions; the population size is derived
    /// from them, not <see cref="PymooSampler" />'s <c>populationSize</c>).</summary>
    Nsga3,

    /// <summary><c>pymoo.algorithms.moo.rnsga2.RNSGA2</c> — reference-point-guided NSGA-II (needs
    /// <see cref="PymooAlgorithmOptions.ReferencePoints" />).</summary>
    RNsga2,

    /// <summary><c>pymoo.algorithms.moo.rnsga3.RNSGA3</c> — reference-point-guided NSGA-III (needs
    /// <see cref="PymooAlgorithmOptions.ReferencePoints" />; the population size is derived from the
    /// reference points × <see cref="PymooAlgorithmOptions.PopulationPerReferencePoint" />).</summary>
    RNsga3,

    /// <summary><c>pymoo.algorithms.moo.unsga3.UNSGA3</c> — unified NSGA-III (needs reference directions;
    /// population size derived from them).</summary>
    UNsga3,

    /// <summary><c>pymoo.algorithms.moo.age.AGEMOEA</c> — adaptive-geometry MOEA.</summary>
    AgeMoea,

    /// <summary><c>pymoo.algorithms.moo.age2.AGEMOEA2</c> — AGE-MOEA-2 (the live-runtime complement to
    /// OptunaSharp's native <c>AgeMoea2Sampler</c>).</summary>
    AgeMoea2,

    /// <summary><c>pymoo.algorithms.moo.rvea.RVEA</c> — reference-vector-guided EA with APD survival
    /// (needs reference directions; population size derived from them).</summary>
    Rvea,

    /// <summary><c>pymoo.algorithms.moo.sms.SMSEMOA</c> — S-metric-selection (hypervolume) EMOA.</summary>
    SmsEmoa,

    /// <summary><c>pymoo.algorithms.moo.spea2.SPEA2</c> — strength-Pareto EA 2.</summary>
    Spea2,

    /// <summary><c>pymoo.algorithms.moo.ctaea.CTAEA</c> — constrained two-archive EA (needs reference
    /// directions; population size derived from them).</summary>
    Ctaea,
}

/// <summary>The reference-direction generation method (<c>pymoo.util.ref_dirs.get_reference_directions</c>)
/// for the reference-direction algorithms (<see cref="PymooAlgorithm.Nsga3" />,
/// <see cref="PymooAlgorithm.UNsga3" />, <see cref="PymooAlgorithm.Rvea" />,
/// <see cref="PymooAlgorithm.Ctaea" />).</summary>
public enum PymooReferenceDirections
{
    /// <summary><c>"das-dennis"</c> — the structured simplex-lattice directions (the default); the number
    /// of directions is set by <see cref="PymooAlgorithmOptions.NumberOfPartitions" />.</summary>
    DasDennis,

    /// <summary><c>"energy"</c> — Riesz s-energy directions of a chosen count
    /// (<see cref="PymooAlgorithmOptions.NumberOfReferenceDirections" />); reproducible under the
    /// sampler's seed.</summary>
    Energy,
}

/// <summary>
///     Algorithm-specific knobs for <see cref="PymooSampler" />, grouped by the family that uses them.
///     Only the fields relevant to the chosen <see cref="PymooAlgorithm" /> are read; the rest are
///     ignored. Everything has a sensible default, except <see cref="ReferencePoints" />, which is
///     required by <see cref="PymooAlgorithm.RNsga2" /> / <see cref="PymooAlgorithm.RNsga3" /> (there is
///     no meaningful default reference point).
/// </summary>
public sealed class PymooAlgorithmOptions
{
    // --- genetic operators (all algorithms) ---

    /// <summary>Simulated-binary-crossover (SBX) distribution index <c>eta</c>. <c>null</c> ⇒ pymoo's
    /// default (15). Higher values keep offspring closer to the parents.</summary>
    public double? CrossoverEta { get; init; }

    /// <summary>The probability that crossover is applied (SBX <c>prob</c>). <c>null</c> ⇒ pymoo's
    /// default (0.9).</summary>
    public double? CrossoverProbability { get; init; }

    /// <summary>Polynomial-mutation (PM) distribution index <c>eta</c>. <c>null</c> ⇒ pymoo's default
    /// (20).</summary>
    public double? MutationEta { get; init; }

    /// <summary>The per-variable mutation probability (PM <c>prob</c>). <c>null</c> ⇒ pymoo's default
    /// (which is <c>1 / n_var</c> inside PM).</summary>
    public double? MutationProbability { get; init; }

    /// <summary>Whether pymoo eliminates duplicate individuals during mating. Default <c>true</c> (pymoo's
    /// default across these algorithms); set <c>false</c> to allow duplicates.</summary>
    public bool EliminateDuplicates { get; init; } = true;

    // --- RVEA (reference-vector-guided EA) ---

    /// <summary>RVEA's penalty-parameter growth rate <c>alpha</c> (used only for
    /// <see cref="PymooAlgorithm.Rvea" />). pymoo's default is <c>2.0</c>.</summary>
    public double RveaAlpha { get; init; } = 2.0;

    /// <summary>RVEA's reference-vector adaptation frequency <c>adapt_freq</c> (used only for
    /// <see cref="PymooAlgorithm.Rvea" />). pymoo's default is <c>0.1</c>.</summary>
    public double RveaAdaptFrequency { get; init; } = 0.1;

    // --- reference-direction algorithms (NSGA-III, U-NSGA-III, RVEA, C-TAEA) ---

    /// <summary>How the reference directions are generated. Default <see cref="PymooReferenceDirections.DasDennis" />.</summary>
    public PymooReferenceDirections ReferenceDirectionsMethod { get; init; } = PymooReferenceDirections.DasDennis;

    /// <summary>
    ///     Das–Dennis number of partitions per objective. <c>null</c> ⇒ a default of 12 (which yields 13
    ///     directions for 2 objectives, 91 for 3, …). The population size of the reference-direction
    ///     algorithms equals the number of reference directions this produces, <b>not</b>
    ///     <see cref="PymooSampler" />'s <c>populationSize</c>.
    /// </summary>
    public int? NumberOfPartitions { get; init; }

    /// <summary>The number of Riesz-energy reference directions (used only when
    /// <see cref="ReferenceDirectionsMethod" /> is <see cref="PymooReferenceDirections.Energy" />).
    /// <c>null</c> ⇒ <see cref="PymooSampler" />'s <c>populationSize</c>.</summary>
    public int? NumberOfReferenceDirections { get; init; }

    // --- reference-point algorithms (R-NSGA-II, R-NSGA-III) ---

    /// <summary>
    ///     The aspiration / reference points that steer <see cref="PymooAlgorithm.RNsga2" /> /
    ///     <see cref="PymooAlgorithm.RNsga3" />, one row of length <c>n_objectives</c> per point (in the
    ///     study's own objective space — the sampler applies the minimization sign flip). Required for
    ///     those two algorithms.
    /// </summary>
    public double[][]? ReferencePoints { get; init; }

    /// <summary>R-NSGA-II <c>epsilon</c> (the niche radius around each reference point). pymoo's default
    /// is <c>0.001</c>. (R-NSGA-III uses <see cref="Mu" /> instead.)</summary>
    public double Epsilon { get; init; } = 0.001;

    /// <summary>R-NSGA-II per-objective weights for the achievement-scalarizing distance
    /// (its <c>weights</c>). <c>null</c> ⇒ pymoo's default (uniform).</summary>
    public double[]? Weights { get; init; }

    /// <summary>R-NSGA-II: whether the current extreme points are added as extra reference points each
    /// generation (its <c>extreme_points_as_reference_points</c>). Default <c>false</c>.</summary>
    public bool ExtremePointsAsReferencePoints { get; init; }

    /// <summary>R-NSGA-III individuals allocated per reference point (its <c>pop_per_ref_point</c>).
    /// Default 1.</summary>
    public int PopulationPerReferencePoint { get; init; } = 1;

    /// <summary>R-NSGA-III spread control <c>mu</c> (the fraction of reference directions kept near each
    /// reference point). pymoo's default is <c>0.05</c>.</summary>
    public double Mu { get; init; } = 0.05;
}

/// <summary>Maps a <see cref="PymooAlgorithm" /> to its pymoo module + class name and the construction
/// family it belongs to (which knobs it consumes). Central so <see cref="PymooSampler" /> and its tests
/// agree on the mapping.</summary>
internal static class PymooAlgorithmMap
{
    internal static (string Module, string Class) Resolve(PymooAlgorithm algorithm) => algorithm switch
    {
        PymooAlgorithm.Nsga2 => ("pymoo.algorithms.moo.nsga2", "NSGA2"),
        PymooAlgorithm.Nsga3 => ("pymoo.algorithms.moo.nsga3", "NSGA3"),
        PymooAlgorithm.RNsga2 => ("pymoo.algorithms.moo.rnsga2", "RNSGA2"),
        PymooAlgorithm.RNsga3 => ("pymoo.algorithms.moo.rnsga3", "RNSGA3"),
        PymooAlgorithm.UNsga3 => ("pymoo.algorithms.moo.unsga3", "UNSGA3"),
        PymooAlgorithm.AgeMoea => ("pymoo.algorithms.moo.age", "AGEMOEA"),
        PymooAlgorithm.AgeMoea2 => ("pymoo.algorithms.moo.age2", "AGEMOEA2"),
        PymooAlgorithm.Rvea => ("pymoo.algorithms.moo.rvea", "RVEA"),
        PymooAlgorithm.SmsEmoa => ("pymoo.algorithms.moo.sms", "SMSEMOA"),
        PymooAlgorithm.Spea2 => ("pymoo.algorithms.moo.spea2", "SPEA2"),
        PymooAlgorithm.Ctaea => ("pymoo.algorithms.moo.ctaea", "CTAEA"),
        _ => throw new ArgumentOutOfRangeException(nameof(algorithm), algorithm, "Unknown pymoo algorithm."),
    };

    /// <summary>Algorithms whose population size and survival come from a reference-direction set.</summary>
    internal static bool UsesReferenceDirections(PymooAlgorithm a)
        => a is PymooAlgorithm.Nsga3 or PymooAlgorithm.UNsga3 or PymooAlgorithm.Rvea or PymooAlgorithm.Ctaea;

    /// <summary>Algorithms steered by user reference points.</summary>
    internal static bool UsesReferencePoints(PymooAlgorithm a)
        => a is PymooAlgorithm.RNsga2 or PymooAlgorithm.RNsga3;

    /// <summary>Algorithms whose population size is the sampler's <c>populationSize</c> (a <c>pop_size</c>
    /// constructor argument).</summary>
    internal static bool UsesPopulationSize(PymooAlgorithm a)
        => a is PymooAlgorithm.Nsga2 or PymooAlgorithm.RNsga2 or PymooAlgorithm.AgeMoea
            or PymooAlgorithm.AgeMoea2 or PymooAlgorithm.SmsEmoa or PymooAlgorithm.Spea2;
}
