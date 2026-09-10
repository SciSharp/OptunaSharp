using System;
using System.Collections.Generic;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Janus;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Unit tests for the host-agnostic JANUS infill driver (<see cref="JanusOptimizer" />) against a
///     trivial mock <see cref="IPopulationHost" /> — the Plan 03 seam that lets a DE or MOO host reuse
///     the same window/harvest/generation machinery the CMA-ES host uses. These exercise the driver's
///     two host-agnostic guarantees in isolation from any concrete optimizer: it tells the host only
///     <b>coherent generations</b> (exactly popsize solutions, in ask order, partial/gapped generations
///     withheld), and it fills each generation from the host's native <c>Ask</c> while <b>reserving</b>
///     slot 0 for its Gauss–Newton elite — and, for a covariance-free host, offers no CMIX candidates
///     (the paper's GN-only configuration). The CMA-ES-hosted numeric parity is covered by
///     <c>JanusSamplerTests</c> / <c>JanusMetricParityTests</c>; here the host is a recorder, so the
///     assertions are about the driver's control flow, not any float trajectory.
/// </summary>
[TestClass]
public class JanusInfillHostTests
{
    // ---- coherent-generation batching ------------------------------------------------------------

    [TestMethod]
    public void Driver_TellsOnlyCoherentGenerations()
    {
        const int p = 3;
        var space = Space2D();
        var trans = new SearchSpaceTransform(space, transformStep: true, transform01: true);
        var host = new RecordingHost(p, 2);
        var infill = NewInfill(trans, host);

        // ask the first generation (trials 0,1,2).
        var completed = new List<FrozenTrial>();
        var asks = new Dictionary<int, OrderedDict<string, object?>>();
        for (int t = 0; t < 3; t++)
        {
            infill.SetCompletedTrials(completed);
            infill.UpdateEs();
            asks[t] = infill.Ask(t);
        }

        // complete only 0 and 1 → a partial generation (2 of 3) must NOT be told.
        completed.Add(Completed(asks[0], 0, space));
        completed.Add(Completed(asks[1], 1, space));
        infill.SetCompletedTrials(completed);
        infill.UpdateEs();
        host.ToldGenerationSizes.Should().BeEmpty("a partial generation is withheld until it is full");

        // complete the third → exactly one coherent generation of popsize is told.
        completed.Add(Completed(asks[2], 2, space));
        infill.SetCompletedTrials(completed);
        infill.UpdateEs();
        host.ToldGenerationSizes.Should().Equal(p);

        // the told generation is the three asked trials, in ask order (X == transform of their params).
        for (int k = 0; k < p; k++)
            host.ToldGenerations[0][k].Should().Equal(trans.Transform(asks[k]).ToArray<double>());

        // ask the second generation (3,4,5) — pending was reset by the tell above.
        for (int t = 3; t < 6; t++)
        {
            infill.SetCompletedTrials(completed);
            infill.UpdateEs();
            asks[t] = infill.Ask(t);
        }

        // a gap: complete 4 and 5 but NOT 3 → ask order must block the later trials.
        completed.Add(Completed(asks[4], 4, space));
        completed.Add(Completed(asks[5], 5, space));
        infill.SetCompletedTrials(completed);
        infill.UpdateEs();
        host.ToldGenerationSizes.Should().Equal(new[] { p }, "the gap at trial 3 blocks batching 4 and 5");

        // fill the gap → the second coherent generation is told.
        completed.Add(Completed(asks[3], 3, space));
        infill.SetCompletedTrials(completed);
        infill.UpdateEs();
        host.ToldGenerationSizes.Should().Equal(p, p);
        host.ToldGenerations[1].Should().HaveCount(p);
    }

    // ---- native slot fill + slot-0 reservation + CMIX gating -------------------------------------

    [TestMethod]
    public void Driver_FillsNativeSlots_ReservesSlotZeroForElite_AndGatesCmixWithoutCovariance()
    {
        const int p = 3, n = 40;
        var keys = new[] { "x0", "x1" };
        var space = Space2D();
        var trans = new SearchSpaceTransform(space, transformStep: true, transform01: true);
        var host = new RecordingHost(p, 2);
        // fuseStart/gnWindow are pinned small so the Gauss–Newton assembly fires early and deterministically.
        var infill = NewInfill(trans, host);

        var completed = new List<FrozenTrial>();
        var asked = new List<double[]>();   // untransformed param values, indexed by trial number
        for (int t = 0; t < n; t++)
        {
            infill.SetCompletedTrials(completed);
            infill.UpdateEs();
            var p0 = infill.Ask(t);
            double[] xv = Vals(p0, keys);
            asked.Add(xv);
            completed.Add(Completed(p0, t, space, value: xv[0] * xv[0] + xv[1] * xv[1]));
        }

        // Every generation drew exactly popsize native candidates from the host (the driver fills its
        // native slots from Ask, then overwrites slot 0 with the elite — the native draw is still made).
        host.AskCount.Should().BeGreaterThan(0);
        (host.AskCount % p).Should().Be(0, "each generation fills exactly popsize native slots from the host");

        int generations = host.AskCount / p;
        int eliteGenerations = 0;
        for (int g = 0; g < generations; g++)
        {
            for (int k = 0; k < p; k++)
            {
                int t = g * p + k;
                if (t >= n) break;
                double[] native = Vals(trans.Untransform(np.array(host.AskReturns[g * p + k])), keys);
                double[] got = asked[t];
                double maxDiff = Math.Max(Math.Abs(got[0] - native[0]), Math.Abs(got[1] - native[1]));
                if (k == 0)
                {
                    if (maxDiff > 1e-9) eliteGenerations++;   // slot 0 was replaced by the injected elite
                }
                else
                {
                    maxDiff.Should().BeLessThan(1e-9,
                        "non-zero slots are the native host draws — a covariance-free host offers no CMIX candidates");
                }
            }
        }

        eliteGenerations.Should().BeGreaterThan(0,
            "the Gauss–Newton elite is injected into slot 0 once the window supports a fit");
    }

    // ---- helpers ---------------------------------------------------------------------------------

    private static OrderedDict<string, BaseDistribution> Space2D() => new()
    {
        ["x0"] = new FloatDistribution(-5, 5),
        ["x1"] = new FloatDistribution(-5, 5),
    };

    private static JanusOptimizer NewInfill(SearchSpaceTransform trans, IPopulationHost host) =>
        new(trans, host, new List<FrozenTrial>(), StudyDirection.Minimize, cmixSeed: 7,
            fuseEvery: 4, fuseStart: 12, kappa: 1.0,
            gnWindow: 40, gnLambda: 1.0, gnRidge: 1.0,
            cmixW: null, cmixLambda: 1.0, cmixWBase: 0.5, cmixWMax: 0.20,
            gnPseudoMode: "linear", cmixPseudoMode: "linear", pseudoTarget: "raw",
            pcaK: "auto", cmixMode: "portfolio", cmixCandidateFrac: 0.4, cmixCandidateCenter: "mean");

    private static FrozenTrial Completed(OrderedDict<string, object?> paramsDict, int number,
        OrderedDict<string, BaseDistribution> space, double value = 0.0)
    {
        var ft = FrozenTrial.CreateTrial(value: value, @params: paramsDict, distributions: space);
        ft.SetNumber(number);
        return ft;
    }

    private static double[] Vals(OrderedDict<string, object?> p, string[] keys)
    {
        var r = new double[keys.Length];
        for (int i = 0; i < keys.Length; i++) r[i] = Convert.ToDouble(p[keys[i]]);
        return r;
    }

    /// <summary>
    ///     A recording <see cref="IPopulationHost" /> that has no proposal covariance (so the driver
    ///     runs GN-only, gating CMIX off), returns deterministic strictly-interior native draws in the
    ///     [0,1] transformed box, and captures every <c>Ask</c> return and every <c>Tell</c> generation.
    /// </summary>
    private sealed class RecordingHost : IPopulationHost
    {
        private readonly int _dim;
        private uint _asks;

        public RecordingHost(int popsize, int dim)
        {
            PopulationSize = popsize;
            _dim = dim;
            Mean = new double[dim];
            for (int i = 0; i < dim; i++) Mean[i] = 0.5;   // center of the [0,1] transformed box
            Sigma = 0.2;
        }

        public int PopulationSize { get; }
        public double[] Mean { get; }
        public double Sigma { get; }

        public int AskCount => (int)_asks;
        public List<double[]> AskReturns { get; } = new();
        public List<int> ToldGenerationSizes { get; } = new();
        public List<double[][]> ToldGenerations { get; } = new();

        public double[] Ask()
        {
            // deterministic, distinguishable, strictly-interior native draw so the test can map a slot
            // back to its native origin; the ask index is folded in so no two draws collide.
            uint k = _asks++;
            var x = new double[_dim];
            for (int i = 0; i < _dim; i++)
            {
                uint h = Mix(k * 2654435761u + (uint)(i * 40503) + 1u);
                x[i] = 0.1 + 0.8 * (h / 4294967295.0);   // in (0.1, 0.9) ⊂ (0,1)
            }
            AskReturns.Add((double[])x.Clone());
            return x;
        }

        public void Tell(IReadOnlyList<(double[] X, double Value)> solutions)
        {
            ToldGenerationSizes.Add(solutions.Count);
            var gen = new double[solutions.Count][];
            for (int i = 0; i < solutions.Count; i++) gen[i] = (double[])solutions[i].X.Clone();
            ToldGenerations.Add(gen);
        }

        public bool HasProposalCovariance => false;   // GN-only host: CMIX must be gated off
        public double ProposalCovarianceTrace => throw new NotSupportedException("no proposal covariance");
        public double WhitenedStepNorm(double[] diff) => 0.0;   // DE-style degradation: inject unclipped
        public double[,]? GetProposalCovariance() => null;      // no covariance ⇒ reshape is a no-op
        public bool TrySetProposalCovariance(double[,] c) => false;

        private static uint Mix(uint x)   // xorshift so successive indices spread across [0,1]
        {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            return x == 0 ? 1u : x;
        }
    }
}
