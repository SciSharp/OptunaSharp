using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.EvolutionStrategy;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     The parallel-ask contract of the NES samplers (OpenAI-ES, SNES, xNES). A generation's member is picked by the
///     count of members already handed out, finished or still running, and an index past the population is an extra
///     candidate drawn further along the generation's noise stream. Parallel workers ask those at a generation's tail;
///     they used to receive the member another worker was already evaluating. The members themselves must not change,
///     any population-sized set of evaluated candidates must still form a generation, and sequential runs must keep
///     their trajectories.
/// </summary>
[TestClass]
public class EvolutionStrategyExtraCandidateTests
{
    private static readonly double[] Mean = { 0.5, 0.5, 0.5 };

    /// <summary>Builds one of the three optimizers over a 3-dimensional box, seeded so every call is reproducible.</summary>
    /// <param name="name"><c>openaies-mirrored</c>, <c>openaies</c>, <c>snes</c> or <c>xnes</c>.</param>
    /// <returns>A fresh optimizer at generation 0.</returns>
    private static IEvolutionStrategyOptimizer Make(string name) => name switch
    {
        "openaies-mirrored" => new OpenEsOptimizer(Mean, 0.1, 0.05, 6, mirrored: true,
            0, 1.0, 0.01, 1.0, 1e-3, 0.9, 0.999, 1e-8, seed: 21),
        "openaies" => new OpenEsOptimizer(Mean, 0.1, 0.05, 6, mirrored: false,
            0, 1.0, 0.01, 1.0, 1e-3, 0.9, 0.999, 1e-8, seed: 21),
        "snes" => new SnesOptimizer(Mean, new[] { 0.1, 0.1, 0.1 }, etaMu: 1.0, etaSigma: SnesOptimizer.DefaultEtaSigma(3),
            popsize: SnesOptimizer.DefaultPopulationSize(3), seed: 21),
        _ => new XnesOptimizer(Mean, 0.1, XnesOptimizer.Identity(3), etaMu: 1.0, etaSigma: XnesOptimizer.DefaultEtaSigma(3),
            etaB: XnesOptimizer.DefaultEtaSigma(3), popsize: XnesOptimizer.DefaultPopulationSize(3), seed: 21),
    };

    /// <summary>A candidate's identity: the exact bits of every coordinate.</summary>
    /// <param name="x">The candidate.</param>
    /// <returns>A string equal for two candidates exactly when every coordinate is bit-identical.</returns>
    private static string Key(double[] x) => string.Join(",", x.Select(v => BitConverter.DoubleToInt64Bits(v)));

    /// <summary>
    ///     Asking past the population continues the generation's noise stream: the members are unchanged by it, every
    ///     extra candidate differs from the members and from the other extras, and one index always gives one candidate.
    ///     Before the change, an index past the population threw <see cref="ArgumentOutOfRangeException" /> and the
    ///     sampler clamped it onto the last member.
    /// </summary>
    /// <param name="name">The optimizer under test (see <see cref="Make" />).</param>
    [TestMethod]
    [DataRow("openaies-mirrored")]
    [DataRow("openaies")]
    [DataRow("snes")]
    [DataRow("xnes")]
    public void Ask_PastThePopulation_ExtendsTheStreamWithoutChangingTheMembers(string name)
    {
        var opt = Make(name);
        int n = opt.PopulationSize;

        var membersBefore = Enumerable.Range(0, n).Select(opt.Ask).ToList();
        var extras = Enumerable.Range(n, 3).Select(opt.Ask).ToList();
        var membersAfter = Enumerable.Range(0, n).Select(opt.Ask).ToList();

        for (int i = 0; i < n; i++)
            membersAfter[i].Should().Equal(membersBefore[i], "asking an extra candidate must not change a member");
        membersBefore.Concat(extras).Select(Key).Should().OnlyHaveUniqueItems(
            "an extra candidate repeats neither a member nor another extra");
        opt.Ask(n + 1).Should().Equal(extras[1], "the same index gives the same candidate");
    }

    /// <summary>
    ///     A generation may include extra candidates: when tail workers finish extras before the last members, the first
    ///     population-sized set of finished candidates is told, and the update recovers each one's noise from its vector.
    ///     The generation advances, and the next one asks finite candidates.
    /// </summary>
    /// <param name="name">The optimizer under test (see <see cref="Make" />).</param>
    [TestMethod]
    [DataRow("openaies-mirrored")]
    [DataRow("openaies")]
    [DataRow("snes")]
    [DataRow("xnes")]
    public void Tell_AcceptsAGenerationThatIncludesExtraCandidates(string name)
    {
        var opt = Make(name);
        int n = opt.PopulationSize;

        // Members n-2 and n-1 are still running; extras n and n+1 finished first and complete the generation.
        var evaluated = Enumerable.Range(0, n - 2).Concat(new[] { n, n + 1 })
            .Select(i =>
            {
                double[] x = opt.Ask(i);
                return (X: x, Value: x.Sum(v => (v - 0.3) * (v - 0.3)));
            })
            .ToList();
        opt.Tell(evaluated);

        opt.Generation.Should().Be(1);
        Enumerable.Range(0, n + 2).Select(opt.Ask)
            .Should().OnlyContain(x => x.Length == 3 && x.All(double.IsFinite));
    }

    /// <summary>
    ///     A seeded sequential study keeps exactly the trajectory it had before members were indexed by running count:
    ///     sequentially no other member is ever running and no ask passes the population, so the member indices, and
    ///     with them every suggested value, are unchanged. The digests were recorded from the code before the change.
    /// </summary>
    /// <param name="name"><c>openaies</c>, <c>snes</c> or <c>xnes</c>.</param>
    /// <param name="expectedDigest">SHA-256 of every suggested value's bits, from the code before the change.</param>
    /// <remarks>
    ///     Pinned to Windows x64, where the digests were recorded. The samplers' updates call <c>Math.Exp</c> and
    ///     <c>Math.Log</c>, which .NET takes from the platform's C library, and xNES decomposes through OpenBLAS; both
    ///     can differ in the last bit on another OS or CPU, so elsewhere the case is inconclusive rather than red.
    ///     <para>
    ///         The xNES digest also needs OpenBLAS's Haswell kernels, the ones it was recorded on. CI pins them except on
    ///         Windows on AMD CPUs with AVX-512, where OpenBLAS 0.3.34's forced Haswell kernels fault
    ///         (<c>.github/scripts/openblas_coretype.sh</c>), and OpenBLAS's own kernels there (Zen 4/5, 2026-11-14)
    ///         changed the xNES digest; the OpenAI-ES and SNES digests, which never reach OpenBLAS, held.
    ///     </para>
    /// </remarks>
    [TestMethod]
    [DataRow("openaies", "015121A1E1D3C3FE7999286C8E9672EC4FEF4F89CC34029129D10412DF004B82")]
    [DataRow("snes", "61D84A7D142AB9AFA7C8C1C8DB8AF387CAA2FDF0E900111FF5D74635076884FF")]
    [DataRow("xnes", "B036282FCA3E2AF23CBC72344EEB69C5EFDB48D3451D85503C8A6E8434DBBC30")]
    public void SeededSequentialStudy_KeepsItsTrajectory(string name, string expectedDigest)
    {
        if (!OperatingSystem.IsWindows() || RuntimeInformation.ProcessArchitecture != Architecture.X64)
            Assert.Inconclusive("The recorded digests are for Windows x64; libm and OpenBLAS may differ in the last bit elsewhere.");
        // Only xNES decomposes through OpenBLAS; openblas_get_corename() names the kernels this process runs.
        if (name == "xnes" && NumSharp.Interop.OpenBLAS.OpenBlasEngine.CoreName is not "Haswell")
            Assert.Inconclusive(
                "The xNES digest was recorded on OpenBLAS's Haswell kernels; this process runs " +
                $"'{NumSharp.Interop.OpenBLAS.OpenBlasEngine.CoreName}' (OPENBLAS_CORETYPE=" +
                $"{Environment.GetEnvironmentVariable("OPENBLAS_CORETYPE") ?? "unset"}), whose last bits differ.");

        BaseSampler sampler = name switch
        {
            "openaies" => new OpenAiEsSampler(seed: 5),
            "snes" => new SnesSampler(seed: 5),
            _ => new XnesSampler(seed: 5),
        };
        var study = Optuna.CreateStudy(sampler: sampler);

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", -10, 10);
            double z = t.SuggestFloat("z", -10, 10);
            return (x - 2) * (x - 2) + (y + 1) * (y + 1) + z * z;
        }, nTrials: 60);

        Digest(study.Trials).Should().Be(expectedDigest);
    }

    /// <summary>Hashes the exact bits of every trial's x, y and z, in trial order.</summary>
    /// <param name="trials">The finished trials.</param>
    /// <returns>An upper-case hexadecimal SHA-256 digest.</returns>
    private static string Digest(IEnumerable<FrozenTrial> trials)
    {
        var text = string.Join(";", trials.Select(t => string.Join(",", new[] { "x", "y", "z" }
            .Select(k => BitConverter.DoubleToInt64Bits(Convert.ToDouble(t.Params[k])).ToString("X16")))));
        return Convert.ToHexString(SHA256.HashData(Encoding.ASCII.GetBytes(text)));
    }
}
