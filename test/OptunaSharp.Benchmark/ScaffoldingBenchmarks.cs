using BenchmarkDotNet.Attributes;
using NumSharp;

namespace OptunaSharp.Benchmark;

/// <summary>
///     PLACEHOLDER benchmarks. No Optuna algorithms are ported yet, so these exercise the dependency
///     chain (OptunaSharp → NumSharp) purely to prove the BenchmarkDotNet harness compiles and runs.
///     As samplers, pruners, and the study loop land, add <c>[Benchmark]</c> methods that measure
///     them — and, where useful, compare against live Optuna through the interop project. This is the
///     template, not a meaningful performance number.
/// </summary>
[MemoryDiagnoser]
public class ScaffoldingBenchmarks
{
    [Params(1_000, 100_000)]
    public int N;

    /// <summary>A representative NumSharp op — the numerical foundation OptunaSharp builds on.</summary>
    [Benchmark]
    public int NumSharp_ArangeAdd()
    {
        using NDArray a = np.arange(N);
        using NDArray b = a + 1;
        return (int)b.size;
    }

    /// <summary>Trivially exercises the OptunaSharp assembly itself.</summary>
    [Benchmark]
    public int OptunaSharp_VersionLength() => OptunaSharpVersion.Version.Length;
}
