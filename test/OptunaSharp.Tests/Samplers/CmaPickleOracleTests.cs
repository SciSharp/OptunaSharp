using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.CmaEs;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     The CMA-ES optimizer state as Optuna stores it (review item P3-2): <c>pickle.dumps(optimizer).hex()</c> of a
///     cmaes <c>CMA</c>, <c>SepCMA</c> or <c>CMAwM</c>. OptunaSharp stored its own JSON there before, so every Python
///     worker failed in <c>bytes.fromhex</c> after a C# worker, and every C# worker failed after a Python one. These
///     tests replay the committed oracle (<c>oracle/cma_pickle_oracle.json</c>: the states Optuna 5.0.0 stored with
///     cmaes 0.13.1 and numpy 2.4.2, from <c>oracle/gen_cma_pickle_oracle.py</c>) with no Python at test time.
/// </summary>
/// <remarks>
///     Each oracle variant keeps a study's first stored state (from a fresh optimizer) and the next ones (from
///     optimizers restored and then told). Python pickles the two kinds differently: the covariance's position and
///     which arrays share a dtype object differ, so both are checked byte for byte. The live counterpart, with real
///     Python workers on one study, is <c>CmaEsMixedFleetInteropTests</c> in the interop suite.
/// </remarks>
[TestClass]
public class CmaPickleOracleTests
{
    private static readonly Lazy<JsonElement> OracleDoc = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "oracle", "cma_pickle_oracle.json");
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    /// <summary>
    ///     The codec alone: every stored state Python wrote reads into the value model and writes back to the same
    ///     bytes, which pins the protocol-4 opcode choice, the memo (dtype objects, interned strings, the
    ///     <c>b''</c>/<c>b'b'</c> singletons) and the framing of small pickles.
    /// </summary>
    [TestMethod]
    public void PythonPickle_RewritesEveryStoredState_ByteForByte()
    {
        foreach (var (variant, index, _, raw) in StoredStates())
            PythonPickle.Dumps(PythonPickle.Loads(raw)).Should().Equal(raw, $"{variant} state #{index} must survive a read and write");
    }

    /// <summary>
    ///     A 196 KB pickle of an untold 125-dimensional cmaes CMA: its 63,000-byte covariance makes a frame pass the 64
    ///     KiB target at a later object boundary, and its 125,000-byte <c>_ESigma</c> is written outside any frame.
    ///     Both are reproduced as CPython's pickler commits them.
    /// </summary>
    [TestMethod]
    public void PythonPickle_FramesALargePickleAsCPythonDoes()
    {
        byte[] raw = CodecCase("cma_125d_untold");
        raw.Length.Should().BeGreaterThan(2 * 64 * 1024, "the case must cross the frame target more than once");

        PythonPickle.Dumps(PythonPickle.Loads(raw)).Should().Equal(raw);
    }

    /// <summary>
    ///     The optimizers' state mappings: every stored state restores into the OptunaSharp optimizer of its class and
    ///     pickles back to Python's bytes, with the first state of a study in the fresh layout and the others in the
    ///     restored one. A one-dimensional "separable" study stores a CMA, and restores as one.
    /// </summary>
    [TestMethod]
    public void CmaPickle_RestoresAndRepicklesEveryStoredState_ByteForByte()
    {
        foreach (var (variant, index, _, raw) in StoredStates())
        {
            ICmaOptimizer optimizer = CmaPickle.Loads(raw);

            optimizer.Should().BeOfType(variant switch
            {
                "sepcma" => typeof(SepCma),
                "cmawm" or "cmawm_continuous" => typeof(CmaWm),
                _ => typeof(Cma),
            }, $"{variant} stores that cmaes class");
            CmaPickle.Dumps(optimizer, restored: index > 0).Should().Equal(raw, $"{variant} state #{index}");
        }
    }

    /// <summary>
    ///     CPython 3.14 pickles with protocol 5 by default, where numpy writes a contiguous array as
    ///     <c>_frombuffer(bytearray, dtype, shape, order)</c>: a state a Python 3.14 worker stores restores to exactly
    ///     the optimizer the protocol-4 state holds (OptunaSharp writes protocol 4, which every Python reads).
    /// </summary>
    [TestMethod]
    public void CmaPickle_RestoresProtocol5States_AsTheProtocol4Ones()
    {
        var cases = OracleDoc.Value.GetProperty("protocol5_cases").EnumerateArray().ToList();
        cases.Should().NotBeEmpty();
        foreach (var c in cases)
        {
            string variant = c.GetProperty("variant").GetString()!;
            int index = c.GetProperty("index").GetInt32();
            byte[] p5 = Convert.FromHexString(c.GetProperty("hex").GetString()!);
            byte[] p4 = StoredStates().Single(s => s.Variant == variant && s.Index == index).Pickle;
            p5[1].Should().Be(5, "the case is a protocol-5 pickle");

            CmaPickle.Dumps(CmaPickle.Loads(p5), restored: index > 0).Should().Equal(p4, $"{variant} state #{index}");
        }
    }

    /// <summary>
    ///     numpy 1.x pickles its array and scalar helpers from <c>numpy.core.multiarray</c>, where numpy 2 writes
    ///     <c>numpy._core.multiarray</c> (and reads both). A state naming the old module restores the same.
    /// </summary>
    [TestMethod]
    public void CmaPickle_ReadsTheModuleNamesOfNumpy1()
    {
        byte[] p4 = StoredStates().First(s => s.Variant == "cma" && s.Index == 1).Pickle;
        byte[] numpy2 = [0x8c, 22, .. "numpy._core.multiarray"u8];
        byte[] numpy1 = [0x8c, 21, .. "numpy.core.multiarray"u8];
        int at = p4.AsSpan().IndexOf(numpy2);
        at.Should().BePositive("the state names numpy 2's module");
        // The FRAME length before it no longer matches; the reader, unlike Python's, does not depend on it.
        byte[] old = [.. p4[..at], .. numpy1, .. p4[(at + numpy2.Length)..]];

        CmaPickle.Dumps(CmaPickle.Loads(old), restored: true).Should().Equal(p4);
    }

    /// <summary>
    ///     End to end: an OptunaSharp study run as the oracle's Python study was stores the same optimizer states at the
    ///     same trials, byte for byte, once the <c>_funhist_values</c> entries cmaes has not written yet are taken from
    ///     Python's state (cmaes allocates them with <c>np.empty</c>, so Python's own bytes there are leftover memory).
    ///     This covers every computed value, Python's scalar types (a user <c>sigma0</c> or a warm start makes
    ///     <c>_tolx</c> a float; learning-rate adaptation makes the rates numpy scalars) and numpy's dtype sharing.
    /// </summary>
    /// <param name="variant">The oracle variant, run with the matching sampler and objective.</param>
    [TestMethod]
    [DataRow("cma")]
    [DataRow("cma_lr_adapt")]
    [DataRow("sepcma")]
    [DataRow("sepcma_1d")]
    [DataRow("cmawm_continuous")]
    [DataRow("cma_user_sigma0")]
    [DataRow("cma_warm_start")]
    [DataRow("cma_infinite")]
    [DataRow("cma_30d")]
    public void CSharpStudy_StoresTheStatesOptunaStores(string variant)
    {
        var (pyStates, csStates) = RunVariant(variant);

        csStates.Select(s => s.Trial).Take(pyStates.Count).Should().Equal(pyStates.Select(s => s.Trial),
            "the optimizer must reach each generation at the same trial");
        for (int i = 0; i < pyStates.Count; i++)
        {
            var cs = (PyInstance)PythonPickle.Loads(csStates[i].Pickle)!;
            var py = (PyInstance)PythonPickle.Loads(pyStates[i].Pickle)!;
            CopyUninitializedFunhist(cs, py);

            Flatten(cs).Should().Equal(Flatten(py), $"{variant} state at trial {pyStates[i].Trial}");
            PythonPickle.Dumps(cs).Should().Equal(pyStates[i].Pickle, $"{variant} state at trial {pyStates[i].Trial}");
        }
    }

    /// <summary>
    ///     With discrete dimensions CMAwM's margin correction rides SciPy's <c>norm.cdf</c>/<c>chi2.ppf</c>, which
    ///     OptunaSharp reproduces to a few ULP (a documented divergence), so the margin study's states agree in layout,
    ///     types and dtype sharing exactly and in value to a relative 1e-12 (measured: at most 1e-14 after four
    ///     generations).
    /// </summary>
    [TestMethod]
    public void CSharpStudy_WithMargin_StoresOptunasLayout_AndValuesToAFewUlp()
    {
        var (pyStates, csStates) = RunVariant("cmawm");

        csStates.Select(s => s.Trial).Take(pyStates.Count).Should().Equal(pyStates.Select(s => s.Trial));
        for (int i = 0; i < pyStates.Count; i++)
        {
            var cs = (PyInstance)PythonPickle.Loads(csStates[i].Pickle)!;
            var py = (PyInstance)PythonPickle.Loads(pyStates[i].Pickle)!;
            CopyUninitializedFunhist(cs, py);
            CopyCloseValues(cs, py, relativeTolerance: 1e-12, path: "");

            PythonPickle.Dumps(cs).Should().Equal(pyStates[i].Pickle, $"cmawm state at trial {pyStates[i].Trial}");
        }
    }

    /// <summary>
    ///     A study whose newest optimizer state is the JSON OptunaSharp stored before 2026-10-29 keeps optimizing: the
    ///     sampler restores the JSON, continues exactly as from the equivalent pickle, and stores its next state as a
    ///     pickle, so Python workers can join from then on.
    /// </summary>
    [TestMethod]
    public void LegacyJsonState_IsStillRestored_AndTheNextStateIsAPickle()
    {
        const int warmup = 20, more = 15;
        var reference = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 3));
        reference.Optimize(Plane2, nTrials: warmup + more);

        // Copy the first `warmup` trials; the newest stored state becomes the legacy JSON of the same optimizer.
        var copied = reference.Trials.Take(warmup).Select(t => t.DeepCopy()).ToList();
        var newest = copied.Last(t => t.SystemAttrs.ContainsKey("cma:optimizer:0"));
        string hex = ConcatOptimizerChunks(newest, "cma:");
        var cma = (Cma)CmaPickle.FromHex(hex);
        string legacy = "cma\n" + OptimizerStateJson.Serialize(cma.ToState());
        foreach (string key in newest.SystemAttrs.Keys.Where(k => k.StartsWith("cma:optimizer:", StringComparison.Ordinal)).ToList())
            newest.SystemAttrsDict.Remove(key);
        newest.SystemAttrsDict["cma:optimizer:0"] = legacy;

        var resumed = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 3));
        resumed.AddTrials(copied);
        var fromPickle = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 3));
        fromPickle.AddTrials(reference.Trials.Take(warmup).Select(t => t.DeepCopy()));
        resumed.Optimize(Plane2, nTrials: more);
        fromPickle.Optimize(Plane2, nTrials: more);

        resumed.Trials.Skip(warmup).Select(t => t.Params["x"]).Should().Equal(fromPickle.Trials.Skip(warmup).Select(t => t.Params["x"]),
            "the JSON holds the same optimizer as the pickle");
        var next = resumed.Trials.Skip(warmup).First(t => t.SystemAttrs.ContainsKey("cma:optimizer:0"));
        string nextState = ConcatOptimizerChunks(next, "cma:");
        nextState.Should().MatchRegex("^[0-9a-f]+$", "the next state is Optuna's hex pickle");
        CmaPickle.FromHex(nextState).Should().BeOfType<Cma>();
    }

    /// <summary>
    ///     Python's <c>bytes.fromhex</c> semantics: either case, whitespace between bytes, and Python 3.12's message,
    ///     including the one a Python worker printed for OptunaSharp's legacy JSON (position 1 of <c>cma\n…</c>).
    /// </summary>
    /// <param name="text">The input.</param>
    /// <param name="badPosition">The position Python reports, or -1 for valid hex.</param>
    [TestMethod]
    [DataRow("8004", -1)]
    [DataRow(" 80 04 \n", -1)]
    [DataRow("80AB", -1)]
    [DataRow("cma\n{\"NDim\":2}", 1)]
    [DataRow("abc", 3)]
    [DataRow("zz", 0)]
    [DataRow("0g", 1)]
    [DataRow("8 0", 1)]
    [DataRow("zé", 1)]
    public void FromHexString_MatchesPythonsBytesFromhex(string text, int badPosition)
    {
        Func<byte[]> read = () => CmaPickle.FromHexString(text);

        if (badPosition < 0)
            read().Should().Equal(Convert.FromHexString(new string(text.Where(char.IsAsciiHexDigit).ToArray())));
        else
            read.Should().Throw<ValueError>().WithMessage($"non-hexadecimal number found in fromhex() arg at position {badPosition}");
    }

    /// <summary>
    ///     The reader never imports anything: a pickle naming a global outside numpy and the cmaes optimizer classes is
    ///     refused before it could be called, as are malformed and truncated pickles.
    /// </summary>
    [TestMethod]
    public void Loads_RefusesForeignGlobals_AndBrokenPickles()
    {
        // pickle.dumps(os.system) at protocol 4, framed, then a REDUCE that would call it.
        byte[] hostile = { 0x80, 0x04, 0x95, 0x11, 0, 0, 0, 0, 0, 0, 0, 0x8c, 0x02, (byte)'o', (byte)'s', 0x94,
            0x8c, 0x06, (byte)'s', (byte)'y', (byte)'s', (byte)'t', (byte)'e', (byte)'m', 0x94, 0x93, 0x94, 0x2e };
        byte[] stored = StoredStates().First().Pickle;

        FluentActions.Invoking(() => CmaPickle.Loads(hostile)).Should().Throw<ValueError>().WithMessage("*os.system is not allowed*");
        FluentActions.Invoking(() => CmaPickle.Loads(stored[..^40])).Should().Throw<ValueError>().WithMessage("*ended before STOP*");
        FluentActions.Invoking(() => CmaPickle.Loads(PythonPickle.Dumps(1.5))).Should().Throw<ValueError>().WithMessage("*not a cmaes optimizer*");
        FluentActions.Invoking(() => CmaPickle.Loads(new byte[] { 0x80, 0x04, 0xff })).Should().Throw<ValueError>().WithMessage("*unsupported opcode 0xff*");

        // A one-element array whose shape claims (2^31 - 1)^3 elements: the count would overflow 64 bits, so the
        // reader must refuse it as malformed, neither wrapping into a size that matches nor allocating.
        byte[] small = PythonPickle.Dumps(new NpArray(new NpDtype("f8"), new[] { 1 }, new[] { 0.0 }));
        byte[] versionAndShape = { 0x4b, 0x01, 0x4b, 0x01, 0x85, 0x94 };  // BININT1 1, then (1,) as BININT1 1 TUPLE1
        byte[] hugeShape = [0x4b, 0x01, 0x4a, 0xff, 0xff, 0xff, 0x7f, 0x4a, 0xff, 0xff, 0xff, 0x7f, 0x4a, 0xff, 0xff, 0xff, 0x7f, 0x87, 0x94];
        int at = small.AsSpan().IndexOf(versionAndShape);
        at.Should().BePositive("the array state starts with the version and the shape");
        byte[] hostileShape = [.. small[..at], .. hugeShape, .. small[(at + versionAndShape.Length)..]];

        PythonPickle.Loads(small).Should().BeOfType<NpArray>();
        FluentActions.Invoking(() => PythonPickle.Loads(hostileShape)).Should().Throw<ValueError>().WithMessage("*does not match its shape*");
    }

    /// <summary>
    ///     A with-margin state from cmaes 0.13.0 or older (materialized <c>z_space</c>/<c>z_lim</c>, no
    ///     <c>_discrete_space_*</c>) is refused with an error that names the version to run, instead of a later failure.
    /// </summary>
    [TestMethod]
    public void Loads_RefusesTheCmawmStateOfOlderCmaes()
    {
        var state = (PyInstance)PythonPickle.Loads(StoredStates().First(s => s.Variant == "cmawm").Pickle)!;
        state.State.RemoveAll(kv => kv.Key.StartsWith("_discrete_space_", StringComparison.Ordinal));
        byte[] old = PythonPickle.Dumps(state);

        FluentActions.Invoking(() => CmaPickle.Loads(old)).Should().Throw<ValueError>().WithMessage("*cmaes 0.13.0 or older*0.13.1*");
    }

    // ---- the oracle -----------------------------------------------------------------------------

    /// <summary>Every stored state of every oracle variant, in study order.</summary>
    /// <returns>The variant name, the state's index in its study, the trial that stored it and its pickle bytes.</returns>
    private static IEnumerable<(string Variant, int Index, int Trial, byte[] Pickle)> StoredStates()
    {
        foreach (var variant in OracleDoc.Value.GetProperty("variants").EnumerateObject())
        {
            int index = 0;
            foreach (var p in variant.Value.GetProperty("pickles").EnumerateArray())
                yield return (variant.Name, index++, p.GetProperty("trial").GetInt32(), Convert.FromHexString(p.GetProperty("hex").GetString()!));
        }
    }

    /// <summary>A codec case: a gzip-compressed, base64-encoded pickle.</summary>
    /// <param name="name">The case name in <c>codec_cases</c>.</param>
    /// <returns>The pickle's bytes.</returns>
    private static byte[] CodecCase(string name)
    {
        string b64 = OracleDoc.Value.GetProperty("codec_cases").GetProperty(name).GetString()!;
        using var gz = new GZipStream(new MemoryStream(Convert.FromBase64String(b64)), CompressionMode.Decompress);
        using var raw = new MemoryStream();
        gz.CopyTo(raw);
        return raw.ToArray();
    }

    /// <summary>
    ///     Runs an oracle variant in OptunaSharp, with the sampler and objective <c>gen_cma_pickle_oracle.py</c> gives
    ///     it, and collects the stored states of both runs.
    /// </summary>
    /// <param name="variant">The oracle variant name.</param>
    /// <returns>Python's kept states and every state the C# study stored, each with its trial number.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The variant is not one the generator defines.</exception>
    private static (List<(int Trial, byte[] Pickle)> Py, List<(int Trial, byte[] Pickle)> Cs) RunVariant(string variant)
    {
        var (sampler, objective, trials) = variant switch
        {
            "cma" => (new CmaEsSampler(seed: 7), (Func<Trial, double>)Mixed, 40),
            "cma_lr_adapt" => (new CmaEsSampler(seed: 7, lrAdapt: true), Mixed, 40),
            "sepcma" => (new CmaEsSampler(seed: 7, useSeparableCma: true), Plane2, 40),
            "sepcma_1d" => (new CmaEsSampler(seed: 7, useSeparableCma: true), OneD, 30),
            "cmawm" => (new CmaEsSampler(seed: 7, withMargin: true), Margin, 40),
            "cmawm_continuous" => (new CmaEsSampler(seed: 7, withMargin: true), Plane2, 30),
            "cma_user_sigma0" => (new CmaEsSampler(seed: 7, popsize: 9, sigma0: 0.7,
                x0: new Dictionary<string, object?> { ["x"] = 1.0, ["y"] = -1.0 }), Plane2, 40),
            "cma_warm_start" => (new CmaEsSampler(seed: 7, sourceTrials: WarmStartSources()), Plane2, 30),
            "cma_infinite" => (new CmaEsSampler(seed: 7), SometimesInfinite, 40),
            "cma_30d" => (new CmaEsSampler(seed: 7), Sphere30, 45),
            _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Not an oracle variant."),
        };
        var oracle = OracleDoc.Value.GetProperty("variants").GetProperty(variant);
        string prefix = oracle.GetProperty("prefix").GetString()!;
        var py = oracle.GetProperty("pickles").EnumerateArray()
            .Select(p => (p.GetProperty("trial").GetInt32(), Convert.FromHexString(p.GetProperty("hex").GetString()!)))
            .ToList();

        var study = Optuna.CreateStudy(sampler: sampler);
        study.Optimize(objective, nTrials: trials);
        var cs = study.Trials
            .Where(t => t.SystemAttrs.ContainsKey(prefix + "optimizer:0"))
            .Select(t => (t.Number, CmaPickle.FromHexString(ConcatOptimizerChunks(t, prefix))))
            .ToList();
        return (py, cs);
    }

    /// <summary>Reassembles a trial's stored optimizer state from its numbered chunks.</summary>
    /// <param name="trial">A trial that stored a state.</param>
    /// <param name="prefix">The sampler's attribute prefix: <c>cma:</c>, <c>sepcma:</c> or <c>cmawm:</c>.</param>
    /// <returns>The concatenated state text.</returns>
    private static string ConcatOptimizerChunks(FrozenTrial trial, string prefix)
    {
        int count = trial.SystemAttrs.Keys.Count(k => k.StartsWith(prefix + "optimizer:", StringComparison.Ordinal));
        return string.Concat(Enumerable.Range(0, count).Select(i => (string)trial.SystemAttrs[$"{prefix}optimizer:{i}"]!));
    }

    // The generator's objectives, operation for operation: Python's `**` on floats is the C library's pow, as is
    // Math.Pow, and an int `(n - 1) ** 2` is exact.

    /// <summary><c>mixed</c>: two linear floats and a log-scaled one.</summary>
    /// <param name="t">The running trial.</param>
    /// <returns>The objective value.</returns>
    private static double Mixed(Trial t)
    {
        double x = t.SuggestFloat("x", -5, 5), y = t.SuggestFloat("y", -5, 5), z = t.SuggestFloat("z", 1e-3, 10, log: true);
        return Math.Pow(x - 1, 2) + Math.Pow(y + 0.5, 2) + Math.Pow(Math.Log(z), 2);
    }

    /// <summary><c>plane2</c>: two linear floats.</summary>
    /// <param name="t">The running trial.</param>
    /// <returns>The objective value.</returns>
    private static double Plane2(Trial t)
    {
        double x = t.SuggestFloat("x", -5, 5), y = t.SuggestFloat("y", -5, 5);
        return Math.Pow(x - 1, 2) + Math.Pow(y + 0.5, 2);
    }

    /// <summary><c>one_d</c>: one float.</summary>
    /// <param name="t">The running trial.</param>
    /// <returns>The objective value.</returns>
    private static double OneD(Trial t) => Math.Pow(t.SuggestFloat("x", -5, 5) - 1, 2);

    /// <summary><c>margin</c>: a float, an integer and a stepped float.</summary>
    /// <param name="t">The running trial.</param>
    /// <returns>The objective value.</returns>
    private static double Margin(Trial t)
    {
        double x = t.SuggestFloat("x", -5, 5);
        long n = t.SuggestInt("n", -3, 3);
        double s = t.SuggestFloat("s", 0, 2, step: 0.5);
        return Math.Pow(x - 1, 2) + (n - 1) * (n - 1) + Math.Pow(s - 1.5, 2);
    }

    /// <summary><c>sphere(30)</c>, summed as CPython 3.12's <c>sum()</c> sums floats: Neumaier's compensated sum.</summary>
    /// <param name="t">The running trial.</param>
    /// <returns>The objective value.</returns>
    private static double Sphere30(Trial t)
    {
        double sum = 0, compensation = 0;
        for (int i = 0; i < 30; i++)
        {
            double x = Math.Pow(t.SuggestFloat($"x{i}", -5, 5) - 0.5, 2);
            double total = sum + x;
            compensation += Math.Abs(sum) >= Math.Abs(x) ? (sum - total) + x : (x - total) + sum;
            sum = total;
        }
        return compensation != 0 && double.IsFinite(compensation) ? sum + compensation : sum;
    }

    /// <summary><c>sometimes_infinite</c>: infinite every fifth trial, so the funhist stores infinities.</summary>
    /// <param name="t">The running trial.</param>
    /// <returns>The objective value.</returns>
    private static double SometimesInfinite(Trial t)
    {
        double x = t.SuggestFloat("x", -5, 5), y = t.SuggestFloat("y", -5, 5);
        return t.Number % 5 == 3 ? double.PositiveInfinity : Math.Pow(x - 1, 2) + Math.Pow(y, 2);
    }

    /// <summary>The generator's warm-start sources: twelve <c>plane2</c> trials drawn from <c>RandomState(3)</c>.</summary>
    /// <returns>The source trials.</returns>
    private static List<FrozenTrial> WarmStartSources()
    {
        var rng = np.random.RandomState(3);
        var trials = new List<FrozenTrial>();
        for (int i = 0; i < 12; i++)
        {
            double x = rng.uniform(-5, 5).GetDouble(), y = rng.uniform(-5, 5).GetDouble();
            trials.Add(FrozenTrial.CreateTrial(
                @params: new OrderedDict<string, object?> { ["x"] = x, ["y"] = y },
                distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(-5, 5), ["y"] = new FloatDistribution(-5, 5) },
                value: Math.Pow(x - 1, 2) + Math.Pow(y + 0.5, 2)));
        }
        return trials;
    }

    // ---- comparison helpers ---------------------------------------------------------------------

    /// <summary>
    ///     Copies Python's values into the <c>_funhist_values</c> entries cmaes has not written yet. Generation
    ///     <c>g</c> writes entries <c>2(g mod term)</c> and the next one; the rest still hold what <c>np.empty</c>
    ///     returned, which OptunaSharp cannot know (it starts them at zero).
    /// </summary>
    /// <param name="cs">OptunaSharp's state, modified in place.</param>
    /// <param name="py">Python's state for the same generation.</param>
    private static void CopyUninitializedFunhist(PyInstance cs, PyInstance py)
    {
        if (cs.TryGet("_cma", out var inner) && inner is PyInstance csInner)
        {
            CopyUninitializedFunhist(csInner, (PyInstance)py.Get("_cma")!);
            return;
        }
        double[] csValues = ((NpArray)cs.Get("_funhist_values")!).Doubles!;
        double[] pyValues = ((NpArray)py.Get("_funhist_values")!).Doubles!;
        long generation = (long)cs.Get("_g")!, term = (long)cs.Get("_funhist_term")!;
        var written = new HashSet<long>();
        for (long g = 1; g <= generation; g++)
        {
            written.Add(2 * (g % term));
            written.Add(2 * (g % term) + 1);
        }
        for (int i = 0; i < csValues.Length && i < pyValues.Length; i++)
            if (!written.Contains(i))
                csValues[i] = pyValues[i];
    }

    /// <summary>
    ///     Asserts that every float of <paramref name="cs" /> is within a relative tolerance of Python's, then copies
    ///     Python's value over, so the remaining comparison sees layout, types and dtype sharing alone.
    /// </summary>
    /// <param name="cs">OptunaSharp's state, modified in place.</param>
    /// <param name="py">Python's state.</param>
    /// <param name="relativeTolerance">The largest relative difference allowed.</param>
    /// <param name="path">The attribute path so far, for failure messages.</param>
    private static void CopyCloseValues(PyInstance cs, PyInstance py, double relativeTolerance, string path)
    {
        for (int k = 0; k < py.State.Count && k < cs.State.Count; k++)
        {
            string where = path + py.State[k].Key;
            switch (py.State[k].Value, cs.State[k].Value)
            {
                case (PyInstance pyInner, PyInstance csInner):
                    CopyCloseValues(csInner, pyInner, relativeTolerance, where + ".");
                    break;
                case (NpFloat64 p, NpFloat64 c):
                    AssertClose(c.Value, p.Value, relativeTolerance, where);
                    cs.State[k] = new(cs.State[k].Key, new NpFloat64(p.Value, c.Dtype));
                    break;
                case (double p, double c):
                    AssertClose(c, p, relativeTolerance, where);
                    cs.State[k] = new(cs.State[k].Key, p);
                    break;
                case (NpArray { Doubles: { } p }, NpArray { Doubles: { } c }) when p.Length == c.Length:
                    for (int i = 0; i < p.Length; i++)
                    {
                        AssertClose(c[i], p[i], relativeTolerance, $"{where}[{i}]");
                        c[i] = p[i];
                    }
                    break;
            }
        }
    }

    /// <summary>Asserts a relative closeness, treating equal values (infinities, NaNs) as exact.</summary>
    /// <param name="actual">OptunaSharp's value.</param>
    /// <param name="expected">Python's value.</param>
    /// <param name="relativeTolerance">The largest relative difference allowed.</param>
    /// <param name="where">The attribute path.</param>
    private static void AssertClose(double actual, double expected, double relativeTolerance, string where)
    {
        if (actual.Equals(expected))
            return;
        Math.Abs(actual - expected).Should().BeLessThanOrEqualTo(relativeTolerance * Math.Max(Math.Abs(expected), 1e-300), where);
    }

    /// <summary>
    ///     Lists a state as one line per value: path, Python type, exact bits and, for arrays, the dtype object's
    ///     first-appearance index, so a failure names the first attribute that differs.
    /// </summary>
    /// <param name="state">An unpickled state.</param>
    /// <returns>The lines.</returns>
    private static List<string> Flatten(PyInstance state)
    {
        var lines = new List<string>();
        var dtypes = new Dictionary<NpDtype, int>(ReferenceEqualityComparer.Instance);
        string Dtype(NpDtype d)
        {
            if (!dtypes.TryGetValue(d, out int i))
                dtypes[d] = i = dtypes.Count;
            return $"{d.Code}#{i}";
        }
        void Walk(PyInstance s, string path)
        {
            lines.Add($"{path}<{s.Module}.{s.QualName}>");
            foreach (var (key, value) in s.State)
            {
                string where = path + key;
                lines.Add(value switch
                {
                    PyInstance inner => Nested(inner, where),
                    NpFloat64 f => $"{where}: numpy.float64 {BitConverter.DoubleToInt64Bits(f.Value):x16} ({f.Value:R})",
                    double d => $"{where}: float {BitConverter.DoubleToInt64Bits(d):x16} ({d:R})",
                    NpArray { Doubles: { } a } arr => $"{where}: {Dtype(arr.Dtype)}({string.Join(",", arr.Shape)}) " +
                        string.Join(" ", a.Select(v => BitConverter.DoubleToInt64Bits(v).ToString("x16"))),
                    NpArray arr => $"{where}: {Dtype(arr.Dtype)}({string.Join(",", arr.Shape)}) {string.Join(" ", arr.Longs!)}",
                    null => $"{where}: None",
                    _ => $"{where}: {value.GetType().Name} {value}",
                });
            }
        }
        string Nested(PyInstance inner, string where)
        {
            Walk(inner, where + ".");
            return $"{where}: end";
        }
        Walk(state, "");
        return lines;
    }
}
