using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Visualization;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the visualization info layer, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). A dump of every info function over seven studies matched
///     Optuna 5.0.0 apart from the case below; each test fails today and passes once fixed, at which point its
///     <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
[TestClass]
public class VisualizationParityBugTests
{
    /// <summary>
    ///     QA-26: the parallel-coordinate data must handle a categorical parameter whose choices include
    ///     <c>None</c>, as Optuna's <c>_get_parallel_coordinate_info</c> does; the port throws
    ///     <see cref="System.ArgumentNullException" /> ("Value cannot be null. (Parameter 'key')").
    /// </summary>
    /// <remarks>
    ///     The port collects the categorical vocabulary in a dictionary keyed by the raw choice value, and a .NET
    ///     dictionary rejects a null key. So <c>plot_parallel_coordinate</c> fails for any study with a
    ///     <c>suggest_categorical(..., [None, ...])</c> parameter, a common way to make an option optional.
    ///     Optuna 5.0.0 for these four trials: the <c>c</c> dimension has values <c>(0, 1, 2, 0)</c>, tickvals
    ///     <c>[0, 1, 2]</c> and ticktext <c>['None', 'a', 'b']</c>.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void ParallelCoordinate_NoneCategoricalChoice_HandledLikePython()
    {
        var study = Optuna.CreateStudy();
        var cd = new CategoricalDistribution("a", CategoricalChoice.None, "b");
        var fd = new FloatDistribution(0, 1);
        foreach (var (c, x, v) in new (string? C, double X, double V)[] { (null, 0.1, 1.0), ("a", 0.5, 2.0), ("b", 0.9, 3.0), (null, 0.3, 0.5) })
        {
            study.AddTrial(FrozenTrial.CreateTrial(
                value: v,
                @params: new OrderedDict<string, object?> { ["c"] = c, ["x"] = x },
                distributions: new OrderedDict<string, BaseDistribution> { ["c"] = cd, ["x"] = fd }));
        }

        var info = ParallelCoordinate.GetInfo(study);

        var dim = info.DimsParams.Single(d => d.Label == "c");
        dim.Values.Should().Equal(0.0, 1.0, 2.0, 0.0);
        dim.Tickvals.Should().Equal(0.0, 1.0, 2.0);
        dim.Ticktext.Should().Equal("None", "a", "b");
    }

    /// <summary>
    ///     QA-44 (labels): parallel-coordinate labels must be truncated by Python's <c>len()</c>, which counts code
    ///     points; <c>TruncateLabel</c> counts UTF-16 code units, so a name with characters above U+FFFF is shortened
    ///     when Python keeps it whole, and the 17-unit cut can split a surrogate pair.
    /// </summary>
    /// <remarks>
    ///     <c>_truncate_label(label)</c> is <c>label if len(label) &lt; 20 else f"{label[:17]}..."</c>. A name of ten
    ///     U+1F600 characters is 10 code points (kept whole by Optuna 5.0.0) but 20 code units, so the port returns
    ///     its first 17 units — eight emoji and a lone high surrogate — followed by <c>...</c>, a string that is not
    ///     valid UTF-16 and renders as a replacement character. Found by the D86 string-semantics check.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void ParallelCoordinate_AstralCharacterLabel_TruncatedByCodePointLikePython()
    {
        string name = string.Concat(Enumerable.Repeat(char.ConvertFromUtf32(0x1F600), 10));
        var study = Optuna.CreateStudy();
        var fd = new FloatDistribution(0, 1);
        foreach (var (x, v) in new[] { (0.1, 1.0), (0.7, 2.0) })
        {
            study.AddTrial(FrozenTrial.CreateTrial(
                value: v,
                @params: new OrderedDict<string, object?> { [name] = x },
                distributions: new OrderedDict<string, BaseDistribution> { [name] = fd }));
        }

        var info = ParallelCoordinate.GetInfo(study);

        info.DimsParams.Single().Label.Should().Be(name);
    }

    /// <summary>Collects the messages of every log record, so a test can assert on Optuna's warning text.</summary>
    private sealed class CollectingHandler : Logging.ILogHandler
    {
        /// <summary>The messages received so far, oldest first.</summary>
        public System.Collections.Generic.List<string> Messages { get; } = new();

        /// <summary>Stores the record's message; never throws, so it cannot disturb the code under test.</summary>
        /// <param name="record">The log record delivered by <see cref="Logging.OptunaLogging" />.</param>
        public void Emit(in Logging.LogRecord record)
        {
            lock (Messages)
                Messages.Add(record.Message);
        }
    }

    /// <summary>
    ///     QA-32: the timeline's reversed-times warning must print the datetimes the way Python's <c>str(datetime)</c>
    ///     does; the port interpolates <see cref="System.DateTime" /> with the current culture's default format.
    /// </summary>
    /// <remarks>
    ///     Optuna logs <c>f"... The start time is {datetime_start} and the end time is {datetime_complete}."</c>,
    ///     which gives <c>2026-01-02 03:10:00</c> and <c>2026-01-02 03:09:59.500000</c> (microseconds only when non-zero).
    ///     The port's <c>{datetimeStart}</c> gives <c>1/2/2026 3:10:00 AM</c> and <c>1/2/2026 3:09:59 AM</c> under en-US,
    ///     <c>01/02/2026 03:10:00</c> under the invariant culture (day-first or dotted under others), and always drops
    ///     the fraction, so two different reversed pairs can print the same text. The timeline bars themselves match
    ///     Optuna exactly.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Timeline_ReversedTimesWarning_FormatsDatetimesLikePython()
    {
        var study = Optuna.CreateStudy();
        study.AddTrial(new FrozenTrial(0, TrialState.Complete, 1.0,
            new System.DateTime(2026, 1, 2, 3, 10, 0), new System.DateTime(2026, 1, 2, 3, 9, 59, 500),
            new OrderedDict<string, object?>(), new OrderedDict<string, BaseDistribution>(),
            new OrderedDict<string, object?>(), new OrderedDict<string, object?>(), new OrderedDict<long, double>(), 0));
        var handler = new CollectingHandler();
        Logging.OptunaLogging.AddHandler(handler);
        try
        {
            Timeline.GetInfo(study);
        }
        finally
        {
            Logging.OptunaLogging.RemoveHandler(handler);
        }

        handler.Messages.Should().Contain(
            "The start and end times for Trial 0 seem to be reversed. "
            + "The start time is 2026-01-02 03:10:00 and the end time is 2026-01-02 03:09:59.500000.");
    }
}
