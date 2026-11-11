using System;
using System.Globalization;
using AwesomeAssertions;
using OptunaSharp.Integration.LightGbm;

namespace OptunaSharp.Tests.Integration.LightGbm;

/// <summary>
///     Open parity bugs in the LightGBM integration found by the 2026-11-10 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-10.md</c>). Each test fails today and passes once fixed, at which point its
///     <c>OpenBugs</c> category should be removed.
/// </summary>
[TestClass]
public class Run2LightGbmParityBugTests
{
    /// <summary>
    ///     QA-76: when the callback's validation name and metric are not in the evaluation results, optuna-integration's
    ///     <c>LightGBMPruningCallback</c> raises a <c>ValueError</c> that ends with <c>str(env.evaluation_result_list)</c>,
    ///     the Python repr of the list of tuples (<c>100.0</c>, <c>1e-05</c>). The port formats each score with C#'s
    ///     default <c>double</c> formatting (<c>{r.Score}</c> in an interpolated string), which prints <c>100</c> and
    ///     <c>1E-05</c>.
    /// </summary>
    /// <remarks>
    ///     Python: <c>str([('valid_0', 'l2', 100.0, False), ('valid_1', 'auc', 1e-05, True)])</c> is
    ///     <c>[('valid_0', 'l2', 100.0, False), ('valid_1', 'auc', 1e-05, True)]</c>. The validation and metric names
    ///     are always single-quoted too, where Python switches to double quotes for a name that contains a single quote,
    ///     and a cross-validation result (five-element tuples with the standard deviation) is printed with four elements.
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void MissingMetric_ListsTheResultsInPythonRepr()
    {
        var trial = Optuna.CreateStudy().Ask();
        var callback = new LightGBMPruningCallback(trial, "missing");
        var results = new[]
        {
            new LightGbmEvaluationResult("valid_0", "l2", 100.0, false),
            new LightGbmEvaluationResult("valid_1", "auc", 1e-05, true),
        };

        Action call = () => callback.Call(0, results);

        call.Should().Throw<NumSharp.ValueError>().WithMessage(
            "The entry associated with the validation name \"valid_0\" and the metric name \"missing\" is not found in " +
            "the evaluation result list [('valid_0', 'l2', 100.0, False), ('valid_1', 'auc', 1e-05, True)].");
    }

    /// <summary>
    ///     QA-76, the culture half: the interpolated score uses the current culture, so under a culture with a decimal
    ///     comma the message reads <c>0,5</c> where Python always prints <c>0.5</c>.
    /// </summary>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void MissingMetric_FormatsScoresWithoutTheCurrentCulture()
    {
        var previous = CultureInfo.CurrentCulture;
        try
        {
            // de-DE writes 0.5 as "0,5"; Python's repr never depends on a locale.
            CultureInfo.CurrentCulture = new CultureInfo("de-DE");
            var callback = new LightGBMPruningCallback(Optuna.CreateStudy().Ask(), "missing");

            Action call = () => callback.Call(0, new[] { new LightGbmEvaluationResult("valid_0", "l2", 0.5, false) });

            call.Should().Throw<NumSharp.ValueError>().WithMessage("*[('valid_0', 'l2', 0.5, False)].");
        }
        finally
        {
            CultureInfo.CurrentCulture = previous;
        }
    }

    /// <summary>
    ///     QA-76, the tuner's alias handling: optuna-integration's <c>_handling_alias_parameters</c> warns with an
    ///     f-string of the duplicated aliases' <c>dict</c> (Python repr: quoted strings, <c>1.0</c>) and
    ///     <c>{param_name}={value}</c> (<c>str</c>: <c>1.0</c>, bare strings). The port's <c>FormatDict</c> interpolates
    ///     the values with C#'s <c>ToString</c>, so <c>'x'</c> loses its quotes and <c>1.0</c> becomes <c>1</c>.
    /// </summary>
    /// <remarks>
    ///     Python (optuna-integration 5.1.0.dev on the vendored source): the parameters
    ///     <c>{'sub_row': 0.5, 'subsample': 1.0, 'min_data': 20, 'min_data_in_leaf': 'x'}</c> warn
    ///     <c>bagging_fraction in param detected multiple identical aliases {'sub_row': 0.5, 'subsample': 1.0}, but we use
    ///     bagging_fraction=1.0.</c> and <c>min_child_samples in param detected multiple identical aliases
    ///     {'min_data': 20, 'min_data_in_leaf': 'x'}, but we use min_child_samples=x.</c> The warning reaches the port's
    ///     log at WARNING level, which is where this test reads it.
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void DuplicatedAliases_WarnWithPythonDictRepr()
    {
        var writer = new System.IO.StringWriter();
        var handler = new OptunaSharp.Logging.StreamLogHandler(writer);
        OptunaSharp.Logging.OptunaLogging.AddHandler(handler);
        try
        {
            Alias.HandleAliasParameters(new OptunaSharp.Collections.OrderedDict<string, object?>
            {
                ["sub_row"] = 0.5, ["subsample"] = 1.0, ["min_data"] = 20L, ["min_data_in_leaf"] = "x",
            });
        }
        finally
        {
            // Detach before asserting so a failure does not leave the handler on the shared logger.
            OptunaSharp.Logging.OptunaLogging.RemoveHandler(handler);
        }

        writer.ToString().Should()
            .Contain("bagging_fraction in param detected multiple identical aliases {'sub_row': 0.5, 'subsample': 1.0}, " +
                     "but we use bagging_fraction=1.0.")
            .And.Contain("min_child_samples in param detected multiple identical aliases {'min_data': 20, " +
                         "'min_data_in_leaf': 'x'}, but we use min_child_samples=x.");
    }

    /// <summary>
    ///     QA-76, the metric type error: optuna-integration's <c>_handling_alias_metrics</c> names the offending type as
    ///     <c>{type(metric)}</c>, i.e. <c>&lt;class 'int'&gt;</c>; the port prints the CLR type name (<c>Int64</c>).
    /// </summary>
    /// <exception cref="AssertFailedException">The message does not match Optuna's.</exception>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void NonIterableMetric_NamesThePythonType()
    {
        Action handle = () => Alias.HandleAliasMetrics(
            new OptunaSharp.Collections.OrderedDict<string, object?> { ["metric"] = 1L });

        handle.Should().Throw<NumSharp.ValueError>().WithMessage(
            "The `metric` parameter is expected to be a string or an iterable object, but got <class 'int'>.");
    }
}
