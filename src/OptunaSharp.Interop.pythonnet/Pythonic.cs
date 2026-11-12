using System;
using System.Collections.Generic;
using Python.Runtime;
using OptunaSharp;
using OptunaSharp.Distributions;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     The <b>pythonic</b> surface for the OptunaHub bridge — a single root <c>static class</c>
///     <c>Python</c> whose nested subclasses form a tree that reaches each fixed <c>optuna</c> /
///     <c>optunahub</c> call, so the consuming C# reads for what it is:
///     <c>Python.optuna.create_study(...)</c>, <c>Python.optuna.Study.ask(study)</c>,
///     <c>Python.optunahub.load_module(...)</c>. It is always spelled <c>Python.*</c> (the class shadows
///     pythonnet's <c>Python.Runtime</c> namespace in scope), and any Python object whose methods the
///     bridge calls is exposed as a <b>nested class</b> taking that object (mirroring Python's own unbound
///     <c>optuna.samplers.BaseSampler.sample_relative(sampler, ...)</c>) — never an extension.
///     <para>
///     Only the FIXED calls live here. The inherently dynamic work — instantiating an arbitrary
///     contributor class (<c>module.GetAttr(className)</c>), <c>HasAttr</c> probes, Python type-name
///     reflection and generic <c>dict.items()</c> / <c>__len__</c> iteration — stays raw at the call
///     site, and primitive CLR⇔Python value marshaling stays in <see cref="PyConvert" />.
///     </para>
///     <para>Every member calls the C-API and MUST run under the GIL (from a
///     <see cref="PythonRuntime.Run(System.Action)" /> body). A returned <see cref="PyObject" /> is owned
///     by the caller (dispose it); the bodies mirror the exact <c>optuna</c>/<c>optunahub</c> calls the
///     bridge has always made (same kwargs, order and types), so behavior is unchanged.</para>
/// </summary>
internal static class Python
{
    /// <summary><c>import optuna</c> — the fixed optuna calls the bridge makes, reached as <c>Python.optuna.*</c>.</summary>
    internal static class optuna
    {
        /// <summary><c>optuna.create_study(directions=[...], study_name=...)</c> — the in-memory shadow study.</summary>
        /// <param name="directions">One direction per objective, in objective-index order.</param>
        /// <param name="studyName">
        ///     The study name, or null for Optuna's generated one. The shadow carries the OptunaSharp study's name: some
        ///     hub objects derive state from it (<c>HyperbandPruner</c> assigns brackets by
        ///     <c>crc32(f"{study.study_name}_{trial.number}")</c>).
        /// </param>
        /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
        /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
        internal static PyObject create_study(IReadOnlyList<StudyDirection> directions, string? studyName = null)
        {
            using var createStudy = OptunaHubRuntime.Optuna.GetAttr("create_study");
            using var kwargs = new PyDict();
            if (studyName is not null)
            {
                using var name = studyName.ToPython();
                kwargs["study_name"] = name;
            }
            var items = new PyObject[directions.Count];
            for (int i = 0; i < items.Length; i++)
                items[i] = (directions[i] == StudyDirection.Maximize ? "maximize" : "minimize").ToPython();
            using (var pyDirs = new PyList(items))
                kwargs["directions"] = pyDirs;
            foreach (var it in items) it.Dispose();
            return createStudy.Invoke(Array.Empty<PyObject>(), kwargs);
        }

        /// <summary><c>optuna.logging.*</c>.</summary>
        internal static class logging
        {
            /// <summary><c>optuna.logging.set_verbosity(optuna.logging.WARNING)</c>.</summary>
            /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
            internal static void set_verbosity_warning()
            {
                using var logging = OptunaHubRuntime.Optuna.GetAttr("logging");
                using var warning = logging.GetAttr("WARNING");
                using var setVerbosity = logging.GetAttr("set_verbosity");
                using var _ = setVerbosity.Invoke(warning);
            }
        }

        /// <summary><c>optuna.trial.*</c>.</summary>
        internal static class trial
        {
            /// <summary>
            ///     <c>optuna.trial.create_trial(state=, values=, params=, distributions=, intermediate_values=)</c>
            ///     — a finished trial for the shadow study. <paramref name="intermediateValues" /> is omitted
            ///     (as Optuna defaults it) when <c>null</c> — the sampler bridge passes none, the pruner bridge does.
            /// </summary>
            /// <param name="state">The trial lifecycle state represented by the Python trial.</param>
            /// <param name="values">The objective values or borrowed Python array assigned to the corresponding field.</param>
            /// <param name="parameters">The named parameter values passed to the Python trial or benchmark.</param>
            /// <param name="distributions">The distributions corresponding to the supplied trial parameters.</param>
            /// <param name="intermediateValues">Optional intermediate reports keyed by step, copied into the Python trial.</param>
            /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
            /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
            internal static PyObject create_trial(
                OptunaSharp.TrialState state,
                IReadOnlyList<double>? values,
                IReadOnlyDictionary<string, object?> parameters,
                IReadOnlyDictionary<string, BaseDistribution> distributions,
                IReadOnlyDictionary<long, double>? intermediateValues = null)
            {
                using var trialModule = OptunaHubRuntime.Optuna.GetAttr("trial");
                using var createTrial = trialModule.GetAttr("create_trial");
                using var kwargs = new PyDict();

                kwargs["state"] = TrialState(state);

                if (values is { Count: > 0 })
                {
                    var items = new PyObject[values.Count];
                    for (int k = 0; k < items.Length; k++)
                        items[k] = values[k].ToPython();
                    using (var pyValues = new PyList(items))
                        kwargs["values"] = pyValues;
                    foreach (var it in items) it.Dispose();
                }

                using (var pyParams = new PyDict())
                using (var pyDists = new PyDict())
                {
                    foreach (var (name, value) in parameters)
                        pyParams[name] = PyConvert.ToPy(value);
                    kwargs["params"] = pyParams;

                    foreach (var (name, dist) in distributions)
                        pyDists[name] = PyConvert.ToPy(dist);
                    kwargs["distributions"] = pyDists;

                    if (intermediateValues is null)
                        return createTrial.Invoke(Array.Empty<PyObject>(), kwargs);

                    using var pyInter = new PyDict();
                    foreach (var (step, value) in intermediateValues)
                    {
                        // Python int keys: a 64-bit step stays exact (an int step key -> float value).
                        using var pyStep = new PyInt(step);
                        using var pyValue = value.ToPython();
                        pyInter.SetItem(pyStep, pyValue);
                    }
                    kwargs["intermediate_values"] = pyInter;
                    return createTrial.Invoke(Array.Empty<PyObject>(), kwargs);
                }
            }

            /// <summary><c>optuna.trial.TrialState.&lt;NAME&gt;</c> — the Python enum member for an OptunaSharp state.</summary>
            /// <param name="state">The trial lifecycle state represented by the Python trial.</param>
            /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
            /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
            internal static PyObject TrialState(OptunaSharp.TrialState state)
            {
                string name = state switch
                {
                    OptunaSharp.TrialState.Complete => "COMPLETE",
                    OptunaSharp.TrialState.Pruned => "PRUNED",
                    OptunaSharp.TrialState.Fail => "FAIL",
                    OptunaSharp.TrialState.Waiting => "WAITING",
                    _ => "RUNNING",
                };
                using var trialModule = OptunaHubRuntime.Optuna.GetAttr("trial");
                using var stateEnum = trialModule.GetAttr("TrialState");
                return stateEnum.GetAttr(name);
            }
        }

        /// <summary><c>optuna.distributions.*</c> constructors.</summary>
        internal static class distributions
        {
            /// <summary><c>optuna.distributions.FloatDistribution(low=, high=, log=, step=)</c>.</summary>
            /// <param name="low">The inclusive lower distribution bound.</param>
            /// <param name="high">The inclusive upper distribution bound, subject to step alignment.</param>
            /// <param name="log">Whether the numerical distribution is sampled on a logarithmic scale.</param>
            /// <param name="step">The discretization interval, or the reported trial step for report operations.</param>
            /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
            /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
            internal static PyObject FloatDistribution(double low, double high, bool log, double? step)
            {
                using var dists = OptunaHubRuntime.Optuna.GetAttr("distributions");
                using var ctor = dists.GetAttr("FloatDistribution");
                using var kwargs = new PyDict();
                kwargs["low"] = low.ToPython();
                kwargs["high"] = high.ToPython();
                kwargs["log"] = log.ToPython();
                kwargs["step"] = step is { } s ? s.ToPython() : ((object?)null).ToPython();
                return ctor.Invoke(Array.Empty<PyObject>(), kwargs);
            }

            /// <summary><c>optuna.distributions.IntDistribution(low=, high=, log=, step=)</c>.</summary>
            /// <param name="low">The inclusive lower distribution bound.</param>
            /// <param name="high">The inclusive upper distribution bound, subject to step alignment.</param>
            /// <param name="log">Whether the numerical distribution is sampled on a logarithmic scale.</param>
            /// <param name="step">The discretization interval, or the reported trial step for report operations.</param>
            /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
            /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
            internal static PyObject IntDistribution(long low, long high, bool log, long step)
            {
                using var dists = OptunaHubRuntime.Optuna.GetAttr("distributions");
                using var ctor = dists.GetAttr("IntDistribution");
                using var kwargs = new PyDict();
                kwargs["low"] = low.ToPython();
                kwargs["high"] = high.ToPython();
                kwargs["log"] = log.ToPython();
                kwargs["step"] = step.ToPython();
                return ctor.Invoke(Array.Empty<PyObject>(), kwargs);
            }

            /// <summary><c>optuna.distributions.CategoricalDistribution(choices)</c> — choices marshaled via <see cref="PyConvert" />.</summary>
            /// <param name="choices">The categorical values in stable choice order.</param>
            /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
            /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
            internal static PyObject CategoricalDistribution(IReadOnlyList<CategoricalChoice> choices)
            {
                using var dists = OptunaHubRuntime.Optuna.GetAttr("distributions");
                using var ctor = dists.GetAttr("CategoricalDistribution");
                var items = new PyObject[choices.Count];
                for (int k = 0; k < items.Length; k++)
                    items[k] = PyConvert.ToPy(choices[k].Value);
                using var pyChoices = new PyList(items);
                foreach (var it in items) it.Dispose();
                return ctor.Invoke(pyChoices);
            }
        }

        /// <summary><c>optuna.study.Study</c> instance members (unbound — take the study), as the bridge calls them.</summary>
        internal static class Study
        {
            /// <summary><c>study.ask()</c> — open a RUNNING trial. Dispose the result.</summary>
            /// <param name="study">The borrowed Python shadow-study handle; use it only while holding the GIL.</param>
            /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
            /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
            internal static PyObject ask(PyObject study)
            {
                using var ask = study.GetAttr("ask");
                return ask.Invoke();
            }

            /// <summary><c>study.add_trial(trial)</c>.</summary>
            /// <param name="study">The borrowed Python shadow-study handle; use it only while holding the GIL.</param>
            /// <param name="trial">The borrowed Python trial handle; use it only while holding the GIL.</param>
            /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
            internal static void add_trial(PyObject study, PyObject trial)
            {
                using var addTrial = study.GetAttr("add_trial");
                using var _ = addTrial.Invoke(trial);
            }

            /// <summary><c>study.get_trials(deepcopy=)</c>. Dispose the result.</summary>
            /// <param name="study">The borrowed Python shadow-study handle; use it only while holding the GIL.</param>
            /// <param name="deepcopy">Whether Python should return independent copies of its stored frozen trials.</param>
            /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
            /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
            internal static PyObject get_trials(PyObject study, bool deepcopy)
            {
                using var getTrials = study.GetAttr("get_trials");
                using var kwargs = new PyDict();
                kwargs["deepcopy"] = deepcopy.ToPython();
                return getTrials.Invoke(Array.Empty<PyObject>(), kwargs);
            }
        }

        /// <summary><c>optuna.trial.Trial</c> / <c>FrozenTrial</c> instance members (unbound — take the trial).</summary>
        internal static class Trial
        {
            /// <summary><c>trial.number</c> as a CLR <see cref="int" />.</summary>
            /// <param name="trial">The borrowed Python trial handle; use it only while holding the GIL.</param>
            /// <returns>The zero-based trial number within the Python study.</returns>
            /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
            internal static int number_t(PyObject trial)
            {
                using var n = trial.GetAttr("number");
                return n.As<int>();
            }

            /// <summary><c>trial.report(value, step)</c>.</summary>
            /// <param name="trial">The borrowed Python trial handle; use it only while holding the GIL.</param>
            /// <param name="value">The parameter value or borrowed Python scalar to convert.</param>
            /// <param name="step">The discretization interval, or the reported trial step for report operations.</param>
            /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
            internal static void report(PyObject trial, double value, long step)
            {
                using var report = trial.GetAttr("report");
                using var v = value.ToPython();
                using var s = step.ToPython();
                using var _ = report.Invoke(v, s);
            }
        }

        /// <summary><c>optuna.samplers.*</c>.</summary>
        internal static class samplers
        {
            /// <summary>
            ///     <c>optuna.samplers.BaseSampler</c> protocol methods (unbound — take the sampler). The
            ///     <c>HasAttr</c> presence probes stay at the call site (dynamic reflection on a
            ///     contributor-supplied sampler).
            /// </summary>
            internal static class BaseSampler
            {
                /// <summary><c>sampler.infer_relative_search_space(study, trial)</c>. Dispose the result.</summary>
                /// <param name="sampler">The Python sampler handle owned by the wrapper, or borrowed by a low-level invocation.</param>
                /// <param name="study">The borrowed Python shadow-study handle; use it only while holding the GIL.</param>
                /// <param name="trial">The borrowed Python trial handle; use it only while holding the GIL.</param>
                /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
                /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
                internal static PyObject infer_relative_search_space(PyObject sampler, PyObject study, PyObject trial)
                {
                    using var infer = sampler.GetAttr("infer_relative_search_space");
                    return infer.Invoke(study, trial);
                }

                /// <summary><c>sampler.sample_relative(study, trial, search_space)</c>. Dispose the result.</summary>
                /// <param name="sampler">The Python sampler handle owned by the wrapper, or borrowed by a low-level invocation.</param>
                /// <param name="study">The borrowed Python shadow-study handle; use it only while holding the GIL.</param>
                /// <param name="trial">The borrowed Python trial handle; use it only while holding the GIL.</param>
                /// <param name="searchSpace">The ordered distributions defining the relative parameter coordinates.</param>
                /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
                /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
                internal static PyObject sample_relative(PyObject sampler, PyObject study, PyObject trial, PyObject searchSpace)
                {
                    using var sampleRelative = sampler.GetAttr("sample_relative");
                    return sampleRelative.Invoke(new[] { study, trial, searchSpace });
                }

                /// <summary><c>sampler.sample_independent(study, trial, param_name, param_distribution)</c>. Dispose the result.</summary>
                /// <param name="sampler">The Python sampler handle owned by the wrapper, or borrowed by a low-level invocation.</param>
                /// <param name="study">The borrowed Python shadow-study handle; use it only while holding the GIL.</param>
                /// <param name="trial">The borrowed Python trial handle; use it only while holding the GIL.</param>
                /// <param name="name">The Python attribute, parameter, or algorithm name required by the operation.</param>
                /// <param name="dist">The distribution defining value conversion or the borrowed Python distribution object.</param>
                /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
                /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
                internal static PyObject sample_independent(PyObject sampler, PyObject study, PyObject trial, PyObject name, PyObject dist)
                {
                    using var sampleIndependent = sampler.GetAttr("sample_independent");
                    return sampleIndependent.Invoke(new[] { study, trial, name, dist });
                }

                /// <summary><c>sampler.before_trial(study, trial)</c>.</summary>
                /// <param name="sampler">The Python sampler handle owned by the wrapper, or borrowed by a low-level invocation.</param>
                /// <param name="study">The borrowed Python shadow-study handle; use it only while holding the GIL.</param>
                /// <param name="trial">The borrowed trial record (the trial's cached record, as <c>Trial.__init__</c> passes it).</param>
                /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
                internal static void before_trial(PyObject sampler, PyObject study, PyObject trial)
                {
                    using var beforeTrial = sampler.GetAttr("before_trial");
                    using var _ = beforeTrial.Invoke(study, trial);
                }

                /// <summary><c>sampler.after_trial(study, trial, state, values)</c>.</summary>
                /// <param name="sampler">The Python sampler handle owned by the wrapper, or borrowed by a low-level invocation.</param>
                /// <param name="study">The borrowed Python shadow-study handle; use it only while holding the GIL.</param>
                /// <param name="trial">The borrowed trial record (the stored record, as <c>_tell_with_warning</c> passes it).</param>
                /// <param name="state">The borrowed <c>optuna.trial.TrialState</c> the trial finishes with.</param>
                /// <param name="values">The borrowed objective-value list, or <c>None</c>.</param>
                /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
                internal static void after_trial(PyObject sampler, PyObject study, PyObject trial, PyObject state, PyObject values)
                {
                    using var afterTrial = sampler.GetAttr("after_trial");
                    using var _ = afterTrial.Invoke(new[] { study, trial, state, values });
                }

                /// <summary><c>sampler.reseed_rng()</c>.</summary>
                /// <param name="sampler">The Python sampler handle owned by the wrapper, or borrowed by a low-level invocation.</param>
                /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
                internal static void reseed_rng(PyObject sampler)
                {
                    using var reseed = sampler.GetAttr("reseed_rng");
                    using var _ = reseed.Invoke();
                }
            }
        }

        /// <summary><c>optuna.pruners.*</c>.</summary>
        internal static class pruners
        {
            /// <summary><c>optuna.pruners.BasePruner</c> protocol methods (unbound — take the pruner).</summary>
            internal static class BasePruner
            {
                /// <summary><c>pruner.prune(study, trial)</c> — returns the raw result (use <c>IsTrue()</c>; a
                /// pruner may return a numpy <c>bool_</c> that <c>As&lt;bool&gt;()</c> rejects). Dispose it.</summary>
                /// <param name="pruner">The Python pruner handle owned by the wrapper, or borrowed by a low-level invocation.</param>
                /// <param name="study">The borrowed Python shadow-study handle; use it only while holding the GIL.</param>
                /// <param name="trial">The borrowed Python trial handle; use it only while holding the GIL.</param>
                /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
                /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
                internal static PyObject prune(PyObject pruner, PyObject study, PyObject trial)
                {
                    using var prune = pruner.GetAttr("prune");
                    return prune.Invoke(study, trial);
                }
            }
        }
    }

    /// <summary>
    ///     The OptunaHub bridges' shadow-study helpers (<see cref="OptunaHubShadow" />): a small Python module defined once
    ///     in the embedded interpreter, whose functions make the storage-level writes a native Optuna trial makes, reached
    ///     as <c>Python.optunasharp_shadow.*</c>.
    /// </summary>
    /// <remarks>
    ///     Python source rather than a chain of C-API calls, so each helper reads as the Optuna code it mirrors
    ///     (<c>Trial._suggest</c>, <c>Trial.report</c>, <c>Trial.set_user_attr</c>, <c>Trial._get_latest_trial</c>,
    ///     <c>_tell_with_warning</c>'s state write) and diffs against it. The module is built on first use and kept for
    ///     the process, like <see cref="PythonRuntime.Import" />'s cache.
    /// </remarks>
    internal static class optunasharp_shadow
    {
        /// <summary>The module's source.</summary>
        private const string Source = """
            import copy

            import optuna
            from optuna.trial._trial import _LazyTrialSystemAttrs


            def add_trial(study, state, values, params, distributions, user_attrs, system_attrs, intermediate_values):
                # One OptunaSharp trial, in any state, as the shadow's next trial; returns its trial id.
                template = optuna.trial.create_trial(
                    state=state, values=values, params=params, distributions=distributions, user_attrs=user_attrs,
                    system_attrs=system_attrs, intermediate_values=intermediate_values)
                return study._storage.create_new_trial(study._study_id, template_trial=template)


            def begin_trial(study):
                # Study.ask: "Sync storage once every trial." Samplers read the history through
                # study._get_trials(use_cache=True), which keeps the first read until this reset.
                study._thread_local.cached_all_trials = None


            def get_trial(study, trial_id):
                return study._storage.get_trial(trial_id)


            def latest(study, cached):
                # Trial._get_latest_trial
                trial = copy.copy(cached)
                trial.system_attrs = _LazyTrialSystemAttrs(cached._trial_id, study._storage)
                return trial


            def set_param(study, cached, trial_id, name, value, distribution):
                # Trial._suggest, after the value is chosen
                study._storage.set_trial_param(trial_id, name, distribution.to_internal_repr(value), distribution)
                if cached is not None:
                    cached.distributions[name] = distribution
                    cached.params[name] = value


            def report(study, cached, trial_id, step, value):
                # Trial.report
                study._storage.set_trial_intermediate_value(trial_id, step, value)
                if cached is not None:
                    cached.intermediate_values[step] = value


            def set_user_attr(study, cached, trial_id, key, value):
                # Trial.set_user_attr
                study._storage.set_trial_user_attr(trial_id, key, value)
                if cached is not None:
                    cached.user_attrs[key] = value


            def set_system_attr(study, trial_id, key, value):
                study._storage.set_trial_system_attr(trial_id, key, value)


            def set_study_user_attr(study, key, value):
                study._storage.set_study_user_attr(study._study_id, key, value)


            def set_study_system_attr(study, key, value):
                study._storage.set_study_system_attr(study._study_id, key, value)


            def set_state(study, trial_id, state, values):
                study._storage.set_trial_state_values(trial_id, state, values)


            def attrs(study, trial_id):
                # Shallow copies of the four dictionaries a hub object may write to.
                storage = study._storage
                return (
                    dict(storage.get_study_user_attrs(study._study_id)),
                    dict(storage.get_study_system_attrs(study._study_id)),
                    dict(storage.get_trial_user_attrs(trial_id)),
                    dict(storage.get_trial_system_attrs(trial_id)),
                )


            def changed(before, after):
                # By identity: a storage write stores the new value object, and identity works for values without a
                # usable `==` (a NumPy array's is element-wise).
                return {key: value for key, value in after.items() if key not in before or before[key] is not value}
            """;

        /// <summary>The module, built from <see cref="Source" /> on first use; process-scoped, never disposed.</summary>
        private static PyModule? _module;

        /// <summary>The module, building it on first use.</summary>
        /// <exception cref="global::Python.Runtime.PythonException">The source fails to run (Optuna missing or changed).</exception>
        private static PyModule Module => _module ??= PyModule.FromString("optunasharp_shadow", Source);

        /// <summary>Calls one of the module's functions and returns its result, owned by the caller.</summary>
        /// <param name="function">The function's name.</param>
        /// <param name="args">The borrowed arguments.</param>
        /// <returns>A new owned reference; dispose it under the GIL.</returns>
        /// <exception cref="global::Python.Runtime.PythonException">The function raises.</exception>
        private static PyObject Call(string function, params PyObject[] args)
        {
            using PyObject fn = Module.GetAttr(function);
            return fn.Invoke(args);
        }

        /// <summary>Calls one of the module's functions for its effect only.</summary>
        /// <param name="function">The function's name.</param>
        /// <param name="args">The borrowed arguments.</param>
        /// <exception cref="global::Python.Runtime.PythonException">The function raises.</exception>
        private static void Do(string function, params PyObject[] args)
        {
            using PyObject _ = Call(function, args);
        }

        /// <summary>A borrowed reference, or <c>None</c> for null, for an optional argument.</summary>
        /// <param name="value">The borrowed reference, or null.</param>
        /// <param name="none">Receives the owned <c>None</c> to dispose, when <paramref name="value" /> is null.</param>
        /// <returns>The argument to pass.</returns>
        private static PyObject OrNone(PyObject? value, out PyObject? none)
        {
            none = value is null ? ((object?)null).ToPython() : null;
            return value ?? none!;
        }

        /// <summary><c>add_trial(study, state, values, params, distributions, user_attrs, system_attrs, intermediate_values)</c>.</summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <param name="state">The borrowed <c>TrialState</c>.</param>
        /// <param name="values">The borrowed value list, or <c>None</c>.</param>
        /// <param name="parameters">The borrowed parameter dict (external values).</param>
        /// <param name="distributions">The borrowed distribution dict.</param>
        /// <param name="userAttrs">The borrowed user-attribute dict.</param>
        /// <param name="systemAttrs">The borrowed system-attribute dict.</param>
        /// <param name="intermediateValues">The borrowed step-to-value dict.</param>
        /// <returns>The new trial's shadow trial id.</returns>
        /// <exception cref="global::Python.Runtime.PythonException">Optuna rejects the trial (<c>create_trial</c>'s validation) or the write fails.</exception>
        internal static int add_trial(PyObject study, PyObject state, PyObject values, PyObject parameters,
            PyObject distributions, PyObject userAttrs, PyObject systemAttrs, PyObject intermediateValues)
        {
            using PyObject id = Call("add_trial",
                study, state, values, parameters, distributions, userAttrs, systemAttrs, intermediateValues);
            return id.As<int>();
        }

        /// <summary>
        ///     <c>begin_trial(study)</c>: clears the study's per-thread trial cache, as <c>Study.ask</c> does for every trial,
        ///     so a sampler reading <c>study._get_trials(use_cache=True)</c> (CMA-ES, TPE, the GA samplers, QMC, GP) sees
        ///     the history as of this trial rather than as of its first read.
        /// </summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <exception cref="global::Python.Runtime.PythonException">The study has no per-thread cache (an incompatible Optuna).</exception>
        internal static void begin_trial(PyObject study) => Do("begin_trial", study);

        /// <summary><c>get_trial(study, trial_id)</c>: the storage's record of the trial.</summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <param name="trialId">The shadow trial id.</param>
        /// <returns>A new owned reference; dispose it under the GIL.</returns>
        /// <exception cref="global::Python.Runtime.PythonException">The read fails.</exception>
        internal static PyObject get_trial(PyObject study, int trialId)
        {
            using var id = new PyInt(trialId);
            return Call("get_trial", study, id);
        }

        /// <summary><c>latest(study, cached)</c>: <c>Trial._get_latest_trial()</c> of a trial's cached record.</summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <param name="cached">The borrowed cached record.</param>
        /// <returns>A new owned reference; dispose it under the GIL.</returns>
        /// <exception cref="global::Python.Runtime.PythonException">The copy fails.</exception>
        internal static PyObject latest(PyObject study, PyObject cached) => Call("latest", study, cached);

        /// <summary><c>set_param(study, cached, trial_id, name, value, distribution)</c>.</summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <param name="cached">The borrowed cached record to update in place, or null.</param>
        /// <param name="trialId">The shadow trial id.</param>
        /// <param name="name">The borrowed parameter name.</param>
        /// <param name="value">The borrowed external value.</param>
        /// <param name="distribution">The borrowed distribution.</param>
        /// <exception cref="global::Python.Runtime.PythonException">The write fails.</exception>
        internal static void set_param(PyObject study, PyObject? cached, int trialId, PyObject name, PyObject value,
            PyObject distribution)
        {
            using var id = new PyInt(trialId);
            PyObject c = OrNone(cached, out PyObject? none);
            using (none)
                Do("set_param", study, c, id, name, value, distribution);
        }

        /// <summary><c>report(study, cached, trial_id, step, value)</c>.</summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <param name="cached">The borrowed cached record to update in place, or null.</param>
        /// <param name="trialId">The shadow trial id.</param>
        /// <param name="step">The borrowed step.</param>
        /// <param name="value">The borrowed value.</param>
        /// <exception cref="global::Python.Runtime.PythonException">The write fails.</exception>
        internal static void report(PyObject study, PyObject? cached, int trialId, PyObject step, PyObject value)
        {
            using var id = new PyInt(trialId);
            PyObject c = OrNone(cached, out PyObject? none);
            using (none)
                Do("report", study, c, id, step, value);
        }

        /// <summary><c>set_user_attr(study, cached, trial_id, key, value)</c>.</summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <param name="cached">The borrowed cached record to update in place, or null.</param>
        /// <param name="trialId">The shadow trial id.</param>
        /// <param name="key">The borrowed key.</param>
        /// <param name="value">The borrowed value.</param>
        /// <exception cref="global::Python.Runtime.PythonException">The write fails.</exception>
        internal static void set_user_attr(PyObject study, PyObject? cached, int trialId, PyObject key, PyObject value)
        {
            using var id = new PyInt(trialId);
            PyObject c = OrNone(cached, out PyObject? none);
            using (none)
                Do("set_user_attr", study, c, id, key, value);
        }

        /// <summary><c>set_system_attr(study, trial_id, key, value)</c>.</summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <param name="trialId">The shadow trial id.</param>
        /// <param name="key">The borrowed key.</param>
        /// <param name="value">The borrowed value.</param>
        /// <exception cref="global::Python.Runtime.PythonException">The write fails.</exception>
        internal static void set_system_attr(PyObject study, int trialId, PyObject key, PyObject value)
        {
            using var id = new PyInt(trialId);
            Do("set_system_attr", study, id, key, value);
        }

        /// <summary><c>set_study_user_attr(study, key, value)</c>.</summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <param name="key">The borrowed key.</param>
        /// <param name="value">The borrowed value.</param>
        /// <exception cref="global::Python.Runtime.PythonException">The write fails.</exception>
        internal static void set_study_user_attr(PyObject study, PyObject key, PyObject value)
            => Do("set_study_user_attr", study, key, value);

        /// <summary><c>set_study_system_attr(study, key, value)</c>.</summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <param name="key">The borrowed key.</param>
        /// <param name="value">The borrowed value.</param>
        /// <exception cref="global::Python.Runtime.PythonException">The write fails.</exception>
        internal static void set_study_system_attr(PyObject study, PyObject key, PyObject value)
            => Do("set_study_system_attr", study, key, value);

        /// <summary><c>set_state(study, trial_id, state, values)</c>.</summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <param name="trialId">The shadow trial id.</param>
        /// <param name="state">The borrowed <c>TrialState</c>.</param>
        /// <param name="values">The borrowed value list, or <c>None</c>.</param>
        /// <exception cref="global::Python.Runtime.PythonException">The write fails (a finished trial cannot change).</exception>
        internal static void set_state(PyObject study, int trialId, PyObject state, PyObject values)
        {
            using var id = new PyInt(trialId);
            Do("set_state", study, id, state, values);
        }

        /// <summary><c>attrs(study, trial_id)</c>: copies of the study's and the trial's user and system attributes.</summary>
        /// <param name="study">The borrowed shadow study.</param>
        /// <param name="trialId">The shadow trial id.</param>
        /// <returns>A new owned 4-tuple (study user, study system, trial user, trial system); dispose it under the GIL.</returns>
        /// <exception cref="global::Python.Runtime.PythonException">The reads fail.</exception>
        internal static PyObject attrs(PyObject study, int trialId)
        {
            using var id = new PyInt(trialId);
            return Call("attrs", study, id);
        }

        /// <summary><c>changed(before, after)</c>: the keys <paramref name="after" /> gained or replaced.</summary>
        /// <param name="before">The borrowed dict before a call.</param>
        /// <param name="after">The borrowed dict after it.</param>
        /// <returns>A new owned dict; dispose it under the GIL.</returns>
        /// <exception cref="global::Python.Runtime.PythonException">The comparison fails.</exception>
        internal static PyObject changed(PyObject before, PyObject after) => Call("changed", before, after);
    }

    /// <summary><c>import optunahub</c> — the fixed optunahub calls, reached as <c>Python.optunahub.*</c>.</summary>
    internal static class optunahub
    {
        /// <summary>
        ///     <c>optunahub.load_module(package, repo_owner=, repo_name=, ref=, force_reload=, base_url=)</c>.
        ///     May raise <see cref="PythonException" /> — the caller wraps it in <see cref="OptunaHubInteropException" />.
        /// </summary>
        /// <param name="package">The registry-relative package name to load.</param>
        /// <param name="repoOwner">The owner of the registry repository.</param>
        /// <param name="repoName">The registry repository name.</param>
        /// <param name="gitRef">The branch, tag, or commit to load.</param>
        /// <param name="forceReload">Whether to fetch the package again instead of reusing the local cache.</param>
        /// <param name="baseUrl">An optional alternate registry host URL.</param>
        /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
        internal static PyObject load_module(
            string package, string repoOwner, string repoName, string gitRef, bool forceReload, string? baseUrl)
        {
            using var loadModule = OptunaHubRuntime.OptunaHub.GetAttr("load_module");
            using var kwargs = new PyDict();
            kwargs["repo_owner"] = repoOwner.ToPython();
            kwargs["repo_name"] = repoName.ToPython();
            kwargs["ref"] = gitRef.ToPython();
            kwargs["force_reload"] = forceReload.ToPython();
            if (baseUrl is not null)
                kwargs["base_url"] = baseUrl.ToPython();
            using var pkg = package.ToPython();
            return loadModule.Invoke(new[] { pkg }, kwargs);
        }

        /// <summary><c>optunahub.load_local_module(package, registry_root=)</c>.</summary>
        /// <param name="package">The registry-relative package name to load.</param>
        /// <param name="registryRoot">The local registry root containing the package tree.</param>
        /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
        /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
        internal static PyObject load_local_module(string package, string registryRoot)
        {
            using var loadLocal = OptunaHubRuntime.OptunaHub.GetAttr("load_local_module");
            using var kwargs = new PyDict();
            kwargs["registry_root"] = registryRoot.ToPython();
            using var pkg = package.ToPython();
            return loadLocal.Invoke(new[] { pkg }, kwargs);
        }

        /// <summary><c>optunahub.benchmarks.*</c>.</summary>
        internal static class benchmarks
        {
            /// <summary><c>optunahub.benchmarks.BaseProblem</c> instance members (unbound — take the problem).</summary>
            internal static class BaseProblem
            {
                /// <summary><c>problem.evaluate(params)</c> — score a parameter dict. Dispose the result.</summary>
                /// <param name="problem">The borrowed Python problem or benchmark instance.</param>
                /// <param name="parameters">The named parameter values passed to the Python trial or benchmark.</param>
                /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
                /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
                internal static PyObject evaluate(PyObject problem, PyObject parameters)
                {
                    using var evaluate = problem.GetAttr("evaluate");
                    return evaluate.Invoke(parameters);
                }

                /// <summary>
                ///     <c>problem.evaluate_constraints(params)</c> (OptunaHub ≥ 0.5.0) — the problem's constraint
                ///     values: a <c>dict</c> keyed by constraint name, or, from a benchmark package written for
                ///     an older OptunaHub, a plain sequence. A trial is feasible when every value is ≤ 0. The
                ///     base implementation returns an empty <c>dict</c>. Dispose the result.
                /// </summary>
                /// <param name="problem">The borrowed Python problem or benchmark instance.</param>
                /// <param name="parameters">The named parameter values passed to the Python trial or benchmark.</param>
                /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
                /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
                internal static PyObject evaluate_constraints(PyObject problem, PyObject parameters)
                {
                    using var evaluateConstraints = problem.GetAttr("evaluate_constraints");
                    return evaluateConstraints.Invoke(parameters);
                }

                /// <summary><c>problem.directions</c>. Dispose the result.</summary>
                /// <param name="problem">The borrowed Python problem or benchmark instance.</param>
                /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
                /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
                internal static PyObject directions(PyObject problem) => problem.GetAttr("directions");

                /// <summary><c>problem.search_space</c>. Dispose the result.</summary>
                /// <param name="problem">The borrowed Python problem or benchmark instance.</param>
                /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
                /// <exception cref="global::Python.Runtime.PythonException">The Python attribute lookup, constructor, or operation fails.</exception>
                internal static PyObject search_space(PyObject problem) => problem.GetAttr("search_space");
            }
        }
    }

    /// <summary><c>import warnings</c> — the fixed warning call the bridge makes, reached as <c>Python.warnings.*</c>.</summary>
    internal static class warnings
    {
        /// <summary>
        ///     <c>warnings.warn(message, category)</c> — raise a genuine Python warning, so a deprecation that
        ///     OptunaHub's own code would raise reaches Python's warning filters and stderr exactly as it does
        ///     when the problem runs in Python. Honors any active <c>warnings</c> filter, including one that
        ///     turns the warning into an exception (which then propagates as a <see cref="PythonException" />).
        /// </summary>
        /// <param name="message">The warning text, passed verbatim.</param>
        /// <param name="category">The borrowed warning class, e.g. <see cref="builtins.FutureWarning" />.</param>
        /// <exception cref="global::Python.Runtime.PythonException">A warnings filter escalates the warning to an error, or the call itself fails.</exception>
        internal static void warn(string message, PyObject category)
        {
            PyObject module = PythonRuntime.Import("warnings");   // process-scoped cache: never dispose
            using var warnFn = module.GetAttr("warn");
            using var text = message.ToPython();
            using PyObject none = warnFn.Invoke(text, category);
        }
    }

    /// <summary><c>builtins</c> — the built-in names the bridge passes into fixed calls, reached as <c>Python.builtins.*</c>.</summary>
    internal static class builtins
    {
        /// <summary>The built-in <c>FutureWarning</c> class. Dispose the result.</summary>
        /// <exception cref="global::Python.Runtime.PythonException">The attribute lookup fails.</exception>
        internal static PyObject FutureWarning => PythonRuntime.Builtins.GetAttr("FutureWarning");

        /// <summary>The built-in <c>TypeError</c> class. Dispose the result.</summary>
        /// <exception cref="global::Python.Runtime.PythonException">The attribute lookup fails.</exception>
        internal static PyObject TypeError => PythonRuntime.Builtins.GetAttr("TypeError");

        /// <summary>The built-in <c>ValueError</c> class. Dispose the result.</summary>
        /// <exception cref="global::Python.Runtime.PythonException">The attribute lookup fails.</exception>
        internal static PyObject ValueError => PythonRuntime.Builtins.GetAttr("ValueError");
    }
}
