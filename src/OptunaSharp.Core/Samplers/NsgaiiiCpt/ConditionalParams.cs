using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Storages;

namespace OptunaSharp.Samplers.NsgaiiiCpt;

/// <summary>
///     Declares <b>conditional parameters</b> — parameters that only exist while an <i>activator</i>
///     parameter holds one of a set of values — the rule set consumed by <see cref="NSGAIIICptSampler" />
///     (NSGA-III-CPT), the only sampler that accepts it. An <b>OptunaSharp extension</b> with no Optuna
///     analog: opt-in, and the plain genetic samplers (<see cref="NSGAIISampler" /> /
///     <see cref="NSGAIIISampler" />) are bit-exact Optuna ports that do not take it.
///
///     <para><b>Why it exists.</b> Optuna's define-by-run API lets an objective suggest a parameter under
///     a branch only (<c>if (optimizer == "adam") t.SuggestFloat("beta1", …)</c>), but the GA samplers
///     evolve the <i>intersection</i> search space — the parameters present in <i>every</i> completed
///     trial. A parameter absent from even one trial leaves the intersection for good and is resampled
///     uniformly at random by the independent sampler on every later trial: no crossover, no mutation,
///     no inheritance — no evolutionary pressure at all. Declaring the parameter here tells the sampler
///     that its absence is <i>expected</i> (it was inactive), so it stays in the relative search space and
///     is evolved like any other parameter whenever it is active, and is simply not present — never
///     blended, never counted — whenever it is not.</para>
///
///     <para><b>Rules.</b> <c>When(activator, values…).Activates(parameters…)</c> declares that each of
///     <c>parameters</c> is active only while <c>activator</c> holds one of <c>values</c> (Python <c>==</c>
///     semantics via <see cref="CategoricalChoice" />: <c>1 == 1.0 == true</c>, strings ordinal, a lone
///     <c>null</c> is <c>None</c>); <c>When(activator, predicate)</c> takes an arbitrary test instead — the
///     value arrives as the trial stores it: a boxed <see cref="long" /> for int parameters, a
///     <see cref="double" /> for float parameters, the raw choice value for categorical ones. Several
///     rules may name the same activator (they OR: any admissible value suffices) and a parameter may be
///     declared under several activators (they AND: every activator must admit). An activator may itself
///     be conditional (chains); self-activation and cycles are rejected at declaration.</para>
///
///     <para><b>What the sampler does with it.</b> (1) Relative search space: a conditional parameter joins
///     it once any completed trial has suggested it, provided every trial that did used the same
///     distribution (a changing distribution excludes it, exactly as the intersection would).
///     (2) Crossover: the categorical uniform crossover and the numerical crossover operator run over the
///     parameters the selected parents all carry; a parameter carried by only some parents is inherited
///     from one of those; one carried by none stays absent. (3) After crossover (or the no-crossover
///     parent copy) every conditional parameter whose activators say "inactive" is removed from the
///     child — it neither counts toward the default <c>1 / n_params</c> mutation probability nor reaches
///     the trial. (4) The objective's own <c>Suggest*</c> calls stay the final word: a proposed parameter
///     the objective never suggests is never recorded, and a suggested parameter the sampler did not
///     propose is drawn independently — the usual define-by-run contract. The sampler warns once per
///     parameter when a completed trial contradicts the rules (present while inactive, or absent while
///     active), which usually means the objective's branches and the declaration drifted apart.</para>
///
///     <example><code>
///     var rules = new ConditionalParams()
///         .When("optimizer", "adam", "adamw").Activates("beta1", "beta2")
///         .When("optimizer", "sgd").Activates("momentum")
///         .When("n_layers", v =&gt; (long)v! &gt;= 3).Activates("width_3");
///     var sampler = new NSGAIIICptSampler(populationSize: 32, seed: 0, conditionalParams: rules);
///     </code></example>
/// </summary>
public sealed class ConditionalParams
{
    /// <summary>One admissible-value test on an activator plus its display form for diagnostics.</summary>
    /// <param name="Predicate">The test that determines whether an external activator value is admissible.</param>
    /// <param name="Display">The readable condition text used in activation diagnostics.</param>
    internal sealed record Condition(Func<object?, bool> Predicate, string Display);

    // dependent -> activator -> conditions. OR within an activator, AND across activators. Insertion
    // ordered so ToString, the dependency walk and every diagnostic are deterministic functions of the
    // declarations.
    private readonly OrderedDict<string, OrderedDict<string, List<Condition>>> _rules = new();
    private List<string>? _order;

    /// <summary>Trial system-attribute key under which a trial's dormant (latent) conditional values live.</summary>
    internal const string DormantAttrKey = "ConditionalParams:dormant";

    /// <summary>
    ///     <b>Opt-in divergence from Optuna (off by default).</b> When <c>true</c>, a conditional parameter
    ///     that is <i>inactive</i> for a trial does not vanish: its last active value is parked in a latent
    ///     side channel (a trial system attribute — never suggested, never scored, never crossed or
    ///     mutated, so still out of every equation), and when the parameter's branch reactivates it
    ///     <b>resumes from that preserved value instead of a fresh random draw</b>. The hyperparameter list
    ///     stays fluid — genes move between the active genome and the dormant channel — but the fluidity is
    ///     non-destructive: a branch toggling off and on again does not throw away what was learned about
    ///     it. Optuna always re-samples, so this only takes effect when you ask for it.
    ///
    ///     <para>A resumed gene is an ordinary active gene again from that point (free to be crossed and
    ///     mutated); dormancy freezes the value only while the branch is off. A gene that has never been
    ///     active has no latent value to resume, so its first activation is still a normal draw. Requires
    ///     the chosen storage to round-trip a <c>{name: value}</c> map through trial system attributes
    ///     (in-memory and JSON storages do).</para>
    /// </summary>
    public bool PreserveDormantValues { get; init; }

    /// <summary>
    ///     Starts a rule: the parameters named by the following <see cref="ActivationBuilder.Activates" />
    ///     are active only while <paramref name="activator" /> holds one of <paramref name="values" />.
    /// </summary>
    /// <param name="activator">The parameter whose value gates the others.</param>
    /// <param name="values">
    ///     Admissible activator values, compared with Python <c>==</c> semantics (<c>1 == 1.0 == true</c>,
    ///     strings ordinal, NaN matches NaN as Optuna's categorical lookup does). A <c>null</c> element is
    ///     the choice <c>None</c> — spell it <c>(object?)null</c>, since a bare <c>null</c> literal is
    ///     ambiguous with the predicate overload.
    /// </param>
    /// <returns>A builder that attaches dependent parameters to the supplied activator condition.</returns>
    /// <exception cref="System.ArgumentNullException">The activator name is null.</exception>
    /// <exception cref="NumSharp.ValueError">The activator name is blank, or no admissible values are supplied.</exception>
    public ActivationBuilder When(string activator, params object?[]? values)
    {
        ValidateName(activator, nameof(activator));
        // A null array (`When("a", (object?[]?)null)`) means the one admissible value None, not "no values".
        values ??= new object?[] { null };
        if (values.Length == 0)
            throw new ValueError($"`When(\"{activator}\", ...)` needs at least one admissible value.");
        var admissible = (object?[])values.Clone();
        string display = "in {" + string.Join(", ", admissible.Select(PythonRepr.Repr)) + "}";
        return new ActivationBuilder(
            this, activator, new Condition(v => admissible.Any(a => ValuesEqual(a, v)), display));
    }

    /// <summary>
    ///     Starts a rule with an arbitrary test on the activator's value — the value arrives as the trial
    ///     stores it: a boxed <see cref="long" /> for int parameters, a <see cref="double" /> for float
    ///     parameters, the raw choice value (<c>string</c>/<c>long</c>/<c>double</c>/<c>bool</c>/<c>null</c>)
    ///     for categorical ones.
    /// </summary>
    /// <param name="activator">The name of the parameter whose value controls whether dependent parameters are active.</param>
    /// <param name="predicate">The admissible-value test applied to the activator's external value.</param>
    /// <returns>A builder that attaches dependent parameters to the supplied activator condition.</returns>
    /// <exception cref="System.ArgumentNullException">The activator name or predicate is null.</exception>
    /// <exception cref="NumSharp.ValueError">The activator name is blank.</exception>
    public ActivationBuilder When(string activator, Func<object?, bool> predicate)
    {
        ValidateName(activator, nameof(activator));
        if (predicate is null)
            throw new ArgumentNullException(nameof(predicate));
        return new ActivationBuilder(this, activator, new Condition(predicate, "satisfies predicate"));
    }

    /// <summary>The second half of a <see cref="When(string, object?[])" /> rule.</summary>
    public readonly struct ActivationBuilder
    {
        private readonly ConditionalParams _owner;
        private readonly string _activator;
        private readonly Condition _condition;

        /// <summary>Binds the fluent declaration to its owning rule set and activator condition.</summary>
        /// <param name="owner">The activation-rule collection that will receive the dependent-parameter declarations.</param>
        /// <param name="activator">The name of the parameter whose value controls whether dependent parameters are active.</param>
        /// <param name="condition">The activation predicate and its diagnostic display text.</param>
        internal ActivationBuilder(ConditionalParams owner, string activator, Condition condition)
        {
            _owner = owner;
            _activator = activator;
            _condition = condition;
        }

        /// <summary>Names the parameters the rule activates (one or more); returns the owner for chaining.</summary>
        /// <param name="parameters">One or more dependent parameter names to attach to this activator rule.</param>
        /// <returns>The owning rule collection, enabling another fluent activation declaration.</returns>
        /// <exception cref="NumSharp.ValueError">The dependent list is empty, a name is blank, or a declaration introduces self-activation or a cycle.</exception>
        /// <exception cref="System.ArgumentNullException">A dependent parameter name is null.</exception>
        public ConditionalParams Activates(params string[] parameters)
        {
            if (parameters is null || parameters.Length == 0)
                throw new ValueError("`Activates(...)` needs at least one parameter name.");
            foreach (var parameter in parameters)
                _owner.Add(parameter, _activator, _condition);
            return _owner;
        }
    }

    /// <summary>The conditional (activated) parameter names, in declaration order.</summary>
    public IReadOnlyCollection<string> Parameters => _rules.Keys;

    /// <summary>Whether <paramref name="name" /> was declared conditional.</summary>
    /// <param name="name">The parameter or attribute name to look up.</param>
    /// <returns>True when the parameter has at least one declared activator.</returns>
    public bool IsConditional(string name) => _rules.ContainsKey(name);

    /// <summary>The activators <paramref name="name" /> was declared under (empty when unconditional).</summary>
    /// <param name="name">The parameter or attribute name to look up.</param>
    /// <returns>The declared activator names for this parameter, or an empty collection for an unconditional parameter.</returns>
    public IReadOnlyCollection<string> ActivatorsOf(string name)
        => _rules.TryGetValue(name, out var byActivator) ? byActivator.Keys : Array.Empty<string>();

    /// <summary>
    ///     Whether <paramref name="name" /> is active for a complete parameter assignment such as
    ///     <see cref="FrozenTrial.Params" />: unconditional parameters always are; a conditional one is
    ///     active when every activator it was declared under is present, admits the value it holds, and is
    ///     itself active.
    /// </summary>
    /// <param name="name">The parameter or attribute name to look up.</param>
    /// <param name="params">The complete external parameter assignment against which activation is evaluated.</param>
    /// <returns>True when the current assignment satisfies every declared activator requirement.</returns>
    /// <exception cref="System.ArgumentNullException">The parameter assignment is null or the lookup name is null.</exception>
    /// <exception cref="System.Exception">A caller-supplied activation predicate throws while evaluating an external parameter value.</exception>
    public bool IsActive(string name, IReadOnlyDictionary<string, object?> @params)
    {
        if (@params is null)
            throw new ArgumentNullException(nameof(@params));
        if (!_rules.TryGetValue(name, out var byActivator))
            return true;
        foreach (var (activator, conditions) in byActivator)
        {
            if (!@params.TryGetValue(activator, out var value))
                return false;
            if (!Admits(conditions, value))
                return false;
            if (!IsActive(activator, @params))
                return false;
        }
        return true;
    }

    /// <summary>
    ///     Whether <paramref name="value" /> of <paramref name="activator" /> admits <paramref name="name" />
    ///     under the declared rules — <c>true</c> when no rule links the two. Unlike <see cref="IsActive" />
    ///     this judges a single activator, so it can be asked of a partial assignment such as a sampler's
    ///     proposal in which other activators are still to be drawn.
    /// </summary>
    /// <param name="name">The parameter or attribute name to look up.</param>
    /// <param name="activator">The name of the parameter whose value controls whether dependent parameters are active.</param>
    /// <param name="value">The proposed external value of the named activator.</param>
    /// <returns>True when no rule links the parameters or at least one condition accepts the activator value.</returns>
    /// <exception cref="System.Exception">A caller-supplied activation predicate throws while evaluating an external parameter value.</exception>
    internal bool ActivatorAdmits(string name, string activator, object? value)
        => !_rules.TryGetValue(name, out var byActivator)
           || !byActivator.TryGetValue(activator, out var conditions)
           || Admits(conditions, value);

    /// <summary>
    ///     Tri-state activity of <paramref name="name" /> against a <i>partial</i> assignment (a sampler's
    ///     in-progress proposal): <c>true</c>/<c>false</c> as <see cref="IsActive" />, or <c>null</c> when an
    ///     activator it depends on is not present in <paramref name="assignment" /> yet (e.g. dropped for
    ///     mutation and still to be drawn), so activity cannot be decided here.
    /// </summary>
    /// <param name="name">The parameter or attribute name to look up.</param>
    /// <param name="assignment">The proposed external parameter values used to resolve activation rules.</param>
    /// <returns>True or false when activation is resolved, or null when an activator value is unavailable.</returns>
    /// <exception cref="System.Exception">A caller-supplied activation predicate throws while evaluating an external parameter value.</exception>
    internal bool? ResolveActive(string name, IReadOnlyDictionary<string, object?> assignment)
    {
        if (!_rules.TryGetValue(name, out var byActivator))
            return true;
        foreach (var (activator, conditions) in byActivator)
        {
            if (!assignment.TryGetValue(activator, out var value))
                return null;
            if (!Admits(conditions, value))
                return false;
            bool? activatorActive = ResolveActive(activator, assignment);
            if (activatorActive is false)
                return false;
            if (activatorActive is null)
                return null;
        }
        return true;
    }

    /// <summary>
    ///     The dormant-value carry (only runs when <see cref="PreserveDormantValues" />). See that
    ///     property: parks the value of every conditional parameter inactive for <paramref name="child" />
    ///     in the trial's <see cref="DormantAttrKey" /> system attribute (out of every equation), and
    ///     seeds any parameter active for <paramref name="child" /> that no parent supplied from the
    ///     latest latent value — so a reactivating branch resumes instead of drawing at random. Consumes
    ///     no sampler RNG; mutates <paramref name="child" /> in place and writes at most one system attr.
    /// </summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="trial">The trial whose recorded state and values are used.</param>
    /// <param name="child">The candidate parameter assignment whose inactive entries are removed.</param>
    /// <param name="parentPopulation">The parent trials contributing active and dormant values to a candidate.</param>
    /// <param name="searchSpace">The parameter distributions in the column order used by the transform or model.</param>
    /// <exception cref="System.Exception">A caller-supplied activation predicate throws while evaluating an external parameter value.</exception>
    internal void ApplyDormantCarry(
        Study study,
        FrozenTrial trial,
        OrderedDict<string, object?> child,
        IReadOnlyList<FrozenTrial> parentPopulation,
        IReadOnlyDictionary<string, BaseDistribution> searchSpace)
    {
        if (!PreserveDormantValues)
            return;

        var latent = GatherLatent(study, parentPopulation);
        if (latent.Count == 0)
            return;   // nothing learned on any branch yet (e.g. generation 0): nothing to carry.

        var dormant = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var name in Order)
        {
            bool? active = ResolveActive(name, child);
            if (active is null)
                continue;   // an activator is still undecided for this child: leave it to independent sampling.

            if (active is true)
            {
                // The branch is on but no selected parent carried the gene: resume the latent value instead
                // of letting it fall through to a fresh random draw, provided it is still in range.
                if (!child.ContainsKey(name) && latent.TryGetValue(name, out var value) && InRange(searchSpace, name, value))
                    child[name] = value;
            }
            else if (latent.TryGetValue(name, out var value))
            {
                // The branch is off: park the value, untouched, out of the active genome.
                dormant[name] = value;
            }
        }

        if (dormant.Count > 0)
            study.Storage.SetTrialSystemAttr(trial.TrialId, DormantAttrKey, dormant);
    }

    // Single-entry latent cache. The parent population changes once per GENERATION but GatherLatent runs on
    // every ASK, and its cost is |parents| × |rules| dictionary probes (tens of thousands per ask for a real
    // genome) — measured as the single largest sampler cost of a cheap-objective study. Parents are
    // completed trials: their params are immutable and their dormant attr was written before completion, so
    // the same parents ⇒ byte-identical latent map. The key is the study (storage object and id) plus every
    // parent's number, storage id and start time in ascending number order, compared in full (no hashing
    // shortcut), so a hit can never alias a different population. Numbers alone did alias: a rule set reused
    // with its sampler for a second study (every fresh in-memory storage starts at id 0) handed the new
    // generation-1 parents the latent values of the old study's parents with the same numbers (pre-release
    // review item H3). One immutable entry, swapped by a single reference write, so concurrent asks never
    // read a key from one generation with the map of another.
    private LatentEntry? _latent;

    /// <summary>One parent's identity inside a latent-cache key.</summary>
    /// <param name="Number">The parent's trial number.</param>
    /// <param name="TrialId">The parent's storage identifier.</param>
    /// <param name="DatetimeStart">The parent's start time; set for every completed trial.</param>
    private readonly record struct ParentKey(int Number, int TrialId, DateTime? DatetimeStart);

    /// <summary>A memoized latent map and the exact study and parents it was gathered from.</summary>
    /// <param name="Storage">The storage of the study, held weakly so the cache does not keep a finished study alive.</param>
    /// <param name="StudyId">The study's identifier within <paramref name="Storage" />.</param>
    /// <param name="Parents">The parents' identities in ascending trial-number order.</param>
    /// <param name="Latent">The gathered latent map; shared with callers, so never mutated.</param>
    private sealed record LatentEntry(
        WeakReference<BaseStorage> Storage,
        int StudyId,
        ParentKey[] Parents,
        Dictionary<string, object?> Latent);

    /// <summary>
    ///     The latest latent value of each conditional parameter across <paramref name="parentPopulation" />
    ///     — a parent's own recorded value where the branch was active, or the value it had parked while
    ///     dormant — with the most recent parent (by trial number) winning.
    /// </summary>
    /// <remarks>
    ///     Memoized on the study and the exact parent set (<c>_latent</c>): repeated asks within one generation
    ///     return the SAME dictionary instance, so callers must treat the result as read-only.
    /// </remarks>
    /// <param name="study">The study the parents belong to; part of the memo key, since trial numbers repeat across studies.</param>
    /// <param name="parentPopulation">The parent trials contributing active and dormant values to a candidate.</param>
    /// <returns>The dormant parameter values collected from the supplied parent trials.</returns>
    private Dictionary<string, object?> GatherLatent(Study study, IReadOnlyList<FrozenTrial> parentPopulation)
    {
        var key = new ParentKey[parentPopulation.Count];
        for (int i = 0; i < parentPopulation.Count; i++)
        {
            FrozenTrial parent = parentPopulation[i];
            key[i] = new ParentKey(parent.Number, parent.TrialId, parent.DatetimeStart);
        }
        Array.Sort(key, static (a, b) => a.Number.CompareTo(b.Number));

        // Read the entry once: another thread may swap it while this one compares.
        LatentEntry? entry = _latent;
        if (entry is not null
            && entry.StudyId == study.StudyId
            && entry.Storage.TryGetTarget(out BaseStorage? storage)
            && ReferenceEquals(storage, study.Storage)
            && entry.Parents.AsSpan().SequenceEqual(key))
            return entry.Latent;

        var latent = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var trial in parentPopulation.OrderBy(t => t.Number))
        {
            foreach (var name in _rules.Keys)
            {
                if (trial.ParamsDict.TryGetValue(name, out var active))
                    latent[name] = active;
            }
            if (trial.SystemAttrsDict.TryGetValue(DormantAttrKey, out var raw) && raw is not null)
            {
                foreach (var (name, value) in CoerceDormant(raw))
                {
                    if (_rules.ContainsKey(name))
                        latent[name] = value;
                }
            }
        }
        _latent = new LatentEntry(
            new WeakReference<BaseStorage>(study.Storage), study.StudyId, key, latent);
        return latent;
    }

    /// <summary>Whether <paramref name="value" /> is a legal value of <paramref name="name" />'s distribution.</summary>
    /// <param name="searchSpace">The parameter distributions in the column order used by the transform or model.</param>
    /// <param name="name">The parameter or attribute name to look up.</param>
    /// <param name="value">The dormant external parameter value whose current distribution support is checked.</param>
    /// <returns>True when the external value is representable within the supplied distribution.</returns>
    private static bool InRange(IReadOnlyDictionary<string, BaseDistribution> searchSpace, string name, object? value)
    {
        if (!searchSpace.TryGetValue(name, out var distribution))
            return false;
        try
        {
            return distribution.Contains(distribution.ToInternalRepr(value));
        }
        catch (Exception)
        {
            return false;   // a value no longer representable in this distribution is not resumable.
        }
    }

    /// <summary>Reads back a stored dormant map regardless of how the storage round-tripped it.</summary>
    /// <param name="raw">The stored dormant-value object to interpret as a parameter dictionary.</param>
    /// <returns>The stored dormant parameter/value entries that can be read as a mapping.</returns>
    private static IEnumerable<KeyValuePair<string, object?>> CoerceDormant(object raw)
    {
        switch (raw)
        {
            case IReadOnlyDictionary<string, object?> typed:
                return typed;
            case System.Collections.IDictionary dict:
                var pairs = new List<KeyValuePair<string, object?>>(dict.Count);
                foreach (System.Collections.DictionaryEntry entry in dict)
                    pairs.Add(new KeyValuePair<string, object?>(entry.Key?.ToString() ?? "", entry.Value));
                return pairs;
            default:
                return Array.Empty<KeyValuePair<string, object?>>();
        }
    }

    /// <summary>Conditional parameter names ordered so that every activator precedes what it activates.</summary>
    internal IReadOnlyList<string> Order => _order ??= ComputeOrder();

    /// <summary>
    ///     Removes from <paramref name="child" /> (a child individual under construction, keyed by the
    ///     relative search space) every conditional parameter that is not active for it. Walks in
    ///     dependency order, so a pruned activator takes its dependents with it. An activator missing from
    ///     the child counts as <i>inactive</i> when the relative search space manages it (it was pruned, or
    ///     no parent carried it) and as <i>unknown</i> otherwise (single-valued, dynamic, or not yet seen —
    ///     it will be sampled independently), in which case the dependent is kept and the objective's own
    ///     suggest calls decide.
    /// </summary>
    /// <param name="child">The candidate parameter assignment whose inactive entries are removed.</param>
    /// <param name="relativeSearchSpace">The distributions of parameters currently eligible for relative sampling.</param>
    /// <exception cref="System.Exception">A caller-supplied activation predicate throws while evaluating an external parameter value.</exception>
    internal void Prune(
        OrderedDict<string, object?> child, IReadOnlyDictionary<string, BaseDistribution> relativeSearchSpace)
    {
        foreach (var name in Order)
        {
            if (!child.ContainsKey(name))
                continue;
            if (!ShouldKeep(name, child, relativeSearchSpace))
                child.Remove(name);
        }
    }

    /// <summary>Returns true when the candidate parameter should remain active under the declared rules.</summary>
    /// <param name="name">The parameter or attribute name to look up.</param>
    /// <param name="child">The candidate parameter assignment whose inactive entries are removed.</param>
    /// <param name="relativeSearchSpace">The distributions of parameters currently eligible for relative sampling.</param>
    /// <returns>True when the candidate parameter should remain active under the declared rules.</returns>
    private bool ShouldKeep(
        string name, OrderedDict<string, object?> child, IReadOnlyDictionary<string, BaseDistribution> relativeSearchSpace)
    {
        foreach (var (activator, conditions) in _rules[name])
        {
            if (child.TryGetValue(activator, out var value))
            {
                if (!Admits(conditions, value))
                    return false;
            }
            else if (relativeSearchSpace.ContainsKey(activator))
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>
    ///     Every way a complete trial's parameter set disagrees with these rules — a conditional parameter
    ///     present while inactive, or absent while active — as (parameter, message) pairs. Empty when the
    ///     trial agrees. Used to warn once per parameter that the rules and the objective's suggest calls
    ///     differ.
    /// </summary>
    /// <param name="trial">The trial whose recorded state and values are used.</param>
    /// <returns>Parameter names and diagnostics for trial contents inconsistent with declared activation rules.</returns>
    /// <exception cref="System.Exception">A caller-supplied activation predicate throws while evaluating an external parameter value.</exception>
    internal IEnumerable<(string Name, string Message)> Contradictions(FrozenTrial trial)
    {
        foreach (var name in Order)
        {
            bool present = trial.ParamsDict.ContainsKey(name);
            bool active = IsActive(name, trial.ParamsDict);
            if (present == active)
                continue;
            yield return (name, present
                ? $"The conditional parameter `{name}` is present in Trial#{trial.Number} although its rule " +
                  $"({Describe(name)}) says it is inactive there"
                : $"The conditional parameter `{name}` is absent from Trial#{trial.Number} although its rule " +
                  $"({Describe(name)}) says it is active there");
        }
    }

    /// <summary>Renders <paramref name="name" />'s rule: <c>`a` in {1, 2} and `b` satisfies predicate</c>.</summary>
    /// <param name="name">The parameter or attribute name to look up.</param>
    /// <returns>The readable activation requirements of the selected parameter.</returns>
    private string Describe(string name)
        => string.Join(" and ", _rules[name].Select(kv =>
            $"`{kv.Key}` " + string.Join(" or ", kv.Value.Select(c => c.Display))));

    /// <summary>Returns a deterministic textual description of the declared activation rules.</summary>
    /// <returns>A deterministic textual description of the declared activation rules.</returns>
    public override string ToString()
        => "ConditionalParams(" + string.Join("; ", _rules.Keys.Select(name => $"`{name}` when {Describe(name)}")) + ")";

    /// <summary>Adds dependent-parameter rules after validating names and rejecting self-dependency or cycles.</summary>
    /// <param name="parameter">The dependent parameter whose activation or stored value is being checked.</param>
    /// <param name="activator">The name of the parameter whose value controls whether dependent parameters are active.</param>
    /// <param name="condition">The activation predicate and its diagnostic display text.</param>
    /// <exception cref="NumSharp.ValueError">The dependent name is blank, self-activation is requested, or the new edge would create a dependency cycle.</exception>
    /// <exception cref="System.ArgumentNullException">The dependent parameter name is null.</exception>
    private void Add(string parameter, string activator, Condition condition)
    {
        ValidateName(parameter, nameof(parameter));
        if (string.Equals(parameter, activator, StringComparison.Ordinal))
            throw new ValueError($"Parameter `{parameter}` cannot activate itself.");
        if (DependsOn(activator, parameter))
            throw new ValueError(
                $"Declaring `{parameter}` as activated by `{activator}` would create a cycle: " +
                $"`{activator}` already depends on `{parameter}`.");

        if (!_rules.TryGetValue(parameter, out var byActivator))
            _rules[parameter] = byActivator = new OrderedDict<string, List<Condition>>();
        if (!byActivator.TryGetValue(activator, out var conditions))
            byActivator[activator] = conditions = new List<Condition>();
        conditions.Add(condition);
        _order = null;
    }

    /// <summary>Whether <paramref name="name" /> is (transitively) activated by <paramref name="candidate" />.</summary>
    /// <param name="name">The parameter or attribute name to look up.</param>
    /// <param name="candidate">The possible transitive activator whose reachability is tested.</param>
    /// <returns>True when following activator dependencies reaches the requested parameter.</returns>
    private bool DependsOn(string name, string candidate)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var stack = new Stack<string>();
        stack.Push(name);
        while (stack.Count > 0)
        {
            var current = stack.Pop();
            if (!_rules.TryGetValue(current, out var byActivator))
                continue;
            foreach (var activator in byActivator.Keys)
            {
                if (string.Equals(activator, candidate, StringComparison.Ordinal))
                    return true;
                if (seen.Add(activator))
                    stack.Push(activator);
            }
        }
        return false;
    }

    /// <summary>Returns a dependency order placing activators before their dependent parameters.</summary>
    /// <returns>A dependency order placing activators before their dependent parameters.</returns>
    private List<string> ComputeOrder()
    {
        // Kahn's algorithm over activator -> dependent edges, ties broken by first mention, so the order
        // is a deterministic function of the declarations. Cycles cannot exist (rejected in Add).
        var nodes = new List<string>();
        var indegree = new Dictionary<string, int>(StringComparer.Ordinal);
        var dependents = new Dictionary<string, List<string>>(StringComparer.Ordinal);

        void Touch(string node)
        {
            if (!indegree.TryAdd(node, 0))
                return;
            nodes.Add(node);
            dependents[node] = new List<string>();
        }

        foreach (var (dependent, byActivator) in _rules)
        {
            foreach (var activator in byActivator.Keys)
                Touch(activator);
            Touch(dependent);
            foreach (var activator in byActivator.Keys)
            {
                dependents[activator].Add(dependent);
                indegree[dependent] += 1;
            }
        }

        var order = new List<string>(_rules.Count);
        var ready = new Queue<string>(nodes.Where(node => indegree[node] == 0));
        while (ready.Count > 0)
        {
            var node = ready.Dequeue();
            if (_rules.ContainsKey(node))
                order.Add(node);
            foreach (var dependent in dependents[node])
            {
                if (--indegree[dependent] == 0)
                    ready.Enqueue(dependent);
            }
        }

        System.Diagnostics.Debug.Assert(order.Count == _rules.Count, "Cycles are rejected at declaration.");
        return order;
    }

    /// <summary>Returns true when at least one of the alternative activation conditions accepts the value.</summary>
    /// <param name="conditions">The alternative admissible-value tests for one activator.</param>
    /// <param name="value">The external activator value passed to each alternative predicate.</param>
    /// <returns>True when at least one of the alternative activation conditions accepts the value.</returns>
    private static bool Admits(List<Condition> conditions, object? value)
    {
        foreach (var condition in conditions)
        {
            if (condition.Predicate(value))
                return true;
        }
        return false;
    }

    /// <summary>Rejects a missing or blank parameter name before it can enter the dependency graph.</summary>
    /// <param name="name">The parameter name to validate before adding it to the dependency graph.</param>
    /// <param name="paramName">The argument name included in an ArgumentNullException diagnostic.</param>
    /// <exception cref="System.ArgumentNullException">The parameter name is null.</exception>
    /// <exception cref="NumSharp.ValueError">The parameter name is empty or whitespace.</exception>
    private static void ValidateName(string name, string paramName)
    {
        if (name is null)
            throw new ArgumentNullException(paramName);
        if (name.Length == 0)
            throw new ValueError("Parameter names must be non-empty.");
    }

    /// <summary>
    ///     Python <c>==</c> between an admissible value and a stored parameter value (NaN matches NaN, as
    ///     Optuna's categorical lookup does); values outside the categorical-choice domain fall back to
    ///     <see cref="object.Equals(object?, object?)" />.
    /// </summary>
    /// <param name="expected">The declared categorical value to compare.</param>
    /// <param name="actual">The observed external value to compare against the declaration.</param>
    /// <returns>True when the values compare equal under the categorical/Python comparison convention.</returns>
    internal static bool ValuesEqual(object? expected, object? actual)
    {
        if (TryChoice(expected, out var e) && TryChoice(actual, out var a))
            return CategoricalChoice.ChoicesEqual(e, a);
        return Equals(expected, actual);
    }

    /// <summary>Returns true when the external value can be represented as a supported categorical choice.</summary>
    /// <param name="value">The external scalar to interpret as a categorical choice.</param>
    /// <param name="choice">Receives the categorical representation when the value can be represented as a categorical choice.</param>
    /// <returns>True when the external value can be represented as a supported categorical choice.</returns>
    private static bool TryChoice(object? value, out CategoricalChoice choice)
    {
        switch (value)
        {
            case null:
            case CategoricalChoice:
            case bool:
            case string:
            case double:
            case float:
            case sbyte:
            case byte:
            case short:
            case ushort:
            case int:
            case uint:
            case long:
                choice = CategoricalChoice.From(value);
                return true;
            case ulong ul when ul <= long.MaxValue:
                choice = CategoricalChoice.From(value);
                return true;
            default:
                choice = default;
                return false;
        }
    }
}
