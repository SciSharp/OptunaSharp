using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.SearchSpace;

/// <summary>
///     Declares <b>conditional parameters</b> — parameters that only exist while an <i>activator</i>
///     parameter holds one of a set of values — for the genetic samplers
///     (<see cref="Samplers.NSGAIISampler" /> / <see cref="Samplers.NSGAIIISampler" />). An
///     <b>OptunaSharp extension</b> with no Optuna analog: opt-in, and the samplers' Optuna-parity path is
///     untouched when none is declared.
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
///     var sampler = new NSGAIIISampler(populationSize: 32, seed: 0, conditionalParams: rules);
///     </code></example>
/// </summary>
public sealed class ConditionalParams
{
    /// <summary>One admissible-value test on an activator plus its display form for diagnostics.</summary>
    internal sealed record Condition(Func<object?, bool> Predicate, string Display);

    // dependent -> activator -> conditions. OR within an activator, AND across activators. Insertion
    // ordered so ToString, the dependency walk and every diagnostic are deterministic functions of the
    // declarations.
    private readonly OrderedDict<string, OrderedDict<string, List<Condition>>> _rules = new();
    private List<string>? _order;

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

        internal ActivationBuilder(ConditionalParams owner, string activator, Condition condition)
        {
            _owner = owner;
            _activator = activator;
            _condition = condition;
        }

        /// <summary>Names the parameters the rule activates (one or more); returns the owner for chaining.</summary>
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
    public bool IsConditional(string name) => _rules.ContainsKey(name);

    /// <summary>The activators <paramref name="name" /> was declared under (empty when unconditional).</summary>
    public IReadOnlyCollection<string> ActivatorsOf(string name)
        => _rules.TryGetValue(name, out var byActivator) ? byActivator.Keys : Array.Empty<string>();

    /// <summary>
    ///     Whether <paramref name="name" /> is active for a complete parameter assignment such as
    ///     <see cref="FrozenTrial.Params" />: unconditional parameters always are; a conditional one is
    ///     active when every activator it was declared under is present, admits the value it holds, and is
    ///     itself active.
    /// </summary>
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
    internal bool ActivatorAdmits(string name, string activator, object? value)
        => !_rules.TryGetValue(name, out var byActivator)
           || !byActivator.TryGetValue(activator, out var conditions)
           || Admits(conditions, value);

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
    private string Describe(string name)
        => string.Join(" and ", _rules[name].Select(kv =>
            $"`{kv.Key}` " + string.Join(" or ", kv.Value.Select(c => c.Display))));

    public override string ToString()
        => "ConditionalParams(" + string.Join("; ", _rules.Keys.Select(name => $"`{name}` when {Describe(name)}")) + ")";

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

    private static bool Admits(List<Condition> conditions, object? value)
    {
        foreach (var condition in conditions)
        {
            if (condition.Predicate(value))
                return true;
        }
        return false;
    }

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
    internal static bool ValuesEqual(object? expected, object? actual)
    {
        if (TryChoice(expected, out var e) && TryChoice(actual, out var a))
            return CategoricalChoice.ChoicesEqual(e, a);
        return Equals(expected, actual);
    }

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
