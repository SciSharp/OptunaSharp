# Distributions

A **distribution** describes the range and shape of a single hyperparameter: its bounds, whether it is
sampled on a log scale, its discretization step, or its set of categorical choices. Every
`Suggest*` call on a [trial](study-and-trials.md) is backed by a distribution, and each is recorded on
the trial so the search space can be reconstructed later. OptunaSharp ports `optuna.distributions`
into the **`OptunaSharp.Distributions`** namespace.

## The three distributions

```csharp
using OptunaSharp.Distributions;

// Float: continuous, optionally log-scaled or stepped (log and step are mutually exclusive).
var lr      = new FloatDistribution(1e-5, 1e-1, log: true);
var dropout = new FloatDistribution(0.0, 0.5, step: 0.1);

// Int: integer range with a step (default 1), optionally log-scaled.
var units   = new IntDistribution(16, 256, log: true);
var layers  = new IntDistribution(1, 5);

// Categorical: an ordered set of choices.
var optimizer = new CategoricalDistribution("adam", "sgd", "rmsprop");
```

| Type | Constructor | Notes |
|------|-------------|-------|
| `FloatDistribution` | `(double low, double high, bool log = false, double? step = null)` | `Low`, `High`, `Log`, `Step` |
| `IntDistribution` | `(long low, long high, bool log = false, long step = 1)` | `Low`, `High`, `Log`, `Step` |
| `CategoricalDistribution` | `(IEnumerable<CategoricalChoice> choices)` / `(params CategoricalChoice[])` | `Choices`, `ChoiceAt(index)` |

You usually never construct these directly — `trial.SuggestFloat/SuggestInt/SuggestCategorical` create
them for you. Construct them explicitly when pre-defining a search space, for example the
`fixedDistributions` argument to `study.Ask(...)` or a `GridSampler` grid.

All three derive from the abstract `BaseDistribution`, which defines the internal/external
representation conversions the samplers use (`ToInternalRepr`, `ToExternalRepr`) and `Single()` (true
when the distribution admits exactly one value).

## Categorical values: `CategoricalChoice`

Categorical choices are not plain objects. Optuna's choices are a Python
`Union[None, bool, int, float, str]` and compare with **Python equality**, where `True == 1`,
`1.0 == 1`, and `NaN != NaN`. C#'s `object.Equals` disagrees on every one of those. OptunaSharp
therefore models a choice as the readonly struct `CategoricalChoice`, which reproduces Python's
semantics exactly — the one place where getting object equality "right" for .NET would be wrong for
Optuna.

```csharp
CategoricalChoice a = true;      // implicit conversion from bool
CategoricalChoice b = 1;         // from int
Console.WriteLine(a == b);       // True  — Python semantics (True == 1)

CategoricalChoice nan = double.NaN;
Console.WriteLine(nan == nan);   // False — NaN != NaN

CategoricalChoice none = CategoricalChoice.None;   // Python None
var made = CategoricalChoice.From(someObject);      // from an arbitrary boxed value
```

Implicit conversions exist from `bool`, `int`, `long`, `double`, and `string`, so the typed
`SuggestCategorical` overloads and collection initializers read naturally. Inspect a value with
`Kind` (`CategoricalChoiceKind.None/Bool/Int/Float/String`), `Value`, `IsNone`, and `IsNaN`.

## Serialization and compatibility

Distributions round-trip to and from Optuna's JSON representation, so a search space written by Python
Optuna is understood by OptunaSharp and vice versa:

```csharp
string json = Distribution.ToJson(lr);
BaseDistribution back = Distribution.FromJson(json);

// Throws when a parameter's distribution changed incompatibly between trials:
Distribution.CheckDistributionCompatibility(oldDist, newDist);
```

## Deprecated distributions

Optuna's pre-3.0 distribution classes are kept as `[Obsolete]` aliases that forward to the modern
three: `UniformDistribution`, `LogUniformDistribution`, `DiscreteUniformDistribution` (→
`FloatDistribution`), and `IntUniformDistribution`, `IntLogUniformDistribution` (→ `IntDistribution`).
Prefer the modern types.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `FloatDistribution(1e-5, 1e-1, log=True)` | `new FloatDistribution(1e-5, 1e-1, log: true)` |
| `IntDistribution(16, 256, step=8)` | `new IntDistribution(16, 256, step: 8)` |
| `CategoricalDistribution(["a", "b"])` | `new CategoricalDistribution("a", "b")` |
| a categorical choice `1 == True` | `CategoricalChoice` equality (`(CategoricalChoice)1 == true`) |
| `distribution_to_json` / `json_to_distribution` | `Distribution.ToJson` / `Distribution.FromJson` |

**Divergence.** Like Optuna, `CategoricalDistribution` cannot distinguish `True` from `1` or `1.0`
from `1` when mapping a value back to its internal index — the choices collapse under Python equality.
This is an accepted upstream behavior, reproduced deliberately and pinned by tests.

**Divergence (kept on purpose).** When a stepped `FloatDistribution`'s range is not divisible by its step, Optuna
lowers `high` to the last grid point with Python's `Decimal` (28 *significant* digits). OptunaSharp uses .NET
`decimal` (28 digits *after the point*, nothing below `1e-28`), and the owner chose on 2026-11-12 to keep it. Ordinary
ranges come out identical. Only tiny magnitudes differ: a bound or step whose shortest repr needs more than 28 decimal
places (a 17-digit value below about `1e-11`) can leave `high` one ULP from Optuna's and `BruteForceSampler`'s grid one
point longer; a bound below `1e-28` counts as zero; and a range whose step count needs more than 28 digits, such as
`(0, 1e20, step: 1e-9)`, is accepted where Optuna raises `decimal.InvalidOperation`. The `FloatDistribution` API docs
list the measured cases.

## See also

[Study and Trials](study-and-trials.md) · [Search Spaces](search-spaces.md) · [Samplers](samplers.md) ·
the [API reference](../api/index.md).
