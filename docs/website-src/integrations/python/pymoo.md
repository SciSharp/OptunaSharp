# pymoo

`OptunaSharp.Interop.Pymoo` exposes eleven pymoo evolutionary algorithms through one
`PymooSampler`. OptunaSharp owns study storage, trial suggestions and objective evaluation; the
embedded Python library performs genetic variation and population survival. This is an OptunaSharp
extension using pymoo's own `ask`/`tell` protocol.

## Install and optimize

```shell
dotnet add package OptunaSharp.Interop.Pymoo
python -m pip install "pymoo==0.6.1.3"
```

The repository's existing bridge documentation identifies pymoo 0.6.1.3 as its developed/tested
baseline. The package targets .NET 8 and .NET 10, shares the [Python.NET runtime](pythonnet.md), and
does not require a Python `optuna_integration` glue package. Its constructor checks that pymoo can
be imported. Compatibility with newer pymoo versions needs the live checks described below.

```csharp
using System;
using OptunaSharp;
using OptunaSharp.Interop.Pymoo;

var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new PymooSampler(PymooAlgorithm.SmsEmoa, populationSize: 20, seed: 0));

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", 0, 1);
    double y = trial.SuggestFloat("y", 0, 1);
    return new[] { x, 1 - x + y };
}, nTrials: 200, nJobs: 1);

foreach (var trial in study.BestTrials)
    Console.WriteLine(string.Join(", ", trial.Values!));
```

`PymooSampler()` defaults to `Nsga2`, population size 50 and an unseeded run. Always give a seed
when reproducibility is required. The constructor requires `populationSize >= 2`, including when
the chosen family ultimately derives its actual population from directions or points.

> **Any thread, any `nJobs`.** The sampler's pymoo calls run on the runtime's
> [Python thread](pythonnet.md#the-python-thread), so it can be driven from any thread and at any
> `nJobs`. Before that, driving it off the main thread crashed the process in scipy's `cdist` (7 of 8
> runs, measured on 2026-11-07).
>
> **macOS on Apple silicon: AGE-MOEA's stack.**
>
> - `AgeMoea` and `AgeMoea2` run pymoo's numba-compiled survival kernels, and the first `tell`
>   JIT-compiles them. On macOS arm64 that needs more stack than a thread .NET creates has by default.
> - The sampler's calls run on the Python thread, which has 16 MB, CPython's own size, so they need
>   nothing more.
> - Running such pymoo code yourself on a .NET thread (the thread pool, `Task.Run`, an ASP.NET request)
>   still needs a thread with a `maxStackSize` of at least 16 MB, or `DOTNET_DefaultStackSize=1000000`
>   (hexadecimal, 16 MB) set before the process starts. OptunaSharp's own `nJobs` threads reserve 16 MB.
> - Windows x64 (measured down to 256 KB) and the Linux x64 CI runner with default stacks are not
>   affected.

## Algorithms and population sizes

| `PymooAlgorithm` | Python class | Population configuration |
|---|---|---|
| `Nsga2` | `NSGA2` | `populationSize` |
| `Nsga3` | `NSGA3` | Derived from reference directions |
| `RNsga2` | `RNSGA2` | `populationSize`; requires `ReferencePoints` |
| `RNsga3` | `RNSGA3` | Derived by pymoo; requires `ReferencePoints` and reads `PopulationPerReferencePoint` |
| `UNsga3` | `UNSGA3` | Derived from reference directions |
| `AgeMoea` | `AGEMOEA` | `populationSize` |
| `AgeMoea2` | `AGEMOEA2` | `populationSize` |
| `Rvea` | `RVEA` | Derived from reference directions |
| `SmsEmoa` | `SMSEMOA` | `populationSize` |
| `Spea2` | `SPEA2` | `populationSize` |
| `Ctaea` | `CTAEA` | Derived from reference directions |

R-NSGA-II uses the supplied population size; it does not share R-NSGA-III's derived-size behavior.
The bridge reads `algorithm.pop_size` from pymoo to determine the generation length, including the
initial independently sampled generation.

## Reference directions and aspiration points

For NSGA-III, U-NSGA-III, RVEA and C-TAEA, configure Das–Dennis directions or Riesz-energy directions:

```csharp
var sampler = new PymooSampler(
    PymooAlgorithm.Nsga3, seed: 0,
    options: new PymooAlgorithmOptions
    {
        ReferenceDirectionsMethod = PymooReferenceDirections.DasDennis,
        NumberOfPartitions = 12
    });
```

Das–Dennis is the default with 12 partitions: for example, this gives 13 directions with two
objectives and 91 with three. For `PymooReferenceDirections.Energy`, set
`NumberOfReferenceDirections`; when omitted it uses `populationSize` as the direction count.

R-NSGA-II and R-NSGA-III require a nonempty `ReferencePoints` matrix, one row per aspiration point and
one column per objective. Write the coordinates in the study's objective directions; maximizing
columns are negated by the bridge for pymoo.

```csharp
var sampler = new PymooSampler(
    PymooAlgorithm.RNsga2, populationSize: 40, seed: 0,
    options: new PymooAlgorithmOptions
    {
        ReferencePoints = new[] { new[] { 0.3, 0.7 }, new[] { 0.7, 0.3 } },
        Epsilon = 0.001
    });
```

R-NSGA-II also reads `Weights` and `ExtremePointsAsReferencePoints` (default `false`). R-NSGA-III
reads `PopulationPerReferencePoint` (default 1) and `Mu` (default 0.05). A missing point matrix or
a row with the wrong objective count is rejected when the algorithm is constructed for sampling.

## Genetic operators and RVEA options

The common options are `CrossoverEta`, `CrossoverProbability`, `MutationEta`,
`MutationProbability` and `EliminateDuplicates`. Explicit crossover settings construct pymoo's SBX;
explicit mutation settings construct its PM. Null settings leave the selected algorithm's own
operator defaults in place. Different algorithms can therefore retain different defaults.

```csharp
var sampler = new PymooSampler(PymooAlgorithm.Nsga2, populationSize: 40, seed: 0,
    options: new PymooAlgorithmOptions
    {
        CrossoverEta = 30,
        CrossoverProbability = 0.9,
        MutationEta = 20,
        MutationProbability = 0.1,
        EliminateDuplicates = true
    });
```

The probability properties are passed to pymoo's `prob` arguments. In particular,
`MutationProbability` does not expose a separate PM `prob_var` argument; consult the backend's
[mutation API](https://pymoo.org/operators/mutation.html) when translating Python operator settings.
`EliminateDuplicates: false` explicitly disables duplicate elimination; `true` leaves the algorithm's
default constructor behavior. There is no `n_offsprings` setting because replay expects a full
population from each `ask`.

RVEA also receives `RveaAlpha` (default 2) and `RveaAdaptFrequency` (default 0.1). The bridge fixes
pymoo's termination to `("n_gen", 1_000_000_000)` and controls the budget using OptunaSharp trials.
There is no public pymoo generation-horizon option. This matters for RVEA's horizon-dependent penalty
and adaptation schedule: `nTrials` does not become pymoo's `n_gen`, so a bounded `pymoo.minimize`
example is not an equivalent configuration.

## Constraints

pymoo's inequality convention is `G <= 0` for feasibility, matching OptunaSharp. Record constraints
on every completed trial, including the initial generation:

```csharp
var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new PymooSampler(PymooAlgorithm.Nsga2, populationSize: 20, seed: 0));
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", 0, 1);
    double y = trial.SuggestFloat("y", 0, 1);
    trial.SetConstraint("budget", x + y - 1.0);
    return new[] { x, 1 - x + y };
}, nTrials: 200, nJobs: 1);
```

Alternatively, pass `constraintsFunc: Func<FrozenTrial, IEnumerable<double>>` to compute constraints
after completed trials. The first completed trial determines the constraint names used for the
matrix. Later replayed trials must contain those names; adding names later does not add new columns.
Keep the schema stable. Support for a particular constraint problem also depends on the chosen
pymoo algorithm; C-TAEA is the specifically constrained two-archive method.

## Replay and parameter geometry

The first full population of compatible completed trials comes from `independentSampler` (a seeded
`RandomSampler` by default). The sampler then creates a fresh pymoo algorithm, injects that generation's
parameters as its initial population, and replays full completed generations by assigning their
stored objective matrix `F` and constraint matrix `G` before `tell`. One row of the next `ask`
population becomes the next OptunaSharp trial. Python objects are disposed within the GIL scope;
the sampler itself is not disposable.

Numeric parameters are transformed to a continuous unit cube, including log scales and grids.
Categoricals become one-hot columns; decoding uses the largest column, while integers and stepped
floats are rounded to their legal values. This supports valid Optuna parameters but changes the
geometry relative to a pymoo mixed-variable operator. Conditional parameters outside the intersection
are independently sampled. Single-value distributions are handled by the trial.

Replay depends on a consistent seed, configuration, history and parameter space. No live Python
algorithm is serialized into storage. Rebuilding from trials avoids a Python-object checkpoint, but
does not restore arbitrary independent-sampler state or make every interruption and parameter-space
change equivalent to an uninterrupted run.

Sequential successful generations are the clearest replay contract. The candidate index counts
compatible trials of any state after the last replayed generation and wraps modulo population size.
Oversubscribed parallel runs can therefore return repeated individuals. Do not infer deterministic
asynchronous, failed-trial or pruned-trial parity from the sequential tests. Use one sampler instance
per study and `nJobs: 1` for reference comparisons.

## Excluded algorithms and validation

MOEA/D and its parallel variant are not exposed: their steady-state cadence does not fit this
uniform full-population replay. Dynamic DNSGA-II and KGB require reevaluating points through a Python
problem, while this bridge evaluates in C#. pymoo's single-objective algorithm families, gradient
methods and arbitrary custom operators are also outside this API. Some exposed evolutionary methods
can still be used with one objective; that does not add all single-objective pymoo algorithms.

The [behavioral tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/PymooSamplerParityTests.cs)
exercise all eleven algorithms on a two-objective front. The same fixture checks selected seeded
repeat runs, one- and three-objective examples, maximize directions, grids, logarithmic and categorical
parameters, dynamic fallback, duplicate settings and required reference points. Direct numerical
replay comparisons use NSGA-II configurations, including custom operators and constraints, with
`1e-9` tolerance. This is not direct numerical parity coverage of every algorithm/configuration.

The comparison starts with the same injected generation zero. It does not establish equality to
`pymoo.minimize`, which can use different initialization, transforms and termination. Nor does it
establish equality to OptunaSharp's managed algorithms: for example, this pymoo SMS-EMOA path is
generational, while the managed `SmsEmoaSampler` uses a steady-state variant.

Live tests become inconclusive without pymoo. Separate
[API tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/PymooApiUnitTests.cs)
check the enum and construction families and the population-size guard.
See [sampler/replay source](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.Pymoo/PymooSampler.cs),
[algorithm options](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.Pymoo/PymooAlgorithm.cs)
and [Python setup](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.Pymoo/Pythonic.cs).
