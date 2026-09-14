# Pythonic search space

OptunaSharp infers the search space *as the objective runs* — the **define-by-run** model. You don't
declare parameters up front; you ask for them with three `Suggest*` methods, and the space is whatever
those calls describe on this execution. In C# that means you use ordinary language constructs —
conditionals, loops, methods — to shape the space.

## The three suggest methods

| Method | For | Optional arguments |
|--------|-----|--------------------|
| `SuggestCategorical` | categorical parameters | — |
| `SuggestInt` | integer parameters | `step`, `log` |
| `SuggestFloat` | floating-point parameters | `step`, `log` |

`step` discretizes the range; `log` samples in the logarithmic domain (they are mutually exclusive):

```csharp
using OptunaSharp;

double Objective(Trial trial)
{
    // Categorical
    string optimizer     = trial.SuggestCategorical("optimizer", new[] { "MomentumSGD", "Adam" });

    // Integer
    long numLayers       = trial.SuggestInt("num_layers", 1, 3);
    long numChannels     = trial.SuggestInt("num_channels", 32, 512, log: true);   // log scale
    long numUnits        = trial.SuggestInt("num_units", 10, 100, step: 5);        // discretized

    // Floating point
    double dropoutRate   = trial.SuggestFloat("dropout_rate", 0.0, 1.0);
    double learningRate  = trial.SuggestFloat("learning_rate", 1e-5, 1e-2, log: true);   // log scale
    double dropPathRate  = trial.SuggestFloat("drop_path_rate", 0.0, 1.0, step: 0.1);    // discretized

    return Train(optimizer, numLayers, numChannels, numUnits, dropoutRate, learningRate, dropPathRate);
}
```

See [Distributions](../distributions.md) and [Search Spaces](../search-spaces.md) for the full
semantics, including `CategoricalChoice` Python-equality rules.

## Defining spaces with control flow

Because the space is built while the objective runs, **branches** and **loops** just work — a
parameter can exist only on some paths.

**Branches** — different models expose different hyperparameters:

```csharp
double Objective(Trial trial)
{
    string classifier = trial.SuggestCategorical("classifier", new[] { "SVC", "RandomForest" });
    if (classifier == "SVC")
    {
        double svcC = trial.SuggestFloat("svc_c", 1e-10, 1e10, log: true);
        return TrainSvc(svcC);
    }
    else
    {
        long rfMaxDepth = trial.SuggestInt("rf_max_depth", 2, 32, log: true);
        return TrainRandomForest(rfMaxDepth);
    }
}
```

**Loops** — size a network by a suggested layer count, naming each layer's parameter uniquely:

```csharp
long nLayers = trial.SuggestInt("n_layers", 1, 3);
long inSize = 28 * 28;
var layers = new List<Layer>();
for (int i = 0; i < nLayers; i++)
{
    long nUnits = trial.SuggestInt($"n_units_l{i}", 4, 128, log: true);   // unique name per layer
    layers.Add(new Linear(inSize, nUnits));
    layers.Add(new ReLU());
    inSize = nUnits;
}
layers.Add(new Linear(inSize, 10));
```

## A note on the number of parameters

The difficulty of optimization grows *roughly exponentially* with the number of parameters — the
trials needed to search well rise sharply as you add dimensions. Prefer to leave out parameters you
don't expect to matter.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])` | `trial.SuggestCategorical("optimizer", new[] { "MomentumSGD", "Adam" })` |
| `trial.suggest_int("num_channels", 32, 512, log=True)` | `trial.SuggestInt("num_channels", 32, 512, log: true)` |
| `trial.suggest_int("num_units", 10, 100, step=5)` | `trial.SuggestInt("num_units", 10, 100, step: 5)` |
| `trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)` | `trial.SuggestFloat("learning_rate", 1e-5, 1e-2, log: true)` |
| `trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)` | `trial.SuggestFloat("drop_path_rate", 0.0, 1.0, step: 0.1)` |
| Python `if`/`for` in the objective | C# `if`/`for` in the objective |

## See also

[Distributions](../distributions.md) · [Search Spaces](../search-spaces.md) ·
[Efficient optimization algorithms](efficient-algorithms.md) ·
[Ablation study with brute force](../recipes/ablation-study.md) (conditional spaces) ·
the [API reference](../../api/index.md).
