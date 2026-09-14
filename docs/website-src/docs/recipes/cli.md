# Command-line interface

OptunaSharp ships a command-line tool, **`optunasharp`** (the `OptunaSharp.Cli` .NET tool), that ports
Optuna's `optuna` CLI command-for-command. It is handy when the objective can't be written as code —
for example a lab measurement or a manual evaluation — and you want to drive an
[ask/tell](ask-and-tell.md) loop from the shell.

| Command | Description |
|---------|-------------|
| `create-study` | Create a new study. |
| `delete-study` | Delete a specified study. |
| `studies` | Show a list of studies. |
| `study-names` | List study names in a storage. |
| `study set-user-attr` | Set a user attribute on a study. |
| `trials` | Show a list of trials. |
| `best-trial` | Show the best trial. |
| `best-trials` | Show trials on the Pareto front. |
| `ask` | Create a new trial and suggest parameters. |
| `tell` | Finish a trial created by `ask`. |
| `storage upgrade` | Upgrade the schema of an RDB storage. |

Install it as a global tool:

```bash
dotnet tool install -g OptunaSharp.Cli
```

## When you *can* write the objective in code

There is nothing wrong with a plain program — a tiny console app that optimizes and prints the best
result:

```csharp
using OptunaSharp;

double Objective(Trial trial)
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}

var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 100);
Console.WriteLine($"Best value: {study.BestValue} (params: {study.BestParams["x"]})");
```

## When you can't: an interactive ask/tell loop

Suppose the objective for a parameter `x` in `[-10, 10]` is measured by hand. Create a persistent
study, then repeatedly *ask* for a parameter, run the experiment, and *tell* the result. The
`--storage sqlite:///example.db` argument persists the study (see [RDB storage](rdb.md)).

```bash
# Create the study; capture its generated name.
STUDY_NAME=`optunasharp create-study --storage sqlite:///example.db`

# Ask for a parameter value to try next.
optunasharp ask --storage sqlite:///example.db --study-name $STUDY_NAME --sampler TPESampler \
    --search-space '{"x": {"name": "FloatDistribution", "attributes": {"step": null, "low": -10.0, "high": 10.0, "log": false}}}'
# -> {"number": 0, "params": {"x": 2.512238141966016}}
```

Run your experiment with the suggested `x`, then report the objective value back for that trial
number:

```bash
optunasharp tell --storage sqlite:///example.db --study-name $STUDY_NAME \
    --trial-number 0 --values 0.7 --state complete
# -> Told trial 0 with values [0.7] and state TrialState.COMPLETE.
```

The `--search-space` JSON is the exact format Optuna uses (a name plus distribution attributes), so a
search space produced by Python's `optuna` CLI works verbatim here, and vice versa.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna create-study --storage …` | `optunasharp create-study --storage …` |
| `optuna ask --study-name … --sampler … --search-space '…'` | `optunasharp ask --study-name … --sampler … --search-space '…'` |
| `optuna tell --trial-number 0 --values 0.7 --state complete` | `optunasharp tell --trial-number 0 --values 0.7 --state complete` |
| `optuna study set-user-attr -k dataset --value MNIST` | `optunasharp study set-user-attr -k dataset --value MNIST` |

## See also

The full [CLI integration guide](../../integrations/cli.md) · [Ask-and-tell interface](ask-and-tell.md) ·
[Saving/Resuming with RDB](rdb.md) · [User attributes](attributes.md).
