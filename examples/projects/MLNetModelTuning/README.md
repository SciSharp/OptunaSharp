# MLNetModelTuning

Tunes a real ML.NET model with OptunaSharp, with every trial trained in-process: no Python and no surrogate
objective. `OptunaSearchCV` samples four FastTree hyperparameters with TPE, and `MLNetEstimator`
cross-validates each candidate on five folds.

```bash
dotnet run --project examples/projects/MLNetModelTuning          # 20 trials, about 20 s
dotnet run --project examples/projects/MLNetModelTuning -- 40    # 40 more, in the same study
```

## What it does

1. Generates **Friedman #1** ([`FriedmanData.cs`](FriedmanData.cs)), a standard regression benchmark:
   `y = 10 sin(π x₀ x₁) + 20 (x₂ − 0.5)² + 10 x₃ + 5 x₄ + ε`, with five more features that are pure noise. It
   uses NumSharp's NumPy-identical random stream, so the 1,000 rows are bit-for-bit the float32 values of
   `sklearn.datasets.make_friedman1(n_samples=1000, noise=1.0, random_state=0)`; all 11,000 values were
   compared. 80% of the rows tune the model and 20% are held out.
2. Opens the study `fasttree-friedman1` in **SQLite** (`out/mlnet-tuning.db`, Optuna's schema), so every run
   continues the previous one's search. Python Optuna can open the file too.
3. Searches `number_of_leaves`, `number_of_trees`, `learning_rate` and `minimum_example_count_per_leaf`,
   maximizing the mean cross-validation R².
4. Refits the best configuration on all tuning rows and scores it on the **held-out** rows. That score, not
   the cross-validation one, is the honest estimate of model quality.
5. Ranks the hyperparameters with **fANOVA**, saves the model (`out/best-model.zip`, loadable with
   `MLContext.Model.Load`), and writes two **Plotly** charts (`out/history.html`, `out/importance.html`).

A first run on this machine (Windows 11, .NET 10.0.1):

```
Best mean cross-validation R²: 0.9219, trial 4 of 20:
    number_of_leaves                 4
    number_of_trees                  136
    learning_rate                    0.176463
    minimum_example_count_per_leaf   26
R² of the refit model on the held-out rows: 0.9188
Hyperparameter importance (fANOVA):
    learning_rate                    0.688
    ...
```

A second run with `-- 5` continued the same study and found a better trial (number 20, R² 0.9240).

## Notes

- **Functional API, no DI.** In a console app the plain constructors are the simplest composition. The
  [TuningService](../TuningService/README.md) and [TuningWorker](../TuningWorker/README.md) show the container.
- **Output folder.** Everything goes to `out/` in this folder: the project sets `RunWorkingDirectory`, so
  `dotnet run --project` starts here from wherever it is called. Run the built executable directly and `out/`
  goes under the current directory.
- **One training thread.** FastTree runs single-threaded. On folds of 640 rows, coordinating many threads
  costs more than it saves: measured here on 32 logical CPUs (with an earlier, wider tree range), the first
  trial took 21 s with FastTree's default threading and 3.5 s single-threaded, with the same results. A
  single-threaded fit is also deterministic.
- **`nJobs` stays at 1.** `MLNetEstimator` shares one `MLContext` between trials, and concurrent training through
  one `MLContext` is not something [`docs/THREADING.md`](../../../docs/THREADING.md) vouches for.
- **The FutureWarning on every trial is expected.** `OptunaSearchCV` reports each trial's fold scores through
  `optuna.terminator`, which Optuna deprecated in v4.9.0. Python's `OptunaSearchCV` prints the same warning on
  every trial too; that was checked with Optuna 5.0.0.
- **x64.** ML.NET's FastTree native code runs on x64.
