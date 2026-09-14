# Recipes

Short, task-focused how-tos that show OptunaSharp features in context. Where the [Guides](../study-and-trials.md)
explain a subsystem end to end, each recipe answers one practical question — "how do I persist a
study?", "how do I write my own sampler?", "how do I feed in hyperparameters I already know are
good?" — and is a direct port of the matching page in Optuna's
[Recipes tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html#recipes)
(`optuna/tutorial/20_recipes/`), rewritten against the C# API.

The objectives here are deliberately small (a quadratic, a mock score) so the *mechanism* — not a
machine-learning library — is what each page teaches. Swap in your real training code wherever a
recipe returns a placeholder value.

## The recipes

| # | Recipe | You'll learn to |
|---|--------|-----------------|
| 1 | [Saving/Resuming a Study with an RDB backend](rdb.md) | Persist a study to SQLite and resume it later |
| 2 | [Multi-objective optimization](multi-objective.md) | Optimize several objectives and read the Pareto front |
| 3 | [User attributes](attributes.md) | Annotate studies and trials with your own metadata |
| 4 | [Command-line interface](cli.md) | Drive an ask/tell loop from the `optunasharp` tool |
| 5 | [User-defined sampler](user-defined-sampler.md) | Implement your own sampling algorithm |
| 6 | [User-defined pruner](user-defined-pruner.md) | Implement your own early-stopping rule |
| 7 | [Callback for `Study.Optimize`](callback.md) | Run code after every trial (and stop the study) |
| 8 | [Specify hyperparameters manually](specify-params.md) | Enqueue known-good params or inject already-evaluated ones |
| 9 | [Ask-and-tell interface](ask-and-tell.md) | Own the optimization loop; define-and-run; batch evaluation |
| 10 | [Re-use the best trial](reuse-best-trial.md) | Re-evaluate the best hyperparameters after tuning |
| 11 | [Journal-file storage](journal-storage.md) | Persist and share a study through a single log file |
| 12 | [Artifacts](artifacts.md) | Attach large files (models, images) to trials |
| 13 | [Wilcoxon pruner](wilcoxon-pruner.md) | Early-stop objectives that average many evaluations |
| 14 | [Ablation study with brute force](ablation-study.md) | Exhaustively sweep a conditional search space |

## How these differ from Guides

A Guide is the reference for a whole area — [Storage](../storage.md) documents every backend;
[Samplers](../samplers.md) documents every sampler. A recipe is a walkthrough of one scenario and
links back to the Guide for the full surface. If you want the complete API, follow the Guide link at
the bottom of each recipe.

## Relationship to Optuna

Each recipe corresponds one-to-one with a file in Optuna's
[`tutorial/20_recipes/`](https://github.com/optuna/optuna/tree/master/tutorial/20_recipes). The C#
port keeps the same narrative and demonstrates the same feature; under a fixed seed the trial
sequence is bit-for-bit identical to Python wherever the path is deterministic.

## See also

[Study and Trials](../study-and-trials.md) · [Samplers](../samplers.md) · [Pruners](../pruners.md) ·
[Storage](../storage.md) · [Artifacts](../artifacts.md) · the [API reference](../../api/index.md).
