# MOEA/D (2007)

> 🚧 **Stub** — a full write-up (idea, pseudocode, OptunaSharp usage) is coming. Back to the [Methods overview](index.md).

**Paper:** [Zhang & Li, IEEE TEC 11(6), 2007](https://doi.org/10.1109/TEVC.2007.892759)


## OptunaHub's `MOEADSampler` is a different sampler

OptunaSharp's `MoeaDSampler` ports pymoo's `MOEAD`. OptunaHub also has a MOEA/D
([`samplers/moead`](https://hub.optuna.org/samplers/moead/), `MOEADSampler`, since 2024-09), and OptunaSharp does
not follow it. A study moved over from the hub package therefore gets other defaults and other trials, even with the
same seed:

| OptunaHub `MOEADSampler` | OptunaSharp `MoeaDSampler` |
|---|---|
| `population_size=100` | `populationSize = 50`, rounded down to a Das-Dennis lattice size |
| `n_neighbors=None` (then `population_size // 10`) | `nNeighbors = 20` |
| `scalar_aggregation_func="tchebycheff"` (also `"weighted_sum"`, `"PBI"`) | `scalarization`: Tchebycheff by default, plus PBI and weighted sum |
| `crossover=None`, meaning `UniformCrossover(swapping_prob)` | `crossover`: `SBXCrossover` by default |
| `mutation=None`, meaning `UniformMutation()` | `mutation`: `PolynomialMutation` by default |
| `crossover_prob=0.9` (otherwise a random parent is copied) | none: every offspring is crossed |
| `mutation_prob=None` (1 / number of parameters), `swapping_prob=0.5`, `seed` | `mutationProb`, `swappingProb`, `seed`, same defaults |
| (mates within the neighborhood) | `neighborSelectionProb = 0.9` (pymoo's `prob_neighbor_mating`), `nReplacements = 2` |
