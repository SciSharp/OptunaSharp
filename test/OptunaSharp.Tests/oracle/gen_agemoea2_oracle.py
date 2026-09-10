"""Generate the AGE-MOEA-2 differential parity corpus from pymoo 0.6.1.3 (numba 0.60.0,
numpy 2.0.2 — Mimicry's pins). Dumps per-operator input/seed/output so the C# port
(OptunaSharp.Samplers.AgeMoea) can replay the same seeds with LazyRandomState and assert
bit-equality with no Python at test time.

Run in the isolated oracle venv:
    K:/source/OptunaSharp/.venv-pymoo/Scripts/python test/oracle/gen_agemoea2_oracle.py

pymoo runs with numba JIT ON (NUMBA_DISABLE_JIT=1 is broken here: find_corner_solutions uses
numba.intp as a numpy dtype). JIT-on is what Mimicry runs; the operator code below is not
jitted, so this only pins the RNG-coupled operator stream (Phase 1)."""
import json
import types
import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.mating import Mating
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.age import find_corner_solutions, normalize, point_2_line_distance


class Box(Problem):
    def __init__(self, n_var, n_obj, xl, xu):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=np.array(xl, float), xu=np.array(xu, float))

    def _evaluate(self, X, out, *args, **kwargs):
        # a simple, deterministic multi-objective map (only used where F is needed)
        out["F"] = np.column_stack([np.sum(X ** 2, axis=1)] + [np.sum((X - 1.0) ** 2, axis=1)] * (self.n_obj - 1))


def _mk_problem(n_var, n_obj=2):
    return Box(n_var, n_obj, [0.0] * n_var, [1.0] * n_var)


def dump_sampling(cases):
    out = []
    for seed, n_var, n in cases:
        prob = _mk_problem(n_var)
        np.random.seed(seed)
        X = FloatRandomSampling()(prob, n).get("X")
        out.append(dict(seed=seed, n_var=n_var, n=n, X=X.tolist()))
    return out


def dump_crossover(cases):
    out = []
    for seed, pop_X, parents in cases:
        n_var = len(pop_X[0])
        prob = _mk_problem(n_var)
        pop = Population.new(X=np.array(pop_X, float))
        cx = SBX(prob=0.9, eta=15)
        np.random.seed(seed)
        off = cx(prob, pop, parents=np.array(parents)).get("X")
        out.append(dict(seed=seed, pop_X=pop_X, parents=parents, off_X=off.tolist()))
    return out


def dump_mutation(cases):
    out = []
    for seed, off_X in cases:
        n_var = len(off_X[0])
        prob = _mk_problem(n_var)
        pop = Population.new(X=np.array(off_X, float))
        mut = PM(eta=20)
        np.random.seed(seed)
        mx = mut(prob, pop).get("X")
        out.append(dict(seed=seed, in_X=off_X, out_X=mx.tolist()))
    return out


def dump_tournament(cases):
    out = []
    for seed, pop_X, ranks, crowd, n_select in cases:
        n_var = len(pop_X[0])
        prob = _mk_problem(n_var)
        pop = Population.new(X=np.array(pop_X, float))
        pop.set("F", np.array([[0.0] * prob.n_obj] * len(pop_X)))
        pop.set("rank", np.array(ranks))
        pop.set("crowding", np.array(crowd, float))
        pop.set("CV", np.zeros((len(pop_X), 1)))
        algo = types.SimpleNamespace(tournament_type="comp_by_rank_and_crowding")
        sel = TournamentSelection(func_comp=binary_tournament)
        np.random.seed(seed)
        parents = sel(prob, pop, n_select, n_parents=2, algorithm=algo, to_pop=False)
        out.append(dict(seed=seed, pop_X=pop_X, ranks=ranks, crowding=crowd,
                        n_select=n_select, parents=np.array(parents).tolist()))
    return out


def dump_mating(cases):
    out = []
    for seed, pop_X, ranks, crowd, n_off in cases:
        n_var = len(pop_X[0])
        prob = _mk_problem(n_var)
        pop = Population.new(X=np.array(pop_X, float))
        pop.set("F", np.array([[0.0] * prob.n_obj] * len(pop_X)))
        pop.set("rank", np.array(ranks))
        pop.set("crowding", np.array(crowd, float))
        pop.set("CV", np.zeros((len(pop_X), 1)))
        algo = types.SimpleNamespace(tournament_type="comp_by_rank_and_crowding")
        mating = Mating(TournamentSelection(func_comp=binary_tournament),
                        SBX(prob=0.9, eta=15), PM(eta=20),
                        eliminate_duplicates=DefaultDuplicateElimination(), n_max_iterations=100)
        np.random.seed(seed)
        off = mating.do(prob, pop, n_off, algorithm=algo).get("X")
        out.append(dict(seed=seed, pop_X=pop_X, ranks=ranks, crowding=crowd,
                        n_off=n_off, off_X=off.tolist()))
    return out


def dump_nds(cases):
    out = []
    for F in cases:
        Fa = np.array(F, float)
        fronts = NonDominatedSorting().do(Fa)  # full ranking (n_stop = 1e8)
        rank = np.full(len(F), -1, dtype=int)
        for i, fr in enumerate(fronts):
            rank[fr] = i
        out.append(dict(F=F, rank=rank.tolist(),
                        fronts=[np.array(fr).tolist() for fr in fronts]))
    return out


def dump_corner(cases):
    out = []
    for front in cases:
        idx = find_corner_solutions(np.array(front, float))
        out.append(dict(front=front, extreme=np.array(idx).tolist()))
    return out


def dump_normalize(cases):
    out = []
    for front in cases:
        fa = np.array(front, float)
        extreme = find_corner_solutions(fa)
        nf, norm = normalize(fa.copy(), extreme)
        out.append(dict(front=front, extreme=np.array(extreme).tolist(),
                        norm_front=nf.tolist(), normalization=norm.tolist()))
    return out


def main():
    rng = np.random.RandomState(12345)
    # fixed random populations/parents for the operator cases
    def randpop(n, d):
        return rng.random((n, d)).tolist()

    sampling_cases = [(42, 6, 8), (7, 4, 10), (0, 8, 20), (2024, 3, 5)]

    crossover_cases = []
    for seed in (42, 7, 0, 2024):
        pop = randpop(8, 6)
        parents = rng.randint(0, 8, size=(4, 2)).tolist()
        crossover_cases.append((seed, pop, parents))

    mutation_cases = [(seed, randpop(8, 6)) for seed in (42, 7, 0, 2024)]

    tournament_cases = []
    for seed in (42, 7, 0, 2024):
        pop = randpop(10, 5)
        ranks = rng.randint(0, 3, size=10).tolist()
        crowd = rng.random(10).tolist()
        tournament_cases.append((seed, pop, ranks, crowd, 5))
    # a tie-heavy case: all rank 0, some equal (inf) crowding -> exercises choice() draws
    tie_pop = randpop(6, 4)
    tie_ranks = [0] * 6
    tie_crowd = [float("inf"), float("inf"), 0.5, 0.5, 0.2, 0.9]
    tournament_cases.append((99, tie_pop, tie_ranks, tie_crowd, 4))

    mating_cases = []
    for seed in (42, 7, 0, 2024):
        pop = randpop(8, 6)
        ranks = rng.randint(0, 3, size=8).tolist()
        crowd = rng.random(8).tolist()
        mating_cases.append((seed, pop, ranks, crowd, 8))

    # Phase 2: non-dominated sort, corner solutions, normalization (deterministic given inputs).
    nds_cases = [rng.random((n, m)).tolist() for (n, m) in [(10, 2), (12, 3), (8, 2), (20, 3), (6, 4)]]
    corner_cases = [rng.random((n, m)).tolist() for (n, m) in [(8, 2), (10, 3), (12, 3), (15, 4)]]
    # include a degenerate m<=n case (returns arange) and the normalize duplicate-extreme fallback path
    corner_cases.append(rng.random((3, 3)).tolist())
    normalize_cases = [rng.random((n, m)).tolist() for (n, m) in [(8, 2), (10, 3), (12, 3), (15, 4)]]

    corpus = dict(
        pymoo_version=__import__("pymoo").__version__,
        numpy_version=np.__version__,
        sampling=dump_sampling(sampling_cases),
        crossover=dump_crossover(crossover_cases),
        mutation=dump_mutation(mutation_cases),
        tournament=dump_tournament(tournament_cases),
        mating=dump_mating(mating_cases),
        nds=dump_nds(nds_cases),
        corner=dump_corner(corner_cases),
        normalize=dump_normalize(normalize_cases),
    )
    import os, math
    def san(o):
        # encode non-finite floats as sentinel strings (JSON has no inf/nan; System.Text.Json
        # rejects Python's Infinity literal). The C# side maps these back.
        if isinstance(o, float):
            if math.isinf(o):
                return "Infinity" if o > 0 else "-Infinity"
            if math.isnan(o):
                return "NaN"
            return o
        if isinstance(o, dict):
            return {k: san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [san(v) for v in o]
        return o

    path = os.path.join(os.path.dirname(__file__), "agemoea2_operators.json")
    with open(path, "w") as f:
        json.dump(san(corpus), f, allow_nan=False)
    print("wrote", path)
    for k in ("sampling", "crossover", "mutation", "tournament", "mating", "nds", "corner", "normalize"):
        print(f"  {k}: {len(corpus[k])} cases")


if __name__ == "__main__":
    main()
