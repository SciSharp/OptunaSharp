"""Generate the AGE-MOEA-2 differential parity corpus from pymoo 0.6.1.3. Dumps per-operator
input/seed/output so the C# port (OptunaSharp.Samplers.AgeMoea) can replay the same seeds with
LazyRandomState and assert bit-equality with no Python at test time.

Run in the repo's main Python env (the same interpreter the interop suite uses):
    python test/OptunaSharp.Tests/oracle/gen_agemoea2_oracle.py

pymoo 0.6.1.3 requires only numpy>=1.15, so it runs on the interop env's numpy 2.4.2 alongside
numba 0.65.1 (install once with `pip install pymoo==0.6.1.3 --no-deps autograd alive_progress`;
numpy/numba/optuna are untouched). The corpus is byte-for-byte IDENTICAL to the original
numpy-2.0.2 / numba-0.60.0 (Mimicry's pins) run — verified field-by-field, 0 ULP — because the
MT19937 stream is version-stable, libm is the machine's, and the fastmath kernels reduce over
2-3 objectives (no reassociation). So no isolated venv is needed.

pymoo runs with numba JIT ON (NUMBA_DISABLE_JIT=1 is broken: find_corner_solutions uses
numba.intp as a numpy dtype). JIT-on is what Mimicry runs."""
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
from pymoo.algorithms.moo.age import find_corner_solutions, normalize, point_2_line_distance, AGEMOEASurvival
from pymoo.algorithms.moo.age2 import find_zero, project_on_manifold, AGEMOEA2Survival


class Box(Problem):
    def __init__(self, n_var, n_obj, xl, xu):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=np.array(xl, float), xu=np.array(xu, float))

    def _evaluate(self, X, out, *args, **kwargs):
        # distinct polynomial objectives — no transcendentals, so C# reproduces them bit-for-bit
        # (numpy X**2 == x*x; np.sum over n_var<128 is sequential). Used only by the driver (minimize).
        cols = [np.sum(X ** 2, axis=1), np.sum((X - 1.0) ** 2, axis=1), np.sum((X - 0.5) ** 2, axis=1)]
        out["F"] = np.column_stack(cols[:self.n_obj])


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


def dump_find_zero(cases):
    out = []
    for point, n in cases:
        p = find_zero(np.array(point, float), n, 0.001)
        out.append(dict(point=point, n=n, p=float(p)))
    return out


def dump_project(cases):
    out = []
    for point, p in cases:
        pr = project_on_manifold(np.array(point, float), p)
        out.append(dict(point=point, p=p, projected=np.array(pr).tolist()))
    return out


def dump_survival_score(cases, survival):
    out = []
    for front1 in cases:
        fa = np.array(front1, float)
        ideal = np.min(fa, axis=0)
        cd, p, norm = survival.survival_score(fa.copy(), ideal)
        out.append(dict(front1=front1, ideal=ideal.tolist(),
                        crowd_dist=cd.tolist(), p=float(p), normalization=np.array(norm).tolist()))
    return out


def dump_do(cases, survival):
    out = []
    for F, n_survive in cases:
        n_var = 4
        prob = _mk_problem(n_var, n_obj=len(F[0]))
        # unique X per individual so survivors can be identified by row identity
        X = np.arange(len(F) * n_var, dtype=float).reshape(len(F), n_var)
        pop = Population.new(X=X, F=np.array(F, float))
        pop.set("CV", np.zeros((len(F), 1)))
        survivors = survival._do(prob, pop, n_survive=n_survive)
        out.append(dict(
            F=F, n_survive=n_survive,
            rank=np.array(pop.get("rank"), dtype=int).tolist(),
            crowding=np.array(pop.get("crowding"), dtype=float).tolist(),
            survivor_F=survivors.get("F").tolist(),
        ))
    return out


def dump_driver(cases):
    from pymoo.algorithms.moo.age2 import AGEMOEA2
    out = []
    for seed, n_var, n_obj, pop_size, n_gen in cases:
        prob = _mk_problem(n_var, n_obj)
        algo = AGEMOEA2(pop_size=pop_size)
        algo.setup(prob, termination=("n_gen", n_gen), seed=seed, verbose=False)
        gens = []
        while algo.has_next():
            algo.next()
            gens.append(dict(X=algo.pop.get("X").tolist(), F=algo.pop.get("F").tolist()))
        out.append(dict(seed=seed, n_var=n_var, n_obj=n_obj, pop_size=pop_size,
                        n_gen=n_gen, n_gens_captured=len(gens), gens=gens))
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

    # Phase 3: survival geometry (find_zero, project_on_manifold, survival_score, _do).
    # find_zero / project points are positive vectors (as after shift+normalize).
    find_zero_cases = [(rng.random(n).tolist(), n) for n in (2, 3, 4)]
    find_zero_cases += [([0.3, 0.7], 2), ([0.1, 0.5, 0.9], 3), ([1.0, 1.0, 1.0], 3)]
    project_cases = []
    for p in (0.5, 1.0, 1.7, 2.5):
        project_cases.append((rng.random(3).tolist(), p))
        project_cases.append((rng.random(4).tolist(), p))
    # survival_score front-0 objectives (2 and 3 objectives), sizes exercising the greedy loop
    ss_cases = [rng.random((n, m)).tolist() for (n, m) in [(10, 2), (12, 3), (8, 2), (15, 3)]]
    # full environmental selection on merged populations (2*pop_size rows), 2 and 3 objectives
    do_cases = [(rng.random((16, m)).tolist(), 8) for m in (2, 3)]
    do_cases += [(rng.random((20, 3)).tolist(), 10), (rng.random((12, 2)).tolist(), 6)]

    # Phase 4: full minimize()-shaped driver — (seed, n_var, n_obj, pop_size, n_gen).
    driver_cases = [
        (42, 6, 2, 8, 5),
        (7, 5, 2, 10, 6),
        (0, 4, 3, 12, 5),
        (2024, 8, 2, 16, 4),
    ]

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
        find_zero=dump_find_zero(find_zero_cases),
        project=dump_project(project_cases),
        survival_score_age2=dump_survival_score(ss_cases, AGEMOEA2Survival()),
        survival_score_age1=dump_survival_score(ss_cases, AGEMOEASurvival()),
        do_age2=dump_do(do_cases, AGEMOEA2Survival()),
        driver=dump_driver(driver_cases),
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
    for k in ("sampling", "crossover", "mutation", "tournament", "mating", "nds", "corner", "normalize",
              "find_zero", "project", "survival_score_age2", "survival_score_age1", "do_age2", "driver"):
        print(f"  {k}: {len(corpus[k])} cases")


if __name__ == "__main__":
    main()
