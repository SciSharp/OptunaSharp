using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     Layer 1 of the AGE-MOEA-2 port: a standalone <c>minimize()</c>-shaped genetic-algorithm loop,
///     the analogue of running <c>pymoo.optimize.minimize(problem, AGEMOEA2(pop_size), ("n_gen",
///     n_gen), seed)</c>. It combines the pymoo-faithful operators (<see cref="FloatRandomSampling" />,
///     <see cref="Mating" />) with the AGE-MOEA-2 survival (<see cref="AgeMoea2Survival" />) under the
///     <c>GeneticAlgorithm</c> schedule (<c>advance_after_initial_infill=True</c>,
///     <c>n_offsprings=pop_size</c>) from <c>pymoo.algorithms.base.genetic</c> / <c>core.algorithm</c>
///     (<c>@0.6.1.3</c>). This is where the byte-perfect guarantee lives: generation-by-generation
///     bit-exact against pymoo.
///
///     Pass an <see cref="AgeMoeaSurvival" /> (the 2019 base) instead of the default
///     <see cref="AgeMoea2Survival" /> to run AGE-MOEA-1.
/// </summary>
internal sealed class AgeMoea2Optimizer
{
    private readonly AgeMoeaProblem _problem;
    private readonly int _popSize;
    private readonly int _nGen;
    private readonly int? _seed;
    private readonly FloatRandomSampling _sampling;
    private readonly Mating _mating;
    private readonly AgeMoeaSurvival _survival;
    private readonly DuplicateElimination? _eliminateDuplicates;

    internal AgeMoea2Optimizer(
        AgeMoeaProblem problem, int popSize, int nGen, int? seed,
        AgeMoeaSurvival? survival = null,
        double crossoverProb = 0.9, double crossoverEta = 15, double mutationEta = 20,
        bool eliminateDuplicates = true)
    {
        _problem = problem;
        _popSize = popSize;
        _nGen = nGen;
        _seed = seed;
        _survival = survival ?? new AgeMoea2Survival();
        _eliminateDuplicates = eliminateDuplicates ? new DuplicateElimination() : null;
        _sampling = new FloatRandomSampling();
        _mating = new Mating(
            new TournamentSelection(),
            new SimulatedBinaryCrossover(prob: crossoverProb, eta: crossoverEta),
            new PolynomialMutation(eta: mutationEta),
            _eliminateDuplicates);
    }

    /// <summary>
    ///     Run the optimizer, returning the population after every generation (the last entry is the
    ///     final result). Generation 1 is the initialized + survived population
    ///     (<c>advance_after_initial_infill</c>); each subsequent generation mates, evaluates, and
    ///     re-applies survival to the merged parents+offspring.
    /// </summary>
    public List<Population> Run()
    {
        NumPyRandom rng = new LazyRandomState(_seed).Rng;
        var history = new List<Population>();

        // Generation 1: initial infill (sample + duplicate-elimination) then survival.
        double[][] x = _sampling.Do(_problem, _popSize, rng);
        Population pop = Population.New(x);
        if (_eliminateDuplicates is not null)
            pop = _eliminateDuplicates.Do(pop);
        Evaluate(pop);
        pop = _survival.Do(pop, pop.Count);
        history.Add(pop);

        // Generations 2..n_gen: mate -> evaluate -> survival on merged parents+offspring.
        for (int gen = 2; gen <= _nGen; gen++)
        {
            Population off = _mating.Do(_problem, pop, _popSize, rng);
            if (off.Count == 0)
                break; // pymoo forces termination when no unique offspring can be produced
            Evaluate(off);
            pop = _survival.Do(Population.Merge(pop, off), _popSize);
            history.Add(pop);
        }

        return history;
    }

    private void Evaluate(Population pop)
    {
        var (f, cv) = _problem.Evaluate(pop.X);
        pop.F = f;
        pop.CV = cv;
    }
}
