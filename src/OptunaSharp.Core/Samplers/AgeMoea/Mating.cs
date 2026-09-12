using System;
using NumSharp;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     The mating / infill criterion — port of <c>pymoo.core.mating.Mating</c> over
///     <c>pymoo.core.infill.InfillCriterion.do</c> (<c>@0.6.1.3</c>). One <see cref="MatingStep" /> is
///     selection → crossover → mutation; <see cref="Do" /> repeats it, eliminating duplicates against
///     the current population and the offspring collected so far, until <c>n_offsprings</c> unique
///     children exist (or <c>n_max_iterations = 100</c> is hit). The retry is the data-dependent RNG
///     coupling the porting plan flags: each extra iteration re-draws a full selection+crossover+mutation.
/// </summary>
internal sealed class Mating
{
    private readonly TournamentSelection _selection;
    private readonly SimulatedBinaryCrossover _crossover;
    private readonly PolynomialMutation _mutation;
    private readonly DuplicateElimination? _eliminateDuplicates;
    private const int NMaxIterations = 100;

    /// <summary>Connects tournament selection, crossover, mutation, and optional duplicate filtering.</summary>
    /// <param name="selection">Tournament strategy used to choose mating parents.</param>
    /// <param name="crossover">Operator combining the selected numerical parent vectors.</param>
    /// <param name="mutation">Numerical mutation operator used when an offspring coordinate is selected for mutation.</param>
    /// <param name="eliminateDuplicates">Duplicate filter or flag controlling whether duplicate offspring are removed.</param>
    public Mating(TournamentSelection selection, SimulatedBinaryCrossover crossover, PolynomialMutation mutation,
        DuplicateElimination? eliminateDuplicates)
    {
        _selection = selection;
        _crossover = crossover;
        _mutation = mutation;
        _eliminateDuplicates = eliminateDuplicates;
    }

    /// <summary>One mating step (<c>Mating._do</c>): select parents, cross, mutate — offspring X only.</summary>
    /// <param name="problem">Decision bounds and objective/constraint evaluation surface.</param>
    /// <param name="pop">Population whose decision and evaluation columns are consumed or updated.</param>
    /// <param name="nOffsprings">Number of offspring requested.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <returns>Offspring after one selection, crossover, and mutation pass.</returns>
    public Population MatingStep(AgeMoeaProblem problem, Population pop, int nOffsprings, NumPyRandom rng)
    {
        int nMatings = (int)Math.Ceiling((double)nOffsprings / _crossover.NOffsprings);
        int[][] parents = _selection.Do(pop, nMatings, _crossover.NParents, rng);
        double[][] offX = _crossover.Do(pop.X, parents, problem, rng);
        offX = _mutation.Do(offX, problem, rng);
        return Population.New(offX);
    }

    /// <summary><c>InfillCriterion.do</c>: the duplicate-eliminating retry loop.</summary>
    /// <param name="problem">Decision bounds and objective/constraint evaluation surface.</param>
    /// <param name="pop">Population whose decision and evaluation columns are consumed or updated.</param>
    /// <param name="nOffsprings">Number of offspring requested.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <returns>A population of offspring after bounded duplicate-removal retries.</returns>
    public Population Do(AgeMoeaProblem problem, Population pop, int nOffsprings, NumPyRandom rng)
    {
        Population off = Population.Empty();
        int nInfills = 0;

        while (off.Count < nOffsprings)
        {
            int nRemaining = nOffsprings - off.Count;

            Population offNew = MatingStep(problem, pop, nRemaining, rng);
            // repair = NoRepair (identity)
            if (_eliminateDuplicates is not null)
                offNew = _eliminateDuplicates.Do(offNew, pop, off);

            if (off.Count + offNew.Count > nOffsprings)
                offNew = offNew.Take(nOffsprings - off.Count);

            off = Population.Merge(off, offNew);
            nInfills++;

            if (nInfills >= NMaxIterations) break;
        }

        return off;
    }
}
