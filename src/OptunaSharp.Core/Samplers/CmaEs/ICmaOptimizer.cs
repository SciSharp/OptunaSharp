using System.Collections.Generic;

namespace OptunaSharp.Samplers.CmaEs;

/// <summary>
///     The ask-and-tell surface <see cref="CmaEsSampler" /> drives, shared by the covariance-matrix
///     optimizers whose <c>ask()</c> yields a single parameter vector — <see cref="Cma" /> (full
///     covariance) and <see cref="SepCma" /> (diagonal / separable). Optuna's <c>cmaes.CMAwM</c>
///     (with-margin) has a two-vector <c>ask()</c> and is handled separately by the sampler.
/// </summary>
internal interface ICmaOptimizer
{
    /// <summary>Number of numerical decision coordinates.</summary>
    int Dim { get; }
    /// <summary>Number of evaluated individuals required for one generation.</summary>
    int PopulationSize { get; }
    /// <summary>Number of completed optimizer updates.</summary>
    int Generation { get; }
    /// <summary>Resets the optimizer random stream for subsequent proposals.</summary>
    /// <param name="seed">Seed used to reset the optimizer random stream.</param>
    void ReseedRng(int seed);
    /// <summary>Draws one candidate from the current proposal without advancing the generation.</summary>
    /// <returns>One candidate vector in transformed coordinates.</returns>
    double[] Ask();
    /// <summary>Updates the optimizer using evaluated candidates from one coherent generation.</summary>
    /// <param name="solutions">Evaluated proposal vectors and minimization-oriented values for one coherent generation.</param>
    void Tell(List<(double[] X, double Value)> solutions);
}
