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
    int Dim { get; }
    int PopulationSize { get; }
    int Generation { get; }
    void ReseedRng(int seed);
    double[] Ask();
    void Tell(List<(double[] X, double Value)> solutions);
}
