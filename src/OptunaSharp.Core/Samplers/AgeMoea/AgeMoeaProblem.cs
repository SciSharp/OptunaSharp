using System;
using NumSharp;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     A box-constrained multi-objective minimization problem, the minimal analogue of pymoo's
///     <c>pymoo.core.problem.Problem</c> needed to drive the AGE-MOEA / AGE-MOEA-2 optimizer
///     (<see cref="AgeMoea2Optimizer" />). Objectives are minimized; constraints (if any) are
///     expressed as a scalar constraint-violation <c>CV ≥ 0</c> (0 = feasible), mirroring pymoo's
///     aggregated <c>CV = Σ max(g, 0)</c>.
/// </summary>
internal abstract class AgeMoeaProblem
{
    /// <summary>Defines the decision dimension, objective dimension, and numerical box for population operators.</summary>
    /// <param name="nVar">Number of decision coordinates per individual.</param>
    /// <param name="nObj">Number of objective coordinates per evaluation.</param>
    /// <param name="xl">Lower decision bounds in coordinate order.</param>
    /// <param name="xu">Upper decision bounds in coordinate order.</param>
    /// <exception cref="NumSharp.ValueError">A bound vector length differs from the decision dimension.</exception>
    protected AgeMoeaProblem(int nVar, int nObj, double[] xl, double[] xu)
    {
        if (xl.Length != nVar || xu.Length != nVar)
            throw new ValueError("`xl`/`xu` length must equal `n_var`.");
        NVar = nVar;
        NObj = nObj;
        Xl = xl;
        Xu = xu;
    }

    /// <summary>Number of decision variables (<c>problem.n_var</c>).</summary>
    public int NVar { get; }

    /// <summary>Number of objectives (<c>problem.n_obj</c>).</summary>
    public int NObj { get; }

    /// <summary>Lower bounds, one per variable (<c>problem.xl</c>).</summary>
    public double[] Xl { get; }

    /// <summary>Upper bounds, one per variable (<c>problem.xu</c>).</summary>
    public double[] Xu { get; }

    /// <summary>
    ///     Evaluate a batch of decision vectors, returning per-row objective values <c>F</c> and the
    ///     aggregated constraint violation <c>CV</c> (all zeros when unconstrained).
    /// </summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>Objective rows and aggregate constraint violations aligned with the input individuals.</returns>
    public abstract (double[][] F, double[] CV) Evaluate(double[][] x);
}

/// <summary>
///     A concrete <see cref="AgeMoeaProblem" /> defined by a delegate — used to wrap synthetic
///     benchmark objectives (ZDT/DTLZ, Mimicry-style boxes) for the parity oracle and tests.
/// </summary>
internal sealed class FunctionProblem : AgeMoeaProblem
{
    private readonly Func<double[], double[]> _f;
    private readonly Func<double[], double>? _cv;

    /// <summary>Adapts objective and optional constraint callbacks to the population evaluation interface.</summary>
    /// <param name="nVar">Number of decision coordinates per individual.</param>
    /// <param name="nObj">Number of objective coordinates per evaluation.</param>
    /// <param name="xl">Lower decision bounds in coordinate order.</param>
    /// <param name="xu">Upper decision bounds in coordinate order.</param>
    /// <param name="f">Callback returning objective values for one decision vector.</param>
    /// <param name="cv">Constraint-violation values or callback supplying them in individual order.</param>
    /// <exception cref="NumSharp.ValueError">A bound vector length differs from the decision dimension.</exception>
    public FunctionProblem(int nVar, int nObj, double[] xl, double[] xu,
        Func<double[], double[]> f, Func<double[], double>? cv = null)
        : base(nVar, nObj, xl, xu)
    {
        _f = f;
        _cv = cv;
    }

    /// <inheritdoc />
    /// <exception cref="System.Exception">An objective or constraint callback fails; its exception propagates.</exception>
    public override (double[][] F, double[] CV) Evaluate(double[][] x)
    {
        var f = new double[x.Length][];
        var cv = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            f[i] = _f(x[i]);
            cv[i] = _cv?.Invoke(x[i]) ?? 0.0;
        }
        return (f, cv);
    }
}
