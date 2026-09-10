using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     Polynomial Mutation, ported draw-for-draw from pymoo <c>@0.6.1.3</c> —
///     <c>pymoo.operators.mutation.pm.PM</c> with its <c>mut_pm</c> / <c>mut_binomial</c> functions and
///     the RNG-gating in <c>pymoo.core.mutation.Mutation.do</c>. This is <b>not</b> Optuna's PM
///     (<c>OptunaSharp.Samplers.Nsgaii.PolynomialMutation</c>), which draws one scalar per gene; pymoo
///     draws whole matrices and uses <c>mut_binomial</c> to pick which variables mutate.
///
///     AGE-MOEA-2 constructs <c>PM(eta=20)</c> (<c>prob = 0.9</c>, <c>at_least_once = False</c>);
///     the per-variable probability is <c>prob_var = min(0.5, 1 / n_var)</c>.
///
///     Draw order per call:
///     <list type="number">
///       <item><c>random((n, n_var))</c> — <c>mut_binomial</c>, which variables mutate.</item>
///       <item><c>random(#mut)</c> — the per-position mutation randoms in <c>mut_pm</c>.</item>
///       <item><c>random(n)</c> — the per-individual mutation gate (<c>Mutation.do</c>), applied AFTER
///             the two draws above.</item>
///     </list>
/// </summary>
internal sealed class PolynomialMutation
{
    private readonly double _eta;
    private readonly double _prob;
    private readonly bool _atLeastOnce;

    public PolynomialMutation(double eta = 20, double prob = 0.9, bool atLeastOnce = false)
    {
        // AGE-MOEA-2 uses at_least_once=False. The True branch (row_at_least_once_true) draws an
        // extra np.random.randint per all-False row inside mut_binomial; not needed on this path.
        if (atLeastOnce)
            throw new System.NotSupportedException("PolynomialMutation with at_least_once=True is not ported (AGE-MOEA-2 does not use it).");
        _eta = eta;
        _prob = prob;
        _atLeastOnce = atLeastOnce;
    }

    /// <summary>
    ///     <c>Mutation.do</c>: run <see cref="MutPm" /> over all individuals (consuming its two draws),
    ///     then draw the per-individual gate and keep the mutated vector only where the gate fires.
    /// </summary>
    public double[][] Do(double[][] popX, AgeMoeaProblem problem, NumPyRandom rng)
    {
        int n = popX.Length;
        double probVar = Math.Min(0.5, 1.0 / problem.NVar);

        double[][] xp = MutPm(popX, problem.Xl, problem.Xu, probVar, rng);

        // prob = 0.9; mut = random(n) <= prob  (drawn AFTER _do)
        double[] gate = NpRandom.Random(rng, n);
        var result = new double[n][];
        for (int i = 0; i < n; i++)
            result[i] = gate[i] <= _prob ? xp[i] : (double[])popX[i].Clone();
        return result;
    }

    /// <summary><c>mut_pm</c> — polynomial mutation over the flattened <c>mut_binomial</c>-selected positions.</summary>
    private double[][] MutPm(double[][] x, double[] xl, double[] xu, double probVar, NumPyRandom rng)
    {
        int n = x.Length;
        int nVar = xl.Length;

        // mut = mut_binomial(n, n_var, prob); mut[:, xl == xu] = False
        double[][] binR = NpRandom.Random(rng, n, nVar);
        var mut = new bool[n][];
        for (int i = 0; i < n; i++)
        {
            mut[i] = new bool[nVar];
            for (int j = 0; j < nVar; j++)
                mut[i][j] = binR[i][j] < probVar && xl[j] != xu[j];
        }

        // Xp = copy(X); selected positions overwritten below.
        var xp = new double[n][];
        for (int i = 0; i < n; i++) xp[i] = (double[])x[i].Clone();

        // collect selected positions in C-order (row-major).
        var sel = new List<(int I, int J)>();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < nVar; j++)
                if (mut[i][j]) sel.Add((i, j));
        int count = sel.Count;

        double etaP1 = _eta + 1.0;
        double mutPow = 1.0 / etaP1;

        // rand = random(#mut)
        double[] rand = NpRandom.Random(rng, count);

        for (int k = 0; k < count; k++)
        {
            (int i, int j) = sel[k];
            double xv = x[i][j];
            double lo = xl[j], hi = xu[j];
            double span = hi - lo;
            double delta1 = (xv - lo) / span;
            double delta2 = (hi - xv) / span;
            double r = rand[k];

            double deltaq;
            if (r <= 0.5)
            {
                double xy = 1.0 - delta1;
                double val = 2.0 * r + (1.0 - 2.0 * r) * Math.Pow(xy, etaP1);
                deltaq = Math.Pow(val, mutPow) - 1.0;
            }
            else
            {
                double xy = 1.0 - delta2;
                double val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * Math.Pow(xy, etaP1);
                deltaq = 1.0 - Math.Pow(val, mutPow);
            }

            double y = xv + deltaq * span;
            // back in bounds if necessary (mut_pm's own clamp, before set_to_bounds_if_outside).
            if (y < lo) y = lo;
            if (y > hi) y = hi;
            xp[i][j] = y;
        }

        // set_to_bounds_if_outside (very-unlikely final repair).
        for (int i = 0; i < n; i++)
            for (int j = 0; j < nVar; j++)
            {
                if (xp[i][j] < xl[j]) xp[i][j] = xl[j];
                if (xp[i][j] > xu[j]) xp[i][j] = xu[j];
            }
        return xp;
    }
}
