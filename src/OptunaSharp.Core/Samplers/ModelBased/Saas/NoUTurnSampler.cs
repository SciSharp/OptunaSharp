// The No-U-Turn Sampler (NUTS) — SAASBO's default inference engine (Eriksson & Jankowiak 2021, §4.2.1,
// "the gold standard for inference in models like ours"). This is a faithful managed implementation of the
// efficient recursive NUTS with dual-averaging step-size adaptation of Hoffman & Gelman, "The No-U-Turn
// Sampler", JMLR 15 (2014), Algorithms 3-6, generalized to a diagonal mass matrix (metric) that is adapted
// from the warmup trace — the same three ingredients (recursive tree doubling, dual-averaging ε, diagonal
// metric) that NumPyro's default NUTS gives SAASBO in the reference implementation.
//
// It is a GENERAL sampler over any differentiable unconstrained log-density; SAASBO drives it with the
// reparameterized SAAS log-joint (SaasModel.LogJointAndGrad). It is seeded through NumSharp's MT19937
// (NumPyRandom), never System.Random, so a fixed seed reproduces the chain on a platform — SAASBO's NUTS
// is validated behaviorally (it recovers the paper's sparse relevant subspace), not byte-exact vs NumPyro.

using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.ModelBased.Saas;

/// <summary>
///     A No-U-Turn Sampler over a differentiable unconstrained target density (Hoffman &amp; Gelman 2014).
///     It draws momentum under a diagonal mass matrix, builds a balanced binary tree of leapfrog states
///     until a U-turn or divergence, and slice-samples a new state from the tree; the leapfrog step size is
///     tuned during warmup by dual averaging toward a target acceptance rate, and the diagonal metric is
///     estimated from the warmup trace. SAASBO uses it to sample the SAAS posterior over kernel
///     hyperparameters (§4.2.1); <see cref="Sample" /> returns the post-warmup draws (the caller thins them
///     to the <c>L</c> retained samples of Eqn (10)).
///
///     <para><b>Correctness vs. efficiency.</b> NUTS targets the correct stationary distribution for any
///     fixed positive step size and metric; the dual-averaging and metric adaptation only improve mixing.
///     The metric is estimated with a Stan-style three-phase warmup (an initial step-size buffer, a metric
///     window, and a terminal step-size buffer), degrading gracefully to step-size-only adaptation when the
///     warmup is too short to estimate a full diagonal metric.</para>
/// </summary>
internal sealed class NoUTurnSampler
{
    private const double DeltaMax = 1000.0;   // divergence threshold on the joint log-density (Hoffman & Gelman §3.2)
    // Nesterov dual-averaging constants (Hoffman & Gelman §3.2.1, their recommended defaults).
    private const double Gamma = 0.05;
    private const double T0 = 10.0;
    private const double Kappa = 0.75;

    private readonly int _numWarmup;
    private readonly int _numSamples;
    private readonly int _maxTreeDepth;
    private readonly double _targetAccept;

    /// <summary>Configures the sampler's warmup/sampling budget and adaptation targets.</summary>
    /// <param name="numWarmup">The number of warmup (burn-in + adaptation) iterations, discarded from the output.</param>
    /// <param name="numSamples">The number of post-warmup iterations to return.</param>
    /// <param name="maxTreeDepth">The maximum tree depth per iteration (SAASBO's paper value is 6, capping the
    ///     trajectory at <c>2⁶</c> leapfrog steps).</param>
    /// <param name="targetAccept">The dual-averaging target acceptance probability (the standard <c>0.8</c>).</param>
    /// <exception cref="System.ArgumentException">A budget is negative, or <paramref name="targetAccept" /> is not in <c>(0, 1)</c>.</exception>
    internal NoUTurnSampler(int numWarmup, int numSamples, int maxTreeDepth = 6, double targetAccept = 0.8)
    {
        if (numWarmup < 0)
            throw new ArgumentException("numWarmup must be non-negative.", nameof(numWarmup));
        if (numSamples < 0)
            throw new ArgumentException("numSamples must be non-negative.", nameof(numSamples));
        if (maxTreeDepth < 1)
            throw new ArgumentException("maxTreeDepth must be at least 1.", nameof(maxTreeDepth));
        if (!(targetAccept > 0.0 && targetAccept < 1.0))
            throw new ArgumentException("targetAccept must be in (0, 1).", nameof(targetAccept));
        _numWarmup = numWarmup;
        _numSamples = numSamples;
        _maxTreeDepth = maxTreeDepth;
        _targetAccept = targetAccept;
    }

    // The target log-density and its gradient — set once per Sample call so the recursive tree builder can
    // reach it without threading the delegate through every frame.
    private Func<double[], (double LogP, double[] Grad)> _target = null!;
    private double[] _invMetric = Array.Empty<double>();   // diagonal M⁻¹ (posterior-variance estimate)
    private double[] _momentumStd = Array.Empty<double>();  // per-coordinate momentum std = √(M) = √(1/invMetric)
    private int _dim;

    /// <summary>
    ///     Runs the sampler from <paramref name="initTheta" /> and returns the post-warmup draws (each a copy
    ///     of the latent vector). Momentum, slice, direction, and acceptance randomness are all drawn from
    ///     <paramref name="rng" />, so the whole chain is a deterministic function of the RNG state.
    /// </summary>
    /// <param name="logJointAndGrad">The unconstrained target: value and gradient of the log-density at a point.</param>
    /// <param name="initTheta">The starting latent vector; must be in the target's support (finite log-density).</param>
    /// <param name="rng">The MT19937 stream driving all sampler randomness.</param>
    /// <returns>The list of <see cref="_numSamples" /> post-warmup latent vectors.</returns>
    /// <exception cref="System.ArgumentException">The starting log-density is not finite.</exception>
    internal List<double[]> Sample(
        Func<double[], (double LogP, double[] Grad)> logJointAndGrad, double[] initTheta, NumPyRandom rng)
    {
        _target = logJointAndGrad;
        _dim = initTheta.Length;

        // Start from an identity metric; the warmup schedule replaces it with the estimated diagonal variance.
        _invMetric = new double[_dim];
        _momentumStd = new double[_dim];
        for (int d = 0; d < _dim; d++) { _invMetric[d] = 1.0; _momentumStd[d] = 1.0; }

        var theta = (double[])initTheta.Clone();
        (double logp, double[] grad) = _target(theta);
        if (!IsFinite(logp))
            throw new ArgumentException("The initial log-density is not finite.", nameof(initTheta));

        // ---- warmup schedule (Stan-style three phases) ----------------------------------------------
        // A metric window is only used when the warmup is long enough to estimate D independent variances.
        int initBuffer = Math.Max(1, (int)Math.Round(0.15 * _numWarmup));
        int termBuffer = Math.Max(1, (int)Math.Round(0.10 * _numWarmup));
        int windowStart = initBuffer;
        int windowEnd = _numWarmup - termBuffer;   // exclusive upper bound of the metric-accumulation window
        bool adaptMetric = _numWarmup >= 20 && windowEnd - windowStart >= _dim;

        // Dual-averaging state for the step size ε.
        double eps = FindReasonableEpsilon(theta, grad, logp, rng);
        double mu = Math.Log(10.0 * eps);
        double logEbar = 0.0;   // log of the averaged step size
        double hbar = 0.0;
        int adaptStep = 0;      // dual-averaging iteration counter (reset on each restart)

        // Welford accumulator for the diagonal metric.
        var welfordMean = new double[_dim];
        var welfordM2 = new double[_dim];
        int welfordCount = 0;

        for (int m = 1; m <= _numWarmup; m++)
        {
            (theta, grad, logp, double alphaStat) = Transition(theta, grad, logp, eps, rng);

            // Dual-average the step size toward the target acceptance rate.
            adaptStep++;
            double eta = 1.0 / (adaptStep + T0);
            hbar = (1.0 - eta) * hbar + eta * (_targetAccept - alphaStat);
            double logEps = mu - Math.Sqrt(adaptStep) / Gamma * hbar;
            eps = Math.Exp(logEps);
            double etaBar = Math.Pow(adaptStep, -Kappa);
            logEbar = etaBar * logEps + (1.0 - etaBar) * logEbar;

            if (adaptMetric && m >= windowStart && m < windowEnd)
            {
                // Online (Welford) diagonal-variance accumulation over the metric window.
                welfordCount++;
                for (int d = 0; d < _dim; d++)
                {
                    double delta = theta[d] - welfordMean[d];
                    welfordMean[d] += delta / welfordCount;
                    welfordM2[d] += delta * (theta[d] - welfordMean[d]);
                }
            }

            if (adaptMetric && m == windowEnd - 1 && welfordCount > 1)
            {
                // Close the metric window: set M⁻¹ to the regularized sample variance, then restart the
                // step-size dual averaging from a fresh reasonable ε under the new metric (as Stan does).
                for (int d = 0; d < _dim; d++)
                {
                    double var = welfordM2[d] / (welfordCount - 1);
                    // Stan's regularization: shrink toward 1e-3 with weight 5 pseudo-observations.
                    double reg = (welfordCount / (welfordCount + 5.0)) * var + (5.0 / (welfordCount + 5.0)) * 1e-3;
                    _invMetric[d] = reg > 0.0 ? reg : 1.0;
                    _momentumStd[d] = Math.Sqrt(1.0 / _invMetric[d]);
                }
                eps = FindReasonableEpsilon(theta, grad, logp, rng);
                mu = Math.Log(10.0 * eps);
                logEbar = 0.0;
                hbar = 0.0;
                adaptStep = 0;
            }
        }

        // After warmup freeze the averaged step size (unless warmup was empty, then keep the reasonable ε).
        if (_numWarmup > 0)
            eps = Math.Exp(logEbar);

        // ---- sampling phase (fixed ε and metric) ----------------------------------------------------
        var samples = new List<double[]>(_numSamples);
        for (int m = 0; m < _numSamples; m++)
        {
            (theta, grad, logp, _) = Transition(theta, grad, logp, eps, rng);
            samples.Add((double[])theta.Clone());
        }
        return samples;
    }

    /// <summary>
    ///     One NUTS transition (Hoffman &amp; Gelman Alg. 6, main loop): sample momentum and a slice, then
    ///     grow a balanced leapfrog tree in alternating random directions until a U-turn or divergence,
    ///     progressively replacing the sample by a uniformly chosen new leaf.
    /// </summary>
    /// <param name="theta0">The current latent vector.</param>
    /// <param name="grad0">The gradient of the log-density at <paramref name="theta0" />.</param>
    /// <param name="logp0">The log-density at <paramref name="theta0" />.</param>
    /// <param name="eps">The current leapfrog step size.</param>
    /// <param name="rng">The MT19937 stream.</param>
    /// <returns>The next latent vector, its gradient and log-density, and the mean Metropolis acceptance statistic.</returns>
    private (double[] Theta, double[] Grad, double Logp, double AlphaStat) Transition(
        double[] theta0, double[] grad0, double logp0, double eps, NumPyRandom rng)
    {
        double[] r0 = SampleMomentum(rng);
        double jointLog0 = logp0 - Kinetic(r0);

        // Slice threshold in log-space: log u = jointLog0 − Exponential(1).
        double logu = jointLog0 - StandardExponential(rng);

        double[] thetaMinus = theta0, rMinus = r0, gradMinus = grad0;
        double[] thetaPlus = theta0, rPlus = r0, gradPlus = grad0;
        double[] thetaSample = theta0, gradSample = grad0;
        double logpSample = logp0;

        int n = 1;            // # of valid (in-slice) points in the tree so far
        bool s = true;        // continue growing?
        double alphaSum = 0.0;
        int nAlpha = 0;

        for (int depth = 0; depth < _maxTreeDepth && s; depth++)
        {
            int v = Uniform(rng) < 0.5 ? -1 : 1;
            Tree tree;
            if (v == -1)
            {
                tree = BuildTree(thetaMinus, rMinus, gradMinus, logu, v, depth, eps, jointLog0, rng);
                thetaMinus = tree.ThetaMinus; rMinus = tree.RMinus; gradMinus = tree.GradMinus;
            }
            else
            {
                tree = BuildTree(thetaPlus, rPlus, gradPlus, logu, v, depth, eps, jointLog0, rng);
                thetaPlus = tree.ThetaPlus; rPlus = tree.RPlus; gradPlus = tree.GradPlus;
            }

            // Metropolis-within-tree: accept the subtree's proposal with prob min(1, n'/n).
            if (tree.SPrime && tree.NPrime > 0 && Uniform(rng) < (double)tree.NPrime / n)
            {
                thetaSample = tree.ThetaProp;
                gradSample = tree.GradProp;
                logpSample = tree.LogpProp;
            }

            alphaSum += tree.AlphaSum;
            nAlpha += tree.NAlpha;
            n += tree.NPrime;
            s = tree.SPrime && NoUTurn(thetaMinus, thetaPlus, rMinus, rPlus);
        }

        double alphaStat = nAlpha > 0 ? alphaSum / nAlpha : 0.0;
        return (thetaSample, gradSample, logpSample, alphaStat);
    }

    /// <summary>The recursive subtree state returned by <see cref="BuildTree" />.</summary>
    private struct Tree
    {
        /// <summary>The subtree's leftmost (backward) endpoint: position, momentum and log-density gradient.</summary>
        public double[] ThetaMinus, RMinus, GradMinus;
        /// <summary>The subtree's rightmost (forward) endpoint: position, momentum and log-density gradient.</summary>
        public double[] ThetaPlus, RPlus, GradPlus;
        /// <summary>The uniformly chosen in-slice proposal (position and its gradient) the subtree contributes.</summary>
        public double[] ThetaProp, GradProp;
        /// <summary>The log-density at <see cref="ThetaProp" />, kept so the proposal is never re-evaluated.</summary>
        public double LogpProp;
        /// <summary>The number of in-slice leaves (<c>n′</c>); weights the uniform merge of two subtrees' proposals.</summary>
        public int NPrime;
        /// <summary>The continue flag (<c>s′</c>): <c>false</c> once a leaf diverged or the span made a U-turn.</summary>
        public bool SPrime;
        /// <summary><c>Σ min(1, exp(jointLog − jointLog0))</c> over the leaves — the dual-averaging acceptance statistic.</summary>
        public double AlphaSum;
        /// <summary>The number of leaves summed into <see cref="AlphaSum" />.</summary>
        public int NAlpha;
    }

    /// <summary>
    ///     Recursively builds one leapfrog subtree in direction <paramref name="v" /> (Hoffman &amp; Gelman
    ///     Alg. 6, BuildTree). At depth zero it takes a single leapfrog step and tests the slice/divergence
    ///     conditions; at higher depth it builds two subtrees, merges their proposals uniformly, and checks
    ///     the no-U-turn condition across the merged span.
    /// </summary>
    /// <param name="theta">The endpoint to extend from.</param>
    /// <param name="r">The momentum at that endpoint.</param>
    /// <param name="grad">The gradient of the log-density at that endpoint.</param>
    /// <param name="logu">The log-slice threshold.</param>
    /// <param name="v">The direction (<c>-1</c> backward, <c>+1</c> forward) in fictitious time.</param>
    /// <param name="depth">The current tree depth (a subtree of <c>2^depth</c> leapfrog steps).</param>
    /// <param name="eps">The leapfrog step size.</param>
    /// <param name="jointLog0">The joint log-density of the iteration's start state (for the acceptance statistic).</param>
    /// <param name="rng">The MT19937 stream (for the uniform proposal merge).</param>
    /// <returns>The merged subtree state.</returns>
    private Tree BuildTree(
        double[] theta, double[] r, double[] grad, double logu, int v, int depth, double eps,
        double jointLog0, NumPyRandom rng)
    {
        if (depth == 0)
        {
            // Base case: one leapfrog step in direction v.
            (double[] thetaP, double[] rP, double[] gradP, double logpP) = Leapfrog(theta, r, grad, v * eps);
            double jointLog = logpP - Kinetic(rP);
            bool finite = IsFinite(jointLog);

            int nPrime = finite && logu <= jointLog ? 1 : 0;
            bool sPrime = finite && logu < jointLog + DeltaMax;
            double alpha = finite ? Math.Min(1.0, Math.Exp(jointLog - jointLog0)) : 0.0;

            return new Tree
            {
                ThetaMinus = thetaP, RMinus = rP, GradMinus = gradP,
                ThetaPlus = thetaP, RPlus = rP, GradPlus = gradP,
                ThetaProp = thetaP, GradProp = gradP, LogpProp = logpP,
                NPrime = nPrime, SPrime = sPrime, AlphaSum = alpha, NAlpha = 1
            };
        }

        // Recursion: build the first half-subtree, then (if it survives) the second in the same direction.
        Tree left = BuildTree(theta, r, grad, logu, v, depth - 1, eps, jointLog0, rng);
        if (!left.SPrime)
            return left;   // a divergence in the first half aborts the whole subtree.

        Tree outer = default;
        if (v == -1)
        {
            Tree second = BuildTree(left.ThetaMinus, left.RMinus, left.GradMinus, logu, v, depth - 1, eps, jointLog0, rng);
            outer.ThetaMinus = second.ThetaMinus; outer.RMinus = second.RMinus; outer.GradMinus = second.GradMinus;
            outer.ThetaPlus = left.ThetaPlus; outer.RPlus = left.RPlus; outer.GradPlus = left.GradPlus;
            outer = MergeSubtrees(outer, left, second, rng);
        }
        else
        {
            Tree second = BuildTree(left.ThetaPlus, left.RPlus, left.GradPlus, logu, v, depth - 1, eps, jointLog0, rng);
            outer.ThetaMinus = left.ThetaMinus; outer.RMinus = left.RMinus; outer.GradMinus = left.GradMinus;
            outer.ThetaPlus = second.ThetaPlus; outer.RPlus = second.RPlus; outer.GradPlus = second.GradPlus;
            outer = MergeSubtrees(outer, left, second, rng);
        }
        return outer;
    }

    /// <summary>
    ///     Merges the two half-subtrees of <see cref="BuildTree" />: choose the proposal uniformly by leaf
    ///     count (prob <c>n''/(n'+n'')</c> for the second half), sum the acceptance statistics and slice
    ///     counts, and combine the divergence flag with the no-U-turn test across the full merged span.
    /// </summary>
    /// <param name="span">A tree pre-filled with the merged left/right endpoints.</param>
    /// <param name="first">The first (already-accepted) half-subtree.</param>
    /// <param name="second">The second half-subtree.</param>
    /// <param name="rng">The MT19937 stream (for the uniform proposal choice).</param>
    /// <returns>The merged subtree state.</returns>
    private Tree MergeSubtrees(Tree span, Tree first, Tree second, NumPyRandom rng)
    {
        int nTotal = first.NPrime + second.NPrime;
        // Default: keep the first half's proposal; switch to the second's with prob n''/(n'+n'').
        span.ThetaProp = first.ThetaProp; span.GradProp = first.GradProp; span.LogpProp = first.LogpProp;
        if (second.NPrime > 0 && nTotal > 0 && Uniform(rng) < (double)second.NPrime / nTotal)
        {
            span.ThetaProp = second.ThetaProp; span.GradProp = second.GradProp; span.LogpProp = second.LogpProp;
        }

        span.NPrime = nTotal;
        span.AlphaSum = first.AlphaSum + second.AlphaSum;
        span.NAlpha = first.NAlpha + second.NAlpha;
        span.SPrime = second.SPrime && NoUTurn(span.ThetaMinus, span.ThetaPlus, span.RMinus, span.RPlus);
        return span;
    }

    /// <summary>
    ///     A single leapfrog (Störmer-Verlet) step of size <paramref name="signedEps" /> under the diagonal
    ///     metric: half-kick the momentum, drift the position by <c>ε·M⁻¹r</c>, then half-kick again with the
    ///     new gradient. Returns the moved state and its (recomputed) gradient/log-density.
    /// </summary>
    /// <param name="theta">The starting position.</param>
    /// <param name="r">The starting momentum.</param>
    /// <param name="grad">The gradient of the log-density at <paramref name="theta" />.</param>
    /// <param name="signedEps">The (direction-signed) step size <c>v·ε</c>.</param>
    /// <returns>The stepped position, momentum, gradient, and log-density.</returns>
    private (double[] Theta, double[] R, double[] Grad, double Logp) Leapfrog(
        double[] theta, double[] r, double[] grad, double signedEps)
    {
        var rHalf = new double[_dim];
        for (int d = 0; d < _dim; d++)
            rHalf[d] = r[d] + 0.5 * signedEps * grad[d];

        var thetaNew = new double[_dim];
        for (int d = 0; d < _dim; d++)
            thetaNew[d] = theta[d] + signedEps * _invMetric[d] * rHalf[d];   // drift by ε·M⁻¹r

        (double logpNew, double[] gradNew) = _target(thetaNew);

        var rNew = new double[_dim];
        for (int d = 0; d < _dim; d++)
            rNew[d] = rHalf[d] + 0.5 * signedEps * gradNew[d];

        return (thetaNew, rNew, gradNew, logpNew);
    }

    /// <summary>
    ///     Finds a reasonable initial step size by doubling/halving <c>ε</c> until one leapfrog step crosses
    ///     the acceptance-probability threshold of <c>0.5</c> (Hoffman &amp; Gelman Alg. 4). The same momentum
    ///     draw is reused across the search so the comparison isolates the effect of <c>ε</c>.
    /// </summary>
    /// <param name="theta">The current position.</param>
    /// <param name="grad">The gradient at <paramref name="theta" />.</param>
    /// <param name="logp">The log-density at <paramref name="theta" />.</param>
    /// <param name="rng">The MT19937 stream (for the trial momentum).</param>
    /// <returns>A step size whose one-step acceptance probability is near <c>0.5</c>.</returns>
    private double FindReasonableEpsilon(double[] theta, double[] grad, double logp, NumPyRandom rng)
    {
        double eps = 1.0;
        double[] r = SampleMomentum(rng);
        double jointLog0 = logp - Kinetic(r);

        (double[] _, double[] rP, double[] __, double logpP) = Leapfrog(theta, r, grad, eps);
        double jointLog = logpP - Kinetic(rP);
        double logRatio = jointLog - jointLog0;

        // a = +1 if the first step already accepts well (ratio > log ½), else -1.
        int a = IsFinite(logRatio) && logRatio > Math.Log(0.5) ? 1 : -1;

        // Keep scaling ε by 2^a until the acceptance probability crosses ½ (i.e. a·logRatio ≤ -a·log2).
        int guard = 0;
        while ((IsFinite(logRatio) ? a * logRatio : double.NegativeInfinity) > -a * Math.Log(2.0) && guard++ < 100)
        {
            eps *= Math.Pow(2.0, a);
            (_, rP, __, logpP) = Leapfrog(theta, r, grad, eps);
            jointLog = logpP - Kinetic(rP);
            logRatio = jointLog - jointLog0;
        }
        return eps;
    }

    /// <summary>The generalized no-U-turn condition under the diagonal metric: the trajectory span still has
    ///     a non-negative projection onto the velocity (<c>M⁻¹r</c>) at both endpoints.</summary>
    /// <param name="thetaMinus">The backward endpoint position.</param>
    /// <param name="thetaPlus">The forward endpoint position.</param>
    /// <param name="rMinus">The backward endpoint momentum.</param>
    /// <param name="rPlus">The forward endpoint momentum.</param>
    /// <returns><c>true</c> to keep growing the tree; <c>false</c> when a U-turn has occurred.</returns>
    private bool NoUTurn(double[] thetaMinus, double[] thetaPlus, double[] rMinus, double[] rPlus)
    {
        double dotMinus = 0.0, dotPlus = 0.0;
        for (int d = 0; d < _dim; d++)
        {
            double span = thetaPlus[d] - thetaMinus[d];
            dotMinus += span * _invMetric[d] * rMinus[d];   // span · velocity at the backward end
            dotPlus += span * _invMetric[d] * rPlus[d];     // span · velocity at the forward end
        }
        return dotMinus >= 0.0 && dotPlus >= 0.0;
    }

    /// <summary>Draws a momentum vector <c>r ~ N(0, M)</c> under the diagonal metric (component std <c>√M</c>).</summary>
    /// <param name="rng">The MT19937 stream.</param>
    /// <returns>A momentum vector with the metric's marginal variances.</returns>
    private double[] SampleMomentum(NumPyRandom rng)
    {
        double[] z = NumPyDraws.Randn(rng, _dim);
        var r = new double[_dim];
        for (int d = 0; d < _dim; d++)
            r[d] = z[d] * _momentumStd[d];
        return r;
    }

    /// <summary>The kinetic energy <c>½ rᵀM⁻¹r = ½ Σ_d invMetric_d · r_d²</c> under the diagonal metric.</summary>
    /// <param name="r">The momentum vector.</param>
    /// <returns>The kinetic energy of the momentum under the current metric.</returns>
    private double Kinetic(double[] r)
    {
        double k = 0.0;
        for (int d = 0; d < _dim; d++)
            k += _invMetric[d] * r[d] * r[d];
        return 0.5 * k;
    }

    /// <summary>Draws a standard Exponential(1) variate <c>−log U</c> for the log-slice threshold.</summary>
    /// <param name="rng">The MT19937 stream.</param>
    /// <returns>An Exponential(1) sample.</returns>
    private static double StandardExponential(NumPyRandom rng)
    {
        // Guard the log against an exact-zero uniform draw (probability ~0 but not impossible).
        double u = Uniform(rng);
        return -Math.Log(u > 0.0 ? u : double.Epsilon);
    }

    /// <summary>Draws one scalar uniform in <c>[0, 1)</c> — the NumSharp <c>rand()</c> returns a scalar NDArray that casts to double.</summary>
    /// <param name="rng">The MT19937 stream.</param>
    /// <returns>A uniform sample in <c>[0, 1)</c>.</returns>
    private static double Uniform(NumPyRandom rng) => (double)rng.rand();

    /// <summary>Whether a value is finite (neither NaN nor infinite) — used to treat divergent leaps as rejections.</summary>
    /// <param name="x">The value to test.</param>
    /// <returns><c>true</c> when <paramref name="x" /> is a finite real number.</returns>
    private static bool IsFinite(double x) => !double.IsNaN(x) && !double.IsInfinity(x);
}
