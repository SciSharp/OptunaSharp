namespace OptunaSharp.Samplers;

/// <summary>
///     The posterior-inference strategy SAASBO uses to fit the SAAS GP each trial (Eriksson &amp; Jankowiak
///     2021, §4.2). The choice trades inference fidelity for runtime and does <b>not</b> change the SAAS
///     function prior itself.
/// </summary>
public enum SaasboInferenceMethod
{
    /// <summary>
    ///     The No-U-Turn Sampler (§4.2.1) — the paper's default and "gold standard". It draws a bag of
    ///     <c>L</c> posterior hyperparameter samples and averages expected improvement over them, which the
    ///     paper finds significantly more robust than MAP (§5.2) because averaging over posterior modes
    ///     mitigates EI's pull toward the domain boundary. It is the more expensive option (Hamiltonian Monte
    ///     Carlo over an <c>O(N³D)</c> gradient per leapfrog step).
    /// </summary>
    Nuts = 0,

    /// <summary>
    ///     Maximum a posteriori (§4.2.2 / §A.2) — the faster point-estimate alternative. It optimizes the log
    ///     posterior with Adam for a few pre-selected shrinkage values <c>τ_s</c> and selects one by
    ///     leave-one-out predictive likelihood, yielding a single hyperparameter estimate rather than a
    ///     sample bag. It needs no <c>α</c> hyperparameter but, per the paper, is less robust than NUTS.
    /// </summary>
    Map = 1
}
