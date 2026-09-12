namespace OptunaSharp.Pruners;

/// <summary>
///     Pruner which never prunes trials. Port of <c>optuna/pruners/_nop.py: NopPruner</c>.
/// </summary>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.NopPruner.html">Official Optuna NopPruner reference.</seealso>
public class NopPruner : BasePruner
{
    /// <inheritdoc />
    public override bool Prune(Study study, FrozenTrial trial) => false;
}
