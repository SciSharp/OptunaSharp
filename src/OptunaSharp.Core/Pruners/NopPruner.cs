namespace OptunaSharp.Pruners;

/// <summary>
///     Pruner which never prunes trials. Port of <c>optuna/pruners/_nop.py: NopPruner</c>.
/// </summary>
public class NopPruner : BasePruner
{
    public override bool Prune(Study study, FrozenTrial trial) => false;
}
