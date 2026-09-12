using System.Runtime.CompilerServices;

// The interop suite exercises the botorch API this package implements at the FUNCTION level — calling
// the internal candidates_func implementations, the enum→impl `Resolve`, and the tensor marshaling
// directly and asserting byte-exact parity against the real optuna_integration.botorch.*_candidates_func
// over the same torch tensors (BoTorchApiParityTests). That reach needs InternalsVisibleTo.
//
// This package is strong-named (repo-root Directory.Build.props), so the friend name carries PublicKey=
// (a keyless one is a hard CS1726). The literal is Microsoft's PUBLISHED open-source key (token
// cc7b13ffcd2ddd51), shared across the SciSharp STACK — identity, not access control.
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Interop, PublicKey=" + OptunaSharp.Interop.BoTorch.BoTorchFriendKey.PublicKey)]

namespace OptunaSharp.Interop.BoTorch;

/// <summary>The public half of the strong-name key this package is signed with (the SciSharp STACK key).</summary>
internal static class BoTorchFriendKey
{
    /// <summary>The full strong-name public key matched by the friend-assembly declaration.</summary>
    internal const string PublicKey =
        "00240000048000009400000006020000002400005253413100040000010001004b86c4cb78549b34" +
        "bab61a3b1800e23bfeb5b3ec390074041536a7e3cbd97f5f04cf0f857155a8928eaa29ebfd11cfbb" +
        "ad3ba70efea7bda3226c6a8d370a4cd303f714486b6ebc225985a638471e6ef571cc92a4613c00b8" +
        "fa65d61ccee0cbe5f36330c9a01f4183559f1bef24cc2917c6d913e3a541333a1d05d9bed22b38cb";
}
