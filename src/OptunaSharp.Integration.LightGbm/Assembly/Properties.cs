using System.Runtime.CompilerServices;

// Friend assemblies. Every name carries PublicKey= because OptunaSharp.Integration.LightGbm is
// strong-named (see Directory.Build.props at the repo root): the compiler rejects a keyless
// InternalsVisibleTo from a signed assembly with CS1726.
//
// The key is Microsoft's PUBLISHED open-source key (token cc7b13ffcd2ddd51), shared with the rest of
// the SciSharp STACK — the same key Core signs with. Its private half is public by design, so this is
// NOT an access control; it is a compile-time convenience. The literal is duplicated from Core's
// OptunaSharpFriendKey (which is internal to Core, hence not reusable across the assembly boundary).

// The unit-test project reaches this package's internal LightGBM backend (LightGbmBackend) + the shared
// MlNetTreeEnsembleAdapter linked into this assembly.
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Integration.LightGbm, PublicKey=" + OptunaSharpLightGbmFriendKey.PublicKey)]

/// <summary>
///     The public half of the strong-name key OptunaSharp.Integration.LightGbm is signed with, as the
///     hex literal <see cref="InternalsVisibleToAttribute" /> expects. Mirrors Core's
///     <c>OptunaSharpFriendKey.PublicKey</c> and the repo-root <c>Directory.Build.props</c>
///     (Microsoft's published open-source key, token <c>cc7b13ffcd2ddd51</c>).
/// </summary>
internal static class OptunaSharpLightGbmFriendKey
{
    internal const string PublicKey =
        "00240000048000009400000006020000002400005253413100040000010001004b86c4cb78549b34" +
        "bab61a3b1800e23bfeb5b3ec390074041536a7e3cbd97f5f04cf0f857155a8928eaa29ebfd11cfbb" +
        "ad3ba70efea7bda3226c6a8d370a4cd303f714486b6ebc225985a638471e6ef571cc92a4613c00b8" +
        "fa65d61ccee0cbe5f36330c9a01f4183559f1bef24cc2917c6d913e3a541333a1d05d9bed22b38cb";
}
