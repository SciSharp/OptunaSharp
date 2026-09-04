using System.Runtime.CompilerServices;

// Friend assemblies. Every name carries PublicKey= because OptunaSharp is strong-named (see
// Directory.Build.props at the repo root): the compiler rejects a keyless InternalsVisibleTo from a
// signed assembly with CS1726.
//
// The key is Microsoft's PUBLISHED open-source key (token cc7b13ffcd2ddd51), shared with the rest of
// the SciSharp STACK. Its private half is public by design, so this list is NOT an access control —
// anyone can build an assembly with a matching name and key. It is a compile-time convenience.
//
// The literal lives once, in OptunaSharpFriendKey.PublicKey below; `"name, PublicKey=" + const` is a
// compile-time constant, which is what an attribute argument requires. Global attributes must precede
// type declarations, so the holder class follows them.

[assembly: InternalsVisibleTo("OptunaSharp.Tests, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// The Python.NET interop suite exercises the same internals while comparing against real Optuna.
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Interop, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// `dotnet run` file-based scripts that override AssemblyName to this (see the repo README ->
// "Scripting with dotnet run"). A script UNDER the repo root inherits Directory.Build.props and is
// signed automatically; one run from elsewhere must pass the signing properties itself.
[assembly: InternalsVisibleTo("OptunaSharp.DotNetRunScript, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// The Redis journal backend (satellite package OptunaSharp.Storage.Redis) serializes logs through
// the same internal PyJson (so the on-wire bytes stay byte-compatible with Python Optuna) and raises
// the experimental-class warning via OptunaWarnings/WarningCategory — both internal to Core.
[assembly: InternalsVisibleTo("OptunaSharp.Storage.Redis, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// Its test project asserts that experimental warning fired via the internal OptunaWarnings.WarningRaised
// event (same reason the interop suite is a friend).
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Storage.Redis, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

/// <summary>
///     The public half of the strong-name key OptunaSharp is signed with, as the hex literal
///     <see cref="InternalsVisibleToAttribute" /> expects.
/// </summary>
/// <remarks>
///     Mirrors the <c>OptunaSharpPublicKey</c> property in the repo-root <c>Directory.Build.props</c>
///     and the key asserted by <c>.github/scripts/verify_strong_name.cs</c>. It is Microsoft's
///     published open-source key (token <c>cc7b13ffcd2ddd51</c>), shared across the SciSharp STACK.
/// </remarks>
internal static class OptunaSharpFriendKey
{
    internal const string PublicKey =
        "00240000048000009400000006020000002400005253413100040000010001004b86c4cb78549b34" +
        "bab61a3b1800e23bfeb5b3ec390074041536a7e3cbd97f5f04cf0f857155a8928eaa29ebfd11cfbb" +
        "ad3ba70efea7bda3226c6a8d370a4cd303f714486b6ebc225985a638471e6ef571cc92a4613c00b8" +
        "fa65d61ccee0cbe5f36330c9a01f4183559f1bef24cc2917c6d913e3a541333a1d05d9bed22b38cb";
}
