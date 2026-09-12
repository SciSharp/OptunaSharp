# In-code documentation

The maintained C# port under `src/` documents public, protected, and internal declarations,
including members of internal implementation types. Methods and constructors also document their
parameters, generic parameters, non-void results, and relevant exception contracts. Private methods
are included in the coverage audit. Generated `obj`/`bin` output and explicitly generated source
are excluded; reference submodules, test fixtures, and examples are separate from the shipped port.

Behavior descriptions are checked against the [official Optuna API reference](https://optuna.readthedocs.io/en/stable/reference/index.html)
and the pinned `refs/optuna` source. The pinned source resolves version differences or discrepancies
between upstream prose and implementation. Integration contracts also use `refs/optuna-integration`.
C# adaptations, runtime ownership, and extensions beyond Optuna are described as such in their code.

Use XML `summary`, `param`, `typeparam`, `returns`, and `exception` elements as applicable. Use
`inheritdoc` for an actual inherited or explicitly referenced contract; document additional
overload parameters and any changed behavior. If a member supplies local parameter tags, supply
the complete parameter list so the C# compiler can validate it. Inline comments explain why
validation order, dispatch priority, copying, synchronization, and cleanup behavior matter.

Exception documentation describes validation and supported failure paths, including propagation
from delegates and external providers. Open-ended provider or callback failures cannot be listed
as a closed set of concrete types; their propagation is described explicitly. Do not claim a
method cannot throw merely because it contains no `throw` statement. Likewise, a caught failure
must not be described as escaping the method. Resource exhaustion and arbitrary violations of
internal invariants are not a useful closed exception contract.

Run the structural check with .NET 10 from the repository root:

```powershell
dotnet run .github/scripts/check_documentation.cs -- src
```

For a review inventory including signatures, missing tags, and explicit throw candidates:

```powershell
dotnet run .github/scripts/check_documentation.cs -- src --inventory documentation-inventory.json
```

The [documentation workflow](../.github/workflows/documentation.yml) also compiles all source
projects for their declared target frameworks with XML documentation warnings promoted to errors,
overriding the legacy warning suppression used by other build jobs.

The syntax audit measures structural coverage, not semantic completeness. It cannot prove the
accuracy of prose, resolve every inheritance relationship, discover every indirectly propagated
exception, or judge whether a method body explains its non-obvious decisions. Those remain code
review responsibilities; a zero-issue audit is one verification step, not a substitute for them.
