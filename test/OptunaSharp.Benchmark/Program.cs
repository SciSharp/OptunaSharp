using System.Reflection;
using BenchmarkDotNet.Running;

// Standard BenchmarkDotNet entry point. Run all benchmarks:
//   dotnet run -c Release -f net8.0 -- --filter '*'
// Run a quick pass (fewer iterations):
//   dotnet run -c Release -f net8.0 -- --filter '*' --job short
BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly()).Run(args);
