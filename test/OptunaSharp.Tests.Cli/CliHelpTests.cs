using System;
using System.IO;
using AwesomeAssertions;
using OptunaSharp.Cli;

namespace OptunaSharp.Tests.Cli;

/// <summary>
///     The help and version surface of the <c>optunasharp</c> tool. Every usage line must name the command users type,
///     as Python's <c>optuna --help</c> starts with <c>usage: optuna</c>. Before the root was named explicitly,
///     System.CommandLine's <c>RootCommand</c> took its name from the entry assembly, so the installed tool printed
///     <c>Usage: OptunaSharp.Cli [command] [options]</c> and an in-process host printed its own name.
/// </summary>
[TestClass]
public class CliHelpTests
{
    /// <summary>Runs the CLI in-process and captures what it writes to standard output.</summary>
    /// <param name="exitCode">The exit code <see cref="CliProgram.Run" /> returned.</param>
    /// <param name="args">The command line, without the executable name.</param>
    /// <returns>Everything written to <see cref="Console.Out" /> during the call.</returns>
    private static string Run(out int exitCode, params string[] args)
    {
        var original = Console.Out;
        var buffer = new StringWriter();
        Console.SetOut(buffer);
        try
        {
            exitCode = CliProgram.Run(args);
        }
        finally
        {
            Console.SetOut(original);
        }
        return buffer.ToString();
    }

    /// <summary>
    ///     Every way to reach the root help (<c>--help</c>, the <c>help</c> subcommand, and no arguments, which Optuna
    ///     maps to <c>help</c>) prints a usage line that starts with <c>optunasharp</c> and never the assembly name.
    /// </summary>
    /// <param name="args">The command line that should print the root help, space-separated; empty for no arguments.</param>
    [TestMethod]
    [DataRow("--help")]
    [DataRow("help")]
    [DataRow("")]
    public void RootHelp_UsageLine_NamesTheToolCommand(string args)
    {
        string output = Run(out int exitCode, args.Length == 0 ? Array.Empty<string>() : args.Split(' '));

        exitCode.Should().Be(0);
        output.Should().Contain($"{CliProgram.CommandName} [command] [options]");
        output.Should().NotContain("OptunaSharp.Cli", "the assembly name is not a command anyone can type");
        CliProgram.CommandName.Should().Be("optunasharp", "it must equal ToolCommandName in OptunaSharp.Cli.csproj");
    }

    /// <summary>A subcommand's usage line is prefixed by the tool command too: <c>optunasharp create-study [options]</c>.</summary>
    [TestMethod]
    public void SubcommandHelp_UsageLine_NamesTheToolCommand()
    {
        string output = Run(out int exitCode, "create-study", "--help");

        exitCode.Should().Be(0);
        output.Should().Contain("optunasharp create-study [options]");
    }

    /// <summary>
    ///     <c>--version</c> still works on the explicitly named root: it exits 0 and prints one non-empty line (the
    ///     entry assembly's version, so the exact text depends on the host running the test).
    /// </summary>
    [TestMethod]
    public void Version_PrintsOneLine_AndSucceeds()
    {
        string output = Run(out int exitCode, "--version");

        exitCode.Should().Be(0);
        output.Trim().Should().NotBeEmpty().And.NotContain("\n");
    }
}
