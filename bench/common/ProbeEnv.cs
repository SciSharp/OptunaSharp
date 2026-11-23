// Shared by every benchmark probe in bench/ (compiled in with <Compile Include>, beside Bench.cs, which holds the
// measurement protocol), so no probe's own measurement code has to know which operating system it runs on. It replaces the
// Windows-only ThrottleOptOut.cs of the 2026-11-15/16 workstation runs.
using System.Diagnostics;
using System.Runtime;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Prepares and describes the benchmark process before <c>Main</c> runs: on Windows it opts the process out of
/// execution-speed power throttling (EcoQoS), and on every system it prints one <c>probe env:</c> line stating the
/// runtime, the garbage collector and the CPUs the process may run on.
/// </summary>
/// <remarks>
/// The worker-machine run pins every process with <c>taskset</c>, so the line's <c>affinity</c> field is the proof
/// that a timing ran on the CPUs the runner meant: a probe whose affinity does not match its tag is discarded by the
/// parser instead of being averaged in. On Windows the opt-out still matters (on a hybrid CPU a background console
/// process's busy thread moves to the efficiency cores after about 2.5 s), so the old behaviour is kept there.
/// </remarks>
internal static class ProbeEnv
{
    /// <summary>Applies the Windows opt-out where it exists and prints the environment line.</summary>
    /// <remarks>
    /// A module initializer, so it runs before any probe code. A failure to apply the opt-out or to read the
    /// affinity is reported in the line, not thrown: the measurement stays usable when the state it ran under is
    /// known, and the parser decides what to keep.
    /// </remarks>
    [ModuleInitializer]
    internal static void Apply()
    {
        string throttle = "n/a";
        if (OperatingSystem.IsWindows() && Environment.GetEnvironmentVariable("NO_THROTTLE_OPTOUT") != "1")
        {
            // ControlMask = EXECUTION_SPEED (bit 0) with StateMask = 0 means "never throttle this process's speed".
            var state = new PowerThrottlingState { Version = 1, ControlMask = 1, StateMask = 0 };
            bool ok = SetProcessInformation(Process.GetCurrentProcess().Handle, ProcessPowerThrottling, ref state,
                (uint)Marshal.SizeOf<PowerThrottlingState>());
            throttle = ok ? "True" : $"False (Win32 error {Marshal.GetLastWin32Error()})";
        }

        Console.WriteLine($"probe env: os={RuntimeInformation.OSDescription.Replace(' ', '_')} " +
                          $"runtime={RuntimeInformation.FrameworkDescription.Replace(' ', '_')} " +
                          $"cpus={Environment.ProcessorCount} affinity={ReadAffinity()} " +
                          $"gc={(GCSettings.IsServerGC ? "server" : "workstation")}/{GCSettings.LatencyMode} " +
                          $"throttle_optout={throttle}");
    }

    /// <summary>The CPUs this process may run on, as the kernel lists them.</summary>
    /// <returns>
    /// On Linux the <c>Cpus_allowed_list</c> field of <c>/proc/self/status</c> (for example <c>2-3</c>), which is
    /// what <c>taskset -c</c> set; elsewhere, or when the file cannot be read, <c>unknown</c>.
    /// </returns>
    /// <remarks>Read from procfs rather than through <c>sched_getaffinity</c>, so no native signature is needed.</remarks>
    private static string ReadAffinity()
    {
        if (!OperatingSystem.IsLinux())
            return "unknown";

        try
        {
            foreach (string line in File.ReadLines("/proc/self/status"))
            {
                if (line.StartsWith("Cpus_allowed_list:", StringComparison.Ordinal))
                    return line["Cpus_allowed_list:".Length..].Trim();
            }
        }
        catch (IOException)
        {
            // procfs unreadable (a sandbox): the parser treats "unknown" as an unpinned process.
        }

        return "unknown";
    }

    /// <summary>PROCESS_INFORMATION_CLASS.ProcessPowerThrottling.</summary>
    private const int ProcessPowerThrottling = 4;

    /// <summary>kernel32 SetProcessInformation; called only on Windows.</summary>
    /// <param name="hProcess">The process to change.</param>
    /// <param name="infoClass">The information class; <see cref="ProcessPowerThrottling"/> here.</param>
    /// <param name="info">The throttling state to apply.</param>
    /// <param name="size">The size of <paramref name="info"/> in bytes.</param>
    /// <returns><see langword="true"/> when Windows accepted the setting; the error is then in GetLastWin32Error.</returns>
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetProcessInformation(IntPtr hProcess, int infoClass, ref PowerThrottlingState info, uint size);

    /// <summary>PROCESS_POWER_THROTTLING_STATE (version 1).</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct PowerThrottlingState
    {
        /// <summary>Must be 1 (PROCESS_POWER_THROTTLING_CURRENT_VERSION).</summary>
        public uint Version;

        /// <summary>Which throttling policies this call controls; bit 0 is execution speed.</summary>
        public uint ControlMask;

        /// <summary>For each controlled policy, 1 turns throttling on and 0 turns it off.</summary>
        public uint StateMask;
    }
}
