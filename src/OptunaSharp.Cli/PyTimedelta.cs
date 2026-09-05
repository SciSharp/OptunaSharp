using System;
using System.Globalization;

namespace OptunaSharp.Cli;

/// <summary>
///     CPython <c>str(datetime.timedelta)</c> — <c>[D day[s], ]H:MM:SS[.ffffff]</c> — for the CLI's
///     <c>duration</c> column (Python reaches it via <c>_format_value</c>'s <c>str(value)</c> else-branch).
///     Trial durations are non-negative (complete − start), which is the case CPython renders simply.
/// </summary>
internal static class PyTimedelta
{
    internal static string Str(TimeSpan ts)
    {
        // CPython: hh:mm:ss from the intra-day seconds; days carried separately; microseconds appended.
        int days = ts.Days;
        string s = string.Format(CultureInfo.InvariantCulture, "{0}:{1:D2}:{2:D2}", ts.Hours, ts.Minutes, ts.Seconds);
        if (days != 0)
            s = string.Format(CultureInfo.InvariantCulture, "{0} day{1}, ", days, Math.Abs(days) == 1 ? "" : "s") + s;
        long microseconds = Math.Abs(ts.Ticks) % TimeSpan.TicksPerSecond / (TimeSpan.TicksPerSecond / 1_000_000);
        if (microseconds != 0)
            s += "." + microseconds.ToString("D6", CultureInfo.InvariantCulture);
        return s;
    }
}
