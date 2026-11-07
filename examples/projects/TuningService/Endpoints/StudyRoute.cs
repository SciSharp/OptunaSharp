namespace TuningService.Endpoints;

/// <summary>Puts study names into URL paths and reads them back out.</summary>
/// <remarks>
///     Optuna study names are free text and may contain a slash (<c>exp/run-1</c>). A link escapes it as
///     <c>%2F</c>, and ASP.NET Core decodes every escape of a request path except that one, so that an encoded
///     slash cannot split a path segment. Every handler therefore passes its <c>{name}</c> route value through
///     <see cref="Name" />, and every link is built with <see cref="Uri" />.
/// </remarks>
internal static class StudyRoute
{
    /// <summary>Restores a study name from its <c>{name}</c> route value.</summary>
    /// <param name="routeValue">The route value ASP.NET Core bound, in which a slash is still <c>%2F</c>.</param>
    /// <returns>The study name.</returns>
    /// <remarks>
    ///     A name that literally contains the three characters <c>%2F</c> cannot be told apart from one with a
    ///     slash and reads as the latter; Optuna itself puts no such restriction on names, but they are rare.
    /// </remarks>
    internal static string Name(string routeValue)
        => routeValue.Replace("%2F", "/", StringComparison.OrdinalIgnoreCase);

    /// <summary>The API location of a study.</summary>
    /// <param name="studyName">The study name.</param>
    /// <returns>The relative URI <c>/api/studies/{name}</c>, with the name escaped.</returns>
    internal static string Uri(string studyName) => $"/api/studies/{Escape(studyName)}";

    /// <summary>Escapes a study name for one URL path segment.</summary>
    /// <param name="studyName">The study name.</param>
    /// <returns>The escaped name; a slash becomes <c>%2F</c>, which <see cref="Name" /> turns back.</returns>
    internal static string Escape(string studyName) => System.Uri.EscapeDataString(studyName);
}
