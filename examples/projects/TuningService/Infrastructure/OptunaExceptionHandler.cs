using Microsoft.AspNetCore.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using NumSharp;
using OptunaSharp;

namespace TuningService.Infrastructure;

/// <summary>
///     Turns the exceptions OptunaSharp raises for bad input into RFC 7807 problem responses with the right
///     status code, so endpoints can call OptunaSharp directly instead of catching around every call.
/// </summary>
/// <remarks>
///     <para>
///         OptunaSharp raises Python's exception kinds, with Optuna's messages: <see cref="KeyError" /> for a
///         study or trial that does not exist, <see cref="ValueError" /> for an invalid argument, and Optuna's
///         own <see cref="DuplicatedStudyError" /> and <see cref="UpdateFinishedTrialError" />. The mapping
///         follows Optuna's gRPC proxy, which turns the same four into NOT_FOUND, INVALID_ARGUMENT,
///         ALREADY_EXISTS and FAILED_PRECONDITION.
///     </para>
///     <para>
///         Anything else is left to the default handler, which answers 500 without the exception's details.
///     </para>
/// </remarks>
public sealed class OptunaExceptionHandler : IExceptionHandler
{
    private readonly IProblemDetailsService _problemDetails;

    /// <summary>Creates the handler.</summary>
    /// <param name="problemDetails">Writes the problem response in the format the app configured.</param>
    /// <exception cref="ArgumentNullException"><paramref name="problemDetails" /> is <c>null</c>.</exception>
    public OptunaExceptionHandler(IProblemDetailsService problemDetails)
    {
        _problemDetails = problemDetails ?? throw new ArgumentNullException(nameof(problemDetails));
    }

    /// <summary>Writes a problem response for the exceptions OptunaSharp raises for bad input.</summary>
    /// <param name="httpContext">The request that failed.</param>
    /// <param name="exception">The exception the endpoint threw.</param>
    /// <param name="cancellationToken">Signaled when the request is aborted.</param>
    /// <returns><c>true</c> when the response was written; <c>false</c> to let the next handler deal with it.</returns>
    public async ValueTask<bool> TryHandleAsync(
        HttpContext httpContext, Exception exception, CancellationToken cancellationToken)
    {
        (int Status, string Title)? mapped = exception switch
        {
            KeyError => (StatusCodes.Status404NotFound, "Not found"),
            DuplicatedStudyError => (StatusCodes.Status409Conflict, "The study already exists"),
            UpdateFinishedTrialError => (StatusCodes.Status409Conflict, "The trial is already finished"),
            // NumSharp's ValueError derives from ArgumentException, so this arm also catches plain argument errors
            ArgumentException => (StatusCodes.Status400BadRequest, "Invalid request"),
            _ => null,
        };
        if (mapped is null)
            return false;

        httpContext.Response.StatusCode = mapped.Value.Status;
        return await _problemDetails.TryWriteAsync(new ProblemDetailsContext
        {
            HttpContext = httpContext,
            Exception = exception,
            ProblemDetails = new ProblemDetails
            {
                Status = mapped.Value.Status,
                Title = mapped.Value.Title,
                // Optuna's own message, e.g. "Record does not exist." or "Cannot tell a COMPLETE trial."
                Detail = exception.Message,
            },
        });
    }
}
