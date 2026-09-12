using Microsoft.Extensions.FileProviders;
using OptunaSharp.Logging;
using OptunaSharp.PresentationHost;

OptunaLogging.SetVerbosity(OptunaSharp.Logging.LogLevel.Warning);

if (args.Contains("--self-test", StringComparer.Ordinal))
{
    SelfTest.Run();
    return;
}

var builder = WebApplication.CreateBuilder(args);
if (string.IsNullOrWhiteSpace(builder.Configuration["urls"]))
    builder.WebHost.UseUrls("http://127.0.0.1:5088");
builder.WebHost.ConfigureKestrel(options => options.Limits.MaxRequestBodySize = 16 * 1024);
builder.Services.AddSingleton<RunStore>();
var app = builder.Build();

// Keep this host reusable: every presentation lives below the adjacent shared wwwroot.
var staticRoot = new[]
{
    Path.GetFullPath(Path.Combine(app.Environment.ContentRootPath, "..", "wwwroot")),
    Path.GetFullPath(Path.Combine(app.Environment.ContentRootPath, "docs", "presentations", "wwwroot")),
    Path.Combine(AppContext.BaseDirectory, "wwwroot")
}.FirstOrDefault(Directory.Exists);
if (staticRoot is not null)
{
    var files = new PhysicalFileProvider(staticRoot);
    app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = files });
    app.UseStaticFiles(new StaticFileOptions { FileProvider = files });
}

app.Use(async (context, next) =>
{
    try { await next(context); }
    catch (ArgumentException ex)
    {
        context.Response.StatusCode = StatusCodes.Status400BadRequest;
        await context.Response.WriteAsJsonAsync(new { error = ex.Message });
    }
    catch (SessionLimitException ex)
    {
        context.Response.StatusCode = StatusCodes.Status429TooManyRequests;
        await context.Response.WriteAsJsonAsync(new { error = ex.Message });
    }
    catch (InvalidOperationException ex) when (ex.Message.StartsWith("This run reached", StringComparison.Ordinal))
    {
        context.Response.StatusCode = StatusCodes.Status409Conflict;
        await context.Response.WriteAsJsonAsync(new { error = ex.Message });
    }
});

app.MapGet("/api/health", () => new { status = "ok", engine = "OptunaSharp", algorithm = "NSGA-III" });
app.MapPost("/api/nsga3/reference-lab", (ReferenceLabRequest request) => SelectionGeometry.ReferenceLab(request));
app.MapPost("/api/nsga3/runs", (RunConfig config, RunStore store) => store.Create(config));
app.MapGet("/api/nsga3/runs/{id}", (string id, RunStore store) =>
    store.Read(id) is { } snapshot ? Results.Ok(snapshot) : Results.NotFound(new { error = "Run not found or expired. Start a new run." }));
app.MapPost("/api/nsga3/runs/{id}/step", (string id, RunStore store) =>
    store.Step(id) is { } snapshot ? Results.Ok(snapshot) : Results.NotFound(new { error = "Run not found or expired. Start a new run." }));
app.MapDelete("/api/nsga3/runs/{id}", (string id, RunStore store) =>
    store.Delete(id) ? Results.NoContent() : Results.NotFound(new { error = "Run not found or expired." }));
app.Run();
