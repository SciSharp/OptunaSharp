#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:project ../src/OptunaSharp.Storage.Sqlite/OptunaSharp.Storage.Sqlite.csproj
#:project ../src/OptunaSharp.Storage.Grpc/OptunaSharp.Storage.Grpc.csproj
#:property PublishAot=false

// Remote workers through Optuna's gRPC storage proxy — the database stays on one host behind
// `run_grpc_proxy_server`; four .NET workers that only know the proxy's address (GrpcStorageProxy) tune a
// retailer's promotion pricing in parallel, and — when a Python with optuna is on PATH — a Python Optuna
// worker joins the same study over the same wire protocol.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.run_grpc_proxy_server.html
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.GrpcStorageProxy.html
//         (the proxy pair: one process owns the RDB storage, workers talk gRPC)
//
// Analytic objective, a standard retail model: weekly demand with constant price elasticity (−2.1), a
// promotion lift that grows with discount depth but decays over a long promo (customers stock up), a
// post-promo dip, and a margin floor. The knobs: list price, discount depth, promo length and how often to
// run it per quarter; the score is the quarter's gross profit in $k (maximized). The proxy fronts a SQLite
// file, so the finished study can also be opened directly afterwards.
//
//   Run:  dotnet run examples/0_PricingRemoteWorkersGrpc.cs

using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

// 1. the database host: SQLite behind the gRPC proxy, on a free local port.
string db = Path.Combine(Path.GetTempPath(), "optunasharp-pricing.db");
File.Delete(db);
using var sqlite = new RDBStorage("sqlite:///" + db.Replace('\\', '/'));
int port = FreePort();
await using var server = GrpcProxyServer.MakeServer(sqlite, host: "127.0.0.1", port: port);
await server.StartAsync();
Console.WriteLine($"gRPC storage proxy listening on 127.0.0.1:{port}, backed by {db}");

// 2. the study is created through the proxy, like any remote client would.
using (var admin = new GrpcStorageProxy("127.0.0.1", port))
    Optuna.CreateStudy(storage: admin, studyName: "q3-promo-pricing", direction: StudyDirection.Maximize);

// 3. four .NET workers, each with its own proxy connection, plus an optional Python worker.
var pythonWorker = Task.Run(() => PythonWorker(port));
var dotnetWorkers = Enumerable.Range(1, 4).Select(id => Task.Run(() =>
{
    using var proxy = new GrpcStorageProxy("127.0.0.1", port);
    var study = Optuna.LoadStudy("q3-promo-pricing", proxy, sampler: new TPESampler(seed: id, nStartupTrials: 10));
    study.Optimize(t => { t.SetUserAttr("worker", $".NET #{id}"); return Promo.QuarterProfit(t); }, nTrials: 15);
})).ToArray();
await Task.WhenAll(dotnetWorkers);
string pythonNote = await pythonWorker;

// 4. read the result back through the proxy.
using (var reader = new GrpcStorageProxy("127.0.0.1", port))
{
    var study = Optuna.LoadStudy("q3-promo-pricing", reader);
    var byWorker = study.Trials.GroupBy(t => (string?)t.UserAttrs.GetValueOrDefault("worker") ?? "?").Select(g => $"{g.Key}: {g.Count()}");
    Console.WriteLine($"trials: {study.Trials.Count} ({string.Join(", ", byWorker)}); {pythonNote}");
    var best = study.BestTrial;
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best quarter profit ${0:N1}k by {1} (trial #{2}): {3}", best.Value, best.UserAttrs["worker"], best.Number,
        string.Join(", ", best.Params.Select(kv => string.Format(CultureInfo.InvariantCulture, "{0}={1:G4}", kv.Key, kv.Value)))));
}
await server.StopAsync();

/// <summary>Finds a free TCP port on the loopback interface.</summary>
/// <returns>A port number that was free a moment ago.</returns>
static int FreePort()
{
    var l = new TcpListener(IPAddress.Loopback, 0);
    l.Start();
    int p = ((IPEndPoint)l.LocalEndpoint).Port;
    l.Stop();
    return p;
}

/// <summary>Runs ten trials from Python Optuna through the same proxy, when Python and optuna are available.</summary>
/// <param name="port">The proxy's port.</param>
/// <returns>A one-line note on what the Python worker did.</returns>
static string PythonWorker(int port)
{
    string code = "import optuna,sys,math; optuna.logging.set_verbosity(optuna.logging.ERROR)\n"
                  + "from optuna.storages import GrpcStorageProxy\n"
                  + "s=optuna.load_study(study_name='q3-promo-pricing', storage=GrpcStorageProxy(host='127.0.0.1', port=int(sys.argv[1])), sampler=optuna.samplers.RandomSampler(seed=9))\n"
                  + "def obj(t):\n"
                  + "    p=t.suggest_float('list_price',8,30); d=t.suggest_float('discount',0.0,0.5); w=t.suggest_int('promo_weeks',1,6); n=t.suggest_int('promos_per_quarter',0,3)\n"
                  + "    t.set_user_attr('worker','Python')\n"
                  + "    weeks=13; promo=min(weeks, n*w); base=1200*(p/15)**-2.1; lift=1+2.5*d*math.exp(-0.25*(w-1)); dip=1-0.35*d*min(n,promo)\n"
                  + "    unit=max(p*(1-d)-6.0,-6.0); full=max(p-6.0,-6.0)\n"
                  + "    return (promo*base*(1-d)**-2.1*lift*unit+(weeks-promo)*base*dip*full)/1000\n"
                  + "s.optimize(obj, n_trials=10)\n";
    try
    {
        var psi = new ProcessStartInfo("python") { RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var a in new[] { "-c", code, port.ToString(CultureInfo.InvariantCulture) })
            psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        p.StandardOutput.ReadToEnd();
        string err = p.StandardError.ReadToEnd();
        p.WaitForExit();
        return p.ExitCode == 0 ? "a Python Optuna worker added its trials over gRPC" : "python worker failed: " + err.Trim().Split('\n').Last();
    }
    catch (System.ComponentModel.Win32Exception)
    {
        return "no python on PATH — the Python worker was skipped";
    }
}

/// <summary>The retail promotion model.</summary>
static class Promo
{
    /// <summary>Gross profit of one quarter (13 weeks) under the trial's pricing plan.</summary>
    /// <param name="t">The trial proposing list price, discount depth, promo length and promo count.</param>
    /// <returns>Profit in thousands of dollars.</returns>
    public static double QuarterProfit(Trial t)
    {
        double price = t.SuggestFloat("list_price", 8, 30);            // $ per unit
        double discount = t.SuggestFloat("discount", 0.0, 0.5);        // promo depth
        long weeksPerPromo = t.SuggestInt("promo_weeks", 1, 6);
        long promos = t.SuggestInt("promos_per_quarter", 0, 3);
        const int Weeks = 13;
        const double UnitCost = 6.0;
        double promoWeeks = Math.Min(Weeks, promos * weeksPerPromo);
        double baseDemand = 1200 * Math.Pow(price / 15, -2.1);          // constant elasticity around $15
        double lift = 1 + 2.5 * discount * Math.Exp(-0.25 * (weeksPerPromo - 1));   // long promos lose their pull
        double dip = 1 - 0.35 * discount * Math.Min(promos, promoWeeks);              // stock-up hangover
        double promoUnit = Math.Max(price * (1 - discount) - UnitCost, -UnitCost);
        double fullUnit = Math.Max(price - UnitCost, -UnitCost);
        return (promoWeeks * baseDemand * Math.Pow(1 - discount, -2.1) * lift * promoUnit
                + (Weeks - promoWeeks) * baseDemand * dip * fullUnit) / 1000;
    }
}
