#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// PID controller tuning — CMA-ES tunes the three gains of a DC-motor speed loop against a real closed-loop
// simulation, warm-started from an engineer's hand-tuned gains.
//
// source: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html
//         (the sampler and its `x0` warm start)
//         https://en.wikipedia.org/wiki/PID_controller (the control law, anti-windup, ITAE-style criteria)
//
// Real objective: every trial runs a 3-second Euler simulation of an armature-controlled DC motor
// (J = 0.01 kg·m², b = 0.1 N·m·s, K = 0.01, R = 1 Ω, L = 0.5 H — the textbook motor) under a PID with a
// filtered derivative, a ±24 V actuator limit and conditional-integration anti-windup, tracking a 1 rad/s
// step. The cost is ITAE plus penalties for overshoot above 5 % and for control effort, so the optimizer has
// to trade speed against overshoot the way a controls engineer would. Gains are searched on a log scale
// (they span orders of magnitude), and CMA-ES starts at the hand-tuned (Kp, Ki, Kd) = (100, 200, 10).
//
//   Run:  dotnet run examples/0_PidControllerTuning.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

// the engineer's starting point: decent, but it overshoots and leans on the actuator limit.
var handTuned = new Dictionary<string, object?> { ["kp"] = 100.0, ["ki"] = 200.0, ["kd"] = 10.0 };

var study = Optuna.CreateStudy(
    sampler: new CmaEsSampler(
        x0: handTuned,                         // start the search distribution at the hand-tuned gains
        sigma0: 0.15,                          // step size in CMA-ES's normalized [0,1] coordinates
        seed: 7),                              // seeded, so the run reproduces
    direction: StudyDirection.Minimize,
    studyName: "dc-motor-pid");

study.EnqueueTrial(new OptunaSharp.Collections.OrderedDict<string, object?>   // evaluate the baseline first
{
    ["kp"] = 100.0, ["ki"] = 200.0, ["kd"] = 10.0,
});
study.Optimize(Objective, nTrials: 150);

var baseline = Simulate(100.0, 200.0, 10.0);
var best = study.BestParams;
var tuned = Simulate((double)best["kp"]!, (double)best["ki"]!, (double)best["kd"]!);

Console.WriteLine($"trials: {study.Trials.Count}   best trial: #{study.BestTrial.Number}");
Console.WriteLine();
Console.WriteLine("                    Kp        Ki        Kd      cost   overshoot  settle(2%)  ss-error  peak |u|");
Print("hand-tuned", 100.0, 200.0, 10.0, baseline);
Print("cma-es", (double)best["kp"]!, (double)best["ki"]!, (double)best["kd"]!, tuned);

/// <summary>The optimization target: one closed-loop simulation per trial.</summary>
/// <param name="t">The trial that proposes the three gains.</param>
/// <returns>The simulation's cost (lower is better); a diverging loop scores a large finite penalty.</returns>
static double Objective(Trial t)
{
    double kp = t.SuggestFloat("kp", 1.0, 1000.0, log: true);   // proportional gain
    double ki = t.SuggestFloat("ki", 1.0, 1000.0, log: true);   // integral gain
    double kd = t.SuggestFloat("kd", 0.01, 100.0, log: true);   // derivative gain
    var r = Simulate(kp, ki, kd);
    t.SetUserAttr("overshoot_pct", r.OvershootPct);           // keep the engineering metrics on the trial
    t.SetUserAttr("settling_s", r.SettlingTime);
    return r.Cost;
}

/// <summary>
///     Simulates the PID-controlled DC motor for a 1 rad/s speed step and scores the response.
/// </summary>
/// <param name="kp">Proportional gain (V per rad/s of error).</param>
/// <param name="ki">Integral gain (V per rad of accumulated error).</param>
/// <param name="kd">Derivative gain (V·s per rad/s of error); filtered with a 1/(N·s) pole, N = 100.</param>
/// <returns>
///     The cost and the step-response metrics. A loop that diverges (speed beyond 100 rad/s or NaN) returns a
///     cost of 1e3 so the sampler sees a bad but finite value instead of a failed trial.
/// </returns>
static StepResponse Simulate(double kp, double ki, double kd)
{
    const double J = 0.01, B = 0.1, K = 0.01, R = 1.0, L = 0.5;   // the motor
    const double Dt = 1e-4, T = 3.0, UMax = 24.0, N = 100.0;      // integration step, horizon, limits
    double omega = 0, current = 0, integral = 0, dFilt = 0, prevErr = 1.0;
    double itae = 0, effort = 0, peak = 0, peakU = 0, settle = 0;
    int steps = (int)(T / Dt);
    for (int k = 0; k < steps; k++)
    {
        double time = k * Dt;
        double err = 1.0 - omega;
        // filtered derivative: d/dt of the error through a first-order low-pass (no derivative kick blow-up).
        dFilt += Dt * N * ((err - prevErr) / Dt - dFilt);
        prevErr = err;
        double uRaw = kp * err + ki * integral + kd * dFilt;
        double u = Math.Clamp(uRaw, -UMax, UMax);
        // conditional integration: stop integrating while saturated in the direction of the error, so the
        // integrator does not wind up behind a clamped actuator.
        if (u == uRaw || Math.Sign(err) != Math.Sign(uRaw))
            integral += err * Dt;
        // motor dynamics: J·dω/dt = K·i − b·ω ;  L·di/dt = V − R·i − K·ω
        double dOmega = (K * current - B * omega) / J;
        double dCurrent = (u - R * current - K * omega) / L;
        omega += Dt * dOmega;
        current += Dt * dCurrent;
        if (double.IsNaN(omega) || Math.Abs(omega) > 100)
            return new StepResponse(1e3, double.PositiveInfinity, T, double.PositiveInfinity, UMax);
        itae += time * Math.Abs(err) * Dt;
        effort += u * u * Dt;
        peak = Math.Max(peak, omega);
        peakU = Math.Max(peakU, Math.Abs(u));
        if (Math.Abs(1.0 - omega) > 0.02)
            settle = time;                                          // last time outside the ±2 % band
    }
    double overshoot = Math.Max(0, (peak - 1.0) * 100.0);
    double ssError = Math.Abs(1.0 - omega);
    // the engineering spec: overshoot ≤ 5 %, settle fast, do not burn the actuator.
    double cost = itae + 0.05 * Math.Max(0, overshoot - 5.0) + 1e-5 * effort + 10 * ssError;
    return new StepResponse(cost, overshoot, settle, ssError, peakU);
}

/// <summary>Prints one row of the comparison table.</summary>
/// <param name="label">The row's label.</param>
/// <param name="kp">The proportional gain shown.</param>
/// <param name="ki">The integral gain shown.</param>
/// <param name="kd">The derivative gain shown.</param>
/// <param name="r">The simulated response for those gains.</param>
static void Print(string label, double kp, double ki, double kd, StepResponse r) =>
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
        "{0,-12} {1,9:F2} {2,9:F2} {3,9:F3} {4,9:F5} {5,9:F2}% {6,9:F3} s {7,9:E1} {8,8:F1} V",
        label, kp, ki, kd, r.Cost, r.OvershootPct, r.SettlingTime, r.SteadyStateError, r.PeakVoltage));

/// <summary>Step-response metrics of one simulated closed loop.</summary>
/// <param name="Cost">ITAE plus the overshoot, effort and steady-state penalties (what the study minimizes).</param>
/// <param name="OvershootPct">Peak speed above the 1 rad/s setpoint, in percent (0 when it never overshoots).</param>
/// <param name="SettlingTime">The last time, in seconds, the speed was outside the ±2 % band.</param>
/// <param name="SteadyStateError">The absolute tracking error at the end of the horizon.</param>
/// <param name="PeakVoltage">The largest absolute actuator voltage applied, after the ±24 V clamp.</param>
record StepResponse(double Cost, double OvershootPct, double SettlingTime, double SteadyStateError, double PeakVoltage);
