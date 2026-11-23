#!/usr/bin/env bash
# The benchmarks behind README.md's Performance section, for OptunaSharp (this checkout's src/, .NET 10 Release,
# framework-dependent linux-x64 builds from bench/publish.sh), Python Optuna 5.0.0 and Rustuna 0.3.1 (CPython 3.12.16, one
# venv, so both Python libraries share numpy and cmaes). Written for the worker machine (Proxmox host `worker`: one Intel Xeon
# E5-2690 v2, 10 cores / 20 threads, logical CPU n and n+10 share core n; 3.0 GHz, turbo 3.6 GHz; governor `performance`);
# any Linux machine with the same layout works. bench/README.md has the method; bench/parse.py turns the output into numbers.
#
# Fairness and statistical validity (the owner, 2026-11-22: "pin ... threads to the benchmark that is being run", "remove
# the upper and lower bounds"; 2026-11-23: "establish golden standard of benchmarking ... ensure that the benchmarks are
# statistically valid"):
#   - Steady state, not first runs. Every probe runs its workload under bench/common/Bench.cs or bench/py/benchlib.py: warm-up
#     runs first, which are printed and never counted (at least BENCH_WARMUP_RUNS runs and BENCH_WARMUP_SECONDS seconds; for
#     .NET at least BENCH_DOTNET_TIERING_RUNS runs, its two tiering steps of 30 calls each; then until the last
#     BENCH_WARMUP_QUIET_SHARE of the runs compiled, or imported, nothing; at most BENCH_WARMUP_MAX_SECONDS), then measured
#     runs (at least BENCH_MEASURE_RUNS, then until BENCH_MEASURE_SECONDS or BENCH_MAX_RUNS). Until 2026-11-23 each workload
#     ran once per fresh process, so the numbers were mostly .NET's JIT and Python's lazy imports; the 10:03 run of that day
#     warmed up for a fixed 5 s, after which OptunaSharp's short rows still compiled methods while being measured.
#   - Two levels of samples: ROUNDS fresh processes (default 5) of every workload, each with its own measured runs, so the
#     parser's confidence intervals cover the variation between processes (JIT decisions, memory layout) as well as within.
#   - Every timed process is pinned with taskset. One-trial-at-a-time workloads get SERIAL_CPUS (default 2,3: two physical
#     cores whose hyperthread siblings 12 and 13 stay idle), so the main thread has a core to itself and the runtime's helpers
#     (.NET's tiered JIT and GC threads, OpenBLAS, any Rust threads) get the other, the same two cores for all three libraries.
#     Parallel workloads get PARALLEL_CPUS (default 0-19, the whole machine).
#   - The libraries alternate within each workload (OptunaSharp, Rustuna, Python on odd rounds, the reverse on even ones), and
#     the rounds are the outer loop, so each workload's five processes are spread over the whole run: a slow drift, a warm
#     core or a busy hour favours nobody.
#   - Every process waits for an idle machine first: monitor.py's gate wants GATE_WINDOWS (3) consecutive 1-second windows
#     under GATE_TOTAL % busy with the guests (containers and VMs) under GATE_GUEST % of one CPU, which also gives every
#     process the same cooled-down start. After GATE_MAX_WAIT seconds without that, the run ABORTS (idle numbers or none).
#     Until 2026-11-23 10:30 the gate ran once per block, and a TraderLab pass in CT 801 that started a minute into block B
#     ran beside its timings for three minutes.
#   - Every process is judged afterwards by the load guard: monitor.py logs the guests', the host services' and the other
#     host sessions' load every LOAD_INTERVAL (2) seconds to load.log for the whole run, and a process during any of whose
#     windows one of them reached its limit (GUARD_GUEST, GUARD_SERVICES, GUARD_OTHERS, in % of one CPU) is voided
#     ("@@ void", the parser forgets its lines) and redone in the same place after GATE_WINDOWS_AFTER_VOID (15) quiet
#     seconds, up to ATTEMPTS (6) attempts; one still loaded after that is kept and logged, and the parser drops it.
#
# Layout: BENCH holds the runtime and the venv (bench/setup.sh); HARNESS (default $BENCH/v2) holds this script, monitor.py,
# parse.py, py/ and out/ (the probe builds, all in one folder). Output, in $BENCH/results/$RUN_NAME/: results.txt (every probe
# line tagged "<block> <lang> <probe> <config> run<N> | ...", "@@ start|end" lines with epoch times and exit codes, "@@ load"
# with the guard's verdict on each kept process and "@@ void" on each redone one, "##" lines for the environment and the
# gates), run.log (progress, ends with "ALL DONE" or "ABORTED") and load.log.
# Start detached (the worker's rule 13): nohup setsid bash run_worker.sh </dev/null >run_worker.out 2>&1 &
set -u
BENCH=${BENCH:-/mnt/drive2/data/optunasharp-bench}
HARNESS=${HARNESS:-$BENCH/v2}
export DOTNET_ROOT=$BENCH/dotnet
export PATH="$DOTNET_ROOT:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
# Library defaults stay in force: no thread caps for OpenBLAS / OpenMP / MKL, as in the workstation runs.
unset OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS OPTUNASHARP_NJOBS_EXECUTOR
# The protocol's budgets, the same for every probe of every library (bench/README.md, "Protocol").
export BENCH_WARMUP_SECONDS=${BENCH_WARMUP_SECONDS:-5}
export BENCH_WARMUP_RUNS=${BENCH_WARMUP_RUNS:-2}
export BENCH_DOTNET_TIERING_RUNS=${BENCH_DOTNET_TIERING_RUNS:-70}
export BENCH_WARMUP_QUIET_SHARE=${BENCH_WARMUP_QUIET_SHARE:-0.25}
export BENCH_WARMUP_MAX_RUNS=${BENCH_WARMUP_MAX_RUNS:-500}
export BENCH_WARMUP_MAX_SECONDS=${BENCH_WARMUP_MAX_SECONDS:-60}
export BENCH_MEASURE_RUNS=${BENCH_MEASURE_RUNS:-3}
export BENCH_MEASURE_SECONDS=${BENCH_MEASURE_SECONDS:-10}
export BENCH_MAX_RUNS=${BENCH_MAX_RUNS:-30}
PY=$BENCH/venv/bin/python
OUT=$HARNESS/out
PYD=$HARNESS/py
ROUNDS=${ROUNDS:-5}
SER=${SERIAL_CPUS:-2,3}
PAR=${PARALLEL_CPUS:-0-19}
MONCPU=${MONITOR_CPUS:-0}
BLOCKS=${BLOCKS:-ABPCDEFG}
GATE_TOTAL=${GATE_TOTAL:-5}
GATE_GUEST=${GATE_GUEST:-50}
GATE_MAX_WAIT=${GATE_MAX_WAIT:-7200}
GATE_WINDOWS=${GATE_WINDOWS:-3}
GATE_WINDOWS_AFTER_VOID=${GATE_WINDOWS_AFTER_VOID:-15}
GUARD_GUEST=${GUARD_GUEST:-$GATE_GUEST}
GUARD_SERVICES=${GUARD_SERVICES:-50}
GUARD_OTHERS=${GUARD_OTHERS:-50}
ATTEMPTS=${ATTEMPTS:-6}
LOAD_INTERVAL=${LOAD_INTERVAL:-2}
NAME=${RUN_NAME:-run-$(date +%Y%m%d-%H%M)}
R=$BENCH/results/$NAME
mkdir -p "$R"
RES=$R/results.txt
LOG=$R/run.log
: > "$RES"
: > "$LOG"

# log <text>: a progress line with the time.
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }

# monitor_start: (re)starts the load logger in the background. The guard judges every process from its windows, so a
# logger that died would leave every later process "unknown" (voided until ATTEMPTS runs out): run() restarts it first.
monitor_start() {
    taskset -c "$MONCPU" "$PY" -I "$HARNESS/monitor.py" log "$R/load.log" "$LOAD_INTERVAL" &
    MON=$!
}
monitor_start
# Single quotes: $MON is read when the trap fires, so a restarted logger is the one stopped.
trap 'kill "$MON" 2>/dev/null' EXIT

# gate <label> [windows]: waits for an idle machine (<windows> consecutive 1-s windows, default GATE_WINDOWS) and records
# the windows it passed on in results.txt; run.log hears of it only when it had to wait (more than 5 s beyond its windows),
# so the log stays readable with a gate before every process. Aborts the run on a timeout (idle numbers or none).
gate() {
    local out waited windows=${2:-$GATE_WINDOWS}
    if out=$(taskset -c "$MONCPU" "$PY" -I "$HARNESS/monitor.py" gate "$GATE_TOTAL" "$GATE_GUEST" "$windows" 1 "$GATE_MAX_WAIT" 2>&1); then
        echo "## gate $1 | $(date +%H:%M:%S) | $out" >> "$RES"
        waited=${out##*waited }
        waited=${waited%s}
        [ "${waited:-0}" -gt $((windows + 5)) ] 2>/dev/null && log "gate $1 | $out"
    else
        echo "## ABORTED before $1 | $(date +%H:%M:%S) | $out" >> "$RES"
        log "ABORTED: machine not idle for $GATE_MAX_WAIT s before: $1 | $out"
        exit 2
    fi
}

# tag <block> <lang> <probe> <config>: prefixes every line of a probe's output.
tag() { sed -u "s#^#$1 $2 $3 $4 run$RUN | #" >> "$RES"; }

# run <block> <lang> <probe> <config> <cpus> <command...>: one fresh, pinned process behind its own gate, bracketed by
# start/end lines, then judged by the load guard (monitor.py check). A process during which other work reached a limit is
# voided ("@@ void": the parser forgets every line it printed) and redone in its place after GATE_WINDOWS_AFTER_VOID quiet
# seconds, so a job that runs in bursts with short pauses is waited out; after ATTEMPTS attempts the last one is kept with
# its verdict ("@@ load ... interfered"), logged, and dropped by the parser. A non-zero exit is the probe's failure, not
# the machine's: it is logged and never redone. RUN_ENV (space-separated NAME=value pairs) goes into the process's
# environment only.
run() {
    local block=$1 lang=$2 probe=$3 cfg=$4 cpus=$5
    shift 5
    local key="$block $lang $probe $cfg run$RUN" attempt=1 windows=$GATE_WINDOWS t0 t1 rc verdict
    while :; do
        kill -0 "$MON" 2>/dev/null || { log "load logger restarted (it had exited)"; monitor_start; }
        gate "$key attempt $attempt" "$windows"
        t0=$(date +%s.%N)
        echo "@@ start $key $t0" >> "$RES"
        # --foreground keeps the probe in the runner's process group (plain timeout makes a group of its own), so stopping
        # the run with kill -- -<runner's pgid> also stops the probe; env and taskset exec, so the probe is timeout's child.
        # shellcheck disable=SC2086 # RUN_ENV is a list of NAME=value words on purpose
        timeout --foreground 3600 env ${RUN_ENV:-} taskset -c "$cpus" "$@" 2>&1 | tag "$block" "$lang" "$probe" "$cfg"
        rc=${PIPESTATUS[0]}
        t1=$(date +%s.%N)
        echo "@@ end $key $t1 rc=$rc" >> "$RES"
        if [ "$rc" -ne 0 ]; then
            log "FAILED rc=$rc: $key"
            return
        fi
        verdict=$(taskset -c "$MONCPU" "$PY" -I "$HARNESS/monitor.py" check "$R/load.log" "$t0" "$t1" \
                  "$GUARD_GUEST" "$GUARD_SERVICES" "$GUARD_OTHERS" "$LOAD_INTERVAL" 2>&1)
        case $verdict in
            clean:*)
                echo "@@ load $key $verdict" >> "$RES"
                return
                ;;
        esac
        if [ "$attempt" -ge "$ATTEMPTS" ]; then
            echo "@@ load $key $verdict" >> "$RES"
            log "KEPT after $attempt attempts, not clean (the parser drops it): $key | $verdict"
            return
        fi
        echo "@@ void $key $verdict" >> "$RES"
        log "VOID attempt $attempt: $key | $verdict"
        attempt=$((attempt + 1))
        windows=$GATE_WINDOWS_AFTER_VOID
    done
}

# cs <block> <probe> <config> <cpus> [args...]: an OptunaSharp probe (framework-dependent apphost in the shared out/ folder).
cs() { local b=$1 p=$2 c=$3 cpus=$4; shift 4; run "$b" cs "$p" "$c" "$cpus" "$OUT/$p" "$@"; }

# pyl <block> <optuna|rustuna> <script> <config> <cpus> [args...]: a Python probe through pyrun.py.
pyl() {
    local b=$1 lib=$2 s=$3 c=$4 cpus=$5 lang=py
    shift 5
    [ "$lib" = rustuna ] && lang=ru
    run "$b" "$lang" "$s" "$c" "$cpus" "$PY" -I "$PYD/pyrun.py" "$PYD/$s.py" "$lib" "$@"
}

# trio "<OptunaSharp call>" "<Rustuna call>" "<Python call>": one workload in this round's order (OptunaSharp first on odd
# rounds, Python first on even ones); an empty call is skipped (a library without that workload).
trio() {
    local calls=("$1" "$2" "$3")
    [ $((RUN % 2)) -eq 0 ] && calls=("$3" "$2" "$1")
    for c in "${calls[@]}"; do
        [ -n "$c" ] && eval "$c"
    done
}

has() { case "$BLOCKS" in *"$1"*) return 0 ;; esac; return 1; }

# --- the blocks -------------------------------------------------------------------------------------------------
# No gate here: run() gates every process (until 2026-11-23 10:30 each block, and each large-study workload, had one).

# A - Parallel trials: 64 trials of a ~50 ms CPU-bound objective at 1/2/4/8/16 workers, one sweep process per library.
block_A() {
    trio "cs A njobs sweep \$PAR" "pyl A rustuna njobs_scaling sweep \$PAR" "pyl A optuna njobs_scaling sweep \$PAR"
}

# B - Samplers: the 6-parameter rows, the 50,000-trial Random row and the wide rows. Rustuna has no GP or QMC sampler.
block_B() {
    local c
    for c in random tpe tpe_mv gp cmaes nsga2 qmc; do
        local ru="pyl B rustuna perf $c \$SER $c"
        case $c in gp | qmc) ru="" ;; esac
        trio "cs B perf $c \$SER $c" "$ru" "pyl B optuna perf $c \$SER $c"
    done
    trio "cs B scale inmem \$SER inmem" "pyl B rustuna scale inmem \$SER inmem" "pyl B optuna scale inmem \$SER inmem"
    for c in random tpe_ind cmaes; do
        trio "cs B wide $c \$SER $c" "pyl B rustuna wide $c \$SER $c" "pyl B optuna wide $c \$SER $c"
    done
}

# P - Waiting trials: 640 trials sleeping 100 ms on 64 workers and 1,280 on 128 (ideal 1.0 s each).
block_P() {
    trio "cs P pool jobs64 \$PAR 64 640 100; cs P pool jobs128 \$PAR 128 1280 100" \
         "pyl P rustuna pool both \$PAR" "pyl P optuna pool both \$PAR"
}

# C - Large studies (40 floats, seed 0, defaults). Warm-up runs use a tenth of the trials (bench/README.md: the same code,
#     and Python's 1,000-trial studies take 38-197 s each). Python's 100,000-trial runs are not made (hours each, as in
#     every earlier run).
block_C() {
    trio "cs C proof tpe \$SER tpe 1000 100 plain" "pyl C rustuna pbench tpe \$SER tpe 1000 1 100" \
         "pyl C optuna pbench tpe \$SER tpe 1000 1 100"
    trio "cs C proof motpe \$SER motpe 1000 100 plain" "pyl C rustuna pbench motpe \$SER motpe 1000 1 100" \
         "pyl C optuna pbench motpe \$SER motpe 1000 1 100"
    trio "cs C proof cmaes \$SER cmaes 10000 1000 plain" "pyl C rustuna pbench cmaes \$SER cmaes 10000 1 1000" \
         "pyl C optuna pbench cmaes \$SER cmaes 10000 1 1000"
    trio "cs C proof nsga2 \$SER nsga2 10000 1000 plain" "pyl C rustuna pbench nsga2 \$SER nsga2 10000 1 1000" \
         "pyl C optuna pbench nsga2 \$SER nsga2 10000 1 1000"
    trio "cs C proof cmaes100k \$SER cmaes 100000 10000 plain" "pyl C rustuna pbench cmaes100k \$SER cmaes 100000 1 10000" ""
    trio "cs C proof nsga2100k \$SER nsga2 100000 10000 plain" "pyl C rustuna pbench nsga2100k \$SER nsga2 100000 1 10000" ""
}

# D - Memory: the 100,000-trial studies with compactFinishedTrials (the plain runs are block C's).
block_D() {
    cs D proof cmaes100k-compact "$SER" cmaes 100000 10000 compact
    cs D proof nsga2100k-compact "$SER" nsga2 100000 10000 compact
}

# E - Executors: DedicatedThreads ("reused") against ThreadPool ("pool"), alternating within each workload. starve measures
#     once per process (a thread pool that has not grown yet), so its samples are the rounds' processes.
block_E() {
    local m mult
    for m in soak tpe manycalls; do
        RUN_ENV="OPTUNASHARP_NJOBS_EXECUTOR=reused" cs E execprobe "reused-$m" "$PAR" "$m"
        RUN_ENV="OPTUNASHARP_NJOBS_EXECUTOR=pool" cs E execprobe "pool-$m" "$PAR" "$m"
    done
    for mult in 2 1; do
        RUN_ENV="OPTUNASHARP_NJOBS_EXECUTOR=reused" cs E execprobe "reused-starve$mult" "$PAR" starve "$mult"
        RUN_ENV="OPTUNASHARP_NJOBS_EXECUTOR=pool" cs E execprobe "pool-starve$mult" "$PAR" starve "$mult"
    done
}

# F - Per-worker samplers: plan 11's mode probe, every configuration under the protocol.
block_F() {
    cs F modeprobe tpe-heavy "$PAR" tpe-heavy
    cs F modeprobe samplers "$PAR" samplers
}

# G - docs/performance.md's compact-storage table: plain and compact, each its own process.
block_G() {
    local spec
    for spec in "random 10000 1000" "cmaes 10000 1000" "tpe 1000 100"; do
        # shellcheck disable=SC2086 # spec is three words on purpose
        set -- $spec
        cs G proof "s12-$1-plain" "$SER" "$@" plain
        cs G proof "s12-$1-compact" "$SER" "$@" compact
    done
}

# --- the run ------------------------------------------------------------------------------------------------------

RUN=0
{
    echo "## env | host $(hostname) | kernel $(uname -r) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')" \
         "| governor $(cat /sys/devices/system/cpu/cpufreq/policy2/scaling_governor 2>/dev/null || echo n/a) | no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo n/a)"
    echo "## env | harness $HARNESS | source $(cat "$HARNESS/SOURCE" 2>/dev/null || echo unknown) | dotnet $("$DOTNET_ROOT/dotnet" --list-runtimes | grep NETCore | tr '\n' ' ')"
    echo "## env | $("$PY" -I -c 'import sys, optuna, numpy, scipy, torch, cmaes, importlib.metadata as m; print("python", sys.version.split()[0], "optuna", optuna.__version__, "rustuna", m.version("rustuna"), "numpy", numpy.__version__, "scipy", scipy.__version__, "torch", torch.__version__, "cmaes", cmaes.__version__)')"
    echo "## env | serial cpus $SER | parallel cpus $PAR | monitor cpu $MONCPU | rounds $ROUNDS | blocks $BLOCKS | gate before every process: $GATE_WINDOWS 1-s windows ($GATE_WINDOWS_AFTER_VOID after a void) <$GATE_TOTAL % busy, guests <$GATE_GUEST % of a CPU"
    echo "## env | protocol warm-up >= $BENCH_WARMUP_RUNS runs and $BENCH_WARMUP_SECONDS s, .NET >= $BENCH_DOTNET_TIERING_RUNS runs, then until the last $BENCH_WARMUP_QUIET_SHARE of the runs compiled/imported nothing (max $BENCH_WARMUP_MAX_RUNS runs, $BENCH_WARMUP_MAX_SECONDS s) | measured >= $BENCH_MEASURE_RUNS runs, then until $BENCH_MEASURE_SECONDS s or $BENCH_MAX_RUNS runs"
    # The guard's limits, in the form parse.py reads them back (it applies the same rule to every counted process).
    echo "## guard | guests<$GUARD_GUEST services<$GUARD_SERVICES others<$GUARD_OTHERS interval=$LOAD_INTERVAL attempts=$ATTEMPTS"
    # Which scripts ran: the probes' builds are named by SOURCE, the scripts (not built) by their hashes.
    echo "## env | scripts $(cd "$HARNESS" && sha256sum run_worker.sh monitor.py parse.py py/*.py | awk '{printf "%s:%s ", $2, substr($1, 1, 12)}')"
    echo "## guests at start | $(pct list 2>/dev/null | awk 'NR>1 {printf "CT%s:%s ", $1, $2}')$(qm list 2>/dev/null | awk 'NR>1 {printf "VM%s:%s ", $1, $3}')"
} >> "$RES"
log "start $NAME (rounds $ROUNDS, blocks $BLOCKS)"

# No uncounted round 0 any more: each process warms itself up before it measures, and a cold page cache only slows the
# process's start, which is never timed.
for RUN in $(seq 1 "$ROUNDS"); do
    log "round $RUN"
    for b in A B P C D E F G; do
        if has "$b"; then
            log "round $RUN block $b"
            "block_$b"
        fi
    done
done

echo "## end | $(date +%H:%M:%S)" >> "$RES"
log "ALL DONE"
