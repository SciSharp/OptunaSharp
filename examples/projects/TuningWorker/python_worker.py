"""A Python Optuna worker that joins a study hosted by the C# TuningService.

The TuningService (examples/projects/TuningService) serves Optuna's own gRPC storage
protocol, so this script is plain Optuna 5: it connects with
optuna.storages.GrpcStorageProxy and optimizes the shared study next to the C#
TuningWorker processes. The service's dashboard shows which worker ran each trial.

    pip install "optuna>=5" grpcio protobuf
    python examples/projects/TuningWorker/python_worker.py --trials 20

The objective is a line-for-line copy of TuningCommon/SimulatedTraining.cs. Keep the
two identical: the same parameter names, distributions and formula. Otherwise the C#
and Python workers optimize different problems under one study name.

source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.GrpcStorageProxy.html
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import time
from collections.abc import Callable

import optuna
from optuna.storages import GrpcStorageProxy

# the number of epochs one trial trains for; also the last reported step
EPOCHS = 20

# the user attribute that records which worker evaluated a trial
WORKER_ATTRIBUTE = "worker"


def validation_loss(
    epoch: int,
    learning_rate: float,
    optimizer: str,
    batch_size: int,
    dropout: float,
    weight_decay: float,
) -> float:
    """Return the simulated validation loss after `epoch` epochs.

    An exponential decay from 1.0 toward a floor: a learning rate far from the
    optimizer's sweet spot slows the decay and raises the floor; dropout, batch size
    and weight decay away from their best values raise the floor. The same formula as
    SimulatedTraining.ValidationLoss in C#.

    Args:
        epoch: The epoch just finished, from 1 to EPOCHS.
        learning_rate: The learning rate, between 1e-5 and 1e-1.
        optimizer: "sgd", "momentum" or "adam".
        batch_size: The batch size: 16, 32, 64 or 128.
        dropout: The dropout rate, between 0 and 0.5.
        weight_decay: The weight decay, between 1e-6 and 1e-2.

    Returns:
        The loss, between the floor (at least 0.05) and 1.0.

    Raises:
        KeyError: `optimizer` is not sgd, momentum or adam.
    """
    # each optimizer has its own best learning rate and top speed
    best_learning_rate, top_speed = {
        "sgd": (3e-2, 0.12),
        "momentum": (1e-2, 0.18),
        "adam": (1e-3, 0.25),
    }[optimizer]

    # decades from the sweet spot, squared; x ** 2.0 matches the C# copy's Math.Pow(x, 2.0) bit for bit
    learning_rate_penalty = math.log10(learning_rate / best_learning_rate) ** 2.0
    speed = top_speed / (1.0 + learning_rate_penalty)
    floor = (
        0.05
        + 0.6 * (dropout - 0.2) ** 2.0
        + 0.015 * abs(math.log2(batch_size / 64.0))
        + 0.04 * learning_rate_penalty
        + 0.005 * (math.log10(weight_decay) + 4.0) ** 2.0
    )
    return floor + (1.0 - floor) * math.exp(-speed * epoch)


def make_objective(worker_name: str, epoch_delay: float) -> Callable[[optuna.Trial], float]:
    """Build the objective: record the worker, suggest five parameters, report every epoch.

    Args:
        worker_name: Stored on every trial as the `worker` user attribute.
        epoch_delay: Seconds each simulated epoch sleeps; 0 runs a trial instantly.

    Returns:
        The objective function that study.optimize calls once per trial.
    """

    def objective(trial: optuna.Trial) -> float:
        """Evaluate one trial, stopping early when the study's pruner says so.

        Args:
            trial: The trial to evaluate.

        Returns:
            The validation loss after the last epoch, which the study minimizes.

        Raises:
            optuna.TrialPruned: The pruner stopped the trial; study.optimize records it as PRUNED.
        """
        trial.set_user_attr(WORKER_ATTRIBUTE, worker_name)

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        optimizer = trial.suggest_categorical("optimizer", ["sgd", "momentum", "adam"])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        loss = math.nan
        for epoch in range(1, EPOCHS + 1):
            if epoch_delay > 0:
                time.sleep(epoch_delay)
            loss = validation_loss(epoch, learning_rate, optimizer, batch_size, dropout, weight_decay)
            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")
        return loss

    return objective


def main() -> None:
    """Join the shared study, run this worker's trials, and print the study's best trial.

    Raises:
        ConnectionError: No storage proxy answered within 15 seconds ("GRPC connection timeout").
    """
    parser = argparse.ArgumentParser(description="Python Optuna worker for the OptunaSharp TuningService.")
    parser.add_argument("--host", default="localhost", help="host of the TuningService's gRPC storage proxy")
    parser.add_argument("--port", type=int, default=13000, help="port of the proxy (Optuna's default)")
    parser.add_argument("--study", default="distributed", help="the study to join; created if missing")
    parser.add_argument("--trials", type=int, default=20, help="how many trials this process runs")
    parser.add_argument("--epoch-delay-ms", type=int, default=25, help="milliseconds per simulated epoch")
    args = parser.parse_args()

    storage = GrpcStorageProxy(host=args.host, port=args.port)
    storage.wait_server_ready(timeout=15)

    # the same pruner settings as the C# workers, so all of them prune by the same rule; the sampler is
    # Optuna's default, an unseeded TPESampler, like the C# workers'
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    worker_name = f"python:{socket.gethostname()}:{os.getpid()}"
    study.optimize(make_objective(worker_name, args.epoch_delay_ms / 1000.0), n_trials=args.trials)

    complete = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if complete:
        best = study.best_trial
        print(
            f"{worker_name} ran {args.trials} trials of study {study.study_name!r}, which now has "
            f"{len(study.trials)}. Best: trial {best.number}, loss {best.value}, "
            f"from {best.user_attrs.get(WORKER_ATTRIBUTE)}."
        )
    storage.close()


if __name__ == "__main__":
    main()
