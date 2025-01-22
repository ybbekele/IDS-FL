import flwr as fl
import sys
import json
import numpy as np
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union
#from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import matplotlib.pyplot as plt

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 32,
        "current_round": server_round,
    }
    return config

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.server_metrics = {"round": [], "loss": [], "accuracy": []}  # Initialize server metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        # Aggregate loss
        aggregated_loss = sum(losses) / sum(examples)
        print(f"Round {server_round} loss aggregated from client results: {aggregated_loss}")

        # Store metrics
        self.server_metrics["round"].append(server_round)
        self.server_metrics["loss"].append(aggregated_loss)
        self.server_metrics["accuracy"].append(aggregated_accuracy)

        # Save metrics to a file
        with open('server_metrics.json', 'w') as f:
            json.dump(self.server_metrics, f)


        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}

# Create strategy and run server
strategy = AggregateCustomMetricStrategy(
    on_fit_config_fn=fit_config,
)
fl.server.start_server(strategy=strategy, server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=2087),
    ) 
