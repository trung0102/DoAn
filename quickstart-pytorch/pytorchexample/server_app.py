"""pytorchexample: A Flower / PyTorch app."""

from typing import List, Tuple, Dict, Iterable
import torch
from flwr.app import MessageType
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
    Context,
)
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import Strategy

from pytorchexample.task import Net

# Create ServerApp
app = ServerApp()

class RingRouterStrategy(Strategy):
    """A custom strategy that routes models between clients in a P2P ring topology."""
    
    def __init__(self, fraction_evaluate=1.0, min_fit_clients=5, min_evaluate_clients=5, min_available_clients=5):
        super().__init__()
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        
        self.node_weights: Dict[int, ArrayRecord] = {}
        
        # Early Stopping variables
        self.patience = 5
        self.patience_counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def summary(self) -> None:
        print("Using RingRouterStrategy for P2P")

    def _construct_message(self, record: RecordDict, node_id: int, message_type: str) -> Message:
        return Message(
            content=record,
            message_type=message_type,
            dst_node_id=node_id,
        )

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self.early_stop:
            return []
            
        node_ids = list(grid.get_node_ids())
        if not node_ids:
            return []

        node_ids = sorted(node_ids)
        
        config["server-round"] = server_round
        
        messages = []
        if server_round == 1:
            for node_id in node_ids:
                record = RecordDict({"arrays": arrays, "config": config})
                messages.append(self._construct_message(record, node_id, MessageType.TRAIN))
        else:
            for i, node_id in enumerate(node_ids):
                prev_node_id = node_ids[(i - 1) % len(node_ids)]
                if prev_node_id in self.node_weights:
                    record = RecordDict({"arrays": self.node_weights[prev_node_id], "config": config})
                else:
                    record = RecordDict({"arrays": arrays, "config": config})
                messages.append(self._construct_message(record, node_id, MessageType.TRAIN))
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        
        if not replies:
            return None, None
            
        valid_replies = [msg for msg in replies if not msg.has_error()]
        if not valid_replies:
            return None, None

        for msg in valid_replies:
            node_id = msg.metadata.src_node_id
            self.node_weights[node_id] = msg.content["arrays"]

        return valid_replies[0].content["arrays"], None

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self.early_stop or self.fraction_evaluate == 0.0:
            return []
            
        node_ids = list(grid.get_node_ids())
        config["server-round"] = server_round
        
        messages = []
        for node_id in node_ids:
            if node_id in self.node_weights:
                record = RecordDict({"arrays": self.node_weights[node_id], "config": config})
            else:
                record = RecordDict({"arrays": arrays, "config": config})
            messages.append(self._construct_message(record, node_id, MessageType.EVALUATE))
        return messages

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        
        valid_replies = [msg for msg in replies if not msg.has_error()]
        if not valid_replies:
            return None

        losses = []
        accuracies = []
        
        print(f"\n--- P2P Round {server_round} Evaluation ---")
        for msg in valid_replies:
            node_id = msg.metadata.src_node_id
            metrics = msg.content["metrics"]
            
            if "eval_loss" in metrics:
                loss = metrics["eval_loss"]
                losses.append(loss)
            if "eval_acc" in metrics:
                acc = metrics["eval_acc"]
                accuracies.append(acc)
            
            
            l = loss if "eval_loss" in metrics else 0.0
            a = acc if "eval_acc" in metrics else 0.0
            print(f"  Node {node_id}: Loss={l:.4f}, Acc={a:.4f}")
            
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        print(f"Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}")
        
        # Early Stopping Logic
        # if avg_loss < self.best_loss:
        #     self.best_loss = avg_loss
        #     self.patience_counter = 0
        # else:
        #     self.patience_counter += 1
        #     print(f"  -> No improvement for {self.patience_counter}/{self.patience} rounds.")
        #     if self.patience_counter >= self.patience:
        #         print(f"\n[EARLY STOPPING] Loss không giảm sau {self.patience} vòng. Đã hội tụ, dừng huấn luyện sớm!")
        #         self.early_stop = True
        
        return MetricRecord({"loss": avg_loss, "accuracy": avg_acc})


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    num_nodes: int = context.run_config["num-nodes"]

    
    initial_model = Net()
    arrays = ArrayRecord(initial_model.state_dict())

    strategy = RingRouterStrategy(
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=num_nodes,
        min_evaluate_clients=num_nodes,
        min_available_clients=num_nodes
    )

    # Start strategy
    print("\nStarting P2P Federated Learning (Ring Topology)...")
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    print("\nP2P Training finished!")
