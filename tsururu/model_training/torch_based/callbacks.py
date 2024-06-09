import heapq
import os
from pathlib import Path

import torch


class Callback:
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, logs=None):
        pass

    def on_batch_end(self, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class ES_Checkpoints_Manager(Callback):
    def __init__(self, monitor="val_loss", verbose=1, save_best_only=True, k=5, patience=5, mode="min"):
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.k = k
        self.patience = patience
        self.mode = mode
        self.best_snapshots = []
        self.best_score = None
        self.worst_best_score = float('inf') if mode == "min" else float('-inf')
        self.early_stopping_counter = 0
        self.stop_training = False

    def _is_improvement(self, current, best):
        if self.mode == "min":
            return current < best
        else:
            return current > best

    def _should_save_checkpoint(self, current_score):
        if len(self.best_snapshots) < self.k:
            return True
        if self.mode == "min":
            return current_score < self.worst_best_score
        else:
            return current_score > self.worst_best_score

    def _update_worst_best_score(self):
        if self.mode == "min":
            self.worst_best_score = max(self.best_snapshots)[0]
        else:
            self.worst_best_score = min(self.best_snapshots)[0]

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        model_path = logs.get("filepath") / f"model_{epoch}.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.save_best_only:
            if self._should_save_checkpoint(current_score):
                if len(self.best_snapshots) == self.k:
                    worst_snapshot = heapq.heappop(self.best_snapshots)
                    os.remove(worst_snapshot[1])
                    if self.verbose:
                        print(f"Removing worst model snapshot: {worst_snapshot[1]}")

                heapq.heappush(self.best_snapshots, (current_score, model_path))
                self._update_worst_best_score()
                torch.save(logs["model_state_dict"], model_path)
                if self.verbose:
                    print(f"Model saved to {model_path}")
        else:
            torch.save(logs["model_state_dict"], model_path)
            if self.verbose:
                print(f"Model saved to {model_path}")

        # Early stopping logic
        if self.best_score is None or self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered")
                self.stop_training = True

    def get_best_snapshots(self):
        return [snapshot[1] for snapshot in sorted(self.best_snapshots, key=lambda x: x[0] if self.mode == "min" else -x[0])]