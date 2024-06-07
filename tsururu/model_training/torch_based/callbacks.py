import heapq
import os

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


class EarlyStopping(Callback):
    def __init__(self, patience=5, verbose=1, stop_by_metric=True):
        self.patience = patience
        self.verbose = verbose
        self.best_score = None
        self.early_stopping_counter = 0
        self.stop_training = False
        self.stop_by_metric = stop_by_metric

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get("val_metric") if self.stop_by_metric else logs.get("val_loss")
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered")
                self.stop_training = True
        else:
            self.best_score = current_score
            self.early_stopping_counter = 0


class ModelCheckpoint:
    def __init__(self, filepath, monitor="val_loss", verbose=1, save_best_only=True, k=5):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.k = k
        self.best_snapshots = []
        self.worst_best_score = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        model_path = f"{self.filepath}_epoch_{epoch}.pth"

        if self.save_best_only:
            if len(self.best_snapshots) < self.k or current_score < self.worst_best_score:
                if len(self.best_snapshots) == self.k:
                    worst_snapshot = heapq.heappop(self.best_snapshots)
                    if self.verbose:
                        print(f"Removing worst model snapshot: {worst_snapshot[1]}")
                    os.remove(worst_snapshot[1])

                heapq.heappush(self.best_snapshots, (current_score, model_path))
                self.worst_best_score = max(self.best_snapshots)[0]
                torch.save(logs["model_state_dict"], model_path)

                if self.verbose:
                    print(f"Model saved to {model_path}")
        else:
            torch.save(logs["model_state_dict"], model_path)
            if self.verbose:
                print(f"Model saved to {model_path}")

    def get_best_snapshots(self):
        return [snapshot[1] for snapshot in sorted(self.best_snapshots)]
