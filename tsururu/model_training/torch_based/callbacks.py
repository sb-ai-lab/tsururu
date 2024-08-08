"""Module for callbacks used in training process."""

import heapq
import os
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None


class Callback:
    """Base class for callbacks, that are used in training process."""

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
    """Manager for early stopping and checkpointing during training.

    Args:
        monitor: metric to monitor for early stopping and checkpointing.
        verbose: verbosity mode, 0 or 1.
        save_k_best: number of best checkpoints to keep.
        early_stopping_patience: number of epochs to wait for an improvement before stopping.
        mode: mode for monitoring, either 'min' or 'max'.
        save_to_dir: whether to save checkpoints to a directory.
        averaging_snapshots: whether to average weights of saved checkpoints at the end of training.

    """

    def __init__(
        self,
        monitor: str = "val_loss",
        verbose: int = 1,
        save_k_best: int = 1,
        early_stopping_patience: int = 5,
        mode: str = "min",
        save_to_dir: bool = True,
        averaging_snapshots: bool = False,
    ):
        self.monitor = monitor
        self.verbose = verbose
        self.save_k_best = save_k_best
        self.early_stopping_patience = early_stopping_patience
        self.mode = mode
        self.save_to_dir = save_to_dir
        self.averaging_snapshots = averaging_snapshots

        self.best_snapshots = []
        self.best_score = None
        self.early_stopping_counter = 0
        self.stop_training = False

    @staticmethod
    def _safe_remove(file_path: str):
        """Safely removes a file if it exists.

        Args:
            file_path: path to the file to be removed.

        """
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def _safe_rename(src_path: str, dst_path: str):
        """Safely renames a file if it exists.

        Args:
            src_path: source name of the file.
            dst_path: new name of the file.

        """
        if os.path.exists(src_path):
            os.rename(src_path, dst_path)

    def _del_inner_params(self):
        """Deletes internal parameters to reset the manager."""
        self.best_snapshots = []
        self.best_score = None
        self.early_stopping_counter = 0
        self.stop_training = False

    def _is_improvement(self, current: float, best: float) -> bool:
        """Checks if the current score is an improvement.

        Args:
            current: current score.
            best: best score.

        Returns:
            whether the current score is an improvement.

        """
        if self.mode == "min":
            return current < best
        else:
            return current > best

    def _should_save_checkpoint(self, current_score: float) -> bool:
        """Checks if the current checkpoint should be saved.

        Args:
            current_score: current score.

        Returns:
            whether the checkpoint should be saved.

        """
        if len(self.best_snapshots) < self.save_k_best:
            return True

        worst_best_score = (
            -self.best_snapshots[0][0] if self.mode == "min" else self.best_snapshots[0][0]
        )
        return self._is_improvement(current_score, worst_best_score)

    def _update_worst_best_score(self):
        """Updates the worst best score from the saved checkpoints."""
        if self.mode == "min":
            self.worst_best_score = -self.best_snapshots[0][0]
        else:
            self.worst_best_score = self.best_snapshots[0][0]

    def _rename_checkpoints(self):
        """Renames checkpoints based on their ranking.

        Notes: is called at the end of training to reflect the ranking of the checkpoints.

        """
        for i, (_, snapshot_info) in enumerate(
            sorted(
                self.best_snapshots,
                key=lambda x: -x[0] if self.mode == "min" else x[0],
                reverse=True,
            )
        ):
            epoch = snapshot_info["epoch"]
            if self.save_to_dir:
                new_model_path = snapshot_info["model"].with_name(
                    f"model_epoch_{epoch}_top_{i+1}.pth"
                )
                new_opt_path = snapshot_info["optimizer"].with_name(
                    f"opt_epoch_{epoch}_top_{i+1}.pth"
                )
                new_sch_path = snapshot_info["scheduler"].with_name(
                    f"sch_epoch_{epoch}_top_{i+1}.pth"
                )

                self._safe_rename(snapshot_info["model"], new_model_path)
                snapshot_info["model"] = new_model_path

                self._safe_rename(snapshot_info["optimizer"], new_opt_path)
                snapshot_info["optimizer"] = new_opt_path

                self._safe_rename(snapshot_info["scheduler"], new_sch_path)
                snapshot_info["scheduler"] = new_sch_path

    def get_best_snapshot(self) -> dict:
        """Returns the best saved snapshot.

        Returns:
            the best model snapshot.

        """
        best_snapshot = [
            snapshot[1]["model"]
            for snapshot in sorted(
                self.best_snapshots, key=lambda x: -x[0] if self.mode == "min" else x[0]
            )
        ][-1]

        if self.save_to_dir:
            return torch.load(best_snapshot)
        return best_snapshot

    def get_average_snapshot(self) -> dict:
        """Returns the average snapshot.

        Returns:
            the average model snapshot.

        Notes:
            - Use simple averaging to combine the weights of the saved checkpoints.
            - Is called by trainer at the end of training if averaging_snapshots is True.

        """
        average_snapshot = None
        for snapshot in self.best_snapshots:
            if average_snapshot is None:
                average_snapshot = torch.load(snapshot[1]["model"])
            else:
                average_snapshot += torch.load(snapshot[1]["model"])
        average_snapshot /= len(self.best_snapshots)
        return average_snapshot

    def on_train_begin(self, logs: Optional[dict] = None):
        """Called at the beginning of training.

        Args:
            logs: dictionary of logs.

        Notes:
            - Resets internal parameters at the beginning of training.

        """
        self._del_inner_params()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Called at the end of each epoch.

        Args:
            epoch: the current epoch.
            logs: dictionary of logs.

        Notes:
            - Saves model checkpoints based on the current score.
            - Triggers early stopping if the score does not improve for a number of epochs.

        """
        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        model_state = logs["model_state_dict"]
        optimizer_state = logs["optimizer_state_dict"]
        scheduler_state = logs.get("scheduler_state_dict", None)

        if self.save_to_dir:
            model_path = Path(logs.get("filepath")) / f"model_{epoch}.pth"
            opt_path = Path(logs.get("filepath")) / f"opt_{epoch}.pth"
            sch_path = Path(logs.get("filepath")) / f"sch_{epoch}.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.save_k_best > 0:
            if self._should_save_checkpoint(current_score):
                if len(self.best_snapshots) == self.save_k_best:
                    worst_snapshot = heapq.heappop(self.best_snapshots)
                    if self.save_to_dir:
                        self._safe_remove(worst_snapshot[1]["model"])
                        self._safe_remove(worst_snapshot[1]["optimizer"])
                        self._safe_remove(worst_snapshot[1]["scheduler"])
                    if self.verbose:
                        print(
                            f"Removing worst model snapshot: from epoch {worst_snapshot[1]['epoch']}"
                        )

                snapshot_info = {
                    "model": model_path if self.save_to_dir else model_state,
                    "optimizer": opt_path if self.save_to_dir else optimizer_state,
                    "scheduler": sch_path if self.save_to_dir else scheduler_state,
                    "epoch": epoch,
                }

                if self.mode == "min":
                    heapq.heappush(self.best_snapshots, (-current_score, snapshot_info))
                else:
                    heapq.heappush(self.best_snapshots, (current_score, snapshot_info))

                self._update_worst_best_score()
                if self.save_to_dir:
                    torch.save(model_state, model_path)
                    torch.save(optimizer_state, opt_path)
                    if scheduler_state:
                        torch.save(scheduler_state, sch_path)
                    if self.verbose:
                        print(f"Model saved to {model_path}")
                        print(f"Optimizer saved to {opt_path}")
                        if scheduler_state:
                            print(f"Scheduler saved to {sch_path}")

        else:
            if self.save_to_dir:
                torch.save(model_state, model_path)
                torch.save(optimizer_state, opt_path)
                if scheduler_state:
                    torch.save(scheduler_state, sch_path)
                if self.verbose:
                    print(f"Model saved to {model_path}")
                    print(f"Optimizer saved to {opt_path}")
                    if scheduler_state:
                        print(f"Scheduler saved to {sch_path}")

        # Early stopping logic
        if self.early_stopping_patience > 0:
            if self.best_score is None or self._is_improvement(current_score, self.best_score):
                self.best_score = current_score
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                print(f"Early stopping counter: {self.early_stopping_counter}")
                if self.early_stopping_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print("Early stopping triggered")
                    self.stop_training = True

    def on_train_end(self, logs: Optional[dict] = None):
        """Called at the end of training.

        Args:
            logs: dictionary of logs.

        Notes:
            - Renames checkpoints to reflect their current ranking at the end of training.

        """
        if self.save_to_dir:
            self._rename_checkpoints()
