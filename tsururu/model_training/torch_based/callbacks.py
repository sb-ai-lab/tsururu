"""Module for callbacks used in training process."""

import heapq
import logging
import os
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None


logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        verbose: int = 1,
        save_k_best: int = 1,
        early_stopping_patience: int = 5,
        mode: str = "min",
        save_to_dir: bool = True,
    ):
        self.monitor = monitor
        self.verbose = verbose
        self.save_k_best = save_k_best
        self.early_stopping_patience = early_stopping_patience
        self.mode = mode
        self.save_to_dir = save_to_dir

        self.best_snapshots = []
        self.best_score = None
        self.early_stopping_counter = 0
        self.stop_training = False

        self.last_snapshot = None  # Для сохранения последней эпохи

    @staticmethod
    def _safe_remove(file_path: str):
        """Safely removes a file if it exists.

        Args:
            file_path: path to the file to be removed.

        """
        if os.path.exists(file_path):
            os.remove(file_path)

    def _del_inner_params(self):
        """Deletes internal parameters to reset the manager."""
        self.best_snapshots = []
        self.best_score = None
        self.early_stopping_counter = 0
        self.stop_training = False
        self.last_snapshot = None

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
        num_snapshots = len(self.best_snapshots)

        for snapshot in self.best_snapshots:
            model_state = torch.load(snapshot[1]["model"])

            if average_snapshot is None:
                average_snapshot = {}
                for key, value in model_state.items():
                    average_snapshot[key] = value.clone()
            else:
                for key, value in model_state.items():
                    average_snapshot[key] += value

        for key in average_snapshot:
            average_snapshot[key] /= num_snapshots

        return average_snapshot

    def get_last_snapshot(self, full_state: bool = False) -> dict:
        """Returns the last saved snapshot.

        Args:
            full_state: Whether to return the full state (model, optimizer, scheduler) or only the model.

        Returns:
            dict: The last saved snapshot with model, optimizer, and scheduler if full_state is True.
                  Otherwise, only the model state.
        """
        if self.last_snapshot is None:
            raise ValueError("No last snapshot saved.")

        if self.save_to_dir:
            model_state = torch.load(self.last_snapshot["model"])
            if full_state:
                optimizer_state = torch.load(self.last_snapshot["optimizer"])
                scheduler_state = (
                    torch.load(self.last_snapshot["scheduler"])
                    if self.last_snapshot["scheduler"]
                    and os.path.exists(self.last_snapshot["scheduler"])
                    else None
                )
                return {
                    "model": model_state,
                    "optimizer": optimizer_state,
                    "scheduler": scheduler_state,
                }
            return model_state

        # Если мы сохраняем без файловой системы
        if full_state:
            return {
                "model": self.last_snapshot["model"],
                "optimizer": self.last_snapshot["optimizer"],
                "scheduler": self.last_snapshot["scheduler"],
            }
        return self.last_snapshot["model"]

    def on_train_begin(self, logs: Optional[dict] = None):
        """Called at the beginning of training."""
        self._del_inner_params()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Called at the end of each epoch."""
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

        # Save the last epoch's snapshot for resuming training
        self.last_snapshot = {
            "model": model_path if self.save_to_dir else model_state,
            "optimizer": opt_path if self.save_to_dir else optimizer_state,
            "scheduler": sch_path if self.save_to_dir else scheduler_state,
            "epoch": epoch,
        }

        if self.save_to_dir:
            torch.save(model_state, model_path)
            torch.save(optimizer_state, opt_path)
            if scheduler_state:
                torch.save(scheduler_state, sch_path)
            if self.verbose:
                logger.info(f"Last epoch model saved to {model_path}")
                logger.info(f"Last epoch optimizer saved to {opt_path}")
                
                if scheduler_state:
                    logger.info(f"Last epoch scheduler saved to {sch_path}")

        # Save top-k best snapshots
        if self.save_k_best > 0 and self._should_save_checkpoint(current_score):
            if len(self.best_snapshots) == self.save_k_best:
                worst_snapshot = heapq.heappop(self.best_snapshots)
                if self.save_to_dir:
                    self._safe_remove(worst_snapshot[1]["model"])
                    self._safe_remove(worst_snapshot[1]["optimizer"])
                    self._safe_remove(worst_snapshot[1]["scheduler"])
                if self.verbose:
                    logger.info(
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
                if self.verbose:
                    logger.info(f"Best model snapshot saved to {model_path}")

        # Early stopping logic
        if self.early_stopping_patience > 0:
            if self.best_score is None or self._is_improvement(current_score, self.best_score):
                self.best_score = current_score
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                logger.info(f"Early stopping counter: {self.early_stopping_counter}")
                if self.early_stopping_counter >= self.early_stopping_patience:
                    if self.verbose:
                        logger.info("Early stopping triggered")
                    self.stop_training = True

        if self.save_to_dir:
            manager_path = Path(logs.get("filepath")) / "es_checkpoint_manager.pth"
            torch.save(self, manager_path)
            if self.verbose:
                logger.info(f"Checkpoint manager saved to {manager_path}")

    def on_train_end(self, logs: Optional[dict] = None):
        """Called at the end of training."""
        if self.verbose:
            logger.info("Training finished.")
