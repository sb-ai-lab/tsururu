import copy
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm.notebook import tqdm

from ..dataset.pipeline import Pipeline
from ..models.base import Estimator
from ..torch_based.callbacks import EarlyStopping, ModelCheckpoint
from ..torch_based.data_provider import nnDataset
from .validator import Validator


class MLTrainer:
    """Class for training and predicting using a model and a validation strategy.

    Args:
        model: the model estimator to be used for training.
        validator: the validation strategy to be used for training.

    """

    def __init__(self, model: Estimator, validator: Validator):
        self.model = deepcopy(model)
        self.validator = deepcopy(validator)

        self.models: List[Estimator] = []
        self.scores: List[float] = []

    def fit(self, data: dict, pipeline: Pipeline, val_data: Optional[dict]) -> "MLTrainer":
        """Fits the models using the input data and pipeline.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            pipeline: fitted data preprocessing pipeline.
            val_data: dictionary with validation data. Structure is similar
                to `data`.

        Returns:
            the fitted models.

        """
        X, y = pipeline.generate(data)
        if val_data:
            X_val, y_val = pipeline.generate(val_data)
        else:
            X_val, y_val = None, None

        # Initialize columns' order and reorder columns
        self.features_argsort = np.argsort(pipeline.output_features)
        X = X[:, self.features_argsort]

        for fold_i, X_train, y_train, X_val, y_val in enumerate(
            self.validator.get_split(X, y, X_val, y_val)
        ):
            model = self.model.fit_one_fold(X_train, y_train, X_val, y_val)
            self.models.append(model)
            self.scores.append(model.score)

            print(f"Fold {fold_i}. Score: {model.score}")

        print(f"Mean score: {np.mean(self.scores).round(4)}")
        print(f"Std: {np.std(self.scores).round(4)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions using the trained model.

        Args:
            X: features array.

        Returns:
            array of predicted values.
        """
        models_preds = [model.predict(X) for model in self.models]
        y_pred = np.mean(models_preds, axis=0)
        return y_pred


class DLTrainer:
    def __init__(
        self,
        model: Estimator,
        validator: Validator,
        n_epochs: int = 10,
        batch_size: int = 32,
        drop_last: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        device_ids: List[int] = [0],
        num_workers: int = 4,
        metric: Callable = torch.nn.MSELoss(),
        criterion: torch.nn.Module = torch.nn.MSELoss(),
        optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_after_epoch: bool = True,
        verbose: int = 1,
        stop_by_metric: bool = False,
        patience: int = 5,
        pretrained_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Union[str, Path] = "checkpoints/",
        callbacks: List[Callable] = [],
    ):
        self.model = deepcopy(model)
        self.validator = deepcopy(validator)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device
        self.device_ids = device_ids
        self.num_workers = num_workers
        self.metric = metric
        self.criterion = criterion
        self.optimizer = deepcopy(optimizer)
        self.scheduler = deepcopy(scheduler)
        self.scheduler_after_epoch = scheduler_after_epoch
        self.verbose = verbose
        # either by loss or metric
        self.stop_by_metric = stop_by_metric
        # how many epochs to wait for improvement before early stopping
        self.patience = patience
        self.pretrained_path = pretrained_path
        self.checkpoint_path = checkpoint_path
        self.callbacks = callbacks

        self.best_model_wts = deepcopy(model.state_dict())
        self.best_score = float("inf")
        self.early_stopping_counter = 0

        self.is_fitted = False

    def _build_model(self, model):
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        else:
            model.to(self.device)

        return model

    def _multivariate_collator(data_list, batch_size):
        date_groups = defaultdict(list)
        for X, y, idx_X in data_list:
            date = idx_X  # Assuming idx_X is date for simplicity, adapt if needed
            date_groups[date].append((X, y, idx_X))

        batches = []
        for date, group in date_groups.items():
            while len(group) >= batch_size:
                batches.append(custom_collator(group[:batch_size]))
                group = group[batch_size:]

        return batches

    def load_state(self, path):
        self.model.load_state_dict(torch.load(path))

    def init_trainer(self):
        self.model = self._build_model(self.model)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters())
        else:
            self.optimizer = self.optimizer(self.model.parameters())

        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer)

    def train_model(self, train_loader, val_loader):
        if self.pretrained_path is not None:
            self.load_state(self.pretrained_path)
        else:
            self.init_trainer()

        for epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            self.model.train()
            running_loss = 0.0

            for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
                for callback in self.callbacks:
                    callback.on_batch_begin()

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                logs = {"loss": loss.item()}
                for cb in self.callbacks:
                    cb.on_batch_end(logs)

                if self.scheduler_batch:
                    self.scheduler_batch.step()

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.4f}")

            val_loss, val_metric = self.validate_model(val_loader)
            print(f"Validation, Loss: {val_loss:.4f}, Metric: {val_metric:.4f}")

            if self.scheduler_epoch:
                self.scheduler_epoch.step(val_loss)

            # Сохранение модели и проверка early stopping
            logs = {
                "epoch": epoch,
                "loss": epoch_loss,
                "val_loss": val_loss,
                "val_metric": val_metric,
                "model_state_dict": self.model.state_dict(),
            }

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs)
                if getattr(cb, "stop_training", False):
                    for cb in self.callbacks:
                        cb.on_train_end()
                    return

        for cb in self.callbacks:
            cb.on_train_end()

    def validate_model(self, val_loader):
        self.model.eval()

        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                all_outputs.append(outputs)
                all_targets.append(targets)

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        loss = self.criterion(all_outputs, all_targets).item()
        metric = self.metric_fn(all_outputs, all_targets).item() if self.metric_fn else 0.0

        print(f"Validation, Loss: {loss:.4f}, Metric: {metric:.4f}")

        return loss, metric

    def fit(self, data: dict, pipeline: Pipeline, val_data: Optional[dict]) -> "DLTrainer":
        """Fits the models using the input data and pipeline.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            pipeline: fitted data preprocessing pipeline.
            val_data: dictionary with validation data. Structure is similar
                to `data`.

        Returns:
            the fitted models.

        """
        train_dataset = nnDataset(data, pipeline)

        if pipeline.multivariate:
            # Custom DataLoader for multivariate case
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size * 4,  # Initial larger batch size
                shuffle=True,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
                collate_fn=lambda batch: self._multivariate_collator(batch, self.batch_size),
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
            )

        if val_data:
            val_dataset = nnDataset(val_data, pipeline)

            if pipeline.multivariate:
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.batch_size * 4,  # Initial larger batch size
                    shuffle=False,
                    drop_last=self.drop_last,
                    num_workers=self.num_workers,
                    collate_fn=lambda batch: self._multivariate_collator(batch, self.batch_size),
                )
            else:
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=self.drop_last,
                    num_workers=self.num_workers,
                )
        else:
            val_loader = None

        self.train_model(train_loader, val_loader)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions using the trained model.

        Args:
            X: features array.

        Returns:
            array of predicted values.
        """
        models_preds = [model.predict(X) for model in self.models]
        y_pred = np.mean(models_preds, axis=0)
        return y_pred
