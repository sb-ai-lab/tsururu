from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm.notebook import tqdm

from ..dataset.pipeline import Pipeline
from ..models.base import Estimator
from .torch_based.callbacks import EarlyStopping, ModelCheckpoint
from .torch_based.data_provider import nnDataset
from .validator import Validator


class MLTrainer:
    """Class for training and predicting using a model and a validation strategy.

    Args:
        model: the model estimator to be used for training.
        model_params: the parameters for the model.
        validator: the validation strategy to be used for training.
        validation_params: the parameters for the validation strategy.

    """

    def __init__(
        self,
        model: Estimator,
        model_params: Dict,
        validator: Optional[Validator] = None,
        validation_params: Dict = {},
    ):
        self.model = model
        self.model_params = model_params
        self.validation = validator
        self.validation_params = validation_params

        self.models: List[Estimator] = []
        self.scores: List[float] = []
        self.columns: List[str] = []

    def fit(self, data: dict, pipeline: Pipeline, val_data: Optional[dict] = None) -> "MLTrainer":
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
            self.validator(**self.validation_params).get_split(X, y, X_val, y_val)
        ):
            model = self.model(**self.model_params["params"])
            model.fit_one_fold(X_train, y_train, X_val, y_val)
            self.models.append(model)
            self.scores.append(model.score)

            print(f"Fold {fold_i}. Score: {model.score}")

        print(f"Mean score: {np.mean(self.scores).round(4)}")
        print(f"Std: {np.std(self.scores).round(4)}")

    def predict(self, data: dict, pipeline: Pipeline) -> np.ndarray:
        """Generates predictions using the trained model.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            pipeline: fitted data preprocessing pipeline.

        Returns:
            array of predicted values.

        """
        X, _ = pipeline.generate(data)

        # Reorder columns
        X = X[:, self.features_argsort]

        models_preds = [model.predict(X) for model in self.models]
        y_pred = np.mean(models_preds, axis=0)

        return y_pred


class DLTrainer:
    def __init__(
        self,
        model: Estimator,
        model_params: Dict,
        validator: Optional[Validator] = None,
        validation_params: Dict = {},
        n_epochs: int = 10,
        batch_size: int = 32,
        drop_last: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        device_ids: List[int] = [0],
        num_workers: int = 4,
        metric: Callable = torch.nn.MSELoss(),
        criterion: torch.nn.Module = torch.nn.MSELoss(),
        optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam,
        optinizer_params: Dict = {},
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_params: Dict = {},
        scheduler_after_epoch: bool = True,
        verbose: int = 1,
        early_stopping: bool = True,
        stop_by_metric: bool = False,
        patience: int = 5,
        pretrained_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Union[str, Path] = "checkpoints/",
    ):
        self.model_base = model
        self.model_params = model_params
        self.validator_base = validator
        self.validation_params = validation_params
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device
        self.device_ids = device_ids
        self.num_workers = num_workers
        self.metric = metric
        self.criterion = criterion
        self.optimizer_base = optimizer
        self.optinizer_params = optinizer_params
        self.scheduler_base = scheduler
        self.scheduler_params = scheduler_params
        self.scheduler_after_epoch = scheduler_after_epoch
        self.verbose = verbose
        self.early_stopping = early_stopping
        # either by loss or metric
        self.stop_by_metric = stop_by_metric
        # how many epochs to wait for improvement before early stopping
        self.patience = patience
        self.pretrained_path = pretrained_path
        self.checkpoint_path = checkpoint_path

        self.callbacks = [
            EarlyStopping(patience=patience, stop_by_metric=stop_by_metric),
            ModelCheckpoint(checkpoint_path),
        ]
        self.models = None
        self.optimizers = None
        self.schedulers = None
        self.is_fitted = False

    def init_trainer_one_fold(self):
        model = self.model_base(**self.model_params)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        else:
            model.to(self.device)

        optimizer = self.optimizer_base(model.parameters(), **self.optinizer_params)
        if self.scheduler_base is not None:
            scheduler = self.scheduler_base(optimizer, **self.scheduler_params)
        return model, optimizer, scheduler

    def train_model(self, train_loader, val_loader):
        if self.pretrained_path is not None:
            self.load_state(self.pretrained_path)
        else:
            self.init_trainer()

        # for epoch in tqdm(range(self.n_epochs), desc="Epochs"):
        for epoch in range(self.n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            self.model.train()
            running_loss = 0.0

            # for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
            for inputs, targets in train_loader:
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

                if self.scheduler_after_epoch is False:
                    self.scheduler.step()

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.4f}")

            val_loss, val_metric = self.validate_model(val_loader)
            print(f"Validation, Loss: {val_loss:.4f}, Metric: {val_metric:.4f}")

            if self.scheduler_after_epoch:
                self.scheduler.step(val_loss)

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
                if getattr(cb, "stop_training", False):  # TODO: Точно False?
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
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                all_outputs.append(outputs)
                all_targets.append(targets)

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        loss = self.criterion(all_outputs, all_targets).item()
        metric = self.metric_fn(all_outputs, all_targets).item() if self.metric_fn else 0.0

        print(f"Validation, Loss: {loss:.4f}, Metric: {metric:.4f}")

        return loss, metric

    def fit(self, data: dict, pipeline: Pipeline, val_data: Optional[dict] = None) -> "DLTrainer":
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
        train_dataset_idx = np.arange(len(train_dataset))
        if val_data:
            val_dataset = nnDataset(val_data, pipeline)
            val_dataset_all_idx = np.arange(len(val_dataset))
        else:
            val_dataset = None
            val_dataset_all_idx = None

        for fold_i, train_dataset_idx, _, val_dataset_idx, _ in enumerate(
            self.validator_base(**self.validation_params).get_split(
                val_dataset_all_idx, val_dataset_all_idx
            )
        ):
            train_subset = Subset(train_dataset, train_dataset_idx)
            val_subset = Subset(val_dataset, val_dataset_idx)
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
            )
            val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
            )

            model, optimizer, scheduler = self.init_trainer_one_fold()
            model, optimizer, scheduler, score = self.fit_one_fold(
                train_loader, val_loader, model, optimizer, scheduler
            )

            self.models.append(model)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
            self.scores.append(score)

            print(f"Fold {fold_i}. Score: {model.score}")

        print(f"Mean score: {np.mean(self.scores).round(4)}")
        print(f"Std: {np.std(self.scores).round(4)}")

    def predict(self, data: dict, pipeline: Pipeline) -> np.ndarray:
        """Generates predictions using the trained model.

        Args:
            data: dictionary with current states of "elongated series",
                arrays with features and targets, name of id, date and target
                columns and indices for features and targets.
            pipeline: fitted data preprocessing pipeline.

        Returns:
            array of predicted values.

        """
        test_dataset = nnDataset(data, pipeline)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

        models_preds = [self.validate_model(test_loader, model) for model in self.models]
        y_pred = np.mean(models_preds, axis=0)

        return y_pred
