"""Module for training and predicting using models and validation strategies."""

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..dataset.pipeline import Pipeline
from ..models.base import Estimator
from .torch_based.callbacks import ES_Checkpoints_Manager
from .torch_based.data_provider import Dataset_NN
from .torch_based.metrics import NegativeMSEMetric
from .validator import Validator

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Subset
except ImportError:
    torch = None
    Subset = None
    nn = None

import logging

logger = logging.getLogger(__name__)


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
        model_params: Dict = {},
        validator: Optional[Validator] = None,
        validation_params: Dict = {},
    ):
        self.model = model
        self.model_params = model_params
        self.validator = validator
        self.validation_params = validation_params

        # Provide by strategy if needed
        self.history = None
        self.horizon = None
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

        if (
            np.isnan(X).sum() != 0
            or (X_val is not None and np.isnan(X_val).sum() != 0)
            or np.isnan(y).sum() != 0
            or (y_val is not None and np.isnan(y_val).sum() != 0)
        ):
            if np.isnan(X).sum() != 0 or np.isnan(X_val).sum() != 0:
                logger.warning("It seems that there are NaN values in the input data.")
            else:
                logger.warning("It seems that there are NaN values in the target data.")
            logger.warning(
                "Try to check pipeline configuration (normalization part, especially)."
                "NaN values can be caused by division by zero in DifferenceNormalizer or LastKnownNormalizer."
            )

        # Initialize columns' order and reorder columns
        self.features_argsort = np.argsort(pipeline.output_features)
        X = X[:, self.features_argsort]

        for fold_i, (X_train, y_train, X_val, y_val) in enumerate(
            self.validator(**self.validation_params).get_split(X, y, X_val, y_val)
        ):
            model = self.model(self.model_params)
            model.fit_one_fold(X_train, y_train, X_val, y_val)
            self.models.append(model)
            self.scores.append(model.score)
            logger.info(f"Fold {fold_i}. Score: {model.score}")

        logger.info(f"Mean score: {np.mean(self.scores).round(4)}")
        logger.info(f"Std: {np.std(self.scores).round(4)}")

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

        y_pred = y_pred.reshape(pipeline.y_original_shape)

        return y_pred


class DLTrainer:
    """Class for training and predicting using a deep learning model and a validation strategy.

    Args:
        model: the model estimator to be used for training.
        model_params: the parameters for the model.
        validator: the validation strategy to be used for training.
        validation_params: the parameters for the validation strategy.
        n_epochs: the number of epochs to train the model.
        batch_size: size of batches during training.
        drop_last: whether to drop the last incomplete batch.
        device: device to run the training on.
        device_ids: list of device IDs for data parallelism.
        num_workers: number of workers for data loading.
        metric: metric function to evaluate the model.
        criterion: loss function for training the model.
        optimizer: optimizer for training the model.
        optimizer_params: parameters for the optimizer.
        scheduler: learning rate scheduler.
        scheduler_params: parameters for the scheduler.
        scheduler_after_epoch: whether to step the scheduler after each epoch.
        pretrained_path: path to the pretrained checkpoints.
        best_by_metric: whether to select the best model by metric instead of loss.
        early_stopping_patience: number of epochs to wait for improvement before early stopping.
            0 for early stopping disable.
        save_k_best: number of best checkpoints to save.
            0 for none, `n_epochs` for all.
        averaging_snapshots: whether to average weights of saved checkpoints at the end of training.
        save_to_dir: whether to save checkpoints to a directory.
        checkpoint_path: path to save checkpoints.
        train_shuffle: whether to shuffle the training data.
        verbose: verbosity level.

    """

    def __init__(
        self,
        model: Estimator,
        model_params: Dict,
        validator: Optional[Validator] = None,
        validation_params: Dict = {},
        n_epochs: int = 10,
        batch_size: int = 32,
        drop_last: bool = False,
        device: Optional["torch.device"] = None,
        device_ids: List[int] = [0],
        num_workers: int = 4,
        metric: Callable = NegativeMSEMetric(),
        criterion: Optional["torch.nn.Module"] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        optimizer_params: Dict = {},
        scheduler: Optional["torch.optim.lr_scheduler.LRScheduler"] = None,
        scheduler_params: Dict = {},
        scheduler_after_epoch: bool = True,
        pretrained_path: Optional[Union[Path, str]] = None,
        # es_checkpoint_manager params
        best_by_metric: bool = False,
        early_stopping_patience: int = 5,
        save_k_best: Union[int] = 5,
        average_snapshots: bool = False,
        save_to_dir: bool = True,
        checkpoint_path: Union[Path, str] = "checkpoints/",
        train_shuffle: bool = True,
        verbose: int = 1,
    ):
        if device is None:
            device = torch.device("cuda")

        if criterion is None:
            criterion = torch.nn.MSELoss()

        if optimizer is None:
            optimizer = torch.optim.Adam

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
        self.optimizer_params = optimizer_params
        self.scheduler_base = scheduler
        self.scheduler_params = scheduler_params
        self.scheduler_after_epoch = scheduler_after_epoch

        self.pretrained_path = pretrained_path

        self.train_shuffle = train_shuffle

        self.es = ES_Checkpoints_Manager(
            monitor="val_metric" if best_by_metric else "val_loss",
            verbose=verbose,
            save_k_best=save_k_best,
            early_stopping_patience=early_stopping_patience,
            mode="max" if best_by_metric else "min",
            save_to_dir=save_to_dir,
        )
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path = checkpoint_path

        if isinstance(pretrained_path, str):
            pretrained_path = Path(pretrained_path)
        self.pretrained_path = pretrained_path

        self.average_snapshots = average_snapshots

        # Provide by strategy if needed
        self.callbacks = [self.es]
        self.history = None
        self.horizon = None
        self.target_len = None

        self.models = []
        self.optimizers = []
        self.schedulers = []
        self.scores = []

    def init_trainer_one_fold(self, num_features: int):
        """Initializes the model, optimizer, and scheduler for one fold.

        Args:
            num_features: Number of features in the input data.

        Returns:
            Initialized model, optimizer, and scheduler.

        """
        self.model_params["seq_len"] = num_features
        self.model_params["pred_len"] = self.horizon

        model = self.model_base(**self.model_params)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        else:
            model.to(self.device)

        optimizer = self.optimizer_base(model.parameters(), **self.optimizer_params)
        if self.scheduler_base is not None:
            scheduler = self.scheduler_base(optimizer, **self.scheduler_params)
        else:
            scheduler = None

        return model, optimizer, scheduler

    def load_trainer_one_fold(
        self,
        fold_i: int,
        model: "nn.Module",
        optimizer: "torch.optim.Optimizer",
        scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"],
    ):
        """Loads pretrained model, optimizer, and scheduler states for one fold.

        Args:
            fold_i: fold index.
            model: model to load the state into.
            optimizer: optimizer to load the state into.
            scheduler: scheduler to load the state into (if exists).

        Returns:
            model, optimizer, and scheduler with loaded states.

        """
        self.es = torch.load(self.pretrained_path / "es_checkpoint_manager.pth")
        pretrained_weights = self.es.get_last_snapshot(full_state=True)

        model.load_state_dict(pretrained_weights["model"])
        optimizer.load_state_dict(pretrained_weights["optimizer"])
        if scheduler:
            scheduler.load_state_dict(pretrained_weights["scheduler"])

        return model, optimizer, scheduler

    def train_model(
        self,
        train_loader: "torch.utils.data.DataLoader",
        val_loader: "torch.utils.data.DataLoader",
        model: "nn.Module",
        optimizer: "torch.optim.Optimizer",
        scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"],
    ) -> Tuple[
        "nn.Module", "torch.optim.Optimizer", Optional["torch.optim.lr_scheduler._LRScheduler"], float
    ]:
        """Trains the model for all epochs.

        Args:
            train_loader: dataLoader for the training data.
            val_loader: dataLoader for the validation data.
            model: model to be trained.
            optimizer: optimizer for training.
            scheduler: learning rate scheduler (if exists).

        Returns:
            trained model, optimizer, scheduler, and validation metric.
        """
        for cb in self.callbacks:
            cb.on_train_begin()

        for epoch in range(self.n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            model.train()
            running_loss = 0.0
            start_time = time.time()

            for inputs, targets in train_loader:
                for callback in self.callbacks:
                    callback.on_batch_begin()

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                if torch.isnan(inputs).sum() != 0 or torch.isnan(targets).sum() != 0:
                    if torch.isnan(inputs).sum() != 0:
                        logger.warning("It seems that there are NaN values in the input data.")
                    else:
                        logger.warning("It seems that there are NaN values in the target data.")
                    logger.warning(
                        "Try to check pipeline configuration (normalization part, especially)."
                        "NaN values can be caused by division by zero in DifferenceNormalizer or LastKnownNormalizer."
                    )

                outputs = model(inputs)
                if self.target_len is None:
                    self.target_len = targets.shape[2]
                outputs = outputs[:, :, : self.target_len]
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                logs = {"loss": loss.item()}
                for cb in self.callbacks:
                    cb.on_batch_end(logs)

                if not self.scheduler_after_epoch and scheduler is not None:
                    scheduler.step()
                    logger.info(f"Updating learning rate to {scheduler.get_last_lr()[0]:.6f}.")

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{self.n_epochs}, cost time: {epoch_time:.2f}s")
            logger.info(f"train loss: {epoch_loss:.4f}")

            val_loss, val_metric = self.validate_model(val_loader, model)
            logger.info(f"val loss: {val_loss:.4f}, val metric: {val_metric:.4f}")

            if self.scheduler_after_epoch and scheduler is not None:
                scheduler.step()
                logger.info(f"Updating learning rate to {scheduler.get_last_lr()[0]:.6f}.")

            # Сохранение модели и проверка early stopping
            logs = {
                "epoch": epoch,
                "filepath": self.checkpoint_path,
                "loss": epoch_loss,
                "val_loss": val_loss,
                "val_metric": val_metric,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            if scheduler:
                logs["scheduler_state_dict"] = scheduler.state_dict()

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs)
                if getattr(cb, "stop_training", False):
                    for cb in self.callbacks:
                        cb.on_train_end()
                    return model, optimizer, scheduler, val_metric

        for cb in self.callbacks:
            cb.on_train_end({"filepath": self.checkpoint_path})

        # return best_model or average_model if `n_epochs` = 0
        if self.n_epochs > 0:
            if self.average_snapshots:
                model.load_state_dict(self.es.get_average_snapshot())
            else:
                model.load_state_dict(self.es.get_best_snapshot())
        else:
            val_metric = np.nan

        return model, optimizer, scheduler, val_metric

    def validate_model(
        self,
        val_loader: "torch.utils.data.DataLoader",
        model: "nn.Module",
        return_outputs: bool = False,
        inference: bool = False,
    ) -> Union[float, Tuple[float, float], Tuple[float, float, "torch.Tensor", "torch.Tensor"]]:
        """Validates the model on the validation data.

        Args:
            val_loader: data loader for the validation data.
            model: model to be validated.
            return_outputs: whether to return the outputs and targets.
            inference: if True, skips logging and assumes test data.

        Returns:
            validation loss, metric, and optionally the outputs and targets.

        Note:
            The same method for both validation and make predictions on test data.
            The are NaN values in metric if test data is used.

        """
        model.eval()

        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                outputs = outputs[:, :, : self.target_len]
                all_outputs.append(outputs)
                all_targets.append(targets)

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        loss = self.criterion(all_outputs, all_targets).item()
        metric = self.metric(all_outputs, all_targets).item() if self.metric else 0.0

        if not inference:
            logger.info(f"Validation, Loss: {loss:.4f}, Metric: {metric:.4f}")

        if return_outputs:
            return loss, metric, all_outputs, all_targets
        else:
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
        train_dataset = Dataset_NN(data, pipeline)
        train_dataset_idx = np.arange(len(train_dataset))
        if val_data:
            val_dataset = Dataset_NN(val_data, pipeline)
            val_dataset_all_idx = np.arange(len(val_dataset))
        else:
            val_dataset = None
            val_dataset_all_idx = None

        for fold_i, (train_dataset_idx, _, val_dataset_idx, _) in enumerate(
            self.validator_base(**self.validation_params).get_split(
                X=train_dataset_idx, X_val=val_dataset_all_idx
            )
        ):
            checkpoint_path = self.checkpoint_path
            pretrained_path = self.pretrained_path

            self.checkpoint_path /= f"fold_{fold_i}"
            if pretrained_path:
                self.pretrained_path /= f"fold_{fold_i}"

            train_subset = Subset(train_dataset, train_dataset_idx)
            if val_dataset is not None:
                val_subset = Subset(val_dataset, val_dataset_idx)
            elif val_dataset_idx is not None:
                val_subset = Subset(train_dataset, val_dataset_idx)
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=self.train_shuffle,
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

            logger.info(f"length of train dataset: {len(train_subset)}")
            logger.info(f"length of val dataset: {len(val_subset)}")

            num_features = train_dataset[0][0].shape[0]

            # load or initialize model, optimizer, scheduler
            model, optimizer, scheduler = self.init_trainer_one_fold(num_features)
            if self.pretrained_path:
                model, optimizer, scheduler = self.load_trainer_one_fold(
                    fold_i, model, optimizer, scheduler
                )
            model, optimizer, scheduler, score = self.train_model(
                train_loader, val_loader, model, optimizer, scheduler
            )

            self.models.append(model)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
            self.scores.append(score)

            logger.info(f"Fold {fold_i}. Score: {score}")
            self.checkpoint_path = checkpoint_path
            self.pretrained_path = pretrained_path

        logger.info(f"Mean score: {np.mean(self.scores).round(4)}")
        logger.info(f"Std: {np.std(self.scores).round(4)}")

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
        test_dataset = Dataset_NN(data, pipeline)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

        logger.info(f"length of test dataset: {len(test_dataset)}")

        models_preds = [
            self.validate_model(test_loader, model, return_outputs=True, inference=True)[2].cpu()
            for model in self.models
        ]

        y_pred = np.mean(models_preds, axis=0)
        y_pred = torch.tensor(y_pred)

        if pipeline.strategy_name == "FlatWideMIMOStrategy":
            full_horizon = pipeline.y_original_shape[1]
            num_series = y_pred.shape[0] // full_horizon
            if pipeline.multivariate:
                y_pred = y_pred.reshape(num_series, full_horizon, -1)
            else:
                y_pred = y_pred.reshape(num_series, full_horizon, 1)

        if pipeline.multivariate:
            y_pred = y_pred.permute(2, 0, 1)
            y_pred = y_pred.reshape(-1, y_pred.shape[2])
        else:
            y_pred = y_pred.squeeze(-1)

        return y_pred.numpy()
