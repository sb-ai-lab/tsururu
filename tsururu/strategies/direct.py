from copy import deepcopy
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tsururu.dataset.dataset import TSDataset
from tsururu.dataset.pipeline import Pipeline
from tsururu.dataset.slice import IndexSlicer
from tsururu.model_training.trainer import DLTrainer, MLTrainer
from tsururu.strategies.recursive import RecursiveStrategy
from tsururu.strategies.utils import timing_decorator

index_slicer = IndexSlicer()


class DirectStrategy(RecursiveStrategy):
    """A strategy that uses an individual model for each point in the
        forecast horizon.

    Args:
        horizon: forecast horizon.
        history: number of previous for feature generating
            (i.e., features for observation y_t are counted from observations
            (y_{t-history}, ..., y_{t-1}).
        trainer: trainer with model params and validation params.
        pipeline: pipeline for feature and target generation.
        step:  in how many points to take the next observation while making
            samples' matrix.
        model_horizon: how many points to predict at a time,
            if model_horizon > 1, then it's an intermediate strategy between
            RecursiveStrategy and MIMOStrategy.
        equal_train_size: if true, all models are trained with the same
            training sample (which is equal to the training sample
            of the last model if equal_train_size=false).

    Notes:
        1. Fit: the models is fitted to predict certain point in the
            forecasting horizon (number of models = horizon).
        2. Inference: each model predict one point.

    """

    def __init__(
        self,
        horizon: int,
        history: int,
        trainer: Union[MLTrainer, DLTrainer],
        pipeline: Pipeline,
        step: int = 1,
        model_horizon: int = 1,
        equal_train_size: bool = False,
    ):
        super().__init__(horizon, history, trainer, pipeline, step, model_horizon)
        self.equal_train_size = equal_train_size
        self.strategy_name = "direct"

    @timing_decorator
    def fit(
        self,
        dataset: TSDataset,
    ) -> "DirectStrategy":
        """Fits the direct strategy to the given dataset.

        Args:
            dataset: The dataset to fit the strategy on.

        Returns:
            self.

        """
        self.trainers = []

        # intrinsic_horizon is a multiple of model_horizon
        intrinsic_horizon = self.model_horizon * (
            (self.horizon + self.model_horizon - 1) // self.model_horizon
        )

        if self.equal_train_size:
            features_idx = index_slicer.create_idx_data(
                dataset.data,
                self.model_horizon,
                self.history,
                self.step,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

            target_idx = index_slicer.create_idx_target(
                dataset.data,
                self.model_horizon,
                self.history,
                self.step,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

            data = self.pipeline.create_data_dict_for_pipeline(dataset, features_idx, target_idx)
            data = self.pipeline.fit_transform(data, self.strategy_name)

            val_dataset = self.trainer.validation_params.get("validation_data")

            if val_dataset:
                val_features_idx = index_slicer.create_idx_data(
                    val_dataset.data,
                    self.model_horizon,
                    self.history,
                    self.step,
                    date_column=val_dataset.date_column,
                    delta=val_dataset.delta,
                )

                val_target_idx = index_slicer.create_idx_target(
                    val_dataset.data,
                    self.model_horizon,
                    self.history,
                    self.step,
                    date_column=val_dataset.date_column,
                    delta=val_dataset.delta,
                )

                val_data = self.pipeline.create_data_dict_for_pipeline(
                    val_dataset, val_features_idx, val_target_idx
                )
                val_data = self.pipeline.transform(val_data)
            else:
                val_data = None

            for model_i, horizon in enumerate(
                range(1, intrinsic_horizon // self.model_horizon + 1)
            ):
                target_idx = index_slicer.create_idx_target(
                    dataset.data,
                    intrinsic_horizon,
                    self.history,
                    self.step,
                    date_column=dataset.date_column,
                    delta=dataset.delta,
                )[:, (horizon - 1) * self.model_horizon : horizon * self.model_horizon]

                data["target_idx"] = target_idx

                if val_dataset:
                    val_target_idx = index_slicer.create_idx_target(
                        val_dataset.data,
                        intrinsic_horizon,
                        self.history,
                        self.step,
                        date_column=val_dataset.date_column,
                        delta=val_dataset.delta,
                    )[
                        :,
                        (horizon - 1) * self.model_horizon : horizon * self.model_horizon,
                    ]

                    val_data["target_idx"] = val_target_idx

                if isinstance(self.trainer, DLTrainer):
                    self.trainer.horizon = self.model_horizon
                    self.trainer.history = self.history

                current_trainer = deepcopy(self.trainer)

                # In Direct strategy, we train several models, one for each model_horizon
                if isinstance(current_trainer, DLTrainer):
                    checkpoint_path = current_trainer.checkpoint_path
                    pretrained_path = current_trainer.pretrained_path

                    current_trainer.checkpoint_path /= f"trainer_{model_i}"
                    if pretrained_path:
                        current_trainer.pretrained_path /= f"trainer_{model_i}"

                current_trainer.fit(data, self.pipeline, val_data)

                if isinstance(current_trainer, DLTrainer):
                    current_trainer.checkpoint_path = checkpoint_path
                    current_trainer.pretrained_path = pretrained_path

                self.trainers.append(current_trainer)

        else:
            for model_i, horizon in enumerate(
                range(1, intrinsic_horizon // self.model_horizon + 1)
            ):
                features_idx = index_slicer.create_idx_data(
                    dataset.data,
                    self.model_horizon * horizon,
                    self.history,
                    self.step,
                    date_column=dataset.date_column,
                    delta=dataset.delta,
                )

                target_idx = index_slicer.create_idx_target(
                    dataset.data,
                    self.model_horizon * horizon,
                    self.history,
                    self.step,
                    date_column=dataset.date_column,
                    delta=dataset.delta,
                    n_last_horizon=self.model_horizon,
                )

                data = self.pipeline.create_data_dict_for_pipeline(
                    dataset, features_idx, target_idx
                )
                data = self.pipeline.fit_transform(data, self.strategy_name)

                val_dataset = self.trainer.validation_params.get("validation_data")

                if val_dataset:
                    val_features_idx = index_slicer.create_idx_data(
                        val_dataset.data,
                        self.model_horizon * horizon,
                        self.history,
                        self.step,
                        date_column=val_dataset.date_column,
                        delta=val_dataset.delta,
                    )

                    val_target_idx = index_slicer.create_idx_target(
                        val_dataset.data,
                        self.model_horizon * horizon,
                        self.history,
                        self.step,
                        date_column=val_dataset.date_column,
                        delta=val_dataset.delta,
                        n_last_horizon=self.model_horizon,
                    )

                    val_data = self.pipeline.create_data_dict_for_pipeline(
                        val_dataset, val_features_idx, val_target_idx
                    )
                    val_data = self.pipeline.transform(val_data)
                else:
                    val_data = None

                if isinstance(self.trainer, DLTrainer):
                    self.trainer.horizon = self.model_horizon
                    self.trainer.history = self.history

                current_trainer = deepcopy(self.trainer)

                # In Direct strategy, we train several models, one for each model_horizon
                if isinstance(current_trainer, DLTrainer):
                    checkpoint_path = current_trainer.checkpoint_path
                    pretrained_path = current_trainer.pretrained_path

                    current_trainer.checkpoint_path /= f"trainer_{model_i}"
                    if pretrained_path:
                        current_trainer.pretrained_path /= f"trainer_{model_i}"

                current_trainer.fit(data, self.pipeline, val_data)

                if isinstance(current_trainer, DLTrainer):
                    current_trainer.checkpoint_path = checkpoint_path
                    current_trainer.pretrained_path = pretrained_path

                self.trainers.append(current_trainer)

        self.is_fitted = True

        return self

    def make_step(
        self,
        step: int,
        horizon: int,
        dataset: TSDataset,
        inverse_transform: bool = True,
    ) -> TSDataset:
        """Make a step in the direct strategy.

        Args:
            step: the step number.
            horizon: the horizon length.
            dataset: the dataset to make the step on.

        Returns:
            the updated dataset.

        """
        assert horizon % self.model_horizon == 0

        test_idx = index_slicer.create_idx_test(
            dataset.data,
            horizon,
            self.history,
            self.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )
        target_idx = index_slicer.create_idx_target(
            dataset.data,
            horizon,
            self.history,
            self.step,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )[:, self.model_horizon * step : self.model_horizon * (step + 1)]

        data = self.pipeline.create_data_dict_for_pipeline(dataset, test_idx, target_idx)
        data = self.pipeline.transform(data)

        pred = self.trainers[step % len(self.trainers)].predict(data, self.pipeline)
        if inverse_transform:
            pred = self.pipeline.inverse_transform_y(pred)

        dataset.data.loc[target_idx.reshape(-1), dataset.target_column] = pred.reshape(-1)

        return dataset

    def _aggregate_shap(self, aggregate_by_folds: bool) -> None:
        """Calls aggregate_feature_importance on every trainer.

        Args:
            aggregate_by_folds: passed through to trainer.aggregate_feature_importance.

        """
        for trainer in self.trainers:
            trainer.aggregate_feature_importance(trainer.feature_name, aggregate_by_folds)

    def _plot_shap_boxplots(self, top_k: int) -> None:
        """Per-fold SHAP boxplots for all trainers laid out in a grid (max 3 columns).

        Args:
            top_k: number of top features to show per subplot.

        """
        importance_data = {}
        for trainer_idx, trainer in enumerate(self.trainers):
            fold_data = {}
            keys = [
                k
                for k in trainer.shap_values["train"].keys()
                if k not in ("feature_name", "test", "feature_importance_aggregated")
            ]
            for fold_key in keys:
                data = trainer.shap_values["train"][fold_key]
                mean_imps = data.mean(axis=(0, 2))
                top_idx = np.argsort(mean_imps)[-top_k:]
                fold_data[fold_key] = {
                    "values": data[:, top_idx, 0],
                    "features": trainer.shap_values["train"]["feature_name"][top_idx],
                }
            importance_data[f"trainer_{trainer_idx + 1}"] = fold_data

        first_trainer_data = next(iter(importance_data.values()))
        n_folds = len(first_trainer_data)
        n_trainers = len(importance_data)

        n_cols = min(3, n_folds)
        n_rows_grid = int(np.ceil(n_folds / n_cols))
        total_rows = n_trainers * n_rows_grid

        fig, axes = plt.subplots(
            nrows=total_rows,
            ncols=n_cols,
            figsize=(6 * n_cols, 6 * total_rows * top_k / 8),
            squeeze=False,
        )

        for trainer_idx, (_, fold_data) in enumerate(importance_data.items()):
            start_row = trainer_idx * n_rows_grid
            for fold_idx, (_, data_entry) in enumerate(fold_data.items()):
                row = start_row + fold_idx // n_cols
                col = fold_idx % n_cols
                ax = axes[row, col]

                bp = ax.boxplot(data_entry["values"], orientation="horizontal", patch_artist=True)
                for patch in bp["boxes"]:
                    patch.set_facecolor("lightblue")

                ax.set_title(
                    f"Trainer {trainer_idx + 1}, Fold {fold_idx + 1}",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_yticklabels(data_entry["features"])
                ax.set_xlabel("SHAP Value")
                ax.grid(True)

        # Hide unused subplots
        for trainer_idx in range(n_trainers):
            start_row = trainer_idx * n_rows_grid
            for idx in range(n_folds, n_rows_grid * n_cols):
                row = start_row + idx // n_cols
                col = idx % n_cols
                axes[row, col].set_visible(False)

        plt.suptitle("SHAP Feature Importance per Trainer and Fold", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def _plot_shap_barh(self, trainer_idx: int, top_k: int, round_to: int) -> None:
        """Aggregated SHAP horizontal bar chart for one trainer.

        Args:
            trainer_idx: index into self.trainers.
            top_k: number of top features to show.
            round_to: decimal places for rounding (0 = integers).

        """
        trainer = self.trainers[trainer_idx]
        agg_imp = trainer.shap_values["train"]["feature_importance_aggregated"]
        mean_agg = np.abs(agg_imp).mean(axis=1)
        top_idx = np.argsort(mean_agg)[-top_k:]
        sorted_imps = mean_agg[top_idx]
        sorted_features = np.array(trainer.shap_values["train"]["feature_name"])[top_idx].ravel()

        sorted_imps = sorted_imps.astype(int) if round_to == 0 else np.round(sorted_imps, round_to)

        bar_container = plt.barh(width=sorted_imps, y=sorted_features)
        plt.bar_label(bar_container, sorted_imps, color="red")
        plt.gcf().set_size_inches(5, top_k / 6 + 1)
        sns.despine()
        plt.title(f"Aggregated shap feature importance by trainer {trainer_idx + 1}")
        plt.show()

    def get_feature_importance(
        self,
        top_k: int = 15,
        aggregate_by_folds: bool = True,
        round_to: int = 2,
        return_explainer: bool = False,
    ) -> Optional[List[Any]]:
        """Generates and visualizes feature importance based on SHAP values.

        Args:
            top_k: number of top features to display.
            aggregate_by_folds:
                True  — one aggregated bar chart per trainer.
                False — per-fold boxplots per trainer.
            round_to: decimal places for rounding (0 = integers).
            return_explainer: if True, returns list of shap_explainer objects.

        Returns:
            List of shap_explainers when return_explainer=True, otherwise None.

        """
        self._aggregate_shap(aggregate_by_folds)

        if not aggregate_by_folds:
            # All trainers + folds in one figure
            self._plot_shap_boxplots(top_k)

        arr_explainers = []
        arr_train_shap = []

        for trainer_idx, trainer in enumerate(self.trainers):
            arr_explainers.append(trainer.shap_explainer)

            if aggregate_by_folds:
                self._plot_shap_barh(trainer_idx, top_k, round_to)

            arr_train_shap.append(
                {
                    "shap_values": trainer.shap_values["train"],
                    "feature_names": trainer.shap_values["train"]["feature_name"],
                }
            )

        self.arr_train_shap = arr_train_shap

        if return_explainer:
            return arr_explainers

    def get_train_shap(self) -> List[dict]:
        """Returns training SHAP values for all trainers.

        Returns:
            List of dicts with 'shap_values' and 'feature_names', one per trainer.

        """
        return self.arr_train_shap

    def get_test_shap(self) -> List[dict]:
        """Returns test SHAP values for all trainers.

        Returns:
            List of test SHAP value dicts, one per trainer.

        """
        return [trainer.shap_values["test"] for trainer in self.trainers]
