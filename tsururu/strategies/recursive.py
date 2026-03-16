from copy import deepcopy
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tsururu.dataset.dataset import TSDataset
from tsururu.dataset.pipeline import Pipeline
from tsururu.dataset.slice import IndexSlicer
from tsururu.model_training.trainer import DLTrainer, MLTrainer
from tsururu.strategies.base import Strategy
from tsururu.strategies.utils import timing_decorator

index_slicer = IndexSlicer()


class RecursiveStrategy(Strategy):
    """Strategy that uses a single model to predict all points in the
        forecast horizon.

    Arguments:
        horizon: forecast horizon.
        history: number of previous for feature generating
            (i.e., features for observation y_t are counted from
            observations (y_{t-history}, ..., y_{t-1}).
        step:  in how many points to take the next observation while
            making samples' matrix.
        trainer: trainer with model params and validation params.
        pipeline: pipeline for feature and target generation.
        model_horizon: how many points to predict at a time,
            if model_horizon > 1, then it's an intermediate strategy
            between RecursiveStrategy and MIMOStrategy.
        reduced: whether to form features for all test observations at
            once, in this case, unavailable values are replaced by NaN.

    Notes:
        1. Fit: the model is fitted to predict one point ahead.
        2. Inference: the model iteratively predicts the next point and
            - use this prediction to build further features
                (`reduced` == False);
            - use NaN instead of prediction (`reduced` == True).

    """

    def __init__(
        self,
        horizon: int,
        history: int,
        trainer: Union[MLTrainer, DLTrainer],
        pipeline: Pipeline,
        step: int = 1,
        model_horizon: int = 1,
        reduced: bool = False,
    ):
        super().__init__(horizon, history, trainer, pipeline, step)
        self.model_horizon = model_horizon
        self.reduced = reduced
        self.strategy_name = "recursive"

    @timing_decorator
    def fit(
        self,
        dataset: TSDataset,
    ) -> "RecursiveStrategy":
        """Fits the recursive strategy to the given dataset.

        Args:
            dataset: The dataset to fit the strategy on.

        Returns:
            self.

        """
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
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

            val_target_idx = index_slicer.create_idx_target(
                val_dataset.data,
                self.model_horizon,
                self.history,
                self.step,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

            val_data = self.pipeline.create_data_dict_for_pipeline(
                val_dataset, val_features_idx, val_target_idx
            )
            val_data = self.pipeline.transform(val_data)
        else:
            val_data = None

        if isinstance(self.trainer, DLTrainer):
            if self.strategy_name == "FlatWideMIMOStrategy":
                self.trainer.horizon = 1
            else:
                self.trainer.horizon = self.model_horizon
            self.trainer.history = self.history

        current_trainer = deepcopy(self.trainer)

        # In Recursive strategy, we train the individual model
        if isinstance(current_trainer, DLTrainer):
            checkpoint_path = current_trainer.checkpoint_path
            pretrained_path = current_trainer.pretrained_path

            current_trainer.checkpoint_path /= "trainer_0"
            if pretrained_path:
                current_trainer.pretrained_path /= "trainer_0"

        current_trainer.fit(data, self.pipeline, val_data)

        if isinstance(current_trainer, DLTrainer):
            current_trainer.checkpoint_path = checkpoint_path
            current_trainer.pretrained_path = pretrained_path

        self.trainers.append(current_trainer)

        self.is_fitted = True

        return self

    def make_step(
        self, step: int, horizon: int, dataset: TSDataset, inverse_transform: bool
    ) -> TSDataset:
        """Make a step in the recursive strategy.

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
            horizon - step * self.model_horizon,
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

        pred = self.trainers[0].predict(data, self.pipeline)
        if inverse_transform:
            pred = self.pipeline.inverse_transform_y(pred)

        num_series = data["num_series"] if self.pipeline.multivariate else 1

        target_idx = target_idx.reshape(num_series, -1, self.model_horizon)
        pred = pred.reshape(num_series, -1, self.model_horizon)

        target_idx = target_idx[:, : pred.shape[1]]

        dataset.data.loc[target_idx.reshape(-1), dataset.target_column] = pred.reshape(-1)

        return dataset

    @timing_decorator
    def predict(
        self,
        dataset: TSDataset,
        horizon: int | None = None,
        test_all: bool = False,
        inverse_transform: bool = True,
    ) -> pd.DataFrame:
        """Predicts the target values for the given dataset.

        Args:
            dataset (TSDataset): the dataset to make predictions on.
            horizon (int, optional): number of steps ahead to predict. If None, defaults to the model's training horizon.
            test_all (bool, default=False): if True, performs rolling window prediction over the entire dataset.
                Otherwise, predicts only the last window.
            inverse_transform (bool, default=True): if True, applies inverse transformations to the predictions
                (e.g., reversing normalization/scaling).

        Returns:
            a pandas DataFrame containing the predicted target values.

        """
        if not self.is_fitted:
            raise ValueError("The strategy is not fitted yet.")

        if horizon is None:
            horizon = self.horizon

        # intrinsic_horizon is a multiple of model_horizon
        intrinsic_horizon = self.model_horizon * (
            (horizon + self.model_horizon - 1) // self.model_horizon
        )

        new_data = dataset.make_padded_test(
            intrinsic_horizon, self.history, test_all=test_all, step=self.step
        )

        new_dataset = TSDataset(new_data, dataset.columns_params, dataset.delta)

        if test_all:
            new_dataset.data = new_dataset.data.sort_values(
                [dataset.id_column, "segment_col", dataset.date_column]
            )

        if self.reduced:
            current_test_ids = index_slicer.create_idx_data(
                new_dataset.data,
                self.model_horizon,
                self.history,
                step=self.model_horizon,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

            target_ids = index_slicer.create_idx_target(
                new_dataset.data,
                intrinsic_horizon,
                self.history,
                step=self.model_horizon,
                date_column=dataset.date_column,
                delta=dataset.delta,
            )

            data = self.pipeline.create_data_dict_for_pipeline(
                new_dataset, current_test_ids, target_ids
            )
            data = self.pipeline.transform(data)

            pred = self.trainers[0].predict(data, self.pipeline)
            if inverse_transform:
                pred = self.pipeline.inverse_transform_y(pred)

            new_dataset.data.loc[target_ids.reshape(-1), dataset.target_column] = pred.reshape(-1)

        else:
            for step in range(intrinsic_horizon // self.model_horizon):
                new_dataset = self.make_step(
                    step, intrinsic_horizon, new_dataset, inverse_transform
                )

        # Get dataframe with predictions only
        pred_df = self._make_preds_df(new_dataset, intrinsic_horizon, self.history)
        return pred_df

    def _aggregate_shap(self, aggregate_by_folds: bool) -> None:
        """Calls aggregate_feature_importance on the single trainer.

        Args:
            aggregate_by_folds: passed through to trainer.aggregate_feature_importance.

        """
        trainer = self.trainers[0]
        trainer.aggregate_feature_importance(trainer.feature_name, aggregate_by_folds)

    def _plot_shap_boxplots(self, top_k: int) -> None:
        """Per-fold SHAP boxplots laid out in a grid (max 3 columns).

        Args:
            top_k: number of top features to show per subplot.

        """
        trainer = self.trainers[0]
        keys = [
            k
            for k in trainer.shap_values["train"].keys()
            if k not in ("feature_name", "feature_importance_aggregated")
        ]

        n_folds = len(keys)
        n_cols = min(3, n_folds)
        n_rows = int(np.ceil(n_folds / n_cols))

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(6 * n_cols, 6 * n_rows * top_k / 8),
            squeeze=False,
        )

        for fold_idx, fold_key in enumerate(keys):
            row, col = divmod(fold_idx, n_cols)
            ax = axes[row, col]

            data = trainer.shap_values["train"][fold_key]
            mean_imps = data.mean(axis=0)
            top_idx = np.argsort(mean_imps)[-top_k:]

            bp = ax.boxplot(data[:, top_idx], orientation="horizontal", patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")

            ax.set_title(f"Fold {fold_idx + 1}", fontsize=12, fontweight="bold")
            ax.set_yticklabels(trainer.shap_values["train"]["feature_name"][top_idx])
            ax.set_xlabel("SHAP Value")
            ax.grid(True)

        # Hide unused subplots
        for idx in range(n_folds, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        plt.suptitle("SHAP Feature Importance per Fold", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def _plot_shap_barh(self, top_k: int, round_to: int) -> None:
        """Aggregated SHAP horizontal bar chart for the single trainer.

        Args:
            top_k: number of top features to show.
            round_to: decimal places for rounding (0 = integers).

        """
        trainer = self.trainers[0]
        top_idx = np.argsort(trainer.shap_values["train"]["feature_importance_aggregated"])[-top_k:]
        sorted_imps = trainer.shap_values["train"]["feature_importance_aggregated"][top_idx]
        sorted_features = trainer.shap_values["train"]["feature_name"][top_idx]

        sorted_imps = sorted_imps.astype(int) if round_to == 0 else np.round(sorted_imps, round_to)

        bar_container = plt.barh(width=sorted_imps, y=sorted_features)
        plt.bar_label(bar_container, sorted_imps, color="red")
        plt.gcf().set_size_inches(5, top_k / 6 + 1)
        sns.despine()
        plt.title("Aggregated shap feature importance")
        plt.show()

    def get_feature_importance(
        self,
        top_k: int = 15,
        aggregate_by_folds: bool = True,
        round_to: int = 2,
        return_explainer: bool = False,
    ) -> Optional[Any]:
        """Generates and visualizes feature importance based on SHAP values.

        Args:
            top_k: number of top features to display.
            aggregate_by_folds:
                True  — one aggregated bar chart.
                False — per-fold boxplots.
            round_to: decimal places for rounding (0 = integers).
            return_explainer: if True, returns the shap_explainer object.

        Returns:
            shap_explainer when return_explainer=True, otherwise None.

        """
        self._aggregate_shap(aggregate_by_folds)

        if aggregate_by_folds:
            self._plot_shap_barh(top_k, round_to)
        else:
            self._plot_shap_boxplots(top_k)

        if return_explainer:
            return self.trainers[0].shap_explainer

    def get_train_shap(self) -> dict:
        """Returns training SHAP values for the single trainer.

        Returns:
            Dict with fold SHAP values, aggregated importance, and feature names.

        """
        return self.trainers[0].shap_values["train"]

    def get_test_shap(self) -> dict:
        """Returns test SHAP values for the single trainer.

        Returns:
            Dict with test SHAP values per feature.

        """
        return self.trainers[0].shap_values["test"]
