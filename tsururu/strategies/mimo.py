from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tsururu.dataset.pipeline import Pipeline
from tsururu.model_training.trainer import DLTrainer, MLTrainer
from tsururu.strategies.recursive import RecursiveStrategy


class MIMOStrategy(RecursiveStrategy):
    """A strategy that uses one model that learns to predict
        the entire prediction horizon.

    Arguments:
        horizon: forecast horizon.
        history: number of previous for feature generating
            (i.e., features for observation y_t are counted from observations
            (y_{t-history}, ..., y_{t-1}).
        trainer: trainer with model params and validation params.
        pipeline: pipeline for feature and target generation.
        step:  in how many points to take the next observation while making
            samples' matrix.

    Notes:
        1. Technically, `MIMOStrategy` is a `RecursiveStrategy` or
            `DirectStrategy` for which the horizon of the individual model
            (`model_horizon`) coincides with the full prediction horizon
            (`horizon`).
        2. Fit: the model is fitted to predict a vector which length is equal
            to the length of the prediction horizon.
        3. Inference: the model makes a vector of predictions.

    """

    def __init__(
        self,
        horizon: int,
        history: int,
        trainer: Union[MLTrainer, DLTrainer],
        pipeline: Pipeline,
        step: int = 1,
    ):
        super().__init__(horizon, history, trainer, pipeline, step, model_horizon=horizon)
        self.strategy_name = "MIMOStrategy"

    def _plot_shap_boxplots(self, top_k: int) -> None:
        """Per-fold SHAP boxplots laid out in a grid (max 3 columns).
        Handles 3D SHAP arrays by averaging over the horizon axis.

        Args:
            top_k: number of top features to show per subplot.

        """
        trainer = self.trainers[0]
        keys = [
            k
            for k in trainer.shap_values["train"].keys()
            if k not in ("feature_name", "test", "feature_importance_aggregated")
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
            # data may be 3D (n_samples, n_features, horizon) — average by horizon
            if data.ndim == 3:
                mean_imps = data.mean(axis=(0, 2))
                plot_data = data[:, :, 0]
            else:
                mean_imps = data.mean(axis=0)
                plot_data = data

            top_idx = np.argsort(mean_imps)[-top_k:]

            bp = ax.boxplot(plot_data[:, top_idx], orientation="horizontal", patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")

            ax.set_title(f"Fold {fold_idx + 1}", fontsize=12, fontweight="bold")
            ax.set_yticklabels(
                np.array(trainer.shap_values["train"]["feature_name"])[top_idx].ravel()
            )
            ax.set_xlabel("SHAP Value")
            ax.grid(True)

        for idx in range(n_folds, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        plt.suptitle("SHAP Feature Importance per Fold", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def _plot_shap_barh(self, top_k: int, round_to: int) -> None:
        """Aggregated SHAP horizontal bar chart.
        Handles 2D feature_importance_aggregated (n_features, horizon) by averaging over horizon.

        Args:
            top_k: number of top features to show.
            round_to: decimal places for rounding (0 = integers).

        """
        trainer = self.trainers[0]
        agg_imp = trainer.shap_values["train"]["feature_importance_aggregated"]

        # agg_imp may be 2D (n_features, horizon) — average absolute value by horizon
        if np.ndim(agg_imp) == 2:
            sorted_vals = np.abs(agg_imp).mean(axis=1)
        else:
            sorted_vals = agg_imp

        top_idx = np.argsort(sorted_vals)[-top_k:]
        sorted_imps = sorted_vals[top_idx]
        sorted_features = np.array(trainer.shap_values["train"]["feature_name"])[top_idx].ravel()

        sorted_imps = sorted_imps.astype(int) if round_to == 0 else np.round(sorted_imps, round_to)

        bar_container = plt.barh(width=sorted_imps, y=sorted_features)
        plt.bar_label(bar_container, sorted_imps, color="red")
        plt.gcf().set_size_inches(5, top_k / 6 + 1)
        sns.despine()
        plt.title("Aggregated shap feature importance")
        plt.show()
