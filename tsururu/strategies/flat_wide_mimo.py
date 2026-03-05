from typing import Union, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tsururu.dataset.pipeline import Pipeline
from tsururu.model_training.trainer import DLTrainer, MLTrainer
from tsururu.strategies.mimo import MIMOStrategy


class FlatWideMIMOStrategy(MIMOStrategy):
    """A strategy that uses a single model for all points
        in the prediction horizon.

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
        1. Fit: mixture of DirectStrategy and MIMOStrategy, fit one
            model, but uses deployed over horizon DirectStrategy's features.
        2. Inference: similarly.

    """

    def __init__(
        self,
        horizon: int,
        history: int,
        trainer: Union[MLTrainer, DLTrainer],
        pipeline: Pipeline,
        step: int = 1,
    ):
        super().__init__(horizon, history, trainer, pipeline, step)
        self.strategy_name = "FlatWideMIMOStrategy"

    def get_feature_importance(
        self,
        top_k=15,
        aggregate_by_folds=True,
        round_to=2,
        return_explainer=False,
    ) -> dict | tuple[dict, Any] | None:
        """Generates and visualizes feature importance using SHAP values from trainer folds.

        Args:
            top_k (int, default=15): number of top features to display in plots.
            aggregate_by_folds (bool, default=True): if True, aggregates importance across folds into single bar plot.
                If False, creates individual boxplots for each fold.
            show_plots (bool, default=True): if False, skips plotting and returns explainer immediately.
            round_to (int, default=2): decimal places to round aggregated importance values (0 = integer).
            return_explainer (bool, default=False): if True, returns (importance_dict, explainer) tuple instead of just dict.

        Returns:
            dict: feature importance dictionary if `return_explainer=False` (default).
            tuple[dict, Any]: (importance_dict, shap_explainer) if `return_explainer=True`.
            None: shap_explainer if `show_plots=False`.
        """
        trainer = self.trainers[0]

        feature_name = trainer.feature_name

        trainer.aggregate_feature_importance(feature_name, aggregate_by_folds)

        keys = [
            k
            for k in trainer.shap_values["train"].keys()
            if k not in ("feature_name", "test")
        ]
        n = len(keys)

        if not aggregate_by_folds:
            _, axes = plt.subplots(n, 1, squeeze=False)
            for i, key in enumerate(keys):
                ax = axes[i, 0]

                data = trainer.shap_values["train"][key]
                mean_imps = data.mean(axis=0)
                top_idx = np.argsort(mean_imps)[-top_k:]
                sorted_imps = data[:, top_idx]
                sorted_features = trainer.shap_values["train"]["feature_name"][top_idx]

                bp = ax.boxplot(
                    sorted_imps, orientation="horizontal", patch_artist=True
                )
                for _, patch in enumerate(bp["boxes"]):
                    patch.set_facecolor("lightblue")

                ax.set_title(
                    f"Shap value features on Fold {i+1}", fontsize=14, fontweight="bold"
                )
                ax.set_yticklabels(sorted_features)
                plt.gcf().set_size_inches(12, 5 * top_k)
                ax.set_ylabel("shap_value")
                ax.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            top_idx = np.argsort(
                trainer.shap_values["train"]["feature_importance_aggregated"]
            )[-top_k:]
            sorted_imps = trainer.shap_values["train"]["feature_importance_aggregated"][
                top_idx
            ]
            sorted_features = trainer.shap_values["train"]["feature_name"][top_idx]

            if round_to == 0:
                sorted_imps = sorted_imps.astype(int)
            else:
                sorted_imps = np.round(sorted_imps, round_to)

            bar_conatiner = plt.barh(width=sorted_imps, y=sorted_features)
            plt.bar_label(bar_conatiner, sorted_imps, color="red")
            plt.gcf().set_size_inches(5, top_k / 6 + 1)
            sns.despine()
            plt.title(f"Aggregated shap feature importance")
            plt.show()

        if return_explainer:
            return trainer.shap_explainer

    def get_train_shap(self) -> dict:
        """Returns SHAP values computed on the training/validation folds.

        Contains feature-level SHAP values for each fold, aggregated importance, and feature names.

        Returns:
            dict: training SHAP values dictionary with keys like fold indices, 'feature_importance_aggregated', 
                'feature_name', containing numpy arrays of SHAP values and metadata.
        """
        return self.trainers[0].shap_values["train"]

    def get_test_shap(self) -> dict:
        """Returns SHAP values computed on the test folds.

        Contains feature-level SHAP values for test set predictions.

        Returns:
            dict: test SHAP values dictionary containing numpy arrays of SHAP values per feature.
        """
        return self.trainers[0].shap_values["test"]
