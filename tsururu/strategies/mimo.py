from typing import Union, Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
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
        super().__init__(
            horizon, history, trainer, pipeline, step, model_horizon=horizon
        )
        self.strategy_name = "MIMOStrategy"

    def get_feature_importance(
        self,
        top_k=15,
        aggregate_by_folds=True,
        round_to=2,
        return_explainer=False,
    ) -> Dict | Tuple[Dict, Any] | None:
        """Generates and visualizes feature importance based on SHAP values from training folds.

        Args:
            top_k (int, default=15): number of top features to display in the plots.
            aggregate_by_folds (bool, default=True):
                if True — aggregates importance across folds into a single bar plot.
                if False — creates separate boxplots for each fold.
            round_to (int, default=2): number of decimal places for rounding
                aggregated importance values (0 = integers).
            return_explainer (bool, default=False):
                if True, returns a tuple (importance_dict, shap_explainer) instead of just the dictionary.

        Returns:
            dict: dictionary with feature importance (default).
            tuple[dict, Any]: (importance_dict, list_of_shap_explainers) when return_explainer=True.
            list[Any]: list of shap_explainers when aggregate_by_folds=False and return_explainer=True.
        """
        arr_explainers = []
        arr_train_shap = []

        for trainer_idx, trainer in enumerate(self.trainers):
            feature_name = trainer.feature_name
            trainer.aggregate_feature_importance(feature_name, aggregate_by_folds)

            keys = [
                k for k in trainer.shap_values["train"].keys() if k != "feature_name"
            ]
            n = len(keys)

            arr_explainers.append(trainer.shap_explainer)

            if not aggregate_by_folds:
                _, axes = plt.subplots(n, 1, squeeze=False)
                for i, key in enumerate(keys):
                    ax = axes[i, 0]

                    data = trainer.shap_values["train"][key]
                    mean_imps = data.mean(axis=(0, 2))
                    top_idx = np.argsort(mean_imps)[-top_k:]
                    sorted_imps = data[:, top_idx, 0]
                    sorted_features = trainer.shap_values["train"]["feature_name"][
                        top_idx
                    ]

                    bp = ax.boxplot(
                        sorted_imps, orientation="horizontal", patch_artist=True
                    )
                    for _, patch in enumerate(bp["boxes"]):
                        patch.set_facecolor("lightblue")

                    ax.set_title(
                        f"Shap value features on Fold {i+1}",
                        fontsize=14,
                        fontweight="bold",
                    )
                    ax.set_yticklabels(sorted_features)
                    plt.gcf().set_size_inches(12, 5 * top_k)
                    ax.set_ylabel("shap_value")
                    ax.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                agg_imp = trainer.shap_values["train"]["feature_importance_aggregated"]
                # averaging by horizon
                mean_agg = np.abs(agg_imp).mean(axis=1)
                top_idx = np.argsort(mean_agg)[-top_k:]
                sorted_imps = mean_agg[top_idx]
                sorted_features = np.array(
                    trainer.shap_values["train"]["feature_name"]
                )[top_idx].ravel()

                if round_to == 0:
                    sorted_imps = sorted_imps.astype(int)
                else:
                    sorted_imps = np.round(sorted_imps, round_to)

                bar_conatiner = plt.barh(width=sorted_imps, y=sorted_features)
                plt.bar_label(bar_conatiner, sorted_imps, color="red")
                plt.gcf().set_size_inches(5, top_k / 6 + 1)
                sns.despine()
                plt.title(
                    f"Aggregated shap feature importance by trainer {trainer_idx+1}"
                )
                plt.show()

            train_shap = {
                "shap_values": trainer.shap_values["train"],
                "feature_names": trainer.shap_values["train"]["feature_name"],
            }
            arr_train_shap.append(train_shap)

        self.arr_train_shap = arr_train_shap
        if return_explainer:
            return arr_explainers

    def get_train_shap(self) -> dict:
        """Returns SHAP values computed on the training/validation folds.

        Contains feature-level SHAP values for each fold, aggregated importance, and feature names.

        Returns:
            dict: training SHAP values dictionary with keys like fold indices, 'feature_importance_aggregated',
                'feature_name', containing numpy arrays of SHAP values and metadata.
        """
        return self.arr_train_shap

    def get_test_shap(self) -> dict:
        """Returns SHAP values computed on the test folds.

        Contains feature-level SHAP values for test set predictions.

        Returns:
            dict: test SHAP values dictionary containing numpy arrays of SHAP values per feature.
        """
        arr_test_shap = []

        for trainer in self.trainers:
            arr_test_shap.append(trainer.shap_values["test"])

        return arr_test_shap
