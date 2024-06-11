from copy import deepcopy
from typing import Optional

from ..dataset import IndexSlicer, Pipeline, TSDataset
from ..models import Estimator
from .base import Strategy
from .utils import timing_decorator

index_slicer = IndexSlicer()


class StatStrategy(Strategy):
    """Strategy that uses a stat model to predict all points in the
        forecast horizon.

    Arguments:
        horizon: forecast horizon.
        model: base model.
        pipeline: pipeline for feature and target generation.

    """

    def __init__(
        self,
        horizon: int,
        model: Estimator,
        pipeline: Pipeline,
        history: Optional[int] = None,
    ):
        super().__init__(
            horizon=horizon,
            history=history,
            step=1,
            model=model,
            pipeline=pipeline,
        )
        self.strategy_name = "stat"

    @timing_decorator
    def fit(self, dataset: TSDataset) -> "StatStrategy":
        data = self.pipeline.create_data_dict_for_pipeline(dataset, None, None)
        data = self.pipeline.fit_transform(data, self.strategy_name)

        model = deepcopy(self.model)

        if isinstance(model, Estimator):
            model.fit(data, self.pipeline)

        self.models.append(model)
        return self

    @timing_decorator
    def predict(self, dataset):
        if self.history is None:
            unique_id = dataset.data[dataset.id_column].unique()
            self.history = dataset.data[dataset.data[dataset.id_column] == unique_id[0]].shape[0]
        new_data = dataset.make_padded_test(self.horizon, self.history)
        new_dataset = TSDataset(new_data, dataset.columns_params, dataset.delta)

        features_idx = index_slicer.create_idx_data(
            new_dataset.data,
            self.horizon,
            self.history,
            step=self.horizon,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )

        target_ids = index_slicer.create_idx_target(
            new_dataset.data,
            self.horizon,
            self.history,
            step=self.horizon,
            date_column=dataset.date_column,
            delta=dataset.delta,
        )

        data = self.pipeline.create_data_dict_for_pipeline(new_dataset, features_idx, target_ids)
        data = self.pipeline.transform(data)

        pred = self.models[0].predict(data, self.pipeline)
        pred = self.pipeline.inverse_transform_y(pred)

        new_dataset.data.loc[target_ids.reshape(-1), dataset.target_column] = pred.reshape(-1)

        # Get dataframe with predictions only
        pred_df = self._make_preds_df(new_dataset, self.horizon, self.history)
        return pred_df
