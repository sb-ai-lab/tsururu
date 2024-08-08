import numpy as np

from ..dataset.slice import IndexSlicer

transformers_masks = {
    "raw": r"^",
    "LAG": r"lag_\d+__",
    "SEASON": r"season_\w+__",
    "TIMETONUM": r"time_to_num__",
    "LABEL": r"label_encoder__",
    "OHE": r"ohe_encoder_\S+__",
}

date_attrs = {
    "y": "year",
    "m": "month",
    "d": "day",
    "wd": "weekday",
    "doy": "dayofyear",
    "hour": "hour",
    "min": "minute",
    "sec": "second",
    "ms": "microsecond",
    "ns": "nanosecond",
}


def _seq_mult_ts(data, idx_data):
    index_slicer = IndexSlicer()
    data_seq = np.array([])

    for idx in range(len(data)):
        current_data_seq = index_slicer.get_slice(data[idx], (idx_data[idx], None))
        if data_seq.shape[0] == 0:
            data_seq = current_data_seq
        else:
            data_seq = np.hstack((data_seq, current_data_seq))
    return data_seq
