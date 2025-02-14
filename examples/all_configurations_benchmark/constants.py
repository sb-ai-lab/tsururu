all_models_params = {
    "ILI": {
        "DLinear": {
            "moving_avg": 25,
        },
        "PatchTST": {
            "e_layers": 3,
            "n_heads": 4,
            "d_model": 16,
            "d_ff": 128,
            "dropout": 0.3,
            "head_dropout": 0,
            "patch_len": 24,
            "stride": 2,
        },
        "GPT4TS": {
            "d_model": 768,
            "patch_len": 24, 
            "stride": 2,
            "gpt_layers": 6,
            "freeze": 1,    
        },
        "TimesNet": {
            "e_layers": 2,
            "d_model": 768,
            "d_ff": 768,
            "top_k": 5,    
        },
        "TimeMixer": {
            "e_layers": 2,
            "down_sampling_layers": 3,
            "down_sampling_window": 2,
            "d_model": 16,
            "d_ff": 32,
            "down_sampling_method": "avg",
        },
        "CycleNet": {
            "model_type": 'linear',
            "cycle_len": 24,
        }
    },
    "ETTh1": {
        "DLinear": {
            "moving_avg": 25,
        },
        "PatchTST": {
            "e_layers": 3,
            "n_heads": 4,
            "d_model": 16,
            "d_ff": 128,
            "dropout": 0.3,
            "head_dropout": 0,
            "patch_len": 16,
            "stride": 8,
        },
        "GPT4TS": {
            "d_model": 768,
            "patch_len": 16, 
            "stride": 8,
            "gpt_layers": 6,
            "freeze": 1,    
        },
        "TimesNet": {
            "e_layers": 2,
            "d_model": 16,
            "d_ff": 32,
            "top_k": 5,    
        },
        "TimeMixer": {
            "e_layers": 2,
            "down_sampling_layers": 3,
            "down_sampling_window": 2,
            "d_model": 16,
            "d_ff": 32,
            "down_sampling_method": "avg",
        },
        "CycleNet": {
            "model_type": 'linear',
            "cycle_len": 24,
        }
    },
}
