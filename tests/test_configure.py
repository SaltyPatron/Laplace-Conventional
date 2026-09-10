from laplace_conventional.configure import derive_training_config


def test_derived_shape_tracks_measured_tokens():
    report = {
        "vocab_size": 8192,
        "splits": {"train": {"tokens": 200_000_000}},
        "train_record_length_tokens": {"p95": 1300},
    }
    cfg = derive_training_config(report)
    assert cfg["derivation"]["target_parameters"] == 10_000_000
    assert cfg["training"]["context_length"] == 2048
    m = cfg["model"]
    assert m["parameter_estimate"] > 0
    assert m["hidden_size"] % m["num_attention_heads"] == 0
