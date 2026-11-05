from gaitlab.config import config_hash, load_features_config, load_pipeline_config


def test_default_yaml_matches_model_defaults():
    from_yaml = load_pipeline_config("configs/default.yaml")
    assert from_yaml.windows.strides == 20
    assert from_yaml.clean.lowpass_hz == 12.0
    assert from_yaml.clean.feature_lowpass_hz == 20.0


def test_thresholds_ship_uncalibrated():
    fc = load_features_config("configs/features.yaml")
    assert set(fc.uncalibrated()) == set(fc.tags), "thresholds must stay null until ADR-0009"


def test_config_hash_is_stable_and_sensitive():
    a = load_pipeline_config("configs/default.yaml")
    b = load_pipeline_config("configs/default.yaml")
    assert config_hash(a) == config_hash(b)
    b.windows.strides = 21
    assert config_hash(a) != config_hash(b)


def test_train_lora_yaml_loads_into_the_typed_model():
    from gaitlab.config import load_train_lora_config

    c = load_train_lora_config("configs/train_lora.yaml")
    assert set(c.models) == {"1b", "270m"}
    assert c.models["270m"] == "google/gemma-3-270m-it"
    assert c.grade.grade_tokens == [" 1", " 2", " 3", " 4", " 5"]
    assert c.train.assistant_only_loss is True
    assert c.consistency.runs == 5 and c.consistency.temperature == 0.7
