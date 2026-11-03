from __future__ import annotations

import pytest

from gaitlab.config import FeaturesConfig, PipelineConfig, load_features_config


@pytest.fixture
def cfg() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture
def features_cfg() -> FeaturesConfig:
    return load_features_config("configs/features.yaml")


@pytest.fixture
def synthetic_features_cfg(features_cfg: FeaturesConfig) -> FeaturesConfig:
    from gaitlab.data.synthetic import synthetic_features_config

    return synthetic_features_config(features_cfg)
