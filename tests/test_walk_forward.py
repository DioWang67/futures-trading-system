"""Walk-Forward 驗證模組單元測試。"""

import pytest
import pandas as pd

from src.backtest.walk_forward import WalkForwardValidator, WalkForwardResult
from src.data.fetcher import generate_sample_data


@pytest.fixture
def sample_data():
    return generate_sample_data(n_bars=600, freq="15T", seed=42)


@pytest.fixture
def config():
    return {
        "backtest": {"commission": 22, "slippage": 1, "size": 1, "point_value": 10},
        "optimization": {"n_trials": 3, "timeout": 30},
        "walk_forward": {"n_splits": 2, "train_ratio": 0.7},
    }


class TestWalkForwardValidator:
    def test_creation(self, sample_data, config):
        validator = WalkForwardValidator(sample_data, config=config)
        assert validator.n_splits == 2

    def test_split_data(self, sample_data, config):
        validator = WalkForwardValidator(sample_data, config=config)
        splits = validator._split_data(sample_data)
        assert len(splits) == 2
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0

    def test_validate_returns_result(self, sample_data, config):
        validator = WalkForwardValidator(sample_data, config=config, n_splits=2)
        result = validator.validate()
        assert isinstance(result, WalkForwardResult)
        assert len(result.segments) > 0

    def test_split_respects_lookback_min_lengths(self, sample_data, config):
        cfg = dict(config)
        cfg["strategy"] = {"swing_lookback": 120, "adx_period": 14, "ob_max_age": 20}
        cfg["walk_forward"] = {"n_splits": 2, "train_ratio": 0.7}
        validator = WalkForwardValidator(sample_data, config=cfg)
        splits = validator._split_data(sample_data)
        # 600 bars / 2 splits = 300 bars per segment; train/test = 210/90
        # lookback=120 => min_test_bars=120, so both segments should be skipped.
        assert len(splits) == 0

    def test_pass_rate(self):
        result = WalkForwardResult()
        assert result.pass_rate == 0.0


class TestWalkForwardResult:
    def test_empty_result(self):
        result = WalkForwardResult()
        assert result.all_passed is False
        assert result.pass_rate == 0.0
        assert len(result.segments) == 0
