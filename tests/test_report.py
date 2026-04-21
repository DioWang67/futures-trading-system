"""報告產生器單元測試。"""

import os
import tempfile
import pytest

from src.backtest.engine import BacktestResult
from src.backtest.walk_forward import WalkForwardResult, WalkForwardSegment
from src.report.generator import (
    generate_backtest_report,
    generate_walkforward_report,
    _metrics_cards,
    _trade_table,
    _equity_curve_chart,
)


@pytest.fixture
def sample_result():
    return BacktestResult(
        total_trades=20,
        winning_trades=12,
        losing_trades=8,
        win_rate=0.60,
        profit_factor=1.8,
        sharpe_ratio=1.4,
        max_drawdown=0.10,
        avg_rr=1.6,
        total_pnl=5000.0,
        trade_log=[
            {"direction": "LONG", "entry_price": 16000, "exit_price": 16050,
             "sl": 15980, "tp": 16070, "pnl": 50, "rr": 2.5, "exit_reason": "TP",
             "entry_bar": 10, "exit_bar": 15},
            {"direction": "SHORT", "entry_price": 16100, "exit_price": 16120,
             "sl": 16120, "tp": 16060, "pnl": -20, "rr": -1.0, "exit_reason": "SL",
             "entry_bar": 20, "exit_bar": 25},
        ],
    )


@pytest.fixture
def sample_wf_result(sample_result):
    seg = WalkForwardSegment(
        segment_id=1,
        train_start="2021-01-01",
        train_end="2021-06-30",
        test_start="2021-07-01",
        test_end="2021-12-31",
        test_result=sample_result,
        passed=True,
    )
    return WalkForwardResult(segments=[seg], all_passed=True)


class TestMetricsCards:
    def test_returns_html(self, sample_result):
        html = _metrics_cards(sample_result)
        assert "<div" in html
        assert "60.00%" in html

    def test_pass_fail_classes(self, sample_result):
        html = _metrics_cards(sample_result)
        assert "pass" in html


class TestTradeTable:
    def test_returns_html_with_rows(self, sample_result):
        html = _trade_table(sample_result.trade_log)
        assert "<table>" in html
        assert "LONG" in html
        assert "SHORT" in html

    def test_empty_log(self):
        html = _trade_table([])
        assert "無交易紀錄" in html


class TestEquityCurve:
    def test_returns_chart_html(self, sample_result):
        html = _equity_curve_chart(sample_result.trade_log)
        assert "Plotly" in html

    def test_empty_log(self):
        html = _equity_curve_chart([])
        assert html == ""


class TestGenerateBacktestReport:
    def test_creates_file(self, sample_result):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            result_path = generate_backtest_report(sample_result, output_path=path)
            assert os.path.exists(result_path)
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "SMC+PA" in content
            assert "60.00%" in content
        finally:
            os.unlink(path)


class TestGenerateWalkforwardReport:
    def test_creates_file(self, sample_wf_result):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            result_path = generate_walkforward_report(sample_wf_result, output_path=path)
            assert os.path.exists(result_path)
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Walk-Forward" in content
            assert "PASS" in content
        finally:
            os.unlink(path)
