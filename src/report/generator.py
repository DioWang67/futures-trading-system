"""
HTML 報告產生器

輸出包含權益曲線圖、績效指標、每筆交易紀錄的互動式報告。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.monte_carlo import MonteCarloResult
from src.backtest.walk_forward import WalkForwardResult

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0f0f23; color: #e0e0e0; padding: 20px; }
        h1 { color: #00d4ff; margin-bottom: 20px; text-align: center; }
        h2 { color: #ffa500; margin: 20px 0 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card {
            background: #1a1a3e; border-radius: 10px; padding: 15px; text-align: center;
            border: 1px solid #333;
        }
        .metric-card .value { font-size: 28px; font-weight: bold; color: #00d4ff; }
        .metric-card .label { font-size: 13px; color: #999; margin-top: 5px; }
        .pass { color: #00ff88; }
        .fail { color: #ff4444; }
        .chart { margin: 20px 0; background: #1a1a3e; border-radius: 10px; padding: 15px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th { background: #1a1a3e; color: #ffa500; padding: 10px; text-align: left; }
        td { padding: 8px 10px; border-bottom: 1px solid #222; }
        tr:hover { background: #1a1a3e; }
        .positive { color: #00ff88; }
        .negative { color: #ff4444; }
        .wf-segment { background: #1a1a3e; border-radius: 10px; padding: 15px; margin: 10px 0; border: 1px solid #333; }
        .timestamp { text-align: center; color: #666; margin-top: 30px; font-size: 12px; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    {{ content }}
    <div class="timestamp">Generated: {{ timestamp }}</div>
</body>
</html>"""


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_float(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def _equity_curve_chart(trade_log: list[dict], initial_equity: float = 1_000_000) -> str:
    """產生權益曲線 Plotly JSON。"""
    if not trade_log:
        return ""

    equity = [initial_equity]
    for t in trade_log:
        pnl = t.get("pnl", 0)
        equity.append(equity[-1] + pnl)

    chart_data = json.dumps([{
        "y": equity,
        "type": "scatter",
        "mode": "lines",
        "name": "Equity",
        "line": {"color": "#00d4ff", "width": 2},
    }])

    chart_layout = json.dumps({
        "title": {"text": "權益曲線", "font": {"color": "#e0e0e0"}},
        "paper_bgcolor": "#1a1a3e",
        "plot_bgcolor": "#0f0f23",
        "xaxis": {"title": "Trade #", "color": "#999", "gridcolor": "#333"},
        "yaxis": {"title": "Equity (NTD)", "color": "#999", "gridcolor": "#333"},
        "font": {"color": "#e0e0e0"},
    })

    div_id = f"equity_{id(trade_log)}"
    return f"""
    <div class="chart">
        <div id="{div_id}"></div>
        <script>
            Plotly.newPlot('{div_id}', {chart_data}, {chart_layout}, {{responsive: true}});
        </script>
    </div>
    """


def _metrics_cards(result: BacktestResult) -> str:
    """產生績效指標卡片 HTML。"""
    cards = [
        ("總交易數", str(result.total_trades), ""),
        ("勝率", _format_pct(result.win_rate), "pass" if result.win_rate >= 0.55 else "fail"),
        ("獲利因子", _format_float(result.profit_factor), "pass" if result.profit_factor >= 1.5 else "fail"),
        ("夏普比率", _format_float(result.sharpe_ratio), "pass" if result.sharpe_ratio >= 1.2 else "fail"),
        ("最大回撤", _format_pct(result.max_drawdown), "pass" if result.max_drawdown <= 0.15 else "fail"),
        ("平均 R:R", _format_float(result.avg_rr), "pass" if result.avg_rr >= 1.5 else "fail"),
        ("總損益", f"${result.total_pnl:,.0f}", "positive" if result.total_pnl > 0 else "negative"),
    ]

    html = '<div class="metrics">'
    for label, value, css_class in cards:
        html += f"""
        <div class="metric-card">
            <div class="value {css_class}">{value}</div>
            <div class="label">{label}</div>
        </div>
        """
    html += "</div>"
    return html


def _trade_table(trade_log: list[dict]) -> str:
    """產生交易紀錄表格。"""
    if not trade_log:
        return "<p>無交易紀錄</p>"

    html = """
    <table>
        <thead>
            <tr>
                <th>#</th><th>方向</th><th>進場價</th><th>出場價</th>
                <th>停損</th><th>停利</th><th>損益</th><th>R:R</th><th>出場原因</th>
            </tr>
        </thead>
        <tbody>
    """

    for i, t in enumerate(trade_log, 1):
        pnl = t.get("pnl", 0)
        pnl_class = "positive" if pnl > 0 else "negative"
        html += f"""
        <tr>
            <td>{i}</td>
            <td>{t.get('direction', '')}</td>
            <td>{t.get('entry_price', 0):.1f}</td>
            <td>{t.get('exit_price', 0):.1f}</td>
            <td>{t.get('sl', 0):.1f}</td>
            <td>{t.get('tp', 0):.1f}</td>
            <td class="{pnl_class}">{pnl:.1f}</td>
            <td>{t.get('rr', 0):.2f}</td>
            <td>{t.get('exit_reason', '')}</td>
        </tr>
        """

    html += "</tbody></table>"
    return html


def generate_backtest_report(
    result: BacktestResult,
    title: str = "SMC+PA 回測報告",
    output_path: Optional[str] = None,
) -> str:
    """產生單次回測的 HTML 報告。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    content = ""
    content += "<h2>績效指標</h2>"
    content += _metrics_cards(result)
    content += "<h2>權益曲線</h2>"
    content += _equity_curve_chart(result.trade_log)
    content += "<h2>交易紀錄</h2>"
    content += _trade_table(result.trade_log)

    html = HTML_TEMPLATE.replace("{{ title }}", title)
    html = html.replace("{{ content }}", content)
    html = html.replace("{{ timestamp }}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(REPORTS_DIR / f"backtest_{timestamp}.html")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"報告已輸出: {output_path}")
    return output_path


def _monte_carlo_charts(mc_results: list[MonteCarloResult]) -> str:
    """產生 Monte Carlo 模擬圖表與統計 HTML。"""
    if not mc_results:
        return ""

    html = ""
    for mc in mc_results:
        # Equity distribution histogram
        hist_data = json.dumps([{
            "x": mc.all_final_equities,
            "type": "histogram",
            "nbinsx": 50,
            "marker": {"color": "#00d4ff", "line": {"color": "#0088aa", "width": 1}},
            "name": "Final Equity",
        }])
        hist_layout = json.dumps({
            "title": {"text": f"最終權益分布 ({mc.method})", "font": {"color": "#e0e0e0"}},
            "paper_bgcolor": "#1a1a3e",
            "plot_bgcolor": "#0f0f23",
            "xaxis": {"title": "Final Equity ($)", "color": "#999", "gridcolor": "#333"},
            "yaxis": {"title": "Count", "color": "#999", "gridcolor": "#333"},
            "font": {"color": "#e0e0e0"},
            "shapes": [{
                "type": "line", "x0": mc.final_equity_median, "x1": mc.final_equity_median,
                "y0": 0, "y1": 1, "yref": "paper",
                "line": {"color": "#ffa500", "width": 2, "dash": "dash"},
            }],
        })
        div_id_hist = f"mc_hist_{mc.method}"

        # MDD distribution histogram
        mdd_data = json.dumps([{
            "x": [d * 100 for d in mc.all_max_drawdowns],
            "type": "histogram",
            "nbinsx": 50,
            "marker": {"color": "#ff4444", "line": {"color": "#aa0000", "width": 1}},
            "name": "Max Drawdown",
        }])
        mdd_layout = json.dumps({
            "title": {"text": f"最大回撤分布 ({mc.method})", "font": {"color": "#e0e0e0"}},
            "paper_bgcolor": "#1a1a3e",
            "plot_bgcolor": "#0f0f23",
            "xaxis": {"title": "Max Drawdown (%)", "color": "#999", "gridcolor": "#333"},
            "yaxis": {"title": "Count", "color": "#999", "gridcolor": "#333"},
            "font": {"color": "#e0e0e0"},
        })
        div_id_mdd = f"mc_mdd_{mc.method}"

        # Equity curves (first 100)
        curves_traces = []
        for i, curve in enumerate(mc.equity_curves[:50]):
            curves_traces.append({
                "y": curve,
                "type": "scatter",
                "mode": "lines",
                "line": {"width": 0.5, "color": "rgba(0,212,255,0.15)"},
                "showlegend": False,
                "hoverinfo": "skip",
            })
        # Add median curve highlight (use the curve closest to median final equity)
        if mc.equity_curves:
            median_final = mc.final_equity_median
            closest_idx = min(
                range(len(mc.equity_curves)),
                key=lambda i: abs(mc.equity_curves[i][-1] - median_final),
            )
            curves_traces.append({
                "y": mc.equity_curves[closest_idx],
                "type": "scatter",
                "mode": "lines",
                "line": {"width": 2, "color": "#ffa500"},
                "name": "Median",
            })

        curves_data = json.dumps(curves_traces)
        curves_layout = json.dumps({
            "title": {"text": f"模擬權益曲線 ({mc.method})", "font": {"color": "#e0e0e0"}},
            "paper_bgcolor": "#1a1a3e",
            "plot_bgcolor": "#0f0f23",
            "xaxis": {"title": "Trade #", "color": "#999", "gridcolor": "#333"},
            "yaxis": {"title": "Equity ($)", "color": "#999", "gridcolor": "#333"},
            "font": {"color": "#e0e0e0"},
            "showlegend": True,
        })
        div_id_curves = f"mc_curves_{mc.method}"

        # Statistics cards
        ruin_class = "pass" if mc.ruin_probability < 0.05 else "fail"
        profit_class = "pass" if mc.profit_probability >= 0.6 else "fail"
        mdd95_class = "pass" if mc.max_dd_95th <= 0.25 else "fail"

        html += f"""
        <h3>Monte Carlo: {mc.method} ({mc.n_simulations:,} simulations)</h3>
        <div class="metrics">
            <div class="metric-card">
                <div class="value">${mc.final_equity_mean:,.0f}</div>
                <div class="label">平均最終權益</div>
            </div>
            <div class="metric-card">
                <div class="value">${mc.final_equity_median:,.0f}</div>
                <div class="label">中位數權益</div>
            </div>
            <div class="metric-card">
                <div class="value">${mc.final_equity_5th:,.0f}</div>
                <div class="label">5th 百分位</div>
            </div>
            <div class="metric-card">
                <div class="value">${mc.final_equity_95th:,.0f}</div>
                <div class="label">95th 百分位</div>
            </div>
            <div class="metric-card">
                <div class="value {mdd95_class}">{mc.max_dd_95th:.1%}</div>
                <div class="label">95th MDD</div>
            </div>
            <div class="metric-card">
                <div class="value">{mc.max_dd_worst:.1%}</div>
                <div class="label">最差 MDD</div>
            </div>
            <div class="metric-card">
                <div class="value {profit_class}">{mc.profit_probability:.1%}</div>
                <div class="label">獲利機率</div>
            </div>
            <div class="metric-card">
                <div class="value {ruin_class}">{mc.ruin_probability:.1%}</div>
                <div class="label">破產機率</div>
            </div>
        </div>
        <div class="chart">
            <div id="{div_id_curves}"></div>
            <script>Plotly.newPlot('{div_id_curves}', {curves_data}, {curves_layout}, {{responsive: true}});</script>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div class="chart">
                <div id="{div_id_hist}"></div>
                <script>Plotly.newPlot('{div_id_hist}', {hist_data}, {hist_layout}, {{responsive: true}});</script>
            </div>
            <div class="chart">
                <div id="{div_id_mdd}"></div>
                <script>Plotly.newPlot('{div_id_mdd}', {mdd_data}, {mdd_layout}, {{responsive: true}});</script>
            </div>
        </div>
        """

    return html


def generate_montecarlo_report(
    mc_results: list[MonteCarloResult],
    backtest_result: Optional[BacktestResult] = None,
    title: str = "SMC+PA Monte Carlo 模擬報告",
    output_path: Optional[str] = None,
) -> str:
    """產生 Monte Carlo 模擬的 HTML 報告。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    content = ""
    if backtest_result:
        content += "<h2>原始回測績效</h2>"
        content += _metrics_cards(backtest_result)
        content += "<h2>權益曲線</h2>"
        content += _equity_curve_chart(backtest_result.trade_log)

    content += "<h2>Monte Carlo 模擬</h2>"
    content += _monte_carlo_charts(mc_results)

    html = HTML_TEMPLATE.replace("{{ title }}", title)
    html = html.replace("{{ content }}", content)
    html = html.replace("{{ timestamp }}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(REPORTS_DIR / f"montecarlo_{timestamp}.html")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Monte Carlo 報告已輸出: {output_path}")
    return output_path


def generate_walkforward_report(
    wf_result: WalkForwardResult,
    title: str = "SMC+PA Walk-Forward 驗證報告",
    output_path: Optional[str] = None,
) -> str:
    """產生 Walk-Forward 驗證的 HTML 報告。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    status = "PASS" if wf_result.all_passed else "FAIL"
    status_class = "pass" if wf_result.all_passed else "fail"

    content = f"""
    <h2>Walk-Forward 驗證結果</h2>
    <div class="metrics">
        <div class="metric-card">
            <div class="value {status_class}">{status}</div>
            <div class="label">整體結果</div>
        </div>
        <div class="metric-card">
            <div class="value">{_format_pct(wf_result.pass_rate)}</div>
            <div class="label">通過率</div>
        </div>
        <div class="metric-card">
            <div class="value">{len(wf_result.segments)}</div>
            <div class="label">測試區段數</div>
        </div>
    </div>
    """

    for seg in wf_result.segments:
        seg_status = "PASS" if seg.passed else "FAIL"
        seg_class = "pass" if seg.passed else "fail"

        content += f"""
        <div class="wf-segment">
            <h2>區段 {seg.segment_id}：<span class="{seg_class}">{seg_status}</span></h2>
            <p>訓練期：{seg.train_start} ~ {seg.train_end}</p>
            <p>測試期：{seg.test_start} ~ {seg.test_end}</p>
        """

        if seg.test_result:
            content += _metrics_cards(seg.test_result)
            content += _equity_curve_chart(seg.test_result.trade_log)
            content += _trade_table(seg.test_result.trade_log)

        if seg.best_params:
            content += "<h3>最佳參數</h3><table>"
            for k, v in seg.best_params.items():
                content += f"<tr><td>{k}</td><td>{v}</td></tr>"
            content += "</table>"

        content += "</div>"

    html = HTML_TEMPLATE.replace("{{ title }}", title)
    html = html.replace("{{ content }}", content)
    html = html.replace("{{ timestamp }}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(REPORTS_DIR / f"walkforward_{timestamp}.html")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"報告已輸出: {output_path}")
    return output_path
