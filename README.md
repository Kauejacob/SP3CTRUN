SP3CTRUM — Quant AI Challenge (Itaú Asset)

A reproducible, research-grade pipeline for building and testing a systematic Brazilian equities strategy that blends event-driven, positioning, and LLM multi-agent signals.

We combine three engines:

-Dividend run-up (pre ex-date),

-Low short-interest long-only selection,

-LLM “investment committee” (Agno + ChatGPT) to synthesize fundamental/technical views.

-Backtests (with explicit costs) show positive alpha vs. CDI with lower volatility/drawdown than any single leg, under guardrails against look-ahead and overfitting.

Strategy Overview

-Event (Dividend Run-Up): Long ~ few trading days before ex-date; exit on/after ex-date.

-Positioning (Short Interest): Prefer names with lower short interest as a proxy for cleaner upside pressure.

-LLM Multi-Agent IC: Multiple agents (research, risk, skepticism, synthesis) debate and produce a final conviction score that adjusts weights rather than overrides rules.

-Portfolio: Equal-weighted across eligible names with risk caps; monthly rebalance; turnover controls.

Data & Universe

-Universe: Highly liquid B3 names (top-liquidity list).

-Inputs: Prices, corporate actions (dividends/JCP), short-interest series, basic fundamentals.

-Costs/Slippage: Applied per-trade cost + conservative slippage.

-Calendars: B3 trading calendar; ex-date alignment checked.

-No proprietary data is shipped. Scripts expect environment variables/API keys where needed.

Methodology (Reproducibility First)

-No look-ahead: Signals built only from information available at the decision time.

-Cross-validation: Rolling/expanding windows to assess stability; OOS splitting.

-Risk controls: Max weight per asset, sector caps, stop-reallocation when liquidity < threshold.

-Evaluation: CAGR, Sharpe, Sortino, Max DD, Calmar, Hit-rate, Turnover, TE vs. CDI/IBOV.

Quickstart
 1) Create env:
    python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
    pip install -r requirements.txt
 2) Set keys (if needed):
    e.g., export OPENAI_API_KEY=...

 3) Run backtest:
    python -m sp3ctrum.backtest.run --start YYYY-MM-DD --end YYYY-MM-DD --universe b3_top_liquidity

 4) Generate report:
    python -m sp3ctrum.backtest.report --last-run

Results (Summary)

-Outperformed CDI with lower volatility and drawdown than individual legs.

-Diversification across engines reduced regime sensitivity.

-Robust to reasonable perturbations (window sizes, costs).

-Exact figures depend on your run, data vendor, and cost assumptions.

Limitations & Next Steps

-Short-interest latency/coverage can vary; validate your source.

-LLM agent scoring is adjunct (not pure AI-driven PM); we keep rule-based transparency.

-Roadmap: sector-aware caps, Bayesian blending of engines, execution-aware sizing, live paper-trade.
