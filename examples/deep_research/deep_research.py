"""Deep research example with live web + market-data tools.

This example gives an RLMFlow live search / fetch / stock tools and asks it
to build a grounded research report. The graph should show the useful RLM
pattern:

1. split a broad question into research focuses,
2. delegate each focus to a sub-agent,
3. verify weak/contradictory claims,
4. synthesize a cited final report.

Tools provided to the agent:
- ``web_search``           — DuckDuckGo HTML search (free, no API key).
- ``fetch_url``            — clean text scrape of any URL.
- ``stock_quote``          — current price + key multiples (yfinance).
- ``stock_history``        — historical OHLCV + computed vol/return (yfinance).
- ``stock_fundamentals``   — income / balance / cashflow summary (yfinance).

Optional dependency: ``pip install yfinance`` to enable the market-data tools.
If it's not installed, the script still runs — only the stock tools are skipped.

Usage:
    python examples/deep_research/deep_research.py
    python examples/deep_research/deep_research.py --query "Will NVDA hit $5T market cap in 2026?"
    python examples/deep_research/deep_research.py --model gpt-4o --fast-model gpt-4o-mini
    python examples/deep_research/deep_research.py --workspace examples/runs/deep_research/sdsk
"""

from __future__ import annotations

import argparse
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from rlmflow import RLMConfig, RLMFlow, Workspace
from rlmflow.llm import AnthropicClient, OpenAIClient
from rlmflow.prompts import DEFAULT_BUILDER
from rlmflow.prompts.default import ROLE_TEXT
from rlmflow.runtime.local import LocalRuntime
from rlmflow.tools import FILE_TOOLS, tool

try:
    import yfinance as _yf
except ImportError:
    _yf = None


DEFAULT_QUERY = """\
Research whether Nvidia (NVDA) stock is likely to reach $300 per share by the
end of 2026.

Build a deep research report with:
- current price, recent price action, and key valuation multiples
- business fundamentals and growth drivers
- competitive landscape and moat
- macro / sector tailwinds and headwinds
- analyst price targets and consensus
- strongest bull case (path to $300)
- strongest bear case (why it doesn't get there)
- open questions and what to watch

For every major claim, include citations or source snippets. Separate evidence
from speculation, and call out contradictions between sources. This is research,
not investment advice. Write a markdown report with all your findings, save in output/deep_research.md
"""


RULES_PATH = Path(__file__).with_name("program.md")


def _request_text(url: str, *, headers: dict[str, str] | None = None) -> str:
    request_headers = {"User-Agent": "Mozilla/5.0 rlmflow-deep-research/0.1"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read(2_000_000)
    return raw.decode("utf-8", errors="replace")


def _clean_html(text: str, *, max_chars: int) -> str:
    text = re.sub(r"(?is)<script.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?is)<(br|p|li|h[1-6]|tr|div)\b[^>]*>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()[:max_chars]


def _duckduckgo_search(query: str, max_results: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({"q": query})
    text = _request_text(f"https://duckduckgo.com/html/?{params}")
    results: list[dict[str, str]] = []
    blocks = re.findall(r'(?is)<div class="result[^"]*">(.*?)</div>\s*</div>', text)
    for block in blocks:
        title_match = re.search(r'(?is)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block)
        if not title_match:
            continue
        href = html.unescape(title_match.group(1))
        parsed = urllib.parse.urlparse(href)
        query_params = urllib.parse.parse_qs(parsed.query)
        url = query_params.get("uddg", [href])[0]
        title = _clean_html(title_match.group(2), max_chars=300)
        snippet_match = re.search(r'(?is)<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', block)
        snippet = _clean_html(snippet_match.group(1), max_chars=500) if snippet_match else ""
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


@tool(
    "Search the web. Returns a JSON object: "
    '{"provider": str, "query": str, "results": [{"title", "url", "snippet"}]}. '
    "Use the 'results' field to iterate."
)
def web_search(query: str, max_results: int = 5) -> str:
    max_results = max(1, min(int(max_results), 10))
    try:
        results = _duckduckgo_search(query, max_results)
    except (urllib.error.URLError, TimeoutError) as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "results": []})
    return json.dumps(
        {"provider": "duckduckgo_html", "query": query, "results": results},
        indent=2,
    )


@tool("Fetch a URL and return cleaned page text with the source URL.")
def fetch_url(url: str, max_chars: int = 12000) -> str:
    max_chars = max(1000, min(int(max_chars), 30000))
    try:
        text = _request_text(url)
    except (urllib.error.URLError, TimeoutError) as exc:
        return json.dumps({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {
            "url": url,
            "text": _clean_html(text, max_chars=max_chars),
        },
        indent=2,
    )


# ── stock data tools (yfinance) ──────────────────────────────────────

_QUOTE_FIELDS = (
    "shortName",
    "longName",
    "exchange",
    "currency",
    "country",
    "sector",
    "industry",
    "currentPrice",
    "previousClose",
    "marketCap",
    "enterpriseValue",
    "sharesOutstanding",
    "floatShares",
    "trailingPE",
    "forwardPE",
    "priceToSalesTrailing12Months",
    "priceToBook",
    "trailingEps",
    "forwardEps",
    "dividendYield",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "averageVolume",
    "beta",
    "longBusinessSummary",
)


def _require_yfinance() -> dict | None:
    if _yf is None:
        return {
            "error": "yfinance is not installed. Run `pip install yfinance` "
            "to enable stock_quote / stock_history / stock_fundamentals."
        }
    return None


@tool(
    "Fetch a stock quote with current price + key valuation multiples + business summary. "
    "Returns JSON. Use this BEFORE web_search for anything price/multiples-related — "
    "it's faster and authoritative. Returns {'error': ...} if the ticker is unknown."
)
def stock_quote(ticker: str) -> str:
    err = _require_yfinance()
    if err is not None:
        return json.dumps(err)
    ticker = ticker.strip().upper()
    try:
        info = _yf.Ticker(ticker).info or {}
    except Exception as exc:
        return json.dumps({"ticker": ticker, "error": f"{type(exc).__name__}: {exc}"})
    if not info or info.get("currentPrice") is None and info.get("regularMarketPrice") is None:
        return json.dumps({"ticker": ticker, "error": "ticker not found on Yahoo Finance"})
    summary = info.get("longBusinessSummary") or ""
    payload = {k: info.get(k) for k in _QUOTE_FIELDS if k in info and info.get(k) is not None}
    if summary:
        payload["longBusinessSummary"] = summary[:1500]
    payload["ticker"] = ticker
    payload["source"] = f"https://finance.yahoo.com/quote/{ticker}"
    return json.dumps(payload, indent=2, default=str)


@tool(
    "Fetch historical OHLCV for a ticker plus computed annualized return + volatility. "
    "period: 1d/5d/1mo/3mo/6mo/1y/2y/5y/10y/ytd/max. "
    "interval: 1d/1wk/1mo. Returns JSON with {'samples': [{'date','close'},...], 'metrics': {...}}. "
    "Capped to 60 evenly-spaced samples to stay readable; full history is summarized in 'metrics'."
)
def stock_history(ticker: str, period: str = "3y", interval: str = "1d") -> str:
    err = _require_yfinance()
    if err is not None:
        return json.dumps(err)
    ticker = ticker.strip().upper()
    try:
        df = _yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    except Exception as exc:
        return json.dumps({"ticker": ticker, "error": f"{type(exc).__name__}: {exc}"})
    if df is None or df.empty:
        return json.dumps({"ticker": ticker, "error": "no history available"})

    closes = df["Close"].astype(float)
    returns = closes.pct_change().dropna()
    if len(returns) > 1:
        ann_factor = {"1d": 252, "1wk": 52, "1mo": 12}.get(interval, 252)
        ann_return = float((1 + returns.mean()) ** ann_factor - 1)
        ann_vol = float(returns.std() * (ann_factor**0.5))
        total_return = float(closes.iloc[-1] / closes.iloc[0] - 1)
    else:
        ann_return = ann_vol = total_return = None

    step = max(1, len(df) // 60)
    sampled = df.iloc[::step]
    samples = [
        {"date": str(idx.date()), "close": round(float(row["Close"]), 4)}
        for idx, row in sampled.iterrows()
    ]
    return json.dumps(
        {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "first_date": str(df.index[0].date()),
            "last_date": str(df.index[-1].date()),
            "first_close": round(float(closes.iloc[0]), 4),
            "last_close": round(float(closes.iloc[-1]), 4),
            "metrics": {
                "total_return": total_return,
                "annualized_return": ann_return,
                "annualized_volatility": ann_vol,
                "n_observations": int(len(df)),
            },
            "samples": samples,
            "source": f"https://finance.yahoo.com/quote/{ticker}/history",
        },
        indent=2,
    )


@tool(
    "Fetch summarized financial statements for a ticker: income / balance / cashflow, last 4 periods. "
    "Returns JSON. Numbers are in the issuer's reporting currency (see stock_quote for the currency)."
)
def stock_fundamentals(ticker: str) -> str:
    err = _require_yfinance()
    if err is not None:
        return json.dumps(err)
    ticker = ticker.strip().upper()
    try:
        t = _yf.Ticker(ticker)
        income = t.financials
        balance = t.balance_sheet
        cash = t.cashflow
    except Exception as exc:
        return json.dumps({"ticker": ticker, "error": f"{type(exc).__name__}: {exc}"})

    def _frame_to_json(frame, max_rows: int = 12) -> dict:
        if frame is None or frame.empty:
            return {}
        cols = [str(c.date()) if hasattr(c, "date") else str(c) for c in frame.columns]
        rows = {}
        for label, series in frame.iterrows():
            rows[str(label)] = [
                round(float(v), 2) if v is not None and not _isnan(v) else None
                for v in series.tolist()
            ]
            if len(rows) >= max_rows:
                break
        return {"periods": cols, "rows": rows}

    return json.dumps(
        {
            "ticker": ticker,
            "income_statement": _frame_to_json(income),
            "balance_sheet": _frame_to_json(balance),
            "cash_flow": _frame_to_json(cash),
            "source": f"https://finance.yahoo.com/quote/{ticker}/financials",
        },
        indent=2,
        default=str,
    )


def _isnan(value) -> bool:
    try:
        return value != value
    except Exception:
        return False


def build_prompt_builder(rules: str):
    return (
        DEFAULT_BUILDER.section(
            "role",
            "You are a recursive deep research analyst. You search, read sources, "
            "split independent research angles into child calls, verify weak claims, "
            f"and produce grounded reports.\n\n{ROLE_TEXT}",
            title="Role",
        )
        .section(
            "deep_research",
            rules,
            title="Deep Research Rules",
            after="strategy",
        )
    )


def make_llm(model: str):
    return AnthropicClient(model) if model.startswith("claude") else OpenAIClient(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a recursive deep research example.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--rules",
        type=Path,
        default=RULES_PATH,
        help="Path to the deep-research rules markdown file.",
    )
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fast-model", default="gpt-5-mini")
    parser.add_argument("--workspace", type=Path, default=Path("./deep-research-runs"))
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=18)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Max sibling agents stepped in parallel (1 = sequential).",
    )
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    rules = args.rules.read_text(encoding="utf-8")

    workspace = Workspace.create(args.workspace)
    runtime = LocalRuntime(workspace=workspace)
    tools = [*FILE_TOOLS, web_search, fetch_url]
    if _yf is not None:
        tools += [stock_quote, stock_history, stock_fundamentals]
    else:
        print(
            "[warn] yfinance not installed — stock_quote / stock_history / "
            "stock_fundamentals tools disabled. Install with `pip install yfinance`."
        )
    runtime.register_tools(tools)

    llm_clients = None
    if args.fast_model:
        llm_clients = {
            "fast": {
                "model": make_llm(args.fast_model),
                "description": "Cheaper/faster model for scoped research subtasks.",
            }
        }

    agent = RLMFlow(
        llm_client=make_llm(args.model),
        runtime=runtime,
        workspace=workspace,
        llm_clients=llm_clients,
        config=RLMConfig(
            max_depth=args.max_depth,
            max_iterations=args.max_iterations,
            max_concurrency=args.max_concurrency,
        ),
        prompt_builder=build_prompt_builder(rules),
    )

    graph = agent.start(args.query)
    while not graph.finished:
        graph = agent.step(graph)
        print(graph.tree())

    result = graph.result()
    print("\n" + "=" * 80)
    print(result or "(no result)")

    print(f"\nWorkspace saved to {workspace.root}")

    if not args.no_viewer:
        try:
            from rlmflow.utils.viewer import save_html

            save_html(workspace, args.workspace / "viewer.html")
            print(f"Viewer saved to {args.workspace / 'viewer.html'}")
        except ImportError as exc:
            print(f"Viewer not saved: {exc}")


if __name__ == "__main__":
    main()
