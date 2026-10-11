import os
import numpy as np
import pandas as pd


def _rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_market_context(df, symbol, capital, risk, horizon):
    close = df["Close"].astype(float)
    chart = df.copy()
    chart["SMA20"] = close.rolling(20).mean()
    chart["SMA50"] = close.rolling(50).mean()
    chart["RSI14"] = _rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    chart["MACD"] = ema12 - ema26
    chart["MACDSignal"] = chart["MACD"].ewm(span=9, adjust=False).mean()
    last = chart.iloc[-1]
    return {
        "symbol": symbol,
        "capital": int(capital),
        "risk": risk,
        "horizon": horizon,
        "last_price": float(last["Close"]),
        "sma20": float(last["SMA20"]) if pd.notna(last["SMA20"]) else float("nan"),
        "sma50": float(last["SMA50"]) if pd.notna(last["SMA50"]) else float("nan"),
        "rsi14": float(last["RSI14"]) if pd.notna(last["RSI14"]) else float("nan"),
        "macd": float(last["MACD"]) if pd.notna(last["MACD"]) else float("nan"),
        "macd_signal": float(last["MACDSignal"]) if pd.notna(last["MACDSignal"]) else float("nan"),
        "return_1m": float(close.pct_change(21).iloc[-1] * 100) if len(close) > 21 else float("nan"),
        "return_3m": float(close.pct_change(63).iloc[-1] * 100) if len(close) > 63 else float("nan"),
        "chart": chart,
    }


# ---------------------------------------------------------------
# 企業分析（Yahoo Finance の財務指標）
# ---------------------------------------------------------------
_FUND_KEYS = [
    "operatingMargins", "returnOnEquity", "revenueGrowth", "earningsGrowth",
    "freeCashflow", "trailingEps", "trailingPE", "priceToBook",
]


def fetch_fundamentals(symbol):
    """財務指標を取得。失敗しても例外は出さず、空の辞書を返す。"""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return {}
    out = {}
    for k in _FUND_KEYS:
        v = info.get(k)
        if isinstance(v, (int, float)) and np.isfinite(v):
            out[k] = float(v)
    return out


def _fund_agent(f):
    if not f:
        return {
            "name": "② 企業分析",
            "signal": "データ不足",
            "finding": "Yahoo Financeから財務指標を取得できませんでした。時間をおいて再実行するか、決算短信で売上・営業利益率・EPS・フリーキャッシュフローを確認してください。",
            "score": None,
        }

    # (表示名, 値の表示, 合格か[True/False/None=データなし], 基準の説明)
    def pct(v):
        return f"{v * 100:.1f}%"

    rows = []
    v = f.get("operatingMargins")
    rows.append(("営業利益率", pct(v) if v is not None else None, None if v is None else v >= 0.10, "10%以上で良好"))
    v = f.get("returnOnEquity")
    rows.append(("ROE", pct(v) if v is not None else None, None if v is None else v >= 0.08, "8%以上で良好"))
    v = f.get("revenueGrowth")
    rows.append(("売上成長率(前年同期比)", pct(v) if v is not None else None, None if v is None else v > 0, "プラスで良好"))
    v = f.get("earningsGrowth")
    rows.append(("利益成長率(前年同期比)", pct(v) if v is not None else None, None if v is None else v > 0, "プラスで良好"))
    v = f.get("freeCashflow")
    rows.append(("フリーCF", f"{v / 1e8:,.0f}億円" if v is not None else None, None if v is None else v > 0, "プラスで良好"))

    notes = []
    for name, shown, ok, _ in rows:
        if ok is None:
            notes.append(f"{name}：データなし")
        else:
            notes.append(f"{name} {shown}（{'○' if ok else '×'}）")

    ref = []
    if "trailingEps" in f:
        ref.append(f"EPS {f['trailingEps']:,.1f}円")
    if "trailingPE" in f:
        ref.append(f"PER {f['trailingPE']:.1f}倍")
    if "priceToBook" in f:
        ref.append(f"PBR {f['priceToBook']:.2f}倍")
    ref_text = "参考：" + "、".join(ref) + "（点数には含めていません）。" if ref else ""

    evaluable = [ok for _, _, ok, _ in rows if ok is not None]
    passed = sum(1 for ok in evaluable if ok)
    if len(evaluable) < 3:
        signal = "データ不足"
    else:
        ratio = passed / len(evaluable)
        signal = "良好" if ratio >= 0.6 else ("中立" if ratio >= 0.4 else "懸念")

    finding = "、".join(notes) + "。" + ref_text
    finding += f"（評価できた{len(evaluable)}項目中{passed}項目が基準を満たしました。基準は簡易的な目安です）"
    return {"name": "② 企業分析", "signal": signal, "finding": finding, "score": (passed, len(evaluable))}


# ---------------------------------------------------------------
# テクニカル分析
# ---------------------------------------------------------------
def _tech_agent(ctx):
    price, sma20, sma50 = ctx["last_price"], ctx["sma20"], ctx["sma50"]
    rsi, macd, signal = ctx["rsi14"], ctx["macd"], ctx["macd_signal"]
    score = 0
    evaluated = 0
    notes = []

    if np.isfinite(sma20):
        evaluated += 1
        if price > sma20:
            score += 1; notes.append("終値は20日移動平均より上")
        else:
            notes.append("終値は20日移動平均以下")
    else:
        notes.append("20日移動平均は株価データ不足で判定できません")

    if np.isfinite(sma50):
        evaluated += 1
        if price > sma50:
            score += 1; notes.append("終値は50日移動平均より上")
        else:
            notes.append("終値は50日移動平均以下")
    else:
        notes.append("50日移動平均は株価データ不足で判定できません")

    if np.isfinite(macd) and np.isfinite(signal):
        evaluated += 1
        if macd > signal:
            score += 1; notes.append("MACDはシグナル線より上")
        else:
            notes.append("MACDはシグナル線以下")
    else:
        notes.append("MACDは株価データ不足で判定できません")

    if np.isfinite(rsi) and rsi >= 70:
        notes.append("RSIは70以上で短期的な過熱に注意")
    elif np.isfinite(rsi) and rsi <= 30:
        notes.append("RSIは30以下で売られ過ぎの可能性")

    if evaluated < 3:
        sig = "データ不足"
    else:
        sig = "強気" if score >= 2 else ("中立" if score == 1 else "弱気")
    return {"name": "① テクニカル分析", "signal": sig, "finding": "。".join(notes) + "。"}


def _rule_agents(ctx, fund=None):
    tech = _tech_agent(ctx)
    corp = _fund_agent(fund or {})
    tech_signal, corp_signal = tech["signal"], corp["signal"]

    # 総合判断：テクニカルが強気でも、企業分析が「懸念」なら購入候補にしない
    if tech_signal == "データ不足":
        verdict = "判断保留（株価データ不足）"
    elif tech_signal == "強気" and corp_signal == "懸念":
        verdict = "様子見"
    elif tech_signal == "強気":
        verdict = "仮想購入候補"
    elif tech_signal == "中立":
        verdict = "様子見"
    else:
        verdict = "新規購入を見送り"

    agents = [
        tech,
        {k: v for k, v in corp.items() if k != "score"},
        {"name": "③ ニュース・市場分析", "signal": "未接続", "finding": "この初期版ではニュースの網羅的な取得・真偽判定を行っていません。決算、為替、金利、自動車関税などの材料を確認してください。"},
        {"name": "④ リスク管理", "signal": "中程度" if ctx["risk"] == "中程度" else ctx["risk"], "finding": f"仮想資金は{ctx['capital']:,}円。単一銘柄への集中、決算発表時の急変、円高・円安の影響に注意。購入数量は資金の20%を上限に試算します。"},
    ]
    corp_note = {
        "良好": "企業分析は良好",
        "中立": "企業分析は中立",
        "懸念": "企業分析で懸念あり（購入候補は見送り）",
        "データ不足": "企業分析は財務データ不足のため未評価",
    }[corp_signal]
    final_text = f"テクニカルは{tech_signal}、{corp_note}。ニュース確認は未実装のため、単独で売買判断に使わないでください。"
    agents.append({"name": "⑤ 最終判断", "signal": verdict, "finding": final_text})

    max_amount = int(ctx["capital"] * 0.20)
    price = ctx["last_price"]
    buy = verdict == "仮想購入候補"
    shares = max(0, int(max_amount // price)) if buy else 0
    amount = int(shares * price)
    plan = {"shares": shares, "amount": amount, "cash": int(ctx["capital"] - amount)}
    return {
        "verdict": verdict,
        "summary": final_text + " 価格指標と簡易財務指標による暫定評価で、完全なAI投資判断ではありません。",
        "agents": agents,
        "plan": plan,
        "llm_used": False,
        "fundamentals": fund or {},
    }


def run_agents(ctx, df, api_key=""):
    fund = fetch_fundamentals(ctx["symbol"])
    result = _rule_agents(ctx, fund)
    if not api_key:
        return result
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        corp = next(a for a in result["agents"] if a["name"].startswith("② "))
        prompt = f"""
あなたは日本株の投資リサーチ補助AIです。断定的な利益保証や、実際の発注指示はしないでください。
次の数値だけを根拠に、簡潔な日本語で暫定判断を説明してください。ニュースはデータ未提供なので「未評価」と明記してください。
企業分析が「データ不足」の場合は、その旨も明記してください。
銘柄: {ctx['symbol']}
終値: {ctx['last_price']:.2f}
SMA20: {ctx['sma20']}
SMA50: {ctx['sma50']}
RSI14: {ctx['rsi14']}
MACD: {ctx['macd']}
MACDシグナル: {ctx['macd_signal']}
1か月騰落率(%): {ctx['return_1m']}
3か月騰落率(%): {ctx['return_3m']}
企業分析（{corp['signal']}）: {corp['finding']}
資金: {ctx['capital']}円、リスク許容度: {ctx['risk']}、期間: {ctx['horizon']}
既存のルール判定: {result['verdict']}
"""
        response = client.responses.create(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), input=prompt)
        commentary = response.output_text.strip()
        result["summary"] = commentary
        result["llm_used"] = True
        result["agents"][-1]["finding"] = commentary
    except Exception as exc:
        result["summary"] += f"（AIコメント取得に失敗したため、ルール判定を表示しています：{type(exc).__name__}）"
    return result
