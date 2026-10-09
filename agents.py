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

def _rule_agents(ctx):
    price, sma20, sma50 = ctx["last_price"], ctx["sma20"], ctx["sma50"]
    rsi, macd, signal = ctx["rsi14"], ctx["macd"], ctx["macd_signal"]
    tech_score = 0
    tech_notes = []
    if np.isfinite(sma20) and price > sma20:
        tech_score += 1; tech_notes.append("終値は20日移動平均より上")
    else:
        tech_notes.append("終値は20日移動平均以下、またはデータ不足")
    if np.isfinite(sma50) and price > sma50:
        tech_score += 1; tech_notes.append("終値は50日移動平均より上")
    else:
        tech_notes.append("終値は50日移動平均以下、またはデータ不足")
    if np.isfinite(macd) and np.isfinite(signal) and macd > signal:
        tech_score += 1; tech_notes.append("MACDはシグナル線より上")
    else:
        tech_notes.append("MACDはシグナル線以下、またはデータ不足")
    if np.isfinite(rsi) and rsi >= 70:
        tech_notes.append("RSIは70以上で短期的な過熱に注意")
    elif np.isfinite(rsi) and rsi <= 30:
        tech_notes.append("RSIは30以下で売られ過ぎの可能性")
    tech_signal = "強気" if tech_score >= 2 else ("中立" if tech_score == 1 else "弱気")

    agents = [
        {"name":"① テクニカル分析", "signal":tech_signal, "finding":"。".join(tech_notes) + "。"},
        {"name":"② 企業分析", "signal":"未接続", "finding":"この初期版では財務諸表・決算短信を自動取得していません。売上、営業利益率、EPS、フリーキャッシュフローを別途確認してください。"},
        {"name":"③ ニュース・市場分析", "signal":"未接続", "finding":"この初期版ではニュースの網羅的な取得・真偽判定を行っていません。決算、為替、金利、自動車関税などの材料を確認してください。"},
        {"name":"④ リスク管理", "signal":"中程度" if ctx["risk"]=="中程度" else ctx["risk"], "finding":f"仮想資金は{ctx['capital']:,}円。単一銘柄への集中、決算発表時の急変、円高・円安の影響に注意。初期版の購入数量は資金の20%を上限に試算します。"},
        {"name":"⑤ 最終判断", "signal":tech_signal, "finding":f"テクニカル指標を中心にした暫定判断です。ファンダメンタルズ・ニュース確認が未実装のため、単独で売買判断に使わないでください。"},
    ]
    # conservative position-sizing: max 20% of capital in one paper position
    max_amount = int(ctx["capital"] * 0.20)
    shares = max(0, int(max_amount // price)) if tech_signal == "強気" else 0
    amount = int(shares * price)
    plan = {"shares": shares, "amount": amount, "cash": int(ctx["capital"] - amount)}
    verdict = "仮想購入候補" if tech_signal == "強気" else ("様子見" if tech_signal == "中立" else "新規購入を見送り")
    return {"verdict": verdict, "summary": f"テクニカル評価は{tech_signal}。これは価格・移動平均・MACDを使った簡易評価で、企業価値やニュースを織り込んだ完全なAI投資判断ではありません。", "agents": agents, "plan": plan, "llm_used": False}

def run_agents(ctx, df, api_key=""):
    result = _rule_agents(ctx)
    if not api_key:
        return result
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = f"""
あなたは日本株の投資リサーチ補助AIです。断定的な利益保証や、実際の発注指示はしないでください。
次の数値だけを根拠に、簡潔な日本語で暫定判断を説明してください。企業分析とニュースはデータ未提供なので「未評価」と明記してください。
銘柄: {ctx['symbol']}
終値: {ctx['last_price']:.2f}
SMA20: {ctx['sma20']}
SMA50: {ctx['sma50']}
RSI14: {ctx['rsi14']}
MACD: {ctx['macd']}
MACDシグナル: {ctx['macd_signal']}
1か月騰落率(%): {ctx['return_1m']}
3か月騰落率(%): {ctx['return_3m']}
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
