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


_FUND_CACHE = {}
_FUND_TTL = 3600  # 秒


def _ok(v):
    return isinstance(v, (int, float, np.floating)) and np.isfinite(v)


def _val(df, names, col):
    """決算書DataFrameから、指定列(期)の値を項目名の候補順に探す。"""
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            try:
                v = df.at[n, col]
            except Exception:
                continue
            if _ok(v):
                return float(v)
    return None


_PEER_GROUPS = {
    "世界の主要自動車メーカー": [
        # 国内
        "7203.T", "7267.T", "7201.T", "7269.T", "7270.T", "7261.T", "7202.T", "7211.T", "7272.T",
        # 海外（米・欧・韓・中）
        "F", "GM", "TSLA", "STLA", "VOW3.DE", "MBG.DE", "BMW.DE", "RNO.PA",
        "005380.KS", "000270.KS", "1211.HK",
    ],
}
_PEER_NAMES = {
    "7203.T": "トヨタ", "7267.T": "ホンダ", "7201.T": "日産", "7269.T": "スズキ", "7270.T": "SUBARU",
    "7261.T": "マツダ", "7202.T": "いすゞ", "7211.T": "三菱自", "7272.T": "ヤマハ発",
    "F": "フォード", "GM": "GM", "TSLA": "テスラ", "STLA": "ステランティス", "VOW3.DE": "VW",
    "MBG.DE": "メルセデス", "BMW.DE": "BMW", "RNO.PA": "ルノー",
    "005380.KS": "現代自", "000270.KS": "起亜", "1211.HK": "BYD",
}
_PEER_CACHE = {}


def _peer_group(symbol):
    s = symbol.strip().upper()
    if s.isdigit():
        s += ".T"
    for name, syms in _PEER_GROUPS.items():
        if s in syms:
            return name, s, syms
    return None, s, []


def _cols_desc(df):
    try:
        return sorted(df.columns, reverse=True)
    except Exception:
        return list(df.columns)


def _operating_margin_of(sym):
    """1銘柄の営業利益率（通期）。決算書から計算し、だめならinfoで代替。"""
    try:
        import yfinance as yf
        t = yf.Ticker(sym)
        try:
            fin = t.financials
            if fin is not None and not fin.empty:
                c0 = _cols_desc(fin)[0]
                rev = _val(fin, ["Total Revenue", "Operating Revenue"], c0)
                op = _val(fin, ["Operating Income", "Total Operating Income As Reported"], c0)
                if rev and op is not None:
                    return op / rev
        except Exception:
            pass
        v = (t.info or {}).get("operatingMargins")
        if _ok(v):
            return float(v)
    except Exception:
        pass
    return None


def _peer_margins(syms):
    import time
    from concurrent.futures import ThreadPoolExecutor
    now = time.time()
    res, todo = {}, []
    for s in syms:
        c = _PEER_CACHE.get(s)
        if c and now - c[0] < _FUND_TTL:
            res[s] = c[1]
        else:
            todo.append(s)
    if todo:
        with ThreadPoolExecutor(max_workers=6) as ex:
            for s, m in zip(todo, ex.map(_operating_margin_of, todo)):
                if m is not None:
                    _PEER_CACHE[s] = (now, m)
                    res[s] = m
    return res


def fetch_fundamentals(symbol):
    """財務指標を取得して (指標の辞書, 失敗理由) を返す。例外は外に出さない。
    通期（直近の決算期）の決算書を優先し、足りない項目だけYahooのinfoで補う。"""
    import time
    cached = _FUND_CACHE.get(symbol)
    if cached and time.time() - cached[0] < _FUND_TTL:
        return cached[1], ""

    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
    except Exception as exc:
        return {}, f"yfinanceを読み込めませんでした（{type(exc).__name__}）"

    out, errors = {}, []

    # (1) 通期の決算書から計算
    try:
        fin = t.financials
        if fin is not None and not fin.empty:
            cols = _cols_desc(fin)
            c0 = cols[0]
            rev0 = _val(fin, ["Total Revenue", "Operating Revenue"], c0)
            op0 = _val(fin, ["Operating Income", "Total Operating Income As Reported"], c0)
            ni0 = _val(fin, ["Net Income", "Net Income Common Stockholders"], c0)
            try:
                out["fiscalLabel"] = f"{c0.year}年{c0.month}月期"
            except Exception:
                pass
            if rev0 and op0 is not None:
                out["operatingMargins"] = op0 / rev0
            if len(cols) >= 2:
                c1 = cols[1]
                rev1 = _val(fin, ["Total Revenue", "Operating Revenue"], c1)
                ni1 = _val(fin, ["Net Income", "Net Income Common Stockholders"], c1)
                if rev0 is not None and rev1:
                    out["revenueGrowth"] = rev0 / rev1 - 1
                if ni0 is not None and ni1 and ni1 > 0:
                    out["earningsGrowth"] = ni0 / ni1 - 1
            if ni0 is not None:
                bs = t.balance_sheet
                if bs is not None and not bs.empty:
                    eq = _val(bs, ["Stockholders Equity", "Common Stock Equity",
                                   "Total Equity Gross Minority Interest"], _cols_desc(bs)[0])
                    if eq and eq > 0:
                        out["returnOnEquity"] = ni0 / eq
    except Exception as exc:
        errors.append(f"損益計算書の取得失敗（{type(exc).__name__}）")

    try:
        cf = t.cashflow
        if cf is not None and not cf.empty:
            c0 = _cols_desc(cf)[0]
            fcf = _val(cf, ["Free Cash Flow"], c0)
            if fcf is None:
                ocf = _val(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], c0)
                capex = _val(cf, ["Capital Expenditure"], c0)
                if ocf is not None and capex is not None:
                    fcf = ocf + capex  # capexは通常マイナス
            if fcf is not None:
                out["freeCashflow"] = fcf
    except Exception as exc:
        errors.append(f"キャッシュフローの取得失敗（{type(exc).__name__}）")

    # (2) 足りない項目はinfoで補完（直近四半期ベースが混じる）
    filled_from_info = []
    try:
        info = t.info or {}
        for k in _FUND_KEYS:
            v = info.get(k)
            if k not in out and _ok(v):
                out[k] = float(v)
                if k in ("operatingMargins", "returnOnEquity", "revenueGrowth", "earningsGrowth", "freeCashflow"):
                    filled_from_info.append(k)
    except Exception as exc:
        errors.append(f"info取得失敗（{type(exc).__name__}）")
    if filled_from_info:
        out["infoFilled"] = "、".join(filled_from_info)

    # (3) 同業他社との営業利益率比較（他社も通期）
    if "operatingMargins" in out:
        try:
            gname, me, syms = _peer_group(symbol)
            if gname:
                all_peers = _peer_margins([s for s in syms if s != me])
                # マイナス（営業赤字）の会社は比較から除外する
                peers = {s: m for s, m in all_peers.items() if m >= 0}
                excluded = {s: m for s, m in all_peers.items() if m < 0}
                if len(peers) >= 3:
                    vals = sorted(peers.values())
                    out["peerGroup"] = gname
                    out["peerMedianMargin"] = float(np.median(vals))
                    out["peerCount"] = float(len(vals))
                    out["peerRank"] = float(1 + sum(1 for x in vals if x > out["operatingMargins"]))
                    dom = [m for s, m in peers.items() if s.endswith(".T")]
                    if len(dom) >= 2:
                        out["peerDomesticMedian"] = float(np.median(dom))
                    out["peerDetail"] = "、".join(
                        f"{_PEER_NAMES.get(s, s)} {m * 100:.1f}%"
                        for s, m in sorted(peers.items(), key=lambda kv: -kv[1]))
                    if excluded:
                        out["peerExcluded"] = "、".join(
                            f"{_PEER_NAMES.get(s, s)} {m * 100:.1f}%"
                            for s, m in sorted(excluded.items(), key=lambda kv: -kv[1]))
        except Exception:
            pass

    if out:
        _FUND_CACHE[symbol] = (time.time(), out)
        return out, ""
    reason = "、".join(errors) if errors else "Yahoo Financeに財務データがありませんでした"
    return {}, reason


def _fund_agent(f, err=""):
    if not f:
        return {
            "name": "② 企業分析",
            "signal": "データ不足",
            "finding": "Yahoo Financeから財務指標を取得できませんでした。" + (f"理由：{err}。" if err else "") + "時間をおいて再実行するか、決算短信で売上・営業利益率・EPS・フリーキャッシュフローを確認してください。",
            "score": None,
        }

    # (表示名, 値の表示, 合格か[True/False/None=データなし], 基準の説明)
    def pct(v):
        return f"{v * 100:.1f}%"

    rows = []
    v = f.get("operatingMargins")
    if v is not None and "peerMedianMargin" in f:
        med = f["peerMedianMargin"]
        n = int(f["peerCount"])
        shown = f"{pct(v)}（{f['peerGroup']}・黒字の{n + 1}社中{int(f['peerRank'])}位、中央値{pct(med)}）"
        rows.append(("営業利益率", shown, v >= med, "同業中央値以上で良好"))
    else:
        rows.append(("営業利益率", pct(v) if v is not None else None, None if v is None else v >= 0.10, "10%以上で良好（同業比較なし）"))
    v = f.get("returnOnEquity")
    rows.append(("ROE", pct(v) if v is not None else None, None if v is None else v >= 0.08, "8%以上で良好"))
    v = f.get("revenueGrowth")
    rows.append(("売上成長率(前期比)", pct(v) if v is not None else None, None if v is None else v > 0, "プラスで良好"))
    v = f.get("earningsGrowth")
    rows.append(("利益成長率(前期比)", pct(v) if v is not None else None, None if v is None else v > 0, "プラスで良好"))
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
    if "peerDomesticMedian" in f:
        ref_text += f"国内メーカーのみの中央値は{pct(f['peerDomesticMedian'])}。"
    if f.get("peerDetail"):
        ref_text += f"比較に使った他社の営業利益率：{f['peerDetail']}。"
    if f.get("peerExcluded"):
        ref_text += f"営業赤字のため比較から除外：{f['peerExcluded']}。"

    evaluable = [ok for _, _, ok, _ in rows if ok is not None]
    passed = sum(1 for ok in evaluable if ok)
    if len(evaluable) < 3:
        signal = "データ不足"
    else:
        ratio = passed / len(evaluable)
        signal = "良好" if ratio >= 0.6 else ("中立" if ratio >= 0.4 else "懸念")

    basis = f"通期（{f['fiscalLabel']}）ベース。" if f.get("fiscalLabel") else ""
    if f.get("infoFilled"):
        ref_text += f"（注：一部項目は決算書を取得できず、直近四半期ベースの値で代替しています）"
    finding = basis + "、".join(notes) + "。" + ref_text
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


def _rule_agents(ctx, fund=None, fund_err=""):
    tech = _tech_agent(ctx)
    corp = _fund_agent(fund or {}, fund_err)
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
    fund, fund_err = fetch_fundamentals(ctx["symbol"])
    result = _rule_agents(ctx, fund, fund_err)
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
