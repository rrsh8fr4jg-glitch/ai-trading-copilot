import os
from datetime import date, timedelta
import pandas as pd
import streamlit as st
import yfinance as yf

from agents import run_agents, build_market_context
from storage import init_db, save_decision, load_history

st.set_page_config(
    page_title="AI株トレード・コパイロット",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)
init_db()

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 850px;}
    div[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: .7rem; border-radius: 12px;}
    .hero {padding: 1rem; border-radius: 16px; background: linear-gradient(120deg, #102a43, #1d4e89); color: white; margin-bottom: 1rem;}
    .hero h1 {font-size: 1.55rem; margin: 0;}
    .hero p {margin: .35rem 0 0 0; opacity: .9;}
    @media (max-width: 600px) {
        .block-container {padding-left: 1rem; padding-right: 1rem;}
        h1 {font-size: 1.5rem !important;}
    }
</style>
<div class="hero">
  <h1>📈 AI株トレード・コパイロット</h1>
  <p>日本株・中期投資｜仮想売買専用｜スマホ対応</p>
</div>
""", unsafe_allow_html=True)

with st.expander("投資条件", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input("銘柄コード", value="7203").strip()
        capital = st.number_input("仮想運用資金（円）", min_value=10000, max_value=100000000, value=1000000, step=100000)
    with c2:
        risk = st.selectbox("リスク許容度", ["低め", "中程度", "高め"], index=1)
        horizon = st.selectbox("投資期間", ["短期", "中期", "長期"], index=1)
    period = st.selectbox("分析期間", ["6か月", "1年", "2年", "5年"], index=1)
    period_map = {"6か月": "6mo", "1年": "1y", "2年": "2y", "5年": "5y"}

symbol = ticker if "." in ticker else f"{ticker}.T"
st.caption(f"データ取得先：Yahoo Finance（{symbol}）。市場データは遅延・欠損の可能性があります。")

if st.button("分析して仮想売買判断を作成", type="primary", use_container_width=True):
    with st.spinner("株価データを取得し、5つの分析役が検討しています…"):
        try:
            raw = yf.download(symbol, period=period_map[period], interval="1d", auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                st.error("株価データを取得できませんでした。銘柄コードや通信状況を確認してください。")
                st.stop()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna(subset=["Close"])
            context = build_market_context(raw, symbol, capital, risk, horizon)
            api_key = os.getenv("OPENAI_API_KEY", "")
            try:
                api_key = api_key or st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                pass
            result = run_agents(context, raw, api_key=api_key)
            st.session_state["last_result"] = result
            st.session_state["last_context"] = context
            save_decision(symbol, capital, risk, horizon, result)
        except Exception as e:
            st.error(f"分析中にエラーが発生しました: {e}")
            st.stop()

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    ctx = st.session_state["last_context"]
    st.divider()
    st.subheader("分析サマリー")
    c1, c2, c3 = st.columns(3)
    c1.metric("直近終値", f"¥{ctx['last_price']:,.1f}")
    c2.metric("RSI（14日）", f"{ctx['rsi14']:.1f}" if pd.notna(ctx["rsi14"]) else "—")
    c3.metric("仮想資金", f"¥{ctx['capital']:,.0f}")
    verdict = result.get("verdict", "様子見")
    st.markdown(f"### 最終判断：{verdict}")
    st.write(result.get("summary", "判断コメントはありません。"))
    if result.get("llm_used"):
        st.caption("AIコメント：OpenAI APIを利用")
    else:
        st.caption("ルールベース分析です。OPENAI_API_KEYが設定されるとAIコメントを追加できます。")

    with st.expander("5つの分析エージェント", expanded=True):
        for agent in result.get("agents", []):
            st.markdown(f"**{agent['name']}**")
            st.write(agent["finding"])
            st.caption(f"評価：{agent['signal']}")
            st.divider()

    st.subheader("仮想売買プラン")
    plan = result.get("plan", {})
    p1, p2 = st.columns(2)
    p1.metric("仮想購入数量", f"{plan.get('shares', 0):,} 株")
    p2.metric("投資予定額", f"¥{plan.get('amount', 0):,.0f}")
    st.write(f"現金余力の目安：¥{plan.get('cash', capital):,.0f}")
    st.warning("これは学習・検証用の参考情報です。将来の利益を保証するものではなく、実際の売買指示ではありません。")

    with st.expander("株価チャート・移動平均"):
        chart = ctx["chart"]
        st.line_chart(chart[["Close", "SMA20", "SMA50"]].rename(columns={"Close":"終値","SMA20":"20日平均","SMA50":"50日平均"}))
    with st.expander("分析に使った主な数値"):
        st.json({k: v for k, v in ctx.items() if k not in ["chart"]})

st.divider()
st.subheader("過去の分析履歴")
history = load_history(limit=10)
if history.empty:
    st.caption("まだ分析履歴はありません。")
else:
    st.dataframe(history, use_container_width=True, hide_index=True)

st.caption("注意：株価データは外部サービスに依存します。API障害、データ欠損、株式分割・配当・取引コスト等の影響を完全には扱いません。投資判断はご自身で確認してください。")
