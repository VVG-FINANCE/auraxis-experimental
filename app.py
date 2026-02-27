# =========================
# Auraxis 2.2 - Experimental Stable
# =========================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Auraxis 2.2", layout="wide")
st.title("Auraxis 2.2 — Simulator Fusion Layer")

# -------------------------
# Sidebar - Configurações
# -------------------------
st.sidebar.header("Configurações do Usuário")

perfil_trader = st.sidebar.selectbox(
    "Perfil de Trader:",
    ["Ultra Conservador", "Conservador", "Moderado", "Agressivo", "Automático"]
)

simulador_tipo = st.sidebar.selectbox(
    "Simulador:",
    ["Manual", "Assistido", "Automático"]
)

leverage_guide = {
    "Ultra Conservador": "1:10",
    "Conservador": "1:20",
    "Moderado": "1:50",
    "Agressivo": "1:100",
    "Automático": "Adaptável"
}

st.sidebar.markdown(f"**Alavancagem orientativa:** {leverage_guide[perfil_trader]}")

pares_disponiveis = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"]
periodo = st.sidebar.selectbox("Timeframe:", ["1m", "5m"])

# -------------------------
# Funções
# -------------------------
def fetch_yahoo_data(symbol, period="7d", interval="1m"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    if 'Datetime' not in df.columns:
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Datetime'}, inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    required = ['Datetime', 'Open', 'High', 'Low', 'Close']
    df = df[[c for c in required if c in df.columns]]

    return df.dropna()


def candle_valido(df):
    return (
        (df['High'] > 0) &
        (df['Low'] > 0) &
        (df['Open'] > 0) &
        (df['Close'] > 0)
    )


def compute_confidence(df):
    body = abs(df['Close'] - df['Open'])
    total_range = df['High'] - df['Low']
    total_range = total_range.replace(0, np.nan)
    conf = body / total_range
    conf = conf.fillna(0.3)
    return np.minimum(0.85, conf * 0.85 + 0.15)


def compute_regime(df):
    diff = df['Close'] - df['Open']
    return np.where(abs(diff) < 0.0005, "Lateral",
           np.where(diff > 0, "Alta", "Baixa"))


def compute_fragmentation(candle):
    zones = {}
    high = candle['High']
    low = candle['Low']
    open_ = candle['Open']
    range_ = high - low

    zones['zona_abertura'] = (open_, open_ + 0.2 * range_)
    zones['zona_expansao'] = (open_ + 0.2 * range_, open_ + 0.5 * range_)
    zones['zona_pullback'] = (open_ + 0.5 * range_, open_ + 0.7 * range_)
    zones['zona_exp_final'] = (open_ + 0.7 * range_, open_ + 0.9 * range_)
    zones['zona_exaustao'] = (open_ + 0.9 * range_, high)

    return zones


def simulate_SL_TP(candle, profile):
    percent_map = {
        "Ultra Conservador": 0.005,
        "Conservador": 0.01,
        "Moderado": 0.02,
        "Agressivo": 0.03,
        "Automático": 0.015
    }

    pct = percent_map.get(profile, 0.01)
    range_ = candle['High'] - candle['Low']
    sl = candle['Close'] - pct * range_
    tp = candle['Close'] + pct * range_

    return sl, tp


def simulate_sweep_IPI(df):
    delta = df['Close'].diff().fillna(0)
    ipi = 0.5 + np.tanh(delta * 100) / 2
    df['IPI'] = np.clip(ipi, 0, 1)
    return df


# -------------------------
# Interface Multi-Par
# -------------------------
tabs = st.tabs([p.replace("=X", "") for p in pares_disponiveis])

for i, par in enumerate(pares_disponiveis):

    with tabs[i]:

        st.subheader(f"Par: {par.replace('=X', '')}")

        df = fetch_yahoo_data(par, period="7d", interval=periodo)

        if df.empty:
            st.warning("Sem dados disponíveis.")
            continue

        df = df[candle_valido(df)]
        df['confidence'] = compute_confidence(df)
        df['regime'] = compute_regime(df)
        df = simulate_sweep_IPI(df)

        if df.empty:
            st.warning("Sem candles válidos.")
            continue

        ultimo_candle = df.iloc[-1]
        sl, tp = simulate_SL_TP(ultimo_candle, perfil_trader)

        # -------------------------
        # Gráfico
        # -------------------------
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df['Datetime'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=f"{par} {periodo}"
        ))

        fig.add_trace(go.Scatter(
            x=df['Datetime'],
            y=df['Close'] * df['IPI'],
            mode='lines',
            name='Sweep/IPI'
        ))

        fig.add_hline(y=sl, line=dict(dash='dash'), annotation_text="SL")
        fig.add_hline(y=tp, line=dict(dash='dash'), annotation_text="TP")

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            title=f"Auraxis 2.2 — {par.replace('=X', '')} ({periodo})"
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # Simulador
        # -------------------------
        if simulador_tipo != "Automático":

            col1, col2 = st.columns(2)

            with col1:
                if st.button(f"Aceitar {par}"):
                    st.success(f"Entrada aceita | SL={sl:.5f} | TP={tp:.5f}")

            with col2:
                if st.button(f"Descartar {par}"):
                    st.warning("Entrada descartada")

        st.dataframe(df.tail(50))
