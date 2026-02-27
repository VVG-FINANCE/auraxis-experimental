# ==========================
# Auraxis RIA - Radar Institucional Multi-Timeframe Expandido
# Funcionalidade completa: sinais, SL/TP, Monte Carlo, Bayes, indicadores institucionais
# Compatível: Android + Streamlit gratuito + GitHub
# ==========================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------------------------
# Configurações da Página
# -------------------------
st.set_page_config(
    page_title="Auraxis Radar MultiTF",
    layout="wide"
)
st.markdown(
    "<h1 style='text-align:center; font-size:2rem; color:#1E90FF;'>🌐 Auraxis — Radar Institucional Multi-Timeframe Expandido</h1>",
    unsafe_allow_html=True
)

# -------------------------
# Configurações do Usuário
# -------------------------
ativos_input = st.text_input("Ativos (ex: EURUSD=X,GBPUSD=X)", value="EURUSD=X,USDJPY=X")
ativos = [a.strip().upper() for a in ativos_input.split(",") if a.strip()]

timeframes = st.multiselect(
    "Timeframes",
    ["1m", "5m", "15m", "1h", "4h", "1d"],
    default=["5m", "15m"]
)

num_simulacoes = st.slider("Número de simulações Monte Carlo", min_value=200, max_value=3000, value=500)
vwap_period = st.slider("Período VWAP", min_value=5, max_value=60, value=20)
ema_period = st.slider("Período EMA", min_value=5, max_value=50, value=14)
rsi_period = st.slider("Período RSI", min_value=5, max_value=50, value=14)
bollinger_period = st.slider("Período Bollinger Bands", min_value=10, max_value=50, value=20)
bollinger_dev = st.slider("Desvio padrão Bollinger", min_value=1.0, max_value=3.0, value=2.0, step=0.1)

# -------------------------
# Funções internas
# -------------------------

@st.cache_data(show_spinner=False)
def fetch_data(ativo, period="7d", interval="5m"):
    try:
        df = yf.download(ativo, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        return df
    except:
        return pd.DataFrame()

def compute_ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def momentum_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if (last['Close'] > last['Open']) & (last['Close'] > prev['Close']):
        return 1  # Compra
    elif (last['Close'] < last['Open']) & (last['Close'] < prev['Close']):
        return -1  # Venda
    else:
        return 0  # Neutro

def monte_carlo_sim(df, num_sim=500):
    log_returns = np.log(df['Close'] / df['Close'].shift()).dropna()
    last_price = df['Close'].iloc[-1]
    resultados = []
    for _ in range(num_sim):
        sim = last_price * np.exp(np.cumsum(np.random.choice(log_returns, size=len(df))))
        resultados.append(sim[-1])
    return np.mean(resultados)

def bayesian_score(preco_atual, pct_changes):
    mean = np.mean(pct_changes)
    std = np.std(pct_changes)
    if std == 0:
        return 50
    z_score = (pct_changes[-1] - mean) / std
    score = 50 + (z_score * 10)  # escala 0-100 aproximada
    return max(0, min(100, score))

def safety_zones(df):
    atr = compute_ATR(df).iloc[-1]
    last_close = df['Close'].iloc[-1]
    sl = last_close - atr
    tp = last_close + atr
    return round(sl,5), round(tp,5)

def compute_VWAP(df, period=20):
    vwap = (df['Close'] * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
    return vwap

def compute_EMA(df, period=14):
    return df['Close'].ewm(span=period, adjust=False).mean()

def compute_RSI(df, period=14):
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rsi = 100 - (100/(1 + ma_up/ma_down))
    return rsi

def compute_Bollinger(df, period=20, dev=2.0):
    sma = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()
    upper = sma + dev*std
    lower = sma - dev*std
    return upper, lower

def paradoxo_radar(df):
    momentum = momentum_signal(df)
    mc = monte_carlo_sim(df, num_sim=num_simulacoes)
    score = bayesian_score(df['Close'].iloc[-1], df['Close'].pct_change().dropna())
    sl, tp = safety_zones(df)
    vwap = compute_VWAP(df, period=vwap_period).iloc[-1]
    ema = compute_EMA(df, period=ema_period).iloc[-1]
    rsi = compute_RSI(df, period=rsi_period).iloc[-1]
    boll_upper, boll_lower = compute_Bollinger(df, period=bollinger_period, dev=bollinger_dev)
    return {
        "Momentum": momentum,
        "MonteCarlo": mc,
        "Score": score,
        "SL": sl,
        "TP": tp,
        "VWAP": round(vwap,5),
        "EMA": round(ema,5),
        "RSI": round(rsi,1),
        "Bollinger_Upper": round(boll_upper.iloc[-1],5),
        "Bollinger_Lower": round(boll_lower.iloc[-1],5)
    }

# -------------------------
# Radar Multi-Timeframe
# -------------------------

results = []

with st.spinner("Analisando ativos e indicadores institucionais..."):
    for ativo in ativos:
        ativo_result = {"Ativo": ativo}
        for tf in timeframes:
            df = fetch_data(ativo, period="7d", interval=tf)
            if df.empty: continue
            radar = paradoxo_radar(df)
            for key, value in radar.items():
                ativo_result[f"{tf}_{key}"] = value
        results.append(ativo_result)

# -------------------------
# Ranking de ativos
# -------------------------
def compute_strength(row):
    # força = Momentum * Score normalizado (0-100)
    strength = 0
    for col in row.index:
        if "Momentum" in col:
            signal = row[col]
            strength += signal * 10
        if "Score" in col:
            strength += row[col]/10
    return round(strength,1)

if results:
    df_results = pd.DataFrame(results)
    df_results['Strength'] = df_results.apply(compute_strength, axis=1)

    def color_signal(val):
        if val == 1: return 'background-color: #90EE90'  # verde compra
        elif val == -1: return 'background-color: #FFB6C1'  # vermelho venda
        else: return ''
    
    st.markdown("### Radar de Sinais, Scores e Indicadores Institucionais")
    st.dataframe(
        df_results.style.applymap(color_signal, subset=[c for c in df_results.columns if 'Momentum' in c]),
        height=700
    )
else:
    st.warning("Nenhum dado disponível para os ativos/timeframes selecionados.")

# -------------------------
# Mensagens estratégicas
# -------------------------
st.markdown("### ⚡ Resumo Estratégico")
if results:
    for idx, row in df_results.iterrows():
        st.markdown(f"**{row['Ativo']}** — Força: {row['Strength']}")
        for tf in timeframes:
            momentum_col = f"{tf}_Momentum"
            score_col = f"{tf}_Score"
            sl_col = f"{tf}_SL"
            tp_col = f"{tf}_TP"
            if momentum_col in row:
                signal = row[momentum_col]
                st.markdown(
                    f"- Timeframe {tf}: Signal: {signal}, Score: {row[score_col]}, SL: {row[sl_col]}, TP: {row[tp_col]}"
                )
