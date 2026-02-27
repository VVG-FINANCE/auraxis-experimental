# =========================
# Auraxis 8.0 — Radar Institucional Multi-Timeframe
# =========================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

# -------------------------
# Página
# -------------------------
st.set_page_config(page_title="Auraxis 8.0", layout="wide")
st.title("🌌 Auraxis 8.0 — Radar Institucional Multi-Timeframe")

# -------------------------
# Configurações
# -------------------------
st.sidebar.header("Configurações do Trader")
perfil_trader = st.sidebar.selectbox("Perfil de Trader:", ["Ultra Conservador","Conservador","Moderado","Agressivo"])
capital_inicial = st.sidebar.number_input("Capital Inicial (USD):", value=1000.0, step=100.0)
risk_per_trade = st.sidebar.slider("Risco por trade (%):", 0.1,5.0,1.0)

timeframes = st.sidebar.multiselect("Timeframes:", ["1m","5m","15m","30m"], default=["1m","5m"])
ativos_disponiveis = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCHF=X"]
max_signals = st.sidebar.slider("Máx. sinais por recarga:",1,10,5)

# -------------------------
# Funções de Mercado
# -------------------------
def fetch_data(symbol, period="7d", interval="1m"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty: return pd.DataFrame()
    df = df.reset_index()
    if "Date" in df.columns: df.rename(columns={"Date":"Datetime"}, inplace=True)
    return df[['Datetime','Open','High','Low','Close']].dropna()

def compute_ATR(df, period=14):
    hl = df['High'] - df['Low']
    hc = abs(df['High'] - df['Close'].shift())
    lc = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([hl,hc,lc],axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def momentum_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last['Close']>last['Open'] and last['Close']>prev['Close']: return 1
    elif last['Close']<last['Open'] and last['Close']<prev['Close']: return -1
    else: return 0

def monte_carlo_sim(df, steps=50, sims=100):
    log_returns = np.log(df['Close']/df['Close'].shift(1)).dropna()
    results = []
    for _ in range(sims):
        sample = np.random.choice(log_returns, steps)
        price_path = df['Close'].iloc[-1] * np.exp(np.cumsum(sample))
        results.append(price_path[-1])
    return np.mean(results)

def bayesian_score(current, historical):
    mean = np.mean(historical)
    std = np.std(historical)
    z = (current-mean)/std if std>0 else 0
    score = 1/(1+np.exp(-z))
    return round(score,2)

def position_size(capital, atr, risk_percent):
    return (capital*(risk_percent/100))/atr

# -------------------------
# Persistência
# -------------------------
signal_file = "signals.csv"
if not os.path.exists(signal_file):
    pd.DataFrame(columns=["Datetime","Ativo","Timeframe","Sinal","SL","TP","Score","PosSize"]).to_csv(signal_file,index=False)
signals_df = pd.read_csv(signal_file)

# -------------------------
# Radar
# -------------------------
st.subheader("📊 Radar Multi-Timeframe")
new_signals = []

for ativo in ativos_disponiveis:
    for tf in timeframes:
        df = fetch_data(ativo, period="7d", interval=tf)
        if df.empty: continue
        atr = compute_ATR(df).iloc[-1]
        signal_raw = momentum_signal(df)
        mc_price = monte_carlo_sim(df)
        score = bayesian_score(df['Close'].iloc[-1], df['Close'].pct_change().dropna())

        if signal_raw==1: sinal="BUY"
        elif signal_raw==-1: sinal="SELL"
        else: continue

        ultimo_preco = df['Close'].iloc[-1]
        sl = ultimo_preco - atr if sinal=="BUY" else ultimo_preco + atr
        tp = ultimo_preco + atr*1.5 if sinal=="BUY" else ultimo_preco - atr*1.5
        pos_size = position_size(capital_inicial, atr, risk_per_trade)

        new_signals.append({
            "Datetime": datetime.now(),
            "Ativo": ativo,
            "Timeframe": tf,
            "Sinal": sinal,
            "SL": sl,
            "TP": tp,
            "Score": score,
            "PosSize": pos_size
        })

# Atualiza CSV
if new_signals:
    signals_df = pd.concat([signals_df, pd.DataFrame(new_signals)], ignore_index=True)
    signals_df = signals_df.tail(max_signals)
    signals_df.to_csv(signal_file, index=False)

# -------------------------
# Exibição
# -------------------------
if not signals_df.empty:
    st.dataframe(signals_df[['Datetime','Ativo','Timeframe','Sinal','SL','TP','Score','PosSize']])
else:
    st.info("Sem sinais no momento.")

# -------------------------
# Alertas
# -------------------------
for idx,row in signals_df.iterrows():
    if row['Sinal']=="BUY":
        st.success(f"🟢 {row['Ativo']} [{row['Timeframe']}]: COMPRAR | SL={row['SL']:.5f} | TP={row['TP']:.5f} | Score={row['Score']}")
    elif row['Sinal']=="SELL":
        st.error(f"🔴 {row['Ativo']} [{row['Timeframe']}]: VENDER | SL={row['SL']:.5f} | TP={row['TP']:.5f} | Score={row['Score']}")

st.markdown("💡 Radar gerado automaticamente usando Momentum, ATR, Monte Carlo e Score Bayesiano.")
