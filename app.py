import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================
# CONFIGURAÇÃO INICIAL
# ==========================

st.set_page_config(
    page_title="Auraxis Radar Institucional",
    layout="wide"
)

st.markdown(
    """
    <style>
    .big-font {font-size:22px !important; font-weight:600;}
    .metric-box {
        padding:15px;
        border-radius:12px;
        background-color:#111827;
        text-align:center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌌 Auraxis — Radar Institucional")

# ==========================
# SIDEBAR
# ==========================

st.sidebar.header("Configurações")

ativo = st.sidebar.selectbox(
    "Ativo",
    ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "XAUUSD=X"]
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["15m", "1h", "4h", "1d"]
)

simulacoes = st.sidebar.slider(
    "Número de Simulações",
    200, 3000, 1000, step=200
)

horizonte = 20  # Escolha arquitetural fixa

# ==========================
# DEFINIR PERÍODO ESTÁVEL
# ==========================

def periodo_por_intervalo(tf):
    if tf == "15m":
        return "30d"
    if tf == "1h":
        return "60d"
    if tf == "4h":
        return "180d"
    if tf == "1d":
        return "1y"

# ==========================
# BAIXAR DADOS
# ==========================

df = yf.download(
    ativo,
    period=periodo_por_intervalo(timeframe),
    interval=timeframe,
    progress=False
)

if df.empty:
    st.error("Dados indisponíveis no momento.")
    st.stop()

df = df.dropna()

# ==========================
# RETORNOS LOG
# ==========================

df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna()

retornos = df["log_ret"].values[-300:]  # janela estatística

# ==========================
# DETECÇÃO DE REGIME
# ==========================

media_ret = np.mean(retornos)
vol = np.std(retornos)

if abs(media_ret) > vol * 0.3:
    regime = "Tendência"
elif vol > np.percentile(np.abs(retornos), 75):
    regime = "Expansão"
else:
    regime = "Lateral"

# ==========================
# PROBABILIDADE CONDICIONAL
# ==========================

historico = df.copy()

if regime == "Tendência":
    filtro = abs(historico["log_ret"].rolling(10).mean()) > vol * 0.3
elif regime == "Expansão":
    filtro = historico["log_ret"].rolling(10).std() > vol
else:
    filtro = abs(historico["log_ret"].rolling(10).mean()) <= vol * 0.3

historico_regime = historico[filtro]

if len(historico_regime) > 30:
    prob_alta = np.mean(historico_regime["log_ret"].shift(-1) > 0)
else:
    prob_alta = np.mean(historico["log_ret"].shift(-1) > 0)

# ==========================
# MONTE CARLO (BOOTSTRAP)
# ==========================

ultimo_preco = df["Close"].iloc[-1]
caminhos_finais = []
projecoes = []

for _ in range(simulacoes):
    amostra = np.random.choice(retornos, size=horizonte, replace=True)
    caminho = ultimo_preco * np.exp(np.cumsum(amostra))
    projecoes.append(caminho)
    caminhos_finais.append(caminho[-1])

caminhos_finais = np.array(caminhos_finais)

media_final = np.mean(caminhos_finais)
p5 = np.percentile(caminhos_finais, 5)
p95 = np.percentile(caminhos_finais, 95)

retorno_esperado = (media_final / ultimo_preco - 1) * 100
risco = (p5 / ultimo_preco - 1) * 100
potencial = (p95 / ultimo_preco - 1) * 100

# ==========================
# PAINEL SUPERIOR
# ==========================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Regime", regime)
col2.metric("Probabilidade Alta", f"{prob_alta*100:.2f}%")
col3.metric("Retorno Esperado (20 candles)", f"{retorno_esperado:.2f}%")
col4.metric("Assimetria (P95 vs P5)", f"{potencial - risco:.2f}%")

# ==========================
# GRÁFICO
# ==========================

fig = go.Figure()

# Histórico
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Preço"
    )
)

# Projeção média
media_caminho = np.mean(projecoes, axis=0)
future_index = pd.date_range(
    start=df.index[-1],
    periods=horizonte+1,
    freq=df.index.inferred_freq
)[1:]

fig.add_trace(
    go.Scatter(
        x=future_index,
        y=media_caminho,
        mode="lines",
        name="Projeção Média",
        line=dict(width=3)
    )
)

# Banda de risco
fig.add_trace(
    go.Scatter(
        x=future_index,
        y=[p5]*len(future_index),
        mode="lines",
        name="P5 (Risco)",
        line=dict(dash="dash")
    )
)

fig.add_trace(
    go.Scatter(
        x=future_index,
        y=[p95]*len(future_index),
        mode="lines",
        name="P95 (Potencial)",
        line=dict(dash="dash")
    )
)

fig.update_layout(
    template="plotly_dark",
    height=750,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("Radar baseado em Monte Carlo + Probabilidade Condicional Empírica.")
