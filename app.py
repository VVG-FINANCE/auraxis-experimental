import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =====================================
# CONFIGURAÇÃO INICIAL
# =====================================

st.set_page_config(
    page_title="Auraxis Radar Institucional",
    layout="wide"
)

st.markdown("""
<style>
.metric-box {
    padding:15px;
    border-radius:12px;
    background-color:#111827;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.title("🌌 Auraxis — Radar Institucional")

# =====================================
# SIDEBAR
# =====================================

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

horizonte = 20  # decisão arquitetural fixa

# =====================================
# PERÍODO ESTÁVEL
# =====================================

def periodo_por_intervalo(tf):
    if tf == "15m":
        return "30d"
    if tf == "1h":
        return "60d"
    if tf == "4h":
        return "180d"
    if tf == "1d":
        return "1y"

# =====================================
# DOWNLOAD DADOS
# =====================================

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

# =====================================
# RETORNOS LOG
# =====================================

df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna()

retornos = df["log_ret"].dropna().values[-300:].astype(float)

if len(retornos) < 50:
    st.error("Dados insuficientes para simulação.")
    st.stop()

# =====================================
# DETECÇÃO DE REGIME
# =====================================

media_ret = float(np.mean(retornos))
vol = float(np.std(retornos))

if abs(media_ret) > vol * 0.3:
    regime = "Tendência"
elif vol > np.percentile(np.abs(retornos), 75):
    regime = "Expansão"
else:
    regime = "Lateral"

# =====================================
# PROBABILIDADE CONDICIONAL EMPÍRICA
# =====================================

historico = df.copy()

rolling_mean = historico["log_ret"].rolling(10).mean()
rolling_std = historico["log_ret"].rolling(10).std()

if regime == "Tendência":
    filtro = abs(rolling_mean) > vol * 0.3
elif regime == "Expansão":
    filtro = rolling_std > vol
else:
    filtro = abs(rolling_mean) <= vol * 0.3

historico_regime = historico[filtro]

if len(historico_regime) > 30:
    prob_alta = float(np.mean(historico_regime["log_ret"].shift(-1) > 0))
else:
    prob_alta = float(np.mean(historico["log_ret"].shift(-1) > 0))

# =====================================
# MONTE CARLO BOOTSTRAP
# =====================================

ultimo_preco = float(df["Close"].iloc[-1])

caminhos_finais = []
projecoes = []

for _ in range(simulacoes):
    amostra = np.random.choice(retornos, size=horizonte, replace=True).astype(float)
    cumul = np.cumsum(amostra)
    caminho = ultimo_preco * np.exp(cumul)
    projecoes.append(caminho)
    caminhos_finais.append(caminho[-1])

caminhos_finais = np.array(caminhos_finais)

media_final = float(np.mean(caminhos_finais))
p5 = float(np.percentile(caminhos_finais, 5))
p95 = float(np.percentile(caminhos_finais, 95))

retorno_esperado = (media_final / ultimo_preco - 1) * 100
risco = (p5 / ultimo_preco - 1) * 100
potencial = (p95 / ultimo_preco - 1) * 100
assimetria = potencial - risco

# =====================================
# PAINEL SUPERIOR
# =====================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Regime", regime)
col2.metric("Probabilidade Alta", f"{prob_alta*100:.2f}%")
col3.metric("Retorno Esperado (20 candles)", f"{retorno_esperado:.2f}%")
col4.metric("Assimetria Estatística", f"{assimetria:.2f}%")

# =====================================
# GRÁFICO
# =====================================

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

# Índice futuro seguro
freq = df.index.inferred_freq
if freq is None:
    freq = "D"

future_index = pd.date_range(
    start=df.index[-1],
    periods=horizonte + 1,
    freq=freq
)[1:]

media_caminho = np.mean(projecoes, axis=0)

# Projeção média
fig.add_trace(
    go.Scatter(
        x=future_index,
        y=media_caminho,
        mode="lines",
        name="Projeção Média",
        line=dict(width=3)
    )
)

# Banda inferior
fig.add_trace(
    go.Scatter(
        x=future_index,
        y=[p5] * len(future_index),
        mode="lines",
        name="P5 (Risco)",
        line=dict(dash="dash")
    )
)

# Banda superior
fig.add_trace(
    go.Scatter(
        x=future_index,
        y=[p95] * len(future_index),
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
st.markdown("Radar baseado em Monte Carlo (Bootstrap) + Probabilidade Condicional Empírica.")
