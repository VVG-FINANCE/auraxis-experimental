import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------
# Configurações da Página
# -------------------------
st.set_page_config(page_title="Auraxis Radar MultiTF", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>🌐 Auraxis — Radar Institucional Multi-Timeframe</h1>", unsafe_allow_html=True)

# -------------------------
# Configurações do Usuário
# -------------------------
st.sidebar.header("Configurações do Radar")
ativo = st.sidebar.selectbox("Ativo:", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"])
timeframes = st.sidebar.multiselect("Timeframes:", ["1m", "5m", "15m", "30m", "1h"])
num_simulacoes = st.sidebar.slider("Número de Simulações (Monte Carlo):", min_value=100, max_value=3000, value=500)

# -------------------------
# Funções de Cálculo
# -------------------------
def fetch_data(symbol, period="7d", interval="1m"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.rename(columns={'Date':'Datetime'}, inplace=True) if 'Date' in df.columns else None
    df = df[['Datetime','Open','High','Low','Close']].dropna()
    return df

def gerar_sinais(df):
    """Exemplo de radar institucional simplificado"""
    sinais = []
    for i in range(len(df)):
        if i==0:
            sinais.append("Lateral")
        else:
            if df['Close'][i] > df['Close'][i-1]:
                sinais.append("Compra")
            elif df['Close'][i] < df['Close'][i-1]:
                sinais.append("Venda")
            else:
                sinais.append("Lateral")
    df['Sinal'] = sinais
    return df

def monte_carlo_sim(df, n_sim=500):
    """Simulação simplificada de Monte Carlo para análise de risco"""
    ult_precos = df['Close'].values[-1]
    retornos = df['Close'].pct_change().dropna()
    resultados = []
    for _ in range(n_sim):
        amostra = np.random.choice(retornos, size=len(retornos))
        simulacao = ult_precos * np.exp(np.cumsum(amostra))
        resultados.append(simulacao[-1])
    return resultados

# -------------------------
# Radar Multi-Timeframe
# -------------------------
st.subheader(f"Radar Institucional — {ativo}")
for tf in timeframes:
    st.markdown(f"### Timeframe: {tf}")
    df = fetch_data(ativo, period="7d", interval=tf)
    if df.empty:
        st.warning("Não foi possível carregar dados para este timeframe.")
        continue

    df = gerar_sinais(df)
    st.dataframe(df.tail(10))  # últimos 10 candles + sinal

    sim_result = monte_carlo_sim(df, num_simulacoes)
    st.markdown(f"**Simulação Monte Carlo:** Preço médio futuro estimado: {np.mean(sim_result):.5f}")
