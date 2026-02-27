import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from streamlit_lightweight_charts import renderLightweightCharts

st.set_page_config(page_title="Auraxis RIA", layout="wide")
st.title("🌌 Auraxis — Radar Institucional Adaptativo")

# ==========================
# CONFIGURAÇÕES
# ==========================

st.sidebar.header("Configurações")

ativo = st.sidebar.selectbox(
    "Ativo",
    ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "XAUUSD=X"]
)

simulacoes = st.sidebar.slider("Simulações Monte Carlo", 500, 3000, 1000, step=250)

horizonte = 20

timeframes = {
    "15m": ("30d", 1),
    "1h": ("60d", 2),
    "4h": ("180d", 3),
    "1d": ("1y", 4),
}

# ==========================
# FUNÇÃO DE ANÁLISE
# ==========================

def analisar_timeframe(tf, periodo, peso):
    df = yf.download(ativo, period=periodo, interval=tf, progress=False)

    if df.empty:
        return None

    df = df.dropna()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()

    retornos = df["log_ret"].values[-300:]
    if len(retornos) < 50:
        return None

    media_ret = np.mean(retornos)
    vol = np.std(retornos)

    if abs(media_ret) > vol * 0.3:
        regime = "Tendência"
    elif vol > np.percentile(np.abs(retornos), 75):
        regime = "Expansão"
    else:
        regime = "Lateral"

    prob_alta = np.mean(df["log_ret"].shift(-1) > 0)

    ultimo_preco = float(df["Close"].iloc[-1])

    finais = []
    for _ in range(simulacoes):
        amostra = np.random.choice(retornos, size=horizonte, replace=True)
        cumul = np.cumsum(amostra)
        caminho = ultimo_preco * np.exp(cumul)
        finais.append(caminho[-1])

    finais = np.array(finais)

    retorno_esperado = (np.mean(finais) / ultimo_preco - 1) * 100
    p5 = (np.percentile(finais, 5) / ultimo_preco - 1) * 100
    p95 = (np.percentile(finais, 95) / ultimo_preco - 1) * 100
    assimetria = p95 - p5

    direcao = 1 if retorno_esperado > 0 else -1

    score = 0

    score += min(abs(retorno_esperado) * 5, 30)
    score += min(assimetria * 2, 30)
    score += prob_alta * 40

    score = min(score, 100)

    return {
        "timeframe": tf,
        "peso": peso,
        "score": score,
        "direcao": direcao,
        "prob": prob_alta,
        "ret_esp": retorno_esperado,
        "assimetria": assimetria,
        "df": df
    }

# ==========================
# EXECUÇÃO MULTI-TF
# ==========================

resultados = []

for tf, (periodo, peso) in timeframes.items():
    r = analisar_timeframe(tf, periodo, peso)
    if r:
        resultados.append(r)

if not resultados:
    st.error("Sem dados suficientes.")
    st.stop()

# ==========================
# SCORE GLOBAL PONDERADO
# ==========================

peso_total = sum(r["peso"] for r in resultados)
score_global = sum(r["score"] * r["peso"] for r in resultados) / peso_total

direcao_global = np.sign(sum(r["direcao"] * r["peso"] for r in resultados))

# ==========================
# CLASSIFICAÇÃO
# ==========================

if score_global >= 85:
    classificacao = "🔥 Oportunidade Forte"
elif score_global >= 75:
    classificacao = "🚀 Oportunidade"
elif score_global >= 60:
    classificacao = "⚠ Preparação"
else:
    classificacao = "Neutro"

# ==========================
# PAINEL
# ==========================

col1, col2, col3 = st.columns(3)

col1.metric("Score Global", f"{score_global:.1f}")
col2.metric("Classificação", classificacao)
col3.metric("Direção Dominante", "Alta" if direcao_global > 0 else "Baixa")

st.markdown("### Scores por Timeframe")

for r in resultados:
    st.write(
        f"{r['timeframe']} | Score: {r['score']:.1f} | "
        f"Prob: {r['prob']*100:.1f}% | "
        f"Retorno Esp: {r['ret_esp']:.2f}% | "
        f"Assimetria: {r['assimetria']:.2f}%"
    )

# ==========================
# GRÁFICO (Timeframe Principal 1h)
# ==========================

principal = next((r for r in resultados if r["timeframe"] == "1h"), resultados[0])
df_chart = principal["df"].reset_index()

candles = []

for _, row in df_chart.iterrows():
    time_value = row[df_chart.columns[0]]
    candles.append({
        "time": pd.to_datetime(time_value).strftime("%Y-%m-%dT%H:%M:%S"),
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(row["Close"]),
    })

chart_config = [{
    "chart": {
        "layout": {
            "background": {"type": "solid", "color": "#000000"},
            "textColor": "#d1d4dc",
        },
        "grid": {
            "vertLines": {"color": "#1c1c1c"},
            "horzLines": {"color": "#1c1c1c"},
        },
        "height": 650,
    },
    "series": [
        {
            "type": "Candlestick",
            "data": candles,
            "options": {
                "upColor": "#26a69a",
                "downColor": "#ef5350",
                "borderUpColor": "#26a69a",
                "borderDownColor": "#ef5350",
                "wickUpColor": "#26a69a",
                "wickDownColor": "#ef5350",
            }
        }
    ]
}]

renderLightweightCharts(chart_config)

st.markdown("---")
st.markdown("Sistema baseado em Consenso Ponderado + Monte Carlo + Probabilidade Condicional.")
