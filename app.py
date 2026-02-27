# ==========================
# Auraxis Radar MultiTF - App-Like Dinâmico
# ==========================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

# -------------------------
# Configurações da Página
# -------------------------
st.set_page_config(page_title="Auraxis Radar Dinâmico", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>🌐 Auraxis — Radar Institucional Dinâmico</h1>", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações do Radar")
ativos_disponiveis = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"]
ativo = st.sidebar.selectbox("Escolha o Ativo:", ativos_disponiveis)
timeframes = ["1m", "5m", "15m"]
selected_timeframes = st.sidebar.multiselect("Timeframes:", timeframes, default=timeframes)
num_simulacoes = st.sidebar.slider("Número de Simulações", 100, 3000, 500)
perfil_trader = st.sidebar.selectbox("Perfil de Trader:", ["Ultra Conservador","Conservador","Moderado","Agressivo","Automático"])
atualizacao_segundos = st.sidebar.slider("Atualização Automática (segundos)", 5, 60, 15)
leverage_guide = {"Ultra Conservador":"1:10","Conservador":"1:20","Moderado":"1:50","Agressivo":"1:100","Automático":"Adaptável"}
st.sidebar.markdown(f"**Alavancagem:** {leverage_guide[perfil_trader]}")

# -------------------------
# Funções
# -------------------------
def fetch_data(symbol, period="7d", interval="1m"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if 'Date' in df.columns:
        df.rename(columns={'Date':'Datetime'}, inplace=True)
    df = df[['Datetime','Open','High','Low','Close']].dropna()
    return df

def compute_confidence(df):
    body = abs(df['Close'] - df['Open'])
    total_range = df['High'] - df['Low']
    total_range = total_range.replace(0,np.nan)
    conf = body/total_range
    conf = conf.fillna(0.3)
    return np.minimum(0.85, conf*0.85+0.15)

def compute_regime(df):
    diff = df['Close'] - df['Open']
    regime = np.where(abs(diff)<0.0005,"NEUTRAL",np.where(diff>0,"BUY","SELL"))
    return regime

def simulate_SL_TP(candle, profile):
    perc_map = {"Ultra Conservador":0.001,"Conservador":0.002,"Moderado":0.004,"Agressivo":0.008,"Automático":0.005}
    pct = perc_map.get(profile,0.004)
    sl = candle['Close'] - pct*candle['Close']
    tp = candle['Close'] + pct*candle['Close']
    return round(sl,5), round(tp,5)

def gerar_sinais(df, profile):
    sinais = []
    confs = compute_confidence(df)
    regimes = compute_regime(df)
    for i,row in df.iterrows():
        conf = confs.iloc[i]
        regime = regimes[i]
        sl,tp = simulate_SL_TP(row, profile)
        if conf>0.7:
            emoji="🔥"
        elif conf>0.5:
            emoji="⚡"
        else:
            emoji="🌫"
        sinais.append({"Horário":row['Datetime'],"Sinal":regime,"Confiança":round(conf,2),"SL":sl,"TP":tp,"Emoji":emoji})
    return pd.DataFrame(sinais)

def color_card(sinal):
    return "#b6fcb6" if sinal=="BUY" else "#fcb6b6" if sinal=="SELL" else "#d3d3d3"

# -------------------------
# Loop Dinâmico (Atualização Automática)
# -------------------------
placeholder = st.empty()
while True:
    with placeholder.container():
        st.subheader("⚡ Radar de Sinais — Feed Vertical")
        for tf in selected_timeframes:
            st.markdown(f"### ⏱ Timeframe: {tf}")
            df_tf = fetch_data(ativo, interval=tf)
            if df_tf.empty:
                st.warning(f"Sem dados para {tf}")
                continue
            sinais_df = gerar_sinais(df_tf, perfil_trader)
            sinais_df = sinais_df.tail(10).sort_values(by="Horário", ascending=False)  # últimos 10 sinais

            for i,row in sinais_df.iterrows():
                cor = color_card(row["Sinal"])
                st.markdown(
                    f"""
                    <div style='background-color:{cor}; padding:12px; border-radius:10px; margin-bottom:6px; transition: background-color 0.5s ease;'>
                    <b>{row['Sinal']} {row['Emoji']}</b><br>
                    Horário: {row['Horário'].strftime("%H:%M")}<br>
                    Confiança: {row['Confiança']}<br>
                    SL: {row['SL']} | TP: {row['TP']}
                    </div>
                    """, unsafe_allow_html=True
                )

        # -------------------------
        # Resumo Consolidado
        # -------------------------
        st.subheader("📊 Resumo Consolidado MultiTF")
        counts = sinais_df['Sinal'].value_counts().to_dict()
        col1,col2,col3 = st.columns(3)
        col1.metric("BUY",counts.get("BUY",0))
        col2.metric("SELL",counts.get("SELL",0))
        col3.metric("NEUTRAL",counts.get("NEUTRAL",0))
        
    time.sleep(atualizacao_segundos)
