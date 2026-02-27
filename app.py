import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURAÇÃO DE INTERFACE ULTRA-CLEAN ---
st.set_page_config(page_title="AURAXIS V7", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;700&display=swap');
    :root { --bg: #020617; --card: #0f172a; --accent: #3b82f6; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: var(--bg); color: #f8fafc; }
    .stApp { background: var(--bg); }
    .card { 
        background: var(--card); border: 1px solid #1e293b; border-radius: 16px; 
        padding: 20px; margin-bottom: 20px; border-left: 5px solid #334155;
    }
    .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.65rem; font-weight: bold; font-family: 'JetBrains Mono'; }
    .scalper { background: #450a0a; color: #f87171; border-left: 5px solid #f87171 !important; }
    .daytrade { background: #1e3a8a; color: #60a5fa; border-left: 5px solid #60a5fa !important; }
    .swing { background: #14532d; color: #4ade80; border-left: 5px solid #4ade80 !important; }
    .position { background: #3b0764; color: #c084fc; border-left: 5px solid #c084fc !important; }
    .zone-entry { background: rgba(0,0,0,0.3); border: 1px dashed #334155; padding: 12px; border-radius: 8px; margin: 15px 0; }
    .metric { font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #94a3b8; }
    </style>
""", unsafe_allow_html=True)

# --- ENGINE DE INTELIGÊNCIA ---
@st.cache_data(ttl=30)
def process_auraxis_engine(symbol):
    try:
        # Fetch Data - Diferentes janelas para diferentes horizontes
        df_long = yf.download(symbol, period="100d", interval="1h", progress=False)
        df_short = yf.download(symbol, period="5d", interval="5m", progress=False)
        
        if df_long.empty or df_short.empty: return None

        price = df_short['Close'].iloc[-1]
        
        # Definição dos Horizontes e Parâmetros Adaptativos
        horizontes = [
            {"id": "Scalper Agressivo", "css": "scalper", "df": df_short, "ema": 9, "atr_m": 1.2, "tp_m": 1.5},
            {"id": "Daytrader Moderado", "css": "daytrade", "df": df_short, "ema": 21, "atr_m": 1.5, "tp_m": 2.5},
            {"id": "Swing Trader Conservador", "css": "swing", "df": df_long, "ema": 50, "atr_m": 2.0, "tp_m": 4.5},
            {"id": "Position Ultraconservador", "css": "position", "df": df_long, "ema": 200, "atr_m": 3.0, "tp_m": 8.0}
        ]

        results = []
        for h in horizontes:
            df = h['df']
            # Média Institucional e Volatilidade
            ema_inst = df['Close'].ewm(span=h['ema']).mean().iloc[-1]
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            
            # Score Bayesiano (Probabilidade de continuidade)
            returns = df['Close'].pct_change().dropna()
            z_score = (returns.iloc[-1] - returns.mean()) / returns.std()
            prob = (1 / (1 + np.exp(-z_score))) * 100
            
            # Lógica de Direção e Zona
            sinal = "COMPRA" if price > ema_inst else "VENDA"
            
            # Zona de Entrada Adaptativa (Persistência)
            # Se o preço foge, a zona "caça" o preço ou aguarda o retorno à média
            base_entrada = price if abs(price - ema_inst) < (atr * 2) else ema_inst
            z_top = base_entrada + (atr * 0.2)
            z_bot = base_entrada - (atr * h['atr_m'] * 0.5) if sinal == "COMPRA" else base_entrada + (atr * h['atr_m'] * 0.5)
            
            # Gerenciamento de Risco
            sl = z_bot - (atr * 0.5) if sinal == "COMPRA" else z_bot + (atr * 0.5)
            tp = base_entrada + (atr * h['tp_m']) if sinal == "COMPRA" else base_entrada - (atr * h['tp_m'])
            
            # Filtro de Confiança: Remove sinal se a probabilidade for contra a tendência
            status = "SINAL ATIVO"
            if (sinal == "COMPRA" and prob < 40) or (sinal == "VENDA" and prob > 60):
                status = "AGUARDANDO CONFLUÊNCIA"
                sinal = "NEUTRO"

            results.append({
                "label": h['id'], "css": h['css'], "sinal": sinal, "status": status,
                "z1": min(z_top, z_bot), "z2": max(z_top, z_bot),
                "sl": sl, "tp": tp, "prob": prob
            })
            
        return {"ativo": symbol, "preço": price, "data": results}
    except: return None

# --- UI RENDERER ---
def main():
    st.markdown("<h1 style='text-align:center; color:#3b82f6;'>AURAXIS <span style='color:#fff;'>V7 SOVEREIGN</span></h1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Comandos")
    ativos_raw = st.sidebar.text_input("Ativos", "EURUSD=X, XAUUSD=X, BTC-USD, ETH-USD, US30")
    banca = st.sidebar.number_input("Banca ($)", value=1000)
    
    lista_ativos = [a.strip().upper() for a in ativos_raw.split(",")]

    for ativo in lista_ativos:
        res = process_auraxis_engine(ativo)
        if not res: continue

        st.markdown(f"#### {ativo} <small style='color:#64748b; font-size:0.8rem;'>• Preço: {res['preço']:.5f}</small>", unsafe_allow_html=True)
        
        for s in res['data']:
            # Lógica de cor dinâmica
            color = "#10b981" if s['sinal'] == "COMPRA" else "#ef4444" if s['sinal'] == "VENDA" else "#94a3b8"
            
            st.markdown(f"""
                <div class="card {s['css'] if s['sinal'] != 'NEUTRO' else ''}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="badge {s['css']}">{s['label']}</span>
                        <b style="color:{color};">{s['sinal']}</b>
                    </div>
                    <div style="margin-top:10px;">
                        <small class="metric">{s['status']}</small>
                    </div>
                    {f'''
                    <div class="zone-entry">
                        <small class="metric">ZONA DE ENTRADA (LIQUIDEZ)</small><br>
                        <b style="font-size:1.1rem; color:#f8fafc;">{s['z1']:.5f} — {s['z2']:.5f}</b>
                    </div>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px;">
                        <div>
                            <small class="metric">STOP LOSS</small><br>
                            <b style="color:#ef4444;">{s['sl']:.5f}</b>
                        </div>
                        <div>
                            <small class="metric">TAKE PROFIT</small><br>
                            <b style="color:#10b981;">{s['tp']:.5f}</b>
                        </div>
                    </div>
                    <div style="margin-top:10px; display:flex; justify-content:space-between;">
                        <span class="metric">Confiança: {s['prob']:.1f}%</span>
                        <span class="metric">Lote Sugerido: {((banca*0.01)/abs(res['preço']-s['sl'] if res['preço']!=s['sl'] else 1)):.2f}</span>
                    </div>
                    ''' if s['sinal'] != "NEUTRO" else ""}
                </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
