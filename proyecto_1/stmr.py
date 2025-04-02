import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, norm, t
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards

# Configuración de página
st.set_page_config(page_title="Análisis de Acciones", page_icon="📈", layout="wide")
st.title("📊 Visualización de Rendimientos de Acciones")
st.markdown("---")

# Función para obtener datos
def obtener_datos(stock):
    df = yf.download(stock, start='2010-01-01', end=datetime.today().strftime('%Y-%m-%d'))['Close']
    return df

# Función para calcular rendimientos
def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Selector de acción
stocks_lista = ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA']
stock_seleccionado = st.selectbox("📌 Selecciona una acción", stocks_lista)

# Descarga y cálculo de datos
with st.spinner("Descargando datos..."):
    df_precios = obtener_datos(stock_seleccionado)
    df_rendimientos = calcular_rendimientos(df_precios)

# Cálculo de métricas del activo
rendimiento_medio = df_rendimientos[stock_seleccionado].mean()
kurtosis_valor = kurtosis(df_rendimientos[stock_seleccionado])
sesgo_valor = skew(df_rendimientos[stock_seleccionado])
stdev = df_rendimientos[stock_seleccionado].std()
n = len(df_rendimientos[stock_seleccionado])

# Métricas visuales
col1, col2, col3 = st.columns(3)
col1.metric("📈 Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
col2.metric("📊 Kurtosis",f"{kurtosis_valor:.2f}")
col3.metric("📉 Sesgo", f"{sesgo_valor:.2f}")
style_metric_cards()


# Función para calcular VaR y ES usando t-Student
def parametric_var_es_tstudent(returns, alpha, nu=5):
    mu = np.mean(returns)
    s2 = np.var(returns, ddof=1)  # Varianza muestral insesgada
    sigma2 = ((nu - 2) / nu) * s2
    sigma = np.sqrt(sigma2)

    VaR = t.ppf(1 - alpha, df=nu, loc=mu, scale=sigma)
    ES = t.expect(lambda x: x, args=(nu,), loc=mu, scale=sigma, lb=-np.inf, ub=VaR)
    return VaR, ES


# Cálculo de VaR y ES
intervalos_confianza = [0.95, 0.975, 0.99]
VaR_results = []
ES_results = []

for alpha in intervalos_confianza:
    # Cálculo de la media y desviación estándar
    mean = np.mean(df_rendimientos[stock_seleccionado])
    stdev = np.std(df_rendimientos[stock_seleccionado])

    # VaR y ES paramétrico Normal
    VaR_norm = norm.ppf(1 - alpha, mean, stdev)
    ES_norm = mean - (stdev * norm.pdf(norm.ppf(1 - alpha)) / (1 - alpha))

    # VaR y ES usando t-Student
    VaR_t, ES_t = parametric_var_es_tstudent(df_rendimientos[stock_seleccionado], alpha)

    # VaR y ES Histórico
    VaR_hist = df_rendimientos[stock_seleccionado].quantile(1 - alpha)
    ES_hist = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_hist].mean()

    # VaR y ES Monte Carlo
    sim_returns = np.random.normal(mean, stdev, 100000)
    VaR_mc = np.percentile(sim_returns, (1 - alpha) * 100)
    ES_mc = sim_returns[sim_returns <= VaR_mc].mean()

    VaR_results.append([VaR_norm, VaR_t, VaR_hist, VaR_mc])
    ES_results.append([ES_norm, ES_t, ES_hist, ES_mc])

# Mostrar tabla de resultados
st.subheader("📊 Resultados de VaR y Expected Shortfall")
VaR_df = pd.DataFrame(VaR_results, columns=[ "VaR Normal", "VaR t-Student", "VaR Histórico", "VaR Monte Carlo"],index=['0.95', '0.975', '0.99'])
ES_df = pd.DataFrame(ES_results, columns=["ES Normal", "ES t-Student", "ES Histórico", "ES Monte Carlo"],index=['0.95', '0.975', '0.99'])
VaR_df.index.name = 'Confianza'
ES_df.index.name = 'Confianza'
#
st.write("### 📉 Value at Risk (VaR)")
st.dataframe(VaR_df.style.format({"VaR Normal": "{:.2%}", "VaR t-Student": "{:.2%}", "VaR Histórico": "{:.2%}", "VaR Monte Carlo": "{:.2%}"}))

st.write("### ⚠️ Expected Shortfall (ES)")
st.dataframe(ES_df.style.format({"ES Normal": "{:.2%}", "ES t-Student": "{:.2%}", "ES Histórico": "{:.2%}", "ES Monte Carlo": "{:.2%}"}))
