import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, norm, t
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot
import cufflinks as cf  # Necesario para iplot
from streamlit.components.v1 import html

# Configuración de la página
st.set_page_config(
    page_title="Cálculo de VaR y ES",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
/* Fondo principal */
.stApp {
    background-color: #2C3E50;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #d8e8f2, #c0d8e5) !important;
    color: #1a5276 !important;
}
[data-testid="stSidebar"] .stSelectbox>div>div>select,
[data-testid="stSidebar"] .stTextInput>div>div>input {
    background-color: white !important;
    border: 1px solid #aed6f1 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stButton>button {
    background-color: #2980b9 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}

/* Títulos */
h1 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}

/* Botones */
.stButton>button {
    background-color: #3498db !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.3s !important;
}

.stButton>button:hover {
    background-color: #2980b9 !important;
    transform: scale(1.05);
}

/* Tarjetas de métricas */
[data-testid="metric-container"] {
    background-color: black;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Inputs */
.stTextInput>div>div>input, .stSelectbox>div>div>select {
    border-radius: 8px !important;
    border: 1px solid #bdc3c7 !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Información del proyecto
    st.title("⚙️ Proyecto 1")
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    st.markdown("""

    **👥 Integrantes:**
    - Oziel Hernández
    - Daniela Borzani
    - Santiago Cruz
    - Ximena Paredes
    """)
    st.markdown("---")
    
    # Selector de activo
    stock_seleccionado = st.selectbox(
        "**📊 Descripción del activo a analizar**", 
        ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA','S&P500'],
        index=0
    )
    
    # Mostrar detalles del activo seleccionado
    if stock_seleccionado == 'GOOGL':
        st.markdown(f"### 📝 {stock_seleccionado} se refiere a las Acciones de Google")
    if stock_seleccionado == 'AAPL':
        st.markdown(f"### 📝 {stock_seleccionado} se refiere a las Acciones de Apple")
    if stock_seleccionado == 'MSFT':
        st.markdown(f"### 📝 {stock_seleccionado} se refiere a las Acciones de Microsoft")
    if stock_seleccionado == 'AMZN':
        st.markdown(f"### 📝 {stock_seleccionado} se refiere a las Acciones de Amazon")
    if stock_seleccionado == 'TSLA':
        st.markdown(f"### 📝 {stock_seleccionado} se refiere a las Acciones de Tesla")
    if stock_seleccionado == "S&P500":
            st.markdown("""
El S&P 500 es un índice compuesto por las 500 empresas más grandes que cotizan en bolsa en EE.UU., seleccionadas por capitalización de mercado, liquidez y representatividad sectorial.
""")
            st.markdown("""
            ### 📈 Importancia financiera:
            - Benchmark: Es el principal referente para fondos de inversión y gestores de activos.
            - Indicador económico: Refleja la salud del mercado accionario estadounidense.
            - Derivados: Base para futuros, ETFs (como SPY) y opciones.    
            """)
    st.markdown("---")

# Función para obtener datos
def obtener_datos(stock):
    df = yf.download(stock, start='2010-01-01', end=datetime.today().strftime('%Y-%m-%d'))['Close']
    return df

# Función para calcular rendimientos
def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Selector de acción
#stocks_lista = ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA']
#stock_seleccionado = st.selectbox("📌 Selecciona una acción", stocks_lista)

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

#Damos cierto estilo
st.markdown("""
<h2 style='color: white; text-align: center;'>Proyecto 1 - Métodos Cuantitativos en Finanzas </h2>
""", unsafe_allow_html=True)
# Configuración para cufflinks
cf.go_offline()

#-------------
#GRAFICO DE LOS VALORES HISTORICOS
#-------------

# 1. Generar la figura con iplot 

fig = px.line(df_precios, title='Histórico de precios')

st.write("#### Gráfico de precios históricos")

# 2. Mostrar en Streamlit usando plotly_chart
st.plotly_chart(fig, use_container_width=True)

# Opcional: Añadir controles interactivos
st.sidebar.header('Opciones de visualización')
log_scale = st.sidebar.checkbox('Escala logarítmica', value=False)

if log_scale:
    # Personalizar la figura según las opciones
    if log_scale:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

#--------------------
#FIN DE GRÁFICO DE PRECIOS HISTÓRICOS
#--------------------

#-----------------
#GRAFICO DE HISTOGRAMA
#-----------------
# Configuración del histograma
st.header('Distribución de Retornos Diarios')

# 1. Crear figura con mejoras visuales
fig = go.Figure()

# 2. Añadir histograma principal
fig.add_trace(
    go.Histogram(
        x=df_rendimientos[stock_seleccionado],
        nbinsx=50,  # Ajustar número de bins
        name='Retornos',
        marker_color='#1f77b4',
        opacity=0.75,
        histnorm='probability density'  # Normalizar como densidad
    )
)

# 3. Añadir curva de densidad KDE (opcional)
from scipy.stats import gaussian_kde
kde = gaussian_kde(df_rendimientos[stock_seleccionado])
x_vals = np.linspace(df_rendimientos[stock_seleccionado].min(), df_rendimientos[stock_seleccionado].max(), 100)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=kde(x_vals),
        name='Densidad',
        line=dict(color='#ff7f0e', width=2)
    )
)

# 4. Personalizar layout
fig.update_layout(
    title_text='Distribución de Retornos Diarios (Log Normal)',
    xaxis_title='Retorno',
    yaxis_title='Distribución de Probabilidad',
    bargap=0.01,  # Espacio entre barras
    template='plotly_white',  # o 'plotly_dark' para tema oscuro
    hovermode='x unified'
)

# 5. Añadir línea de media
mean_return = df_rendimientos[stock_seleccionado].mean()
fig.add_vline(
    x=mean_return, 
    line_dash="dash",
    line_color="red",
    annotation_text=f"Media: {mean_return:.4f}", 
    annotation_position="top right"
)

# 6. Mostrar en Streamlit con controles
col1, col2 = st.columns(2)
with col1:
    bin_size = st.slider('Número de bins', 10, 100, 50)
with col2:
    show_kde = st.checkbox('Mostrar curva de densidad', True)

fig.update_traces(nbinsx=bin_size, selector={'type':'histogram'})
if not show_kde:
    fig.for_each_trace(lambda trace: trace.update(visible=False) if trace.name == 'Distribución' else None)

st.plotly_chart(fig, use_container_width=True)

#-----------------
#GRAFICO DE HISTOGRAMA
#-----------------

#Damos cierto estilo
st.markdown("""
<h2 style='color: white; text-align: center;'>📊 Métricas de Rendimientos</h2>
""", unsafe_allow_html=True)

# Columnas con métricas
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📈 Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
    st.markdown("Rendimiento promedio diario a la fecha actual")

with col2:
    st.metric("📊 Kurtosis", f"{kurtosis_valor:.2f}")
    st.markdown("La kurtosis hace referencia a colas pesadas, para valores mayores a 3, y colas ligeras para valores menores a 3")
with col3:
    st.metric("📉 Sesgo", f"{sesgo_valor:.2f}")
    st.markdown("El sesgo hace referencia a si los retornos tienden a ser positivos, para valores negativos del sesgo. Y tienden a ser negativos, para valores positivos del sesgo")

# CSS personalizado para tema oscuro
st.markdown("""
<style>
/* Estilo general de las tarjetas */
[data-testid="stMetric"] {
    background-color: #1E1E1E;
    border: 1px solid #444;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Estilo del valor */
[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 24px;
    font-weight: bold;
}

/* Estilo de la etiqueta */
[data-testid="stMetricLabel"] {
    color: #AAAAAA !important;
    font-size: 16px;
}

/* Estilo del delta (si lo usas) */
[data-testid="stMetricDelta"] {
    color: white !important;
}

/* Iconos más grandes */
.metric-icon {
    font-size: 24px !important;
}

/* Efecto hover */
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    transition: all 0.3s ease;
    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# Estilo adicional con streamlit-extras
style_metric_cards(background_color="#1E1E1E", border_left_color="#666666", border_color="#444444")

# Función para calcular VaR y ES usando t-Student
def parametric_var_es_tstudent(returns, alpha, nu=20):
    mu = np.mean(returns)
    s2 = np.var(returns, ddof=1)  # Varianza muestral insesgada
    sigma2 = ((nu - 2) / nu) * s2
    sigma = np.sqrt(sigma2)

    VaR = t.ppf(1 - alpha, df=nu, loc=mu, scale=sigma)
    
    def safe_quantile_mean(alpha, nu, mu, sigma, num_points=1000):
        cuantiles = t.ppf(np.linspace(1e-6, alpha-1e-6, num_points), df=nu, loc=mu, scale=sigma)
        return np.mean(cuantiles)
    ES = safe_quantile_mean(1 - alpha, nu, mu, sigma)
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
st.write("#### 📉 Value at Risk (VaR)")
st.dataframe(VaR_df.style.format({"VaR Normal": "{:.2%}", "VaR t-Student": "{:.2%}", "VaR Histórico": "{:.2%}", "VaR Monte Carlo": "{:.2%}"}))

st.write("#### ⚠️ Expected Shortfall (ES)")
st.dataframe(ES_df.style.format({"ES Normal": "{:.2%}", "ES t-Student": "{:.2%}", "ES Histórico": "{:.2%}", "ES Monte Carlo": "{:.2%}"}))

### PASAMOS AL EJERCICIO D en adelante 

window = 252
alphas = [0.95, 0.99]

def historical_es(returns, alpha):

    """
    Calcula el Expected Shortfall (ES) histórico.

    """

    var_threshold = returns.quantile(1 - alpha)
    return returns[returns <= var_threshold].mean()

# ------------------------------------------
# 4. ROLLING WINDOW PARA VAR Y ES
# ------------------------------------------

#Inicializamos el DataFrame para guardar resultados
results = pd.DataFrame(index=df_rendimientos.index)
results['returns'] = df_rendimientos[stock_seleccionado]

with st.spinner("Calculando datos para graficar..."):
    for alpha in alphas:
        #Columnas para guardar los resultados
        var_param_col = f'VaR_param_{int(alpha*100)}'
        es_param_col = f'ES_param_{int(alpha*100)}'
        var_hist_col = f'VaR_hist_{int(alpha*100)}'
        es_hist_col = f'ES_hist_{int(alpha*100)}'

            #VaR y ES paramétrico con t-Student
        results[var_param_col] = results['returns'].rolling(window).apply(
            lambda x: parametric_var_es_tstudent(x, alpha)[0], raw=False)
        results[es_param_col] = results['returns'].rolling(window).apply(
            lambda x: parametric_var_es_tstudent(x, alpha)[1], raw=False)

            #VaR histórico (percentil)
        results[var_hist_col] = results['returns'].rolling(window).quantile(1 - alpha)

            #ES histórico (percentil)
        results[es_hist_col] = results['returns'].rolling(window).apply(
            lambda x: historical_es(x, alpha), raw=False)
    window_data = results['returns'].iloc[-252:]
    var95, es95 = parametric_var_es_tstudent(window_data, 0.95, nu=5)
    var99, es99 = parametric_var_es_tstudent(window_data, 0.99, nu=5)

st.write('#### Resultados de VaR y ES paramétrico (t-Student, ν=20)')

# Crear dataframe con los resultados
data = pd.DataFrame({
    'Métrica': ['VaR 95%', 'ES 95%', 'VaR 99%', 'ES 99%'],
    'Valor': [var95, es95, var99, es99],
    'Tipo': ['VaR', 'ES', 'VaR', 'ES']
})

# Mostrar métricas y gráfico
st.write('#### Comparación de VaR y ES, a modo de muestra, de los recientes 252 días')

fig = px.bar(data, x='Métrica', y='Valor', color='Tipo',
             text_auto='.4f', 
             title="VaR vs Expected Shortfall",
             labels={'Valor': 'Valor del riesgo'})
st.plotly_chart(fig, use_container_width=True)

# Mostrar valores exactos
st.dataframe(data.style.format({'Valor': '{:.4f}'}), hide_index=True)

# ------------------------------------------
# 5. VISUALIZACIÓN EN LA GRÁFICA
# ------------------------------------------

st.write("### Gráfico de VaR y ES para los niveles de 0.95 y 0.99")

#Colores por tipo y nivel de confianza
colors = {
    'VaR_param_95': '#d62728',    # rojo oscuro
    'ES_param_95': '#ff9896',     # rojo claro
    'VaR_hist_95': '#1f77b4',     # azul oscuro
    'ES_hist_95': '#aec7e8',      # azul claro
    'VaR_param_99': '#2ca02c',    # verde oscuro
    'ES_param_99': '#98df8a',     # verde claro
    'VaR_hist_99': '#9467bd',     # morado oscuro
    'ES_hist_99': '#c5b0d5'       # morado claro
}

# Configuración de la figura (puedes ajustar el tamaño según necesites)
fig, ax = plt.subplots(figsize=(32, 16))
ax.plot(results['returns'], label='Returns', color='black', alpha=0.4)

# Dibujamos líneas de VaR y ES para cada nivel de confianza
for alpha in alphas:
    a = int(alpha * 100)
    ax.plot(results[f'VaR_param_{a}'], label=f'VaR Paramétrico {a}%', linestyle='--', color=colors[f'VaR_param_{a}'])
    ax.plot(results[f'ES_param_{a}'], label=f'ES Paramétrico {a}%', linestyle='-.', color=colors[f'ES_param_{a}'])
    ax.plot(results[f'VaR_hist_{a}'], label=f'VaR Histórico {a}%', linestyle=':', color=colors[f'VaR_hist_{a}'])
    ax.plot(results[f'ES_hist_{a}'], label=f'ES Histórico {a}%', linestyle='-', linewidth=1.2, color=colors[f'ES_hist_{a}'])

# Configuración del gráfico
ax.set_title(f'Rolling VaR & ES ({stock_seleccionado}) de 252 días', fontsize=20)
ax.set_xlabel('Fecha', fontsize=16)
ax.set_ylabel('Retornos', fontsize=16)
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

# ------------------------------------------
# Ahora calculamos los errores de estimación que hubo
# ------------------------------------------

sum_r=[]

for col in results: #iteramos sobre cada encabezado
  sum = 0
  if col != 'returns': #solo lo hacemos para los VaR y ES
    for i in range(len(results) - 1): #iteramos sobre todos los renglones
      if results[col][i] != np.nan: #iteramos sobre el rango de los no nulos
        if results['returns'][i+1] < results[col][i]: #comparamos si estimó bien o no
          sum += 1
    sum_r.append(sum)


sum_r=np.array(sum_r)#convertimos a arreglo de numpy

porc = (sum_r/len(results))*100 #calculamos el porcentaje de aciertos

estim = ['Buena estimación' if i < 2.5 else 'Mala estimación' for i in porc]

## 1. Crear el DataFrame original (como en tu código)
fallas_1 = pd.DataFrame(
    {
        'VaR_param_95': [porc[0], sum_r[0], estim[0]],
        'ES_param_95': [porc[1], sum_r[1], estim[1]],
        'VaR_hist_95': [porc[2], sum_r[2], estim[2]],
        'ES_hist_95': [porc[3], sum_r[3], estim[3]],
    },
    index=['Porcentaje de fallas', 'Número de fallas', 'Tipo de estimación']
)

fallas_2 = pd.DataFrame(
    {
        'VaR_param_99': [porc[4], sum_r[4], estim[4]],
        'ES_param_99': [porc[5], sum_r[5], estim[5]],
        'VaR_hist_99': [porc[6], sum_r[6], estim[6]],
        'ES_hist_99': [porc[7], sum_r[7], estim[7]],
    },
    index=['Porcentaje de fallas', 'Número de fallas', 'Tipo de estimación']
)

## 2. Transponer y filtrar por nivel de confianza
fallas_transposed_1 = fallas_1.transpose()
fallas_transposed_2 = fallas_2.transpose()

## 3. Crear pestañas con los datos filtrados
tab1, tab2 = st.tabs(["Nivel 95%", "Nivel 99%"])

with tab1:
    st.write("#### Métricas para 95% de confianza")
    st.dataframe(
        fallas_transposed_1[['Porcentaje de fallas', 'Número de fallas', 'Tipo de estimación']]
        .style
        .format("{:.2f}%", subset=['Porcentaje de fallas'])
        .set_properties(**{'background-color': '#1E1E1E', 'color': 'white'}),
        use_container_width=True
    )

with tab2:
    st.write("#### Métricas para 99% de confianza")
    st.dataframe(
        fallas_transposed_2[['Porcentaje de fallas', 'Número de fallas', 'Tipo de estimación']]
        .style
        .format("{:.2f}%", subset=['Porcentaje de fallas'])
        .set_properties(**{'background-color': '#1E1E1E', 'color': 'white'}),
        use_container_width=True
    )


# Mostrar métricas y gráfico
st.write('### Cálculos del VaR Asumiendo distribución normal')

# f. FUNCIÓN VaR NORMAL
# Calculamos el VaR usando una distribución normal
def var_normal(returns, alpha):
    media = returns.mean()
    desviacion = returns.std()
    return norm.ppf(1 - alpha, loc=media, scale=desviacion) 

# Calculamos el VaR para cada nivel de confianza
for alpha in alphas:
    col_name = f'VaR_Norm_{int(alpha*100)}'  # Nombre de la nueva columna
    results[col_name] = results['returns'].rolling(window).apply(
        lambda x: var_normal(x, alpha), raw=False
    )

colors['VaR_Norm_95'] = '#ff008a'
colors['VaR_Norm_99'] = '#6cff3f'

# Gráfica de retornos y sus VaR
fig_1, ax_1 = plt.subplots(figsize=(30, 14))
ax_1.plot(results['returns'], label='Returns', color='black', alpha=0.4)

for alpha in alphas:
    a = int(alpha * 100)
    col_name = f'VaR_Norm_{a}'
    ax_1.plot(results[col_name],
             label=f'VaR Normal Mov.Vol {a}%',
             linestyle='--',
             color=colors[col_name])


# Configuración del gráfico
ax_1.set_title(f'Rolling VaR & ES ({stock_seleccionado}) de 252 días', fontsize=20)
ax_1.set_xlabel('Fecha', fontsize=16)
ax_1.set_ylabel('Retornos', fontsize=16)
ax_1.legend(fontsize=14)
ax_1.grid(True, alpha=0.3)
plt.tight_layout()
# Mostrar el gráfico en Streamlit
st.pyplot(fig_1)

# Contamos las fallas
sum_r_Norm = []
for col in [f'VaR_Norm_{int(a*100)}' for a in alphas]:
    violaciones = 0
    for i in range(len(results) - 1):
        if not np.isnan(results[col].iloc[i]):
            if results['returns'].iloc[i+1] < results[col].iloc[i]:
                violaciones += 1
    sum_r_Norm.append(violaciones)

# evaluamos si fue una buena o mala estimación
porc_Norm = (np.array(sum_r_Norm) / len(results)) * 100
estim_Norm = ['Buena estimación' if x < 2.5 else 'Mala estimación' for x in porc_Norm]


fallas_Norm = pd.DataFrame({
    'VaR_Norm_95': [porc_Norm[0], sum_r_Norm[0], estim_Norm[0]],
    'VaR_Norm_99': [porc_Norm[1], sum_r_Norm[1], estim_Norm[1]]
},
index=['Porcentaje de fallas','Número de fallas','Tipo de estimación'])

fallas_transposed_norm = fallas_Norm.transpose()


st.write("#### Fallas resgistradas para el caso de la normal")
st.dataframe(
        fallas_transposed_norm[['Porcentaje de fallas','Número de fallas','Tipo de estimación']]
        .style
        .format("{:.2f}%", subset=['Porcentaje de fallas'])
        .set_properties(**{'background-color': '#1E1E1E', 'color': 'white'}),
        use_container_width=True
    )

# Footer personalizado
html("""
<div style="text-align: center; padding: 20px; margin-top: 30px; background-color: #4A4A4A; color: white; border-radius: 10px;">
    <p> © 2025 Métodos Cuantitativos en Finanzas | Proyecto 1 </p>
</div>
""")