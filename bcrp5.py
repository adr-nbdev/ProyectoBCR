import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import warnings
import requests 
import time

# Ignorar advertencias menores de pandas
warnings.filterwarnings('ignore')

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(layout="wide", page_title="Simulador BCRP - TradingView Style", page_icon="🏦")

# 2. FUNCIÓN DE DESCARGA DE DATOS (CON CACHÉ Y TEMPORALIDAD)
@st.cache_data(ttl=43200) 
def cargar_datos_mercado(temporalidad):
    # Sistema de reintentos: Intentará 3 veces si Yahoo rechaza la conexión
    for intento in range(3):
        try:
            # TRUCO NINJA: Disfrazamos la petición para no ser bloqueados
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            
            ticker = yf.Ticker("USDPEN=X", session=session)
            
            # Selector de temporalidad
            if temporalidad == "1 Hora (Últimos 2 años)":
                df = ticker.history(period="700d", interval="1h")
            else:
                df = ticker.history(start="2015-01-01", interval="1d")
            
            if not df.empty:
                df = pd.DataFrame(df['Close'])
                df.columns = ['Precio_Cierre']
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df['Tipo'] = 'Datos Reales'
                return df.dropna()
                
        except Exception as e:
            if intento < 2:
                time.sleep(2) # Espera 2 segundos antes del siguiente intento
            else:
                st.error(f"Error de conexión con el mercado tras 3 intentos: {e}")
                
    return pd.DataFrame()

# 3. BARRA LATERAL (CONTROLES DEL USUARIO)
st.sidebar.title("⚙️ Panel de Control")

st.sidebar.markdown("### ⏱️ Temporalidad")
temporalidad = st.sidebar.selectbox("Resolución del Gráfico", ["Diario (Desde 2015)", "1 Hora (Últimos 2 años)"], 
                                    help="Yahoo Finance limita los datos por horas a los últimos 730 días.")

st.sidebar.markdown("### 📊 Parámetros Estadísticos")
st.sidebar.markdown("Se puede ajustar la sensibilidad al alza, a la baja y el periodo de días")

k_ventas = st.sidebar.slider("Sensibilidad Venta (Techo)", min_value=0.5, max_value=4.0, value=2.0, step=0.1, 
                                help="Factor k para subidas. Valores bajos disparan alertas rojas rápido.")
k_compras = st.sidebar.slider("Sensibilidad Compra (Piso)", min_value=0.5, max_value=4.0, value=3.0, step=0.1, 
                                help="Factor k para caídas. Valores altos hacen que el modelo ignore caídas leves.")
dias_ventana = st.sidebar.slider("Ventana Móvil (Periodos)", min_value=10, max_value=100, value=30, step=5,
                                 help="Periodo de tiempo para calcular el 'ruido' del mercado.")

st.sidebar.markdown("### Filtro")
filtro_ruido = st.sidebar.number_input("Movimiento mínimo permitido (S/)", min_value=0.001, max_value=0.050, value=0.008, format="%0.3f",
                                       help="Evita intervenciones en momentos muertos. Si el mercado varía menos que esto, el modelo lo ignora.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔮 Predicción y Escenarios")
st.sidebar.markdown("Modifica los precios de los próximos 10 periodos:")

# Cargar la historia dependiendo de la temporalidad elegida
df_hist = cargar_datos_mercado(temporalidad)

if not df_hist.empty:
    ultimo_precio = float(df_hist['Precio_Cierre'].iloc[-1])
    ultima_fecha = df_hist.index[-1]
    
    # Crear fechas futuras dependiendo de la temporalidad
    if temporalidad == "Diario (Desde 2015)":
        fechas_futuras = pd.bdate_range(start=ultima_fecha + datetime.timedelta(days=1), periods=10)
        formato_fecha = '%Y-%m-%d'
        texto_periodo = "día"
    else:
        fechas_futuras = pd.date_range(start=ultima_fecha + datetime.timedelta(hours=1), periods=10, freq='h')
        formato_fecha = '%Y-%m-%d %H:00'
        texto_periodo = "hora"
    
    # Crear tabla editable para el usuario
    df_input = pd.DataFrame({
        'Precio Proyectado (S/)': [round(ultimo_precio, 4)] * 10
    }, index=fechas_futuras.strftime(formato_fecha))
    
    df_editado = st.sidebar.data_editor(df_input, use_container_width=True)

    # 4. PROCESAMIENTO MATEMÁTICO DEL MODELO
    df_futuro = pd.DataFrame({
        'Precio_Cierre': df_editado['Precio Proyectado (S/)'].values,
        'Tipo': 'Simulación Futura'
    }, index=pd.to_datetime(df_editado.index))
    
    df_modelo = pd.concat([df_hist, df_futuro])
    
    # APLICACIÓN DE LAS FÓRMULAS
    df_modelo['Velocidad'] = df_modelo['Precio_Cierre'].diff()
    df_modelo['Media_Movil'] = df_modelo['Velocidad'].rolling(window=dias_ventana).mean()
    df_modelo['Desviacion_Movil'] = df_modelo['Velocidad'].rolling(window=dias_ventana).std()
    
    # Bandas Asimétricas + Filtro Antirruido
    limite_alza = np.maximum(k_ventas * df_modelo['Desviacion_Movil'], filtro_ruido)
    limite_baja = np.maximum(k_compras * df_modelo['Desviacion_Movil'], filtro_ruido)
    
    df_modelo['Banda_Alta'] = df_modelo['Media_Movil'] + limite_alza
    df_modelo['Banda_Baja'] = df_modelo['Media_Movil'] - limite_baja
    
    # Traducción a Bandas de Bollinger (Panel de Precio)
    df_modelo['Precio_Banda_Alta'] = df_modelo['Precio_Cierre'].shift(1) + df_modelo['Banda_Alta']
    df_modelo['Precio_Banda_Baja'] = df_modelo['Precio_Cierre'].shift(1) + df_modelo['Banda_Baja']

    # Lógica de Intervención (1 = Vender, -1 = Comprar)
    df_modelo['Alerta'] = np.where(df_modelo['Velocidad'] > df_modelo['Banda_Alta'], 1,
                          np.where(df_modelo['Velocidad'] < df_modelo['Banda_Baja'], -1, 0))

    df_limpio = df_modelo.dropna().copy()
    hist_limpio = df_limpio[df_limpio['Tipo'] == 'Datos Reales']
    
    # 5. DISEÑO DE LA INTERFAZ PRINCIPAL
    st.title("📈 Simulador Dinámico de Intervención Cambiaria")
    st.markdown("Analiza la historia e inyecta escenarios ficticios para evaluar la reacción algorítmica.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cotización Actual", f"S/ {ultimo_precio:.3f}")
    c2.metric("Tolerancia Alza (Límite)", f"+ S/ {hist_limpio['Banda_Alta'].iloc[-1]:.3f} / {texto_periodo}")
    c3.metric("Tolerancia Baja (Límite)", f"- S/ {abs(hist_limpio['Banda_Baja'].iloc[-1]):.3f} / {texto_periodo}")
    
    alertas_activas = abs(df_modelo[df_modelo['Tipo'] == 'Simulación Futura']['Alerta']).sum()
    if alertas_activas > 0:
        c4.metric("Diagnóstico del Escenario", "⚠️ ALERTA", delta_color="inverse")
    else:
        c4.metric("Diagnóstico del Escenario", "✅ ESTABLE", delta="Dentro del rango")

    # 6. CONSTRUCCIÓN DEL GRÁFICO (ESTILO TRADINGVIEW)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.6, 0.4], vertical_spacing=0.03)

    real = df_limpio[df_limpio['Tipo'] == 'Datos Reales']
    simulado = df_limpio[df_limpio['Tipo'] == 'Simulación Futura']
    intervencion_venta = df_limpio[df_limpio['Alerta'] == 1]
    intervencion_compra = df_limpio[df_limpio['Alerta'] == -1]

    # --- PANEL SUPERIOR: PRECIO Y BANDAS TIPO BOLLINGER ---
    fig.add_trace(go.Scatter(x=df_limpio.index, y=df_limpio['Precio_Banda_Alta'], line=dict(color='rgba(255, 152, 0, 0.4)', width=1, dash='dot'), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_limpio.index, y=df_limpio['Precio_Banda_Baja'], fill='tonexty', fillcolor='rgba(255, 152, 0, 0.08)', line=dict(color='rgba(255, 152, 0, 0.4)', width=1, dash='dot'), name='Túnel de Precio Permitido'), row=1, col=1)

    # Línea de Precio Histórico
    fig.add_trace(go.Scatter(x=real.index, y=real['Precio_Cierre'], name='Precio Real', line=dict(color='#2962FF', width=2)), row=1, col=1)
    
    if not simulado.empty:
        puente = pd.concat([real.tail(1), simulado])
        fig.add_trace(go.Scatter(x=puente.index, y=puente['Precio_Cierre'], name='Tu Simulación', line=dict(color='#00E676', dash='dot', width=3)), row=1, col=1)
        fig.add_vline(x=ultima_fecha, line_dash="dash", line_color="gray", opacity=0.7, row=1, col=1)
        fig.add_vline(x=ultima_fecha, line_dash="dash", line_color="gray", opacity=0.7, row=2, col=1)

    # Marcadores de Intervención
    fig.add_trace(go.Scatter(x=intervencion_venta.index, y=intervencion_venta['Precio_Cierre'], mode='markers', name='BCRP: Vender USD', marker=dict(symbol='triangle-down', color='#FF5252', size=12, line=dict(width=1, color='white'))), row=1, col=1)
    fig.add_trace(go.Scatter(x=intervencion_compra.index, y=intervencion_compra['Precio_Cierre'], mode='markers', name='BCRP: Comprar USD', marker=dict(symbol='triangle-up', color='#00E676', size=12, line=dict(width=1, color='white'))), row=1, col=1)

    # --- PANEL INFERIOR: VELOCIDAD Y BANDAS DE CHEBYSHEV ---
    fig.add_trace(go.Scatter(x=df_limpio.index, y=df_limpio['Banda_Alta'], line=dict(color='rgba(255, 152, 0, 0.4)', width=1), showlegend=False, hoverinfo='skip'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_limpio.index, y=df_limpio['Banda_Baja'], fill='tonexty', fillcolor='rgba(255, 152, 0, 0.15)', line=dict(color='rgba(255, 152, 0, 0.4)', width=1), name='Tolerancia en Velocidad'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_limpio.index, y=df_limpio['Velocidad'], name=f'Velocidad del Dólar (por {texto_periodo})', line=dict(color='#B0BEC5', width=1.5)), row=2, col=1)

    # Diseño general
    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified', 
        height=800,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1) 

    fig.update_yaxes(title_text="Cotización (S/)", row=1, col=1)
    fig.update_yaxes(title_text=f"Variación (S/ {texto_periodo})", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.info("🔄 Conectando con los servidores del mercado financiero. Por favor, espera...")