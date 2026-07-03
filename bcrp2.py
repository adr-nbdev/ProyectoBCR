import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import warnings
import requests

# Ignorar advertencias menores de pandas
warnings.filterwarnings('ignore')

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(layout="wide", page_title="Simulador BCRP - TradingView Style", page_icon="🏦")

# 2. FUNCIÓN DE DESCARGA DE DATOS (CON CACHÉ PARA MAYOR VELOCIDAD)
@st.cache_data(ttl=3600)
def cargar_datos_desde_2015():
    try:
        ticker = yf.Ticker("USDPEN=X")
        df = ticker.history(start="2015-01-01")
        
        if df.empty:
            return pd.DataFrame()
        
        df = pd.DataFrame(df['Close'])
        df.columns = ['Precio_Cierre']
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df['Tipo'] = 'Datos Reales'
        return df.dropna()
    except Exception as e:
        st.error(f"Error de conexión con el mercado: {e}")
        return pd.DataFrame()

# 3. BARRA LATERAL (CONTROLES DEL USUARIO)
st.sidebar.title("⚙️ Panel de Control")

st.sidebar.markdown("### 📊 Parámetros Estadísticos")
k_chebyshev = st.sidebar.slider("Valor de 'k' (Sensibilidad)", min_value=0.5, max_value=4.0, value=2.0, step=0.1, 
                                help="Amplitud de la banda. Valores bajos hacen al modelo más sensible.")
dias_ventana = st.sidebar.slider("Ventana Móvil (Días)", min_value=10, max_value=100, value=30, step=5,
                                 help="Periodo de tiempo para calcular el 'ruido' del mercado.")

st.sidebar.markdown("---")
st.sidebar.markdown("###  Predicción y Escenarios")
st.sidebar.markdown("Modifica los precios de los próximos 10 días:")

# Cargar la historia
df_hist = cargar_datos_desde_2015()

if not df_hist.empty:
    ultimo_precio = float(df_hist['Precio_Cierre'].iloc[-1])
    ultima_fecha = df_hist.index[-1]
    
    # Crear fechas futuras
    fechas_futuras = pd.bdate_range(start=ultima_fecha + datetime.timedelta(days=1), periods=10)
    
    # Crear tabla editable para el usuario
    df_input = pd.DataFrame({
        'Precio Proyectado (S/)': [round(ultimo_precio, 4)] * 10
    }, index=fechas_futuras.strftime('%Y-%m-%d'))
    
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
    
    # Bandas de Tolerancia en Velocidad (Panel Inferior)
    umbral = k_chebyshev * df_modelo['Desviacion_Movil']
    df_modelo['Banda_Alta'] = df_modelo['Media_Movil'] + umbral
    df_modelo['Banda_Baja'] = df_modelo['Media_Movil'] - umbral
    
    # TRADUCCIÓN A BANDAS TIPO BOLLINGER (Panel Superior)
    # Precio permitido hoy = Precio de ayer + Velocidad máxima permitida hoy
    df_modelo['Precio_Banda_Alta'] = df_modelo['Precio_Cierre'].shift(1) + df_modelo['Banda_Alta']
    df_modelo['Precio_Banda_Baja'] = df_modelo['Precio_Cierre'].shift(1) + df_modelo['Banda_Baja']

    # Lógica de Intervención
    df_modelo['Alerta'] = np.where(df_modelo['Velocidad'] > df_modelo['Banda_Alta'], 1,
                          np.where(df_modelo['Velocidad'] < df_modelo['Banda_Baja'], -1, 0))

    df_limpio = df_modelo.dropna().copy()
    hist_limpio = df_limpio[df_limpio['Tipo'] == 'Datos Reales']
    
    # 5. DISEÑO DE LA INTERFAZ PRINCIPAL
    st.title("Simulador Dinámico de Intervención Cambiaria")
    st.markdown("Modelo matemático-económico para marcar los puntos de compra y venta de dólares del BCRP.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cotización Actual", f"S/ {ultimo_precio:.3f}")
    c2.metric("Tolerancia Alza (Límite)", f"+ S/ {hist_limpio['Banda_Alta'].iloc[-1]:.3f} / día")
    c3.metric("Tolerancia Baja (Límite)", f"- S/ {abs(hist_limpio['Banda_Baja'].iloc[-1]):.3f} / día")
    
    alertas_activas = abs(df_modelo[df_modelo['Tipo'] == 'Simulación Futura']['Alerta']).sum()
    if alertas_activas > 0:
        c4.metric("Diagnóstico del Escenario", "⚠️ ALERTA DE PÁNICO", delta_color="inverse")
    else:
        c4.metric("Diagnóstico del Escenario", "✅ MERCADO ESTABLE", delta="Dentro del rango")

    # 6. CONSTRUCCIÓN DEL GRÁFICO (ESTILO TRADINGVIEW)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.6, 0.4], vertical_spacing=0.03)

    real = df_limpio[df_limpio['Tipo'] == 'Datos Reales']
    simulado = df_limpio[df_limpio['Tipo'] == 'Simulación Futura']
    intervencion_venta = df_limpio[df_limpio['Alerta'] == 1]
    intervencion_compra = df_limpio[df_limpio['Alerta'] == -1]

    # --- PANEL SUPERIOR: PRECIO Y BANDAS TIPO BOLLINGER ---
    # Bandas de precio traducidas
    fig.add_trace(go.Scatter(x=df_limpio.index, y=df_limpio['Precio_Banda_Alta'], line=dict(color='rgba(255, 152, 0, 0.4)', width=1, dash='dot'), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_limpio.index, y=df_limpio['Precio_Banda_Baja'], fill='tonexty', fillcolor='rgba(255, 152, 0, 0.08)', line=dict(color='rgba(255, 152, 0, 0.4)', width=1, dash='dot'), name='Túnel de Precio Permitido'), row=1, col=1)

    # Línea de Precio
    fig.add_trace(go.Scatter(x=real.index, y=real['Precio_Cierre'], name='Precio Real', line=dict(color='#2962FF', width=2)), row=1, col=1)
    
    if not simulado.empty:
        puente = pd.concat([real.tail(1), simulado])
        fig.add_trace(go.Scatter(x=puente.index, y=puente['Precio_Cierre'], name='Tu Simulación', line=dict(color='#00E676', dash='dot', width=3)), row=1, col=1)
        fig.add_vline(x=ultima_fecha, line_dash="dash", line_color="gray", opacity=0.7, row=1, col=1)
        fig.add_vline(x=ultima_fecha, line_dash="dash", line_color="gray", opacity=0.7, row=2, col=1)

    # Marcadores de intervención
    fig.add_trace(go.Scatter(x=intervencion_venta.index, y=intervencion_venta['Precio_Cierre'], mode='markers', name='BCRP: Vender USD', marker=dict(symbol='triangle-down', color='#FF5252', size=12, line=dict(width=1, color='white'))), row=1, col=1)
    fig.add_trace(go.Scatter(x=intervencion_compra.index, y=intervencion_compra['Precio_Cierre'], mode='markers', name='BCRP: Comprar USD', marker=dict(symbol='triangle-up', color='#00E676', size=12, line=dict(width=1, color='white'))), row=1, col=1)

    # --- PANEL INFERIOR: VELOCIDAD Y BANDAS DE CHEBYSHEV ---
    fig.add_trace(go.Scatter(x=df_limpio.index, y=df_limpio['Banda_Alta'], line=dict(color='rgba(255, 152, 0, 0.4)', width=1), showlegend=False, hoverinfo='skip'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_limpio.index, y=df_limpio['Banda_Baja'], fill='tonexty', fillcolor='rgba(255, 152, 0, 0.15)', line=dict(color='rgba(255, 152, 0, 0.4)', width=1), name='Tolerancia en Velocidad'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_limpio.index, y=df_limpio['Velocidad'], name='Velocidad del Dólar', line=dict(color='#B0BEC5', width=1.5)), row=2, col=1)

    # Diseño y formato "TradingView"
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
    fig.update_yaxes(title_text="Variación Diaria (S/)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.info("🔄 Conectando con los servidores del mercado financiero. Por favor, espera...")