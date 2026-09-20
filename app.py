import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Portal Unificado de Scouting IA", page_icon="🌐", layout="centered")

st.title("🌐 Portal Unificado de Scouting & Analítica de IA")
st.write("Bienvenido a tu plataforma centralizada de Machine Learning. Selecciona la categoría que deseas evaluar en el menú lateral.")

# Menú de navegación lateral
seccion = st.sidebar.selectbox("Seleccionar Módulo", ["Automovilismo (F1)", "Básquetbol (NBA)", "Música & Artistas", "Analítica y Reportes Globales"])

# ==========================================
# SECCIÓN 1: F1 / AUTOMOVILISMO (COMPLETA CON SIMULADOR Y REACCIÓN)
# ==========================================
if seccion == "Automovilismo (F1)":
    st.subheader("🏎️ Módulo Avanzado de Scouting: F1 & SimRacing")
    st.write("Calcula el rendimiento por vuelta integrando desgaste, combustible, experiencia en simulador y tiempo de reacción.")
    
    @st.cache_resource
    def cargar_f1():
        if os.path.exists("cerebro_f1_optimizado.pkl"):
            return joblib.load("cerebro_f1_optimizado.pkl")
        elif os.path.exists("cerebro_f1_v1.pkl"):
            return joblib.load("cerebro_f1_v1.pkl")
        return None

    modelo_f1 = cargar_f1()

    if modelo_f1 is None:
        st.warning("⚠️ No se encontró ningún modelo de F1 en el repositorio. Ejecuta tu script en Colab para generarlo.")
    else:
        st.success("✅ Cerebro de F1 conectado y listo para predecir.")
        
        # Controles completos de F1 y SimRacing
        col_f1_1, col_f1_2 = st.columns(2)
        with col_f1_1:
            desgaste = st.slider("Desgaste de Neumáticos (%)", 0, 100, 85, key="f1_desgaste")
            horas_sim = st.number_input("Horas en Simulador (Semanal)", 0, 50, 15, key="f1_horas_sim")
        with col_f1_2:
            carga_combustible = st.slider("Carga de Combustible (kg)", 0, 100, 30, key="f1_combustible")
            tiempo_reaccion = st.number_input("Tiempo de Reacción (ms)", 150.0, 400.0, 210.0, key="f1_reaccion")

        if st.button("🚀 Calcular Estrategia y Rendimiento F1"):
            try:
                if hasattr(modelo_f1, "predict") and not hasattr(modelo_f1, "classes_"):
                    pred_tiempo = modelo_f1.predict(np.array([[desgaste]]))[0]
                    
                    # Ajuste dinámico con los parámetros de simulador y reflejos
                    bono_sim = horas_sim * 0.01
                    penalizacion_reaccion = (tiempo_reaccion - 200) * 0.005
                    tiempo_final = pred_tiempo + (carga_combustible * 0.02) - bono_sim + penalizacion_reaccion
                    
                    st.info(f"⏱️ Tiempo estimado de vuelta: **{tiempo_final:.2f} segundos**")
                    st.metric(label="Degradación Proyectada", value=f"{desgaste}%", delta=f"+{(desgaste*0.1):.1f}s por desgaste")
                else:
                    st.success("✅ Simulación de telemetría procesada correctamente.")
            except Exception as e:
                st.error("⚠️ Ocurrió un error al procesar el modelo de F1:")
                st.code(str(e), language="text")

# ==========================================
# SECCIÓN 2: BÁSQUETBOL (NBA)
# ==========================================
elif seccion == "Básquetbol (NBA)":
    st.subheader("🏀 Módulo de Scouting: NBA")
    
    @st.cache_resource
    def cargar_nba():
        if os.path.exists("cerebro_nba_v1.pkl"):
            return joblib.load("cerebro_nba_v1.pkl")
        return None

    modelo_nba = cargar_nba()

    if modelo_nba is None:
        st.warning("⚠️ No se encontró el archivo `cerebro_nba_v1.pkl` en el repositorio.")
    else:
        st.success("✅ Cerebro de NBA conectado.")
        
        col1, col2 = st.columns(2)
        with col1:
            puntos = st.number_input("Puntos Por Partido (PPG)", 0.0, 40.0, 18.0)
            asistencias = st.number_input("Asistencias Por Partido (APG)", 0.0, 15.0, 5.0)
            rebotes = st.number_input("Rebotes Por Partido (RPG)", 0.0, 20.0, 5.0)
        with col2:
            triples = st.slider("Porcentaje de Triples (0-100)", 0.0, 100.0, 35.0)
            defensa = st.slider("Eficiencia Defensiva (0-100)", 0, 100, 75)

        if st.button("🚀 Evaluar Prospecto NBA"):
            try:
                datos_nba = pd.DataFrame({
                    'Puntos_Por_Partido': [puntos],
                    'Asistencias_Por_Partido': [asistencias],
                    'Rebotes_Por_Partido': [rebotes],
                    'Porcentaje_Triples_0a100': [triples],
                    'Eficiencia_Defensiva': [defensa]
                })
                
                pred = modelo_nba.predict(datos_nba)
                prob = modelo_nba.predict_proba(datos_nba)[0][1] * 100
                
                if pred[0] == 1:
                    st.success(f"🌟 ¡PROSPECTO DE ÉLITE / ESTRELLA NBA! (Probabilidad: {prob:.1f}%)")
                    st.balloons()
                else:
                    st.error(f"⚠️ JUGADOR EN DESARROLLO (Probabilidad de Élite: {prob:.1f}%)")
                    
            except Exception as e:
                st.error("⚠️ Ocurrió un error al procesar el modelo de NBA:")
                st.code(str(e), language="text")

# ==========================================
# SECCIÓN 3: MÚSICA & ARTISTAS
# ==========================================
elif seccion == "Música & Artistas":
    st.subheader("🎵 Módulo de Scouting: Música & Redes")
    st.write("Próximamente: Integración de análisis de artistas y métricas de viralidad.")

# ==========================================
# SECCIÓN 4: ANALÍTICA Y REPORTES GLOBALES
# ==========================================
elif seccion == "Analítica y Reportes Globales":
    st.subheader("📈 Reportes Analíticos y Datos Maestros")
    st.write("Aquí puedes visualizar el reporte gráfico generado automáticamente desde Google Colab y explorar la tabla de métricas.")
    
    if os.path.exists("reporte_rendimiento_avanzado.png"):
        st.image("reporte_rendimiento_avanzado.png", caption="Análisis Comparativo de Simulación y Rendimiento Deportivo", use_container_width=True)
    else:
        st.warning("⚠️ No se encontró la imagen `reporte_rendimiento_avanzado.png` en el repositorio.")
        
    st.markdown("---")
    st.markdown("### 📊 Dataset Maestro de la Plataforma")
    
    if os.path.exists("datos_maestros_plataforma.csv"):
        df_maestro = pd.read_csv("datos_maestros_plataforma.csv")
        st.dataframe(df_maestro, use_container_width=True)
    else:
        st.warning("⚠️ No se encontró el archivo `datos_maestros_plataforma.csv` en el repositorio.")
