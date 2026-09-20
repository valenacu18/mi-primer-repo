import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Portal Unificado de Scouting IA", page_icon="🌐", layout="centered")

st.title("🌐 Portal Unificado de Scouting & Analítica de IA")
st.write("Bienvenido a tu plataforma centralizada de Machine Learning. Selecciona la categoría que deseas evaluar en el menú lateral.")

# Menú lateral optimizado
seccion = st.sidebar.selectbox("Seleccionar Módulo", ["Automovilismo (F1)", "Básquetbol (NBA)", "Música & Artistas"])

# ==========================================
# SECCIÓN 1: F1 / AUTOMOVILISMO
# ==========================================
if seccion == "Automovilismo (F1)":
    st.subheader("🏎️ Módulo de Scouting: F1 & SimRacing")
    st.write("Calcula el rendimiento por vuelta integrando práctica mensual, consistencia, gestión y tiempo de reacción.")
    
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
        
        # Controles de entrada con las categorías exactas del entrenamiento
        horas_sim_mensual = st.number_input("Horas de Simulador Mensual", 0, 300, 125, key="f1_horas_sim")
        consistencia = st.slider("Consistencia de Ritmo (0-100)", 0, 100, 95, key="f1_consistencia")
        gestion_neumaticos = st.slider("Gestión de Neumáticos (0-100)", 0, 100, 55, key="f1_gestion")
        tiempo_reaccion = st.number_input("Tiempo de Reacción (ms)", 100.0, 500.0, 150.0, key="f1_reaccion")
        
        categoria_actual = st.selectbox("Categoría Actual", ["F3", "F2", "F1"], key="f1_categoria")

        if st.button("🚀 Ejecutar Predicción F1"):
            try:
                # Construcción del DataFrame adaptado exactamente a los nombres esperados en el fit
                input_data = pd.DataFrame({
                    'Horas_Practica_Mensual': [horas_sim_mensual],
                    'Consistencia_Ritmo_0a100': [consistencia],
                    'Gestion_Neumaticos_0a100': [gestion_neumaticos],
                    'Tiempo_Reaccion_ms': [tiempo_reaccion],
                    'Categoria_Actual_F1': [1 if categoria_actual == "F1" else 0],
                    'Categoria_Actual_F2': [1 if categoria_actual == "F2" else 0],
                    'Categoria_Actual_F3': [1 if categoria_actual == "F3" else 0]
                })
                
                pred = modelo_f1.predict(input_data)
                resultado_pred = pred[0]
                
                st.success(f"✅ Predicción de rendimiento F1 procesada con éxito.")
                st.info(f"⏱️ Resultado del modelo: {resultado_pred}")

                # Actualización automática y dinámica del Dataset Maestro (CSV)
                archivo_csv = "datos_maestros_plataforma.csv"
                nueva_fila = {
                    'Horas_Simulador_Mensual': horas_sim_mensual,
                    'Consistencia_Ritmo_0a100': consistencia,
                    'Gestion_Neumaticos_0a100': gestion_neumaticos,
                    'Tiempo_Reaccion_ms': tiempo_reaccion,
                    'Categoria_Actual': categoria_actual,
                    'Prediccion': resultado_pred
                }
                
                if os.path.exists(archivo_csv):
                    df_maestro = pd.read_csv(archivo_csv)
                    df_maestro = pd.concat([df_maestro, pd.DataFrame([nueva_fila])], ignore_index=True)
                else:
                    df_maestro = pd.DataFrame([nueva_fila])
                
                df_maestro.to_csv(archivo_csv, index=False)
                st.success("💾 ¡Nueva predicción guardada y agregada a la tabla de valores en tiempo real!")

            except Exception as e:
                st.error("⚠️ Ocurrió un error al procesar el modelo de F1:")
                st.code(str(e), language="text")

    # ==========================================
    # ANÁLISIS GRÁFICO INTEGRADO
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Análisis Gráfico de SimRacing")
    
    if os.path.exists("reporte_rendimiento_avanzado.png"):
        st.image("reporte_rendimiento_avanzado.png", caption="Análisis Comparativo de Simulación y Rendimiento en Pista", use_container_width=True)
    else:
        st.warning("⚠️ No se encontró la imagen `reporte_rendimiento_avanzado.png` en el repositorio.")

    # ==========================================
    # TABLA DE VALORES / DATASET MAESTRO DINÁMICO
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Tabla de Valores y Registro Maestro")
    
    if os.path.exists("datos_maestros_plataforma.csv"):
        df_maestro = pd.read_csv("datos_maestros_plataforma.csv")
        st.dataframe(df_maestro, use_container_width=True)
    else:
        st.warning("⚠️ Aún no hay registros guardados. Ejecuta tu primera predicción de F1 arriba para poblar la tabla.")

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
