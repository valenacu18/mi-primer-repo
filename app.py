import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Portal Multideporte de Scouting IA", page_icon="🏆", layout="centered")

st.title("🏆 Portal Multideporte de Scouting IA")
st.write("Selecciona la disciplina deportiva que deseas evaluar con los modelos de Machine Learning en producción.")

# Menú desplegable para elegir el deporte
deporte = st.selectbox("Selecciona el modelo a utilizar:", ["Automovilismo (F1)", "Básquetbol (NBA)"])

if deporte == "Automovilismo (F1)":
    st.subheader("🏎️ Módulo de Scouting: F1 & SimRacing")
    
    # Cargar modelo de F1
    @st.cache_resource
    def cargar_f1():
        archivo = "cerebro_f1_v1.pkl"
        if os.path.exists(archivo):
            return joblib.load(archivo)
        return None

    modelo_f1 = cargar_f1()

    if modelo_f1 is None:
        st.warning("⚠️ No se encontró el archivo `cerebro_f1_v1.pkl` en el repositorio.")
    else:
        st.success("✅ Cerebro de F1 conectado con éxito.")
        
        col1, col2 = st.columns(2)
        with col1:
            horas = st.number_input("Horas de Simulador Mensual", 10, 300, 150)
            neumaticos = st.slider("Gestión de Neumáticos (0-100)", 0, 100, 85)
        with col2:
            consistencia = st.slider("Consistencia de Ritmo (0-100)", 0, 100, 90)
            reflejos = st.number_input("Tiempo de Reacción (ms)", 100, 300, 180)

        cat = st.selectbox("Categoría Actual", ["F3", "F2", "F1"])

        if st.button("🚀 Ejecutar Predicción F1"):
            datos = pd.DataFrame({
                'Horas_Practica_Mensual': [horas],
                'Gestion_Neumaticos_0a100': [neumaticos],
                'Consistencia_Ritmo_0a100': [consistencia],
                'Tiempo_Reaccion_ms': [reflejos],
                'Categoria_Actual_F1': [1 if cat == "F1" else 0],
                'Categoria_Actual_F2': [1 if cat == "F2" else 0],
                'Categoria_Actual_F3': [1 if cat == "F3" else 0]
            })
            try:
                pred = modelo_f1.predict(datos)
                prob = modelo_f1.predict_proba(datos)[0][1] * 100
                if pred[0] == 1:
                    st.success(f"🌟 ¡APROBADO PARA ASIENTO TOP! (Probabilidad: {prob:.1f}%)")
                    st.balloons()
                else:
                    st.error(f"⚠️ AÚN EN DESARROLLO (Probabilidad: {prob:.1f}%)")
            except Exception as e:
                st.error(f"Error al procesar: {e}")

elif deporte == "Básquetbol (NBA)":
    st.subheader("🏀 Módulo de Scouting: NBA")
    
    # Búsqueda dinámica del archivo de NBA subido
    @st.cache_resource
    def cargar_nba():
        for f in os.listdir("."):
            if "nba" in f.lower() and f.endswith(".pkl"):
                return joblib.load(f)
        return None

    modelo_nba = cargar_nba()

    if modelo_nba is None:
        st.warning("⚠️ No se encontró ningún archivo `.pkl` de NBA en el repositorio. Asegúrate de que su nombre incluya 'nba'.")
    else:
        st.success("✅ Cerebro de NBA conectado con éxito.")
        st.write("¡El modelo de la NBA está listo para recibir estadísticas de jugadores y evaluar su proyección en la liga!")
