import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Portal Unificado de Scouting IA", page_icon="🌐", layout="centered")

st.title("🌐 Portal Unificado de Scouting & Analítica de IA")
st.write("Bienvenido a tu plataforma centralizada de Machine Learning. Selecciona la categoría que deseas evaluar en el menú lateral.")

# Menú de navegación lateral
seccion = st.sidebar.selectbox("Seleccionar Modelo", ["Automovilismo (F1)", "Básquetbol (NBA)", "Música & Artistas"])

# ==========================================
# SECCIÓN 1: F1 / AUTOMOVILISMO
# ==========================================
if seccion == "Automovilismo (F1)":
    st.subheader("🏎️ Módulo de Scouting: F1 & SimRacing")
    
    @st.cache_resource
    def cargar_f1():
        if os.path.exists("cerebro_f1_v1.pkl"):
            return joblib.load("cerebro_f1_v1.pkl")
        return None

    modelo_f1 = cargar_f1()

    if modelo_f1 is None:
        st.warning("⚠️ No se encontró el archivo `cerebro_f1_v1.pkl` en el repositorio.")
    else:
        st.success("✅ Cerebro de F1 conectado.")
        
        col1, col2 = st.columns(2)
        with col1:
            horas = st.number_input("Horas de Simulador Mensual", 10, 300, 150, key="f1_h")
            neumaticos = st.slider("Gestión de Neumáticos (0-100)", 0, 100, 85, key="f1_n")
        with col2:
            consistencia = st.slider("Consistencia de Ritmo (0-100)", 0, 100, 90, key="f1_c")
            reflejos = st.number_input("Tiempo de Reacción (ms)", 100, 300, 180, key="f1_r")

        cat = st.selectbox("Categoría Actual", ["F3", "F2", "F1"], key="f1_cat")

        if st.button("🚀 Ejecutar Predicción F1"):
            try:
                datos = pd.DataFrame({
                    'Horas_Practica_Mensual': [horas],
                    'Gestion_Neumaticos_0a100': [neumaticos],
                    'Consistencia_Ritmo_0a100': [consistencia],
                    'Tiempo_Reaccion_ms': [reflejos],
                    'Categoria_Actual_F1': [1 if cat == "F1" else 0],
                    'Categoria_Actual_F2': [1 if cat == "F2" else 0],
                    'Categoria_Actual_F3': [1 if cat == "F3" else 0]
                })
                
                pred = modelo_f1.predict(datos)
                prob = modelo_f1.predict_proba(datos)[0][1] * 100
                
                if pred[0] == 1:
                    st.success(f"🌟 ¡APROBADO PARA ASIENTO TOP! (Probabilidad: {prob:.1f}%)")
                    st.balloons()
                else:
                    st.error(f"⚠️ AÚN EN DESARROLLO (Probabilidad: {prob:.1f}%)")
                    
            except Exception as e:
                st.error("⚠️ Ocurrió un error al procesar el modelo de F1.")
                st.write("Código del error exacto para depuración:")
                st.exception(e)

# ==========================================
# SECCIÓN 2: MÚSICA Y ARTISTAS
# ==========================================
elif seccion == "Música & Artistas":
    st.subheader("🎵 Módulo de Scouting: Música & Redes")
    
    @st.cache_resource
    def cargar_musica():
        for f in os.listdir("."):
            if "musica" in f.lower() or "artistas" in f.lower() or f.endswith(".pkl"):
                try:
                    m = joblib.load(f)
                    if hasattr(m, "feature_names_in_") and any("Instagram" in col for col in m.feature_names_in_):
                        return m
                except:
                    continue
        return None

    modelo_musica = cargar_musica()

    if modelo_musica is None:
        st.warning("⚠️ No se detectó un modelo con las variables de música/redes sociales en el repositorio.")
    else:
        st.success("✅ Cerebro de Música conectado.")
        
        col1, col2 = st.columns(2)
        with col1:
            instagram = st.number_input("Interacciones en Instagram", 0, 1000000, 50000, key="m_ig")
            presupuesto = st.number_input("Presupuesto USD", 0, 50000, 5000, key="m_pres")
            soundcloud = st.number_input("SoundCloud Plays", 0, 500000, 20000, key="m_sc")
        with col2:
            tiktok = st.number_input("Videos en TikTok", 0, 10000, 150, key="m_tk")
            twitter = st.number_input("Menciones en Twitter", 0, 50000, 1000, key="m_tw")

        if st.button("🚀 Evaluar Potencial Musical"):
            try:
                datos_musica = pd.DataFrame(columns=modelo_musica.feature_names_in_)
                datos_musica.loc[0] = 0
                
                for col in datos_musica.columns:
                    col_lower = col.lower()
                    if "instagram" in col_lower: datos_musica.loc[0, col] = instagram
                    elif "presupuesto" in col_lower: datos_musica.loc[0, col] = presupuesto
                    elif "soundcloud" in col_lower: datos_musica.loc[0, col] = soundcloud
                    elif "tiktok" in col_lower: datos_musica.loc[0, col] = tiktok
                    elif "twitter" in col_lower: datos_musica.loc[0, col] = twitter

                pred = modelo_musica.predict(datos_musica)
                prob = modelo_musica.predict_proba(datos_musica)[0][1] * 100
                
                if pred[0] == 1:
                    st.success(f"🌟 ¡HIT POTENCIAL EN TENDENCIA! (Probabilidad: {prob:.1f}%)")
                    st.balloons()
                else:
                    st.error(f"⚠️ DESARROLLO ARTÍSTICO REQUERIDO (Probabilidad: {prob:.1f}%)")
                    
            except Exception as e:
                st.error("⚠️ Ocurrió un error al procesar el modelo de música.")
                st.write("Código del error exacto para depuración:")
                st.exception(e)

# ==========================================
# SECCIÓN 3: NBA / BÁSQUETBOL
# ==========================================
elif seccion == "Básquetbol (NBA)":
    st.subheader("🏀 Módulo de Scouting: NBA")
    st.write("Próximamente: Configuración de estadísticas de franquicia y rendimiento de jugadores de la NBA.")
