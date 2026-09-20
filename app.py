import streamlit as st
import pandas as pd
import joblib
import os

# Configuración visual de la página
st.set_page_config(page_title="Scouting IA - F1 & Deporte", page_icon="🏎️", layout="centered")

st.title("🏎️ Portal de Scouting IA: Automovilismo & Élite")
st.write("Utiliza modelos de Machine Learning (Random Forest) para predecir el éxito de un piloto en base a su telemetría y rendimiento.")

# 1. CARGAR EL MODELO ENTRENADO (.pkl)
# Nota: Asegúrate de que el archivo .pkl esté subido en la misma carpeta en GitHub
@st.cache_resource
def cargar_modelo():
    # Intenta cargar tu archivo .pkl (cambia el nombre si usaste otro)
    archivo_modelo = "cerebro_f1_v1.pkl"
    if os.path.exists(archivo_modelo):
        return joblib.load(archivo_modelo)
    else:
        return None

modelo = cargar_modelo()

if modelo is None:
    st.warning("⚠️ No se encontró el archivo del modelo (`.pkl`) en el repositorio. Súbelo a GitHub para habilitar las predicciones reales.")
else:
    st.success("✅ ¡Cerebro de la IA cargado y listo en producción!")

    # 2. SECCIÓN DE ENTRADA DE DATOS (INTERFAZ DE USUARIO)
    st.subheader("📊 Ingrese las Métricas del Piloto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        horas_practica = st.number_input("Horas de Simulador Mensual", min_value=10, max_value=300, value=150)
        gestion_neumaticos = st.slider("Gestión de Neumáticos (0-100)", 0, 100, 85)
        
    with col2:
        consistencia = st.slider("Consistencia de Ritmo (0-100)", 0, 100, 90)
        reflejos = st.number_input("Tiempo de Reacción (ms)", min_value=100, max_value=300, value=180)

    # Selección de categoría
    categoria = st.selectbox("Categoría Actual", ["F3", "F2", "F1"])

    # 3. BOTÓN DE PREDICCIÓN
    if st.button("🚀 Ejecutar Predicción de IA"):
        
        # Preparamos los datos con la misma estructura que usó el modelo
        # (Asegurándonos de incluir las columnas Dummy si el modelo las requiere)
        datos_usuario = pd.DataFrame({
            'Horas_Practica_Mensual': [horas_practica],
            'Gestion_Neumaticos_0a100': [gestion_neumaticos],
            'Consistencia_Ritmo_0a100': [consistencia],
            'Tiempo_Reaccion_ms': [reflejos],
            'Categoria_Actual_F1': [1 if categoria == "F1" else 0],
            'Categoria_Actual_F2': [1 if categoria == "F2" else 0],
            'Categoria_Actual_F3': [1 if categoria == "F3" else 0]
        })
        
        # Ejecutamos la predicción
        try:
            prediccion = modelo.predict(datos_usuario)
            probabilidad = modelo.predict_proba(datos_usuario)[0][1] * 100
            
            st.markdown("---")
            st.subheader("🎯 Resultado del Scouting")
            
            if prediccion[0] == 1:
                st.success(f"🌟 ¡APROBADO PARA ASIENTO TOP! (Probabilidad de éxito: {probabilidad:.1f}%)")
                st.balloons()
                st.write("El modelo detecta métricas de élite compatibles con la máxima categoría.")
            else:
                st.error(f"⚠️ AÚN EN DESARROLLO (Probabilidad de éxito: {probabilidad:.1f}%)")
                st.write("Las métricas actuales sugieren más tiempo de rodaje en categorías inferiores.")
                
        except Exception as e:
            st.error(f"Error al procesar la predicción: Revise que las columnas coincidan con el entrenamiento. Detalle: {e}")
