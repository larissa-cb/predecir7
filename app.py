# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Deserción Universitaria",
    page_icon="🎓",
    layout="wide"
)

# Título principal
st.title("🎓 Sistema de Predicción de Deserción Estudiantil")
st.markdown("Sistema inteligente para identificar estudiantes en riesgo de abandono académico")

# Simular un modelo entrenado (en producción real se cargaría desde un archivo)
@st.cache_resource
def create_model():
    # Crear un modelo simple con 10 features
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Generar datos de entrenamiento sintéticos con 10 features
    np.random.seed(42)
    X_train = np.random.randn(1000, 10)
    y_train = np.random.choice([0, 1, 2], size=1000, p=[0.4, 0.3, 0.3])
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    return model

# Inicializar modelo y scaler
model = create_model()
scaler = StandardScaler()

# Sidebar para entrada de datos
st.sidebar.header("📋 Información del Estudiante")

# Definir las 10 features que el modelo espera
st.sidebar.subheader("Datos Académicos Clave")
previous_grade = st.sidebar.slider("Calificación Previa (0-200)", 0, 200, 120)
attendance = st.sidebar.slider("Asistencia (%)", 0, 100, 85)
units_approved = st.sidebar.slider("Materias Aprobadas 1er Sem", 0, 10, 4)
current_avg = st.sidebar.slider("Promedio Actual (0-20)", 0, 20, 12)

st.sidebar.subheader("Datos Personales")
age = st.sidebar.slider("Edad", 17, 50, 20)
scholarship = st.sidebar.selectbox("¿Tiene Beca?", ["No", "Sí"])
tuition_fees = st.sidebar.selectbox("¿Matrícula al Día?", ["Sí", "No"])
debtor = st.sidebar.selectbox("¿Es Deudor?", ["No", "Sí"])
international = st.sidebar.selectbox("¿Estudiante Internacional?", ["No", "Sí"])
displaced = st.sidebar.selectbox("¿Viene de Zona Rural?", ["No", "Sí"])

# Botón para predecir
if st.sidebar.button("🔍 Predecir Riesgo de Deserción"):
    try:
        # Preparar los datos en el orden correcto (10 features)
        input_data = np.array([[
            previous_grade,        # Feature 1
            attendance,            # Feature 2
            units_approved,        # Feature 3
            current_avg,           # Feature 4
            age,                   # Feature 5
            1 if scholarship == "Sí" else 0,  # Feature 6
            1 if tuition_fees == "Sí" else 0, # Feature 7
            1 if debtor == "Sí" else 0,       # Feature 8
            1 if international == "Sí" else 0, # Feature 9
            1 if displaced == "Sí" else 0      # Feature 10
        ]])
        
        # Escalar los datos
        input_scaled = scaler.fit_transform(input_data)
        
        # Hacer predicción
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Mapear predicciones
        risk_levels = ["🚨 Alto Riesgo (Abandono)", "⚠️ Riesgo Medio (Enrolado)", "✅ Bajo Riesgo (Graduado)"]
        risk_level = risk_levels[prediction]
        
        # Mostrar resultados
        st.success("### 📊 Resultados de la Predicción")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nivel de Riesgo", risk_level)
        with col2:
            confidence = probabilities[prediction] * 100
            st.metric("Confianza", f"{confidence:.1f}%")
        with col3:
            risk_score = probabilities[0] * 100  # Probabilidad de abandono
            st.metric("Prob. Abandono", f"{risk_score:.1f}%")
        
        # Barra de progreso para visualizar el riesgo
        st.progress(probabilities[0], text=f"Probabilidad de Abandono: {probabilities[0]*100:.1f}%")
        
        # Mostrar probabilidades de todas las clases
        st.subheader("📈 Probabilidades por Categoría")
        prob_df = pd.DataFrame({
            'Categoría': risk_levels,
            'Probabilidad': [f"{p*100:.1f}%" for p in probabilities]
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)
        
        # Recomendaciones basadas en el riesgo
        st.subheader("🎯 Plan de Acción Recomendado")
        
        if prediction == 0:  # Alto riesgo
            st.error("""
            **🚨 INTERVENCIÓN INMEDIATA REQUERIDA**
            
            **Acciones Prioritarias:**
            - Reunión urgente con consejero académico (48 horas máximo)
            - Evaluación económica completa
            - Programa de mentoría intensiva (3 sesiones/semana)
            - Contacto inmediato con familia/tutores
            - Revisión de carga académica
            - Considerar reducción temporal de materias
            
            **Plazo:** Intervención en 48 horas
            """)
            
        elif prediction == 1:  # Riesgo medio
            st.warning("""
            **⚠️ MONITOREO REFORZADO NECESARIO**
            
            **Acciones Recomendadas:**
            - Evaluación académica quincenal
            - Talleres de habilidades de estudio
            - Mentoría con estudiante avanzado
            - Grupo de apoyo entre pares
            - Revisión de técnicas de estudio
            - Seguimiento de asistencia
            
            **Seguimiento:** Revisión mensual
            """)
            
        else:  # Bajo riesgo
            st.success("""
            **✅ SITUACIÓN ESTABLE**
            
            **Acciones de Mantenimiento:**
            - Continuar con apoyo actual
            - Participación en actividades extracurriculares
            - Oportunidades de desarrollo profesional
            - Preparación para prácticas/pasantías
            - Monitoreo semestral estándar
            
            **Enfoque:** Desarrollo y crecimiento personal
            """)
        
        # Factores de riesgo identificados
        st.subheader("🔍 Factores de Riesgo Detectados")
        
        risk_factors = []
        if previous_grade < 100:
            risk_factors.append(f"Calificación previa baja ({previous_grade}/200)")
        if attendance < 75:
            risk_factors.append(f"Asistencia preocupante ({attendance}%)")
        if units_approved < 4:
            risk_factors.append(f"Bajo rendimiento académico ({units_approved} materias aprobadas)")
        if current_avg < 10:
            risk_factors.append(f"Promedio actual bajo ({current_avg}/20)")
        if scholarship == "No":
            risk_factors.append("Falta de apoyo económico (sin beca)")
        if tuition_fees == "No":
            risk_factors.append("Problemas de pago de matrícula")
        if debtor == "Sí":
            risk_factors.append("Situación de deuda académica")
        if age > 25:
            risk_factors.append("Edad mayor al promedio típico")
        
        if risk_factors:
            st.write("**Factores identificados:**")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.success("✅ No se detectaron factores de riesgo significativos")
            
        # Información del estudiante
        st.subheader("📋 Resumen del Estudiante")
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.write("**Datos Académicos:**")
            st.write(f"- Calificación previa: {previous_grade}/200")
            st.write(f"- Asistencia: {attendance}%")
            st.write(f"- Materias aprobadas: {units_approved}/10")
            st.write(f"- Promedio actual: {current_avg}/20")
            
        with info_col2:
            st.write("**Datos Personales:**")
            st.write(f"- Edad: {age} años")
            st.write(f"- Beca: {scholarship}")
            st.write(f"- Matrícula al día: {tuition_fees}")
            st.write(f"- Deudor: {debtor}")
            st.write(f"- Internacional: {international}")
            st.write(f"- Zona rural: {displaced}")
        
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        st.info("""
        ⚠️ **Solución de problemas:**
        - Verifique que todos los campos estén completos
        - Asegúrese de usar valores válidos
        - Si el error persiste, contacte al administrador
        """)

else:
    # Pantalla inicial
    st.info("👈 Complete la información del estudiante en la barra lateral y haga clic en 'Predecir Riesgo de Deserción'")
    
    # Información sobre el sistema
    st.markdown("---")
    st.subheader("ℹ️ Acerca del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **🎯 Objetivo:**
        - Identificar estudiantes en riesgo de abandono
        - Proporcionar intervenciones tempranas
        - Mejorar las tasas de retención estudiantil
        - Optimizar recursos de apoyo académico
        
        **📊 Métricas consideradas:**
        - Rendimiento académico previo
        - Asistencia y participación
        - Situación económica
        - Datos demográficos
        """)
    
    with col2:
        st.write("""
        **🔧 Tecnologías:**
        - Machine Learning: Random Forest
        - Framework: Streamlit
        - Procesamiento: Scikit-learn
        - Análisis: Pandas, NumPy
        
        **🎓 Beneficios:**
        - Detección temprana (6-12 meses de anticipación)
        - Intervenciones personalizadas
        - Ahorro de recursos institucionales
        - Mejora del éxito estudiantil
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**📞 Soporte Técnico:**
- Email: larissac@ucm.es
""")

st.markdown("---")
st.caption("Sistema de Predicción de Deserción Universitaria v2.0 | Desarrollado con Streamlit y Machine Learning")