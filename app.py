import numpy as np
import pickle
import streamlit as st

# Ruta del modelo preentrenado
MODEL_PATH = 'XGB_model.pkl'

# Función para realizar predicciones
def model_prediction(x_in, model):
    x = np.asarray(x_in).reshape(1, -1)
    preds = model.predict(x)
    return preds

def mensaje_traductor(val):
    if val==0:
        return "No fraude"
    else: 
        return "Es fraude"

def main():
    # Cargar el modelo
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    
    # Título
    st.markdown(
        "<h1 style='color:#181082;text-align:center;'>Detector de Fraude</h1>", 
        unsafe_allow_html=True
    )

    # Entrada de datos
    st.write("Ingrese los valores para las características:")
    feature_labels = [
        "a", "b", "d", "e", "f", "g", "h", "i", 
        "l", "m", "n", "o", "p", "q", "r", "s", "monto"
    ]
    inputs = []
    for label in feature_labels:
        value = st.text_input(f"{label}:")
        inputs.append(value)

    # Botón de predicción
    if st.button("Realizar Predicción"):
        try:
            # Convertir las entradas a flotantes
            x_in = [float(val) for val in inputs]
            # Realizar la predicción
            prediction = model_prediction(x_in, model)
            mensaje=mensaje_traductor(prediction[0])
            st.success(f"El resultado de la predicción es: {mensaje.upper()}")
        except ValueError:
            st.error("Por favor, ingrese solo valores numéricos válidos.")

if __name__ == '__main__':
    main()