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
        "a", "b", "c","d", "e", "f", "g", "h", "i","j","k", 
        "l", "m", "n", "o", "p", "q", "r", "s", "monto"
    ]
    inputs = []
    
    inputs = []
    j_value = None  # Variable para almacenar el valor de "j"
    
    for label in feature_labels:
        value = st.text_input(f"{label} :")
        
        # Validación específica para "j", debe ser un string, no un número
        if label == "j":
            # Verificar si el valor ingresado no es un número
            if value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                st.error("El valor para 'j' no puede ser un número. Por favor ingrese un texto.")
                j_value = None  # Reseteamos el valor de "j"
            else:
                j_value = value  # Asignamos el valor de "j" si es correcto
        else:
            inputs.append(value)
    
    # Incluir el valor de "j" si fue ingresado correctamente
    if j_value is not None:
        inputs.insert(feature_labels.index("j"), j_value)
    
    inputs = [value for label, value in zip(feature_labels, inputs) if label not in ["c", "j", "k"]]

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