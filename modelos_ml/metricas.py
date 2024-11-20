import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve,auc, roc_auc_score

def Accuracy(model,data,data_target,name_data):
    for n_data in range(len(name_data)):
        message=f"Accuracy on {name_data[n_data]} set: {model.score(data[n_data],data_target[n_data])}"
        print(message) 
    return None 

def predicciones(model, data, name_data):
    predictions = {}
    for n_data in range(len(data)):
        predictions[name_data[n_data]] = model.predict(data[n_data])
    return predictions

def matriz_confusion(target_real, prediccion, name_data):
    for i, conjunto in enumerate(name_data):
        print(f"Conjunto {name_data[i]}")
        matriz = confusion_matrix(target_real[i], prediccion[i])
        print(pd.DataFrame(matriz))
        print("\n" + "="*30 + "\n")
    return None

def metricas(target_real,prediccion,name_data):
    cont=0
    for i in name_data:
        print("-----------------------------------------------------------------------------------------")
        print(f"la presicion de el conjunto de {i} es de {precision_score(target_real[0],prediccion[0])}")
        print(f"el recall de el conjunto de {i} es de {recall_score(target_real[0],prediccion[0])}")
        print(f"el valor de f1 de el conjunto de {i} es de {f1_score(target_real[0],prediccion[0])}")
        print("-----------------------------------------------------------------------------------------")
        cont+=1
    return None


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def predicciones_probabilidad(model, data, name_data):
    """
    Genera predicciones de probabilidad para varios conjuntos de datos.
    
    Args:
        model: Modelo entrenado que tiene implementado `predict_proba`.
        data: Lista de conjuntos de datos de entrada.
        name_data: Lista de nombres correspondientes a los conjuntos de datos.
        
    Returns:
        Diccionario donde las claves son los nombres de los conjuntos de datos
        y los valores son las probabilidades de la clase positiva.
    """
    predictions = {}
    for n_data in range(len(data)):
        try:
            # Obtén las probabilidades para la clase positiva (usualmente la clase 1)
            probabilidad = model.predict_proba(data[n_data])[:, 1]
            predictions[name_data[n_data]] = probabilidad
        except AttributeError:
            raise ValueError("El modelo no tiene implementado 'predict_proba'.")
        except IndexError:
            raise ValueError(f"Error al predecir probabilidades para el conjunto {name_data[n_data]}.")
    return predictions


def graficar_curvas_roc(predicciones, etiquetas, nombres_datasets):
    """
    Grafica las curvas ROC y calcula el AUC para varios conjuntos de datos.
    
    Args:
        predicciones: Diccionario con nombres de conjuntos de datos como claves
                      y las probabilidades de la clase positiva como valores.
        etiquetas: Lista de arrays con las etiquetas reales de los conjuntos de datos.
        nombres_datasets: Lista de nombres correspondientes a los conjuntos de datos.
    """
    plt.figure(figsize=(10, 8))
    
    for i, nombre in enumerate(nombres_datasets):
        y_true = etiquetas[i]
        y_pred = predicciones[nombre]
        
        # Calcular la curva ROC y el AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        # Graficar cada curva ROC
        plt.plot(fpr, tpr, label=f'{nombre} (AUC = {auc:.2f})')
    
    # Línea diagonal como referencia (modelo aleatorio)
    plt.plot([0, 1], [0, 1], 'r--', label='Línea aleatoria (AUC = 0.5)')
    
    # Configuración de la gráfica
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()




def ejecutar_evaluacion_completa(model, data, data_target, name_data):
    # Paso 1: Calcular y mostrar el Accuracy
    print("Cálculo de Accuracy:")
    Accuracy(model, data, data_target, name_data)
    
    # Paso 2: Realizar predicciones
    print("\nRealizando predicciones...")
    prediccion = predicciones(model, data, name_data)
    prediccion_con_p=predicciones_probabilidad(model, data, name_data)
    
    # Paso 3: Mostrar matriz de confusión
    print("\nMatriz de confusión:")
    matriz_confusion(data_target, list(prediccion.values()), name_data)
    
    # Paso 4: Calcular y mostrar métricas
    print("\nCálculo de métricas:")
    metricas(data_target, list(prediccion.values()), name_data)
    
    # Paso 5: Graficar curva ROC
    print("\nGraficando Curva ROC:")
    graficar_curvas_roc(predicciones=prediccion_con_p, etiquetas=data_target, nombres_datasets=name_data)


def Accuracy_r_f_p(model, data, data_target, name_data):
    print("Cálculo de Accuracy:")
    Accuracy(model, data, data_target, name_data)

    print("\nRealizando predicciones...")
    prediccion = predicciones(model, data, name_data)

    print("\nCálculo de métricas:")
    metricas(data_target, list(prediccion.values()), name_data)
    
