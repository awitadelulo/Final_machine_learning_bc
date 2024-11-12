import warnings
warnings.filterwarnings('ignore')  # Ignorar advertencias si lo deseas
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

# Función para inicializar el modelo Random Forest
def random_forest(n_estimators=100, max_depth=None, random_state=42):
    """
    Inicializa un modelo RandomForestClassifier con los parámetros especificados.

    Parámetros
    ----------
    n_estimators : int, opcional (default=100)
        Número de árboles en el bosque.

    max_depth : int, opcional (default=None)
        Profundidad máxima de cada árbol. Si es None, los nodos se expanden hasta que todas las hojas son puras.

    random_state : int, opcional (default=42)
        Estado aleatorio para la reproducibilidad.

    Retorna
    -------
    RandomForestClassifier
        Instancia del modelo Random Forest.
    """
    rforest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    return rforest

# Función para entrenar el modelo
def random_forest_fit(data_train, data_train_target, model):
    """
    Entrena el modelo RandomForestClassifier con los datos proporcionados.

    Parámetros
    ----------
    data_train : DataFrame o array-like
        Datos de entrenamiento para las características.

    data_train_target : Series o array-like
        Variable objetivo (target) para el entrenamiento.

    model : RandomForestClassifier
        Modelo de Random Forest previamente inicializado.

    Retorna
    -------
    RandomForestClassifier
        Modelo entrenado.
    """
    return model.fit(data_train, data_train_target)

# Función para visualizar la importancia de las características
def plot_importance_RF(data, model_fit):
    """
    Grafica la importancia de las características basada en un modelo entrenado.

    Parámetros
    ----------
    data : DataFrame
        DataFrame con las características del modelo (sin la variable objetivo).

    model_fit : RandomForestClassifier
        Modelo de Random Forest entrenado con el método `fit`.

    Retorna
    -------
    None
    """
    # Obtenemos la importancia de las características
    df_importance = pd.DataFrame({'feature': data.columns, 'importance': model_fit.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df_importance, palette='viridis')
    plt.title('Importancia de las Características (Random Forest)', fontsize=16)
    plt.xlabel('Importancia', fontsize=12)
    plt.ylabel('Características', fontsize=12)
    plt.show()
