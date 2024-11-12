import warnings
warnings.filterwarnings('ignore')  # Ignorar advertencias si lo deseas

from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Función para inicializar el modelo AdaBoost
def adaboost_model(n_estimators=50, learning_rate=1.0, random_state=42):
    """
    Inicializa un modelo AdaBoostClassifier con los parámetros especificados.

    Parámetros
    ----------
    n_estimators : int, opcional (default=50)
        Número máximo de estimadores de base en el ensamble.

    learning_rate : float, opcional (default=1.0)
        Tasa de aprendizaje que afecta el peso de cada clasificador.

    random_state : int, opcional (default=42)
        Estado aleatorio para la reproducibilidad.

    Retorna
    -------
    AdaBoostClassifier
        Instancia del modelo AdaBoost.
    """
    adaboost = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    return adaboost

# Función para entrenar el modelo
def adaboost_fit(data_train, data_train_target, model):
    """
    Entrena el modelo AdaBoostClassifier con los datos proporcionados.

    Parámetros
    ----------
    data_train : DataFrame o array-like
        Datos de entrenamiento para las características.

    data_train_target : Series o array-like
        Variable objetivo (target) para el entrenamiento.

    model : AdaBoostClassifier
        Modelo de AdaBoost previamente inicializado.

    Retorna
    -------
    AdaBoostClassifier
        Modelo entrenado.
    """
    return model.fit(data_train, data_train_target)

# Función para visualizar la importancia de las características
def plot_importance_AB(data, model_fit):
    """
    Grafica la importancia de las características basada en un modelo entrenado.

    Parámetros
    ----------
    data : DataFrame
        DataFrame con las características del modelo (sin la variable objetivo).

    model_fit : AdaBoostClassifier
        Modelo de AdaBoost entrenado con el método `fit`.

    Retorna
    -------
    None
    """
    # Obtenemos la importancia de las características
    df_importance = pd.DataFrame({'feature': data.columns, 'importance': model_fit.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df_importance, palette='viridis')
    plt.title('Importancia de las Características (AdaBoost)', fontsize=16)
    plt.xlabel('Importancia', fontsize=12)
    plt.ylabel('Características', fontsize=12)
    plt.show()
