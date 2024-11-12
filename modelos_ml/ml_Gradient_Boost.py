import warnings
warnings.filterwarnings('ignore')  # Ignorar advertencias si lo deseas

from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Función para inicializar el modelo Gradient Boosting
def gradient_boost_model(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    """
    Inicializa un modelo GradientBoostingClassifier con los parámetros especificados.

    Parámetros
    ----------
    n_estimators : int, opcional (default=100)
        Número de árboles en el ensamble.

    learning_rate : float, opcional (default=0.1)
        Tasa de aprendizaje, que reduce la contribución de cada árbol adicional.

    max_depth : int, opcional (default=3)
        Profundidad máxima de los árboles individuales. Controla la complejidad del modelo.

    random_state : int, opcional (default=42)
        Estado aleatorio para la reproducibilidad.

    Retorna
    -------
    GradientBoostingClassifier
        Instancia del modelo Gradient Boosting.
    """
    gboost = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    return gboost

# Función para entrenar el modelo
def gradient_boost_fit(data_train, data_train_target, model):
    """
    Entrena el modelo GradientBoostingClassifier con los datos proporcionados.

    Parámetros
    ----------
    data_train : DataFrame o array-like
        Datos de entrenamiento para las características.

    data_train_target : Series o array-like
        Variable objetivo (target) para el entrenamiento.

    model : GradientBoostingClassifier
        Modelo de Gradient Boosting previamente inicializado.

    Retorna
    -------
    GradientBoostingClassifier
        Modelo entrenado.
    """
    return model.fit(data_train, data_train_target)

# Función para visualizar la importancia de las características
def plot_importance_GB(data, model_fit):
    """
    Grafica la importancia de las características basada en un modelo entrenado.

    Parámetros
    ----------
    data : DataFrame
        DataFrame con las características del modelo (sin la variable objetivo).

    model_fit : GradientBoostingClassifier
        Modelo de Gradient Boosting entrenado con el método `fit`.

    Retorna
    -------
    None
    """
    # Obtenemos la importancia de las características
    df_importance = pd.DataFrame({'feature': data.columns, 'importance': model_fit.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df_importance, palette='viridis')
    plt.title('Importancia de las Características (Gradient Boosting)', fontsize=16)
    plt.xlabel('Importancia', fontsize=12)
    plt.ylabel('Características', fontsize=12)
    plt.show()
