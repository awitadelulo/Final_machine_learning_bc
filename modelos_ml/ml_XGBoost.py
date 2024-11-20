import warnings
warnings.filterwarnings('ignore')  # Ignorar advertencias si lo deseas

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Función para inicializar el modelo XGBoost
def xgboost_model(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42,
    colsample_bytree=1.0,
    subsample=1.0,
    scale_pos_weight=None
):
    """
    Inicializa un modelo XGBoost con parámetros ajustables.
    """
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        use_label_encoder=False,  # Deprecado en scikit-learn >= 1.3
        eval_metric='logloss',   # Métrica por defecto para clasificación binaria
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        scale_pos_weight=scale_pos_weight  # Para ajustar desbalance de clases
    )
    return xgb


# Función para calcular scale_pos_weight y entrenar el modelo
def xgboost_fit(data_train, data_train_target, model):
    """
    Calcula scale_pos_weight y entrena un modelo XGBoost.
    
    Parámetros:
    - data_train: Features de entrenamiento.
    - data_train_target: Etiquetas de entrenamiento (0 y 1).
    - model: Instancia del modelo XGBoost inicializado.
    
    Devuelve:
    - Modelo XGBoost ajustado.
    """
    # Calcular el número de ejemplos positivos y negativos
    n_positive = sum(data_train_target == 1)
    n_negative = sum(data_train_target == 0)
    
    # Calcular scale_pos_weight
    scale_pos_weight = n_negative / n_positive
    
    # Asignar scale_pos_weight al modelo
    model.set_params(scale_pos_weight=scale_pos_weight)
    
    # Entrenar el modelo
    model.fit(data_train, data_train_target)
    
    return model


# Función para visualizar la importancia de las características
def plot_importance_XGB(data, model_fit):
    # Obtenemos la importancia de las características
    df_importance = pd.DataFrame({'feature': data.columns, 'importance': model_fit.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df_importance, palette='viridis')
    plt.title('Importancia de las Características (XGBoost)', fontsize=16)
    plt.xlabel('Importancia', fontsize=12)
    plt.ylabel('Características', fontsize=12)
    plt.show()

def xgboost_hyperparameter_tuning(data_train, data_train_target, param_grid):
    # Crear el modelo base
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Configurar GridSearchCV
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)

    # Realizar la búsqueda
    grid_search.fit(data_train, data_train_target)

    # Resultados
    print("Mejores Hiperparámetros:", grid_search.best_params_)
    print("Mejor Puntuación (ROC AUC):", grid_search.best_score_)
    return grid_search.best_estimator_