import warnings
warnings.filterwarnings('ignore')  # Ignorar advertencias si lo deseas


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def adaboost_model(
    n_estimators=50, 
    learning_rate=1.0, 
    random_state=42,
    max_depth=2,
    class_weight='balanced',
    min_samples_split=2):
    
    # Crear el estimador base con los parámetros especificados
    base_estimator = DecisionTreeClassifier(
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        class_weight=class_weight
    )
    
    # Crear el modelo AdaBoost con el estimador base configurado
    adaboost = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    return adaboost


# Función para entrenar el modelo
def adaboost_fit(data_train, data_train_target, model):
    return model.fit(data_train, data_train_target)

# Función para visualizar la importancia de las características
def plot_importance_AB(data, model_fit):
    # Obtenemos la importancia de las características
    df_importance = pd.DataFrame({'feature': data.columns, 'importance': model_fit.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df_importance, palette='viridis')
    plt.title('Importancia de las Características (AdaBoost)', fontsize=16)
    plt.xlabel('Importancia', fontsize=12)
    plt.ylabel('Características', fontsize=12)
    plt.show()
