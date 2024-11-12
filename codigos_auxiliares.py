from sklearn.model_selection import StratifiedShuffleSplit

def data_split(data, test_size, columna_estratificar):
    """
    Divide un DataFrame en un conjunto de entrenamiento y un conjunto de prueba, 
    asegurando una distribución proporcional de la columna especificada (estratificación).

    Esta función utiliza `StratifiedShuffleSplit` de Scikit-learn para realizar una división 
    estratificada, es decir, mantiene la misma distribución de la variable especificada 
    en la columna de estratificación en ambos conjuntos (entrenamiento y prueba).

    Parámetros
    ----------
    data : pandas.DataFrame
        El DataFrame que se desea dividir en conjuntos de entrenamiento y prueba.

    test_size : float
        La proporción de datos que se debe asignar al conjunto de prueba. 
        Debe ser un valor entre 0 y 1 (por ejemplo, 0.2 indica un 20% para prueba y 80% para entrenamiento).

    columna_estratificar : str
        El nombre de la columna en el DataFrame que se utilizará para la estratificación. 
        La estratificación asegura que la distribución de los valores en esta columna sea representativa 
        tanto en el conjunto de entrenamiento como en el de prueba.

    Retorna
    -------
    tuple
        Una tupla que contiene dos DataFrames:
        - El conjunto de entrenamiento (strat_train_set)
        - El conjunto de prueba (strat_test_set)
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in split.split(data, data[columna_estratificar]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    
    # Restablecer los índices y asignar de nuevo a las variables
    strat_train_set = strat_train_set.reset_index(drop=True)
    strat_test_set = strat_test_set.reset_index(drop=True)
    
    return strat_train_set, strat_test_set


def data_target_split(data, target):
    """
    Separa las características (X) del objetivo (y) en un DataFrame.

    Esta función toma un DataFrame y una columna objetivo (`target`), y devuelve dos objetos:
    - Un DataFrame con las características (todas las columnas excepto la del objetivo).
    - Un Serie con los valores del objetivo (`target`).

    Parámetros
    ----------
    data : pandas.DataFrame
        El DataFrame que contiene tanto las características como la columna objetivo.

    target : str
        El nombre de la columna que se desea separar como objetivo (target).

    Retorna
    -------
    tuple
        Una tupla que contiene:
        - Un DataFrame con las características (todas las columnas excepto `target`).
        - Una Serie con los valores del objetivo (`target`).
    """
    target_values = data[target]
    data = data.drop(target, axis=1)
    return data, target_values
