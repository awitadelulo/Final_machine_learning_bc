##Proyecto fraude 
---
El siguiente trabajo consiste en realizar el proyecto final de la materia de Machine Learning. 
En este, se nos presenta un problema que requiere elegir un modelo de Machine Learning para realizar predicciones 
sobre una base de datos que no cuenta con nombres en las características (features). 
El contenido de esta base de datos está relacionado con registros de fraudes en línea.

--- 
1. Entrenamiento del modelo.

Para abordar este problema, se decidió utilizar el modelo **XGBoost** debido a sus características avanzadas y 
su capacidad para manejar datos desbalanceados de manera eficiente. **XGBoost (Extreme Gradient Boosting)** es un modelo de aprendizaje 
automático basado en árboles de decisión que utiliza el enfoque de boosting, lo que significa que combina varios modelos débiles para formar un modelo robusto.
Entre las razones principales para su elección destacan:

1. **Manejo de Datos Desbalanceados**:  
   XGBoost permite la ponderación de clases en la función de pérdida, lo cual es esencial en problemas donde una clase es mucho más frecuente que la otra.
   Esto lo hace particularmente adecuado para problemas como el fraude, donde los datos suelen estar desbalanceados.

3. **Eficiencia Computacional**:  
   XGBoost es conocido por su alta velocidad y rendimiento debido a optimizaciones como la paralelización y el uso eficiente de recursos de memoria.
    Esto lo convierte en una opción ideal para problemas que requieren iteraciones rápidas.

5. **Regularización**:  
   El modelo incorpora regularización \(L1\) y \(L2\), que ayuda a prevenir el sobreajuste,
   una preocupación común en datasets desbalanceados donde el modelo podría memorizar los datos mayoritarios.

7. **Interpretabilidad**:  
   Las métricas como la importancia de las características proporcionadas por XGBoost permiten analizar
   cuáles variables tienen más impacto en las predicciones, facilitando la explicación de los resultados.

9. **Flexibilidad**:  
   XGBoost soporta múltiples funciones de pérdida y ofrece parámetros ajustables,
   lo que permite adaptar el modelo específicamente a las necesidades del problema.

   esto esta mas explicado a detalle en el siguiente link que es elink del notebook dentro del Github [Final_machine_learning_jhonatan.ipynb](https://github.com/awitadelulo/Final_machine_learning_bc/blob/master/Final_machine_learning_jhonatan.ipynb)
---
2. servidor local
Creamos un entorno con conda he instalamos python 3.7, y instalamos las dependencias necesarias.
$   conda create -n Apifraude
$   conda activate Apifraude
$   conda install python=3.9
$   pip install -r requirements.txt
$   streamlit run app.py

