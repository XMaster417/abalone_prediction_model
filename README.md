# Proyecto módulo 2: clasificación del conjunto Abalone

Este proyecto analiza el conjunto de datos **Abalone** y compara dos modelos
de clasificación multiclase para predecir el sexo del abulón:

- `I`: infant.
- `M`: male.
- `F`: female.

Los modelos incluidos son una regresión logística implementada manualmente con
NumPy y un Random Forest construido con scikit-learn.

## Estructura del proyecto

```text
.
├── abalone.data
├── data_transform.py
├── eda.py
├── main.py
├── models/
│   ├── manual_logistic_regression.py
│   └── random_forest.py
├── requirements.txt
└── utils.py
```

- `main.py`: ejecuta el flujo completo del proyecto.
- `abalone.data`: contiene los datos originales sin encabezados.
- `eda.py`: realiza el análisis exploratorio y genera sus gráficas.
- `data_transform.py`: limpia los datos y prepara una versión adecuada para
  cada modelo.
- `models/manual_logistic_regression.py`: contiene la regresión logística
  multiclase desarrollada manualmente.
- `models/random_forest.py`: contiene el modelo de framework y su búsqueda de
  hiperparámetros.
- `requirements.txt`: enumera las dependencias de Python.

## Requisitos

- Python 3.10 o superior.
- NumPy.
- Pandas.
- Matplotlib.
- Seaborn.
- scikit-learn.

## Instalación

Desde la carpeta raíz del proyecto, crea un entorno virtual:

```powershell
python -m venv .venv
```

Actívalo en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instala las dependencias del archivo `requirements.txt`:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecución

Con el entorno virtual activado, ejecuta:

```powershell
python main.py
```

El archivo `abalone.data` debe permanecer en la misma carpeta que `main.py`.
Durante la ejecución se imprimen los resultados en consola y se abren las
gráficas del análisis exploratorio y de ambos modelos.

## Datos utilizados

Los dos modelos utilizan las mismas ocho características:

```text
Length
Diameter
Height
Whole weight
Shucked weight
Viscera weight
Shell weight
Rings
```

Antes del entrenamiento se eliminan observaciones inválidas o atípicas
relacionadas con altura, diámetro y consistencia de los pesos. Posteriormente
los datos se dividen aproximadamente en:

```text
70 % entrenamiento
15 % validación
15 % prueba
```

La variable `Sex` se prepara de manera diferente para cada modelo:

- La regresión logística utiliza las columnas `Sex_I`, `Sex_M` y `Sex_F`.
- Random Forest conserva inicialmente la columna categórica y después codifica
  sus clases con `LabelEncoder`.

## Flujo general

Al ejecutar `main.py` se realizan los siguientes pasos:

1. Se carga `abalone.data` y se asignan nombres a sus columnas.
2. Se ejecuta el análisis exploratorio sobre una copia de los datos.
3. Se limpian y preparan los datos para la regresión logística manual.
4. Se muestra el mapa de correlaciones.
5. Se buscan los mejores hiperparámetros de la regresión logística y se evalúa
   el modelo ganador.
6. Se prepara una segunda versión de los datos sin one-hot encoding para la
   variable objetivo.
7. Se buscan los mejores hiperparámetros de Random Forest y se evalúa el mejor
   estimador de scikit-learn.

## Regresión logística manual

Este modelo está implementado con NumPy y utiliza una estrategia one-vs-all
para distinguir las tres clases. Las características se estandarizan usando
exclusivamente la media y desviación estándar del conjunto de entrenamiento.

### Búsqueda de hiperparámetros

La búsqueda prueba directamente el siguiente grid:

```python
LOGISTIC_REGRESSION_PARAM_GRID = {
    "epochs": [1000, 3000, 5000, 10000],
    "learning_rate": [0.0001, 0.001, 0.01, 0.05],
}
```

En total se evalúan dieciseis combinaciones. Cada una se entrena con el conjunto de
entrenamiento y se compara con el conjunto de validación. Se elige primero la
combinación con mayor exactitud de validación; si existe un empate, se conserva
la que tenga menor BCE de validación.

El conjunto de prueba no participa en la selección. Después de encontrar los
mejores hiperparámetros, el modelo ganador se vuelve a entrenar para calcular
las métricas y los historiales finales.

### Salidas

La consola muestra:

- Resultados de las nueve combinaciones.
- Mejores valores de `epochs` y `learning_rate`.
- Exactitud y BCE de validación usados durante la selección.
- Tamaño de los conjuntos.
- BCE final de entrenamiento, validación y prueba.
- Exactitud de entrenamiento, validación y prueba.

El modelo genera:

- Una matriz de confusión sobre el conjunto de prueba.
- Una gráfica del BCE por época para entrenamiento, validación y prueba.

## Random Forest con scikit-learn

El modelo de framework utiliza `RandomForestClassifier`. La búsqueda de
hiperparámetros se realiza con `GridSearchCV` y validación cruzada estratificada
de cinco particiones sobre el conjunto de entrenamiento.

El criterio de selección es la exactitud promedio de validación cruzada. Al
terminar, `GridSearchCV` reajusta automáticamente la mejor configuración con
todo el conjunto de entrenamiento. Los conjuntos externos de validación y
prueba permanecen fuera de la búsqueda.

### Grid de hiperparámetros

```python
RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [5, 8, 10, 15],
    "max_depth": [2, 3, 5, 10, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}
```

Este grid contiene 180 combinaciones. Cada combinación se evalúa en cinco
particiones, por lo que la búsqueda realiza 900 ajustes internos y un ajuste
final del mejor modelo.

La búsqueda usa todos los núcleos disponibles mediante `n_jobs=-1` y semillas
fijas para obtener resultados reproducibles.

### Salidas

La consola muestra:

- Mejores hiperparámetros encontrados.
- Exactitud media de validación cruzada de la configuración ganadora.
- Tamaño de los conjuntos.
- Exactitud de entrenamiento, validación y prueba.
- BCE de entrenamiento, validación y prueba.
- Reporte de clasificación de validación con precision, recall y F1-score.

El modelo genera:

- Un mapa de calor del reporte de clasificación de validación.
- Una matriz de confusión sobre validación.
- Una gráfica de barras con el BCE de entrenamiento, validación y prueba.

## Comparación de los modelos

| Aspecto | Regresión logística manual | Random Forest |
| --- | --- | --- |
| Implementación | NumPy | scikit-learn |
| Preparación de `Sex` | One-hot encoding | `LabelEncoder` |
| Estandarización | Sí | No es necesaria |
| Selección | Validación reservada | Validación cruzada de 5 folds |
| Parámetros buscados | Épocas y learning rate | Árboles, profundidad, hojas y variables |
| Matriz de confusión | Prueba | Validación |

Los mejores hiperparámetros y las métricas se mantienen en variables durante
la ejecución y se imprimen en consola; no se guardan en archivos adicionales.

## Personalización de los grids

Para probar otras configuraciones, modifica las listas declaradas al inicio de:

- `models/manual_logistic_regression.py` para la regresión logística.
- `models/random_forest.py` para Random Forest.

Al volver a ejecutar `python main.py`, cada modelo realizará la búsqueda con
los nuevos valores y todas sus métricas y gráficas corresponderán al modelo
ganador.
