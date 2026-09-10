import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [5, 8, 10, 15],
    "max_depth": [2, 3, 5, 10, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}


"""
================================================================================
    Función: binary_cross_entropy
    Calcula el BCE promedio para el problema multiclase con una estrategia de 
    one vs all.

    @param y_true -> np.array: valores reales de cada observación.
    @param y_probability -> matrix: matriz con las predicciones del modelo.
    @param number_classes -> int: numero de clases que tiene el dataset (3)
    @return num: promedio del error o perdida para la observacion.
================================================================================
"""
def binary_cross_entropy(y_true, y_probability, number_classes):
    y_one_hot = pd.get_dummies(y_true, dtype=float)
    y_one_hot = y_one_hot.reindex(
        columns=range(number_classes), fill_value=0.0
    )
    y_probability = np.clip(y_probability, 1e-15, 1.0 - 1e-15)
    losses = -(
        y_one_hot * np.log(y_probability)
        + (1.0 - y_one_hot) * np.log(1.0 - y_probability)
    )
    return float(np.mean(losses))
"""
================================================================================
    Función: plot_classification_report
    Calcula metricas de clasificación para cada clase, como precision, recall y 
    F1-score como mapa de calor.

    @param y_true -> List: valores reales de clase.
    @param y_pred -> List: valores predichos por el modelo.
    @param class_names -> List: nombre de las clases del modelo.
================================================================================
"""
def plot_classification_report(y_true, y_pred, class_names):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_dataframe = pd.DataFrame(report).transpose()
    class_metrics = report_dataframe.loc[
        class_names, ["precision", "recall", "f1-score"]
    ]

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        class_metrics,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
    )
    plt.title("Reporte de clasificacion - Validacion")
    plt.xlabel("Metrica")
    plt.ylabel("Clase")
    plt.tight_layout()
    plt.show()

"""
================================================================================
    Función: plot_confusion_matrix
    Muestra la matriz de confusion como un mapa de calor para las clases del
    modelo.

    @param y_true -> List: valores reales de clase.
    @param y_pred -> List: valores predichos por el modelo.
    @param class_names -> List: nombre de las clases del modelo.
================================================================================
"""
def plot_confusion_matrix(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Matriz de confusion - Validacion")
    plt.xlabel("Prediccion")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.show()

"""
================================================================================
    Función: plot_bce
    Grafica el BCE para cada conjunto, test, train y val. Los muestra en una 
    grafica de barras en lugar de una grafica con epocas por la forma de 
    entrenar al modelo.

    @param bce_results -> Dictionary: diccionario con los bce para cada conjunto
================================================================================
"""
def plot_bce(bce_results):
    names = list(bce_results.keys())
    values = list(bce_results.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.bar_label(bars, fmt="%.4f", padding=3)
    plt.title("BCE del modelo Random Forest por conjunto")
    plt.xlabel("Conjunto")
    plt.ylabel("Error BCE")
    plt.ylim(0, max(values) * 1.15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
"""
================================================================================
    Funcion: random_forest_analysis
    Entrena y evalua un modelo con el algoritmo de Random Forest para clasificar 
    la variable Sex.

    Separa los datos en 70% para train, 15% para val y 15% para test. La 
    variable objetivo se codifica ajustando LabelEncoder. Después se entrena el 
    modelo y predicen valores con los diferentes conjutos. 

    Finalmente se imprimen metricas para verificar el rendimiento del modelo.

    @param df -> pd.dataFrame: dataframe con el dataset
================================================================================
"""
def random_forest_analysis(df):
    x = df[[
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        "Rings",
    ]]
    y = df["Sex"]

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)


    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=RANDOM_FOREST_PARAM_GRID,
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(x_train, y_train_encoded)

    model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    datasets = {
        "Entrenamiento": (x_train, y_train_encoded),
        "Validacion": (x_val, y_val_encoded),
        "Test": (x_test, y_test_encoded),
    }
    predictions = {}
    probabilities = {}
    accuracies = {}
    bce_results = {}

    for name, (x_split, y_split) in datasets.items():
        predictions[name] = model.predict(x_split)
        probabilities[name] = model.predict_proba(x_split)
        accuracies[name] = accuracy_score(y_split, predictions[name])
        bce_results[name] = binary_cross_entropy(
            y_split,
            probabilities[name],
            len(label_encoder.classes_),
        )

    # Estadisticas de random forest
    print("\nResumen de Random Forest")

    print("\nMejores hiperparametros (GridSearchCV):")
    for parameter, value in best_params.items():
        print(f"  {parameter}: {value}")
    print(
        "  Exactitud media de validacion cruzada: "
        f"{grid_search.best_score_:.4f}"
    )

    ## Tamaño de cada conjunto (train, val y test )
    print("\nTamanos de los conjuntos:")
    for name, (x_split, _) in datasets.items():
        print(f"  {name}: {len(x_split)}")

    ## Accuracy por cada conjunto (train, val y test )
    print("\nExactitud por conjunto:")
    for name, value in accuracies.items():
        print(f"  {name}: {value:.4f}")

    ## BCE por cada conjunto (train, val y test )
    print("\nBCE por conjunto:")
    for name, value in bce_results.items():
        print(f"  {name}: {value:.6f}")

    ## Reportar estadisticas como f-1, recall, precision
    print("\nReporte de clasificacion - Test:")
    print(
        classification_report(
            y_test_encoded,
            predictions["Test"],
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    # Graficar metricas
    plot_classification_report(
        y_test_encoded,
        predictions["Test"],
        label_encoder.classes_,
    )
    plot_confusion_matrix(
        y_test_encoded,
        predictions["Test"],
        label_encoder.classes_,
    )
    plot_bce(bce_results)
