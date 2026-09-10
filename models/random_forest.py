import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


RANDOM_FOREST_PARAM_GRID = {
    # "n_estimators": [5, 8, 10, 15],
    # "max_depth": [2, 3, 5, 10, None],
    # "min_samples_leaf": [1, 2, 4],
    # "max_features": ["sqrt", "log2", None],
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
    Funcion: negative_binary_cross_entropy_scorer
    Calcula el BCE negativo de una particion de validacion cruzada. Se devuelve
    negativo porque GridSearchCV siempre busca maximizar las metricas.

    @param estimator -> RandomForestClassifier: modelo ajustado a evaluar.
    @param x -> pd.dataframe: variables independientes de la particion.
    @param y -> np.array: clases reales codificadas de la particion.
    @return num: BCE negativo de las probabilidades predichas.
================================================================================
"""
def negative_binary_cross_entropy_scorer(estimator, x, y):
    probabilities = estimator.predict_proba(x)
    bce = binary_cross_entropy(y, probabilities, len(estimator.classes_))
    return -bce
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
    plt.title("Reporte de clasificacion - Test")
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
    plt.title("Matriz de confusion - Test")
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

    Para cada combinacion de hiperparametros, se usa validacion cruzada
    estratificada y se reportan macro F1, BCE y tiempo promedio de ajuste. La
    mejor combinacion se selecciona por mayor macro F1 y menor BCE en caso de
    empate.

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
    scoring = {
        "macro_f1": "f1_macro",
        "bce": negative_binary_cross_entropy_scorer,
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=RANDOM_FOREST_PARAM_GRID,
        scoring=scoring,
        cv=cross_validation,
        n_jobs=-1,
        refit=False,
    )
    grid_search.fit(x_train, y_train_encoded)

    search_results = []
    for index, parameters in enumerate(grid_search.cv_results_["params"]):
        search_results.append(
            {
                "parameters": parameters,
                "validation_macro_f1": grid_search.cv_results_[
                    "mean_test_macro_f1"
                ][index],
                "validation_bce": -grid_search.cv_results_[
                    "mean_test_bce"
                ][index],
                "mean_fit_time_seconds": grid_search.cv_results_[
                    "mean_fit_time"
                ][index],
            }
        )

    best_result = None
    for result in search_results:
        if best_result is None:
            best_result = result
            continue

        same_macro_f1 = np.isclose(
            result["validation_macro_f1"],
            best_result["validation_macro_f1"],
        )
        better_result = (
            result["validation_macro_f1"]
            > best_result["validation_macro_f1"]
        )
        if same_macro_f1:
            better_result = (
                result["validation_bce"]
                < best_result["validation_bce"]
            )

        if better_result:
            best_result = result

    best_params = best_result["parameters"]
    refit_start_time = time.perf_counter()
    model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(x_train, y_train_encoded)
    refit_time_seconds = time.perf_counter() - refit_start_time

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

    print("\nResultados de la busqueda de hiperparametros:\n")
    print(
        f"  {'Hiperparametros':<75}  {'Macro F1 CV':>11}  "
        f"{'BCE CV':>10}  {'Tiempo medio (s)':>17}"
    )
    for result in search_results:
        parameters_text = ", ".join(
            f"{parameter}={value}"
            for parameter, value in result["parameters"].items()
        )
        print(
            f"  {parameters_text:<75}  "
            f"{result['validation_macro_f1']:>11.4f}  "
            f"{result['validation_bce']:>10.6f}  "
            f"{result['mean_fit_time_seconds']:>17.4f}"
        )

    print("\nMejores hiperparametros (GridSearchCV):")
    for parameter, value in best_params.items():
        print(f"  {parameter}: {value}")
    print(
        "  Macro F1 medio de validacion cruzada: "
        f"{best_result['validation_macro_f1']:.4f}"
    )
    print(
        "  Tiempo de reconstruccion final (s): "
        f"{refit_time_seconds:.4f}"
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
