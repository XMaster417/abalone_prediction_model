import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


LOGISTIC_REGRESSION_PARAM_GRID = {
    "epochs": [1000, 3000, 5000, 10000],
    "learning_rate": [0.0001, 0.001, 0.01, 0.05],
}


"""
================================================================================
    Funcion: sigmoid
    Funcion de activacion sigmoide, para retornar valores entre 0 y 1.

    @param z: funcion lineal a convertir con la sigmoide.
    @return num: valor convertido en el rango de 0 a 1.
================================================================================
"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
"""
================================================================================
    Funcion: binary_cross_entropy
    Funcion para calcular el error de los valores predichos.

    @param y_true -> num: valor real de y.
    @param y_hat -> num: valor predicho con la sigmoide de y.
    @return num: error o perdida por cada dato.
================================================================================
"""
def binary_cross_entropy(y_true, y_hat):
    return -(
        y_true * np.log(y_hat)
        + (1.0 - y_true) * np.log(1.0 - y_hat)
    )
"""
================================================================================
    Funcion: distributed_split
    Devuelve las proporciones distribuidas para los conjuntos de train, test y
    val.

    @param x -> matrix: valores de las variables x
    @param y -> matrix: valores de la variable y
    @param train_size -> float: porcentaje de tamaño asignado al conjunto train
    @param val_size -> float: porcentaje de tamaño asignado al conjunto de val
    @param seed -> num: valor escogido para la aleatoriedad
    @return result -> list<tuple>: conjuntos de train, val y test.
================================================================================
"""
def distributed_split(x, y, train_size=0.70, val_size=0.15, seed=1):
    rng = np.random.default_rng(seed)
    splits = [[], [], []]
    num_class = y.shape[1]

    for class_index in range(num_class):
        class_indices = [
            row_index
            ## Regresa los indices con su valor con ayuda de enumerate
            for row_index, value in enumerate(y[:, class_index])
            if value == 1
        ]
        rng.shuffle(class_indices)
        train_end = int(len(class_indices) * train_size)
        val_end = train_end + int(len(class_indices) * val_size)
        splits[0].extend(class_indices[:train_end])
        splits[1].extend(class_indices[train_end:val_end])
        splits[2].extend(class_indices[val_end:])

    result = []
    for indices in splits:
        rng.shuffle(indices)
        result.append((x[indices], y[indices]))
    return result
"""
================================================================================
    Funcion: train.
    Train entrena al modelo para identificar si es I, M, F almacena los pesos y
    bias en una lista para evaluar la opcion mas probable utilizando la tecnica
    de one vs all.

    @param x_train -> matriz: matriz con las variables independientes para train.
    @param y_train -> matriz: matriz las respuestas para train.
    @param datasets -> list<list>: lista con los conjuntos de train, test y val.
    @param epochs -> int: numero de veces que el modelo entrena.
    @param alpha -> float: tasa de aprendizaje del modelo, ajusta el movimiento.
    @return weights, biases, history -> pesos y bias del modelo, asi como los
        errores en history para graficar el BCE.
================================================================================
"""
def train(x_train, y_train, datasets, epochs=3000, alpha=0.001):
    n_samples, n_features = x_train.shape
    n_classes = y_train.shape[1]
    weights = np.zeros((n_features, n_classes))
    biases = np.zeros(n_classes)
    history = {name: [] for name in datasets}

    for _ in range(epochs):
        predictions = sigmoid(np.dot(x_train, weights) + biases)
        loss = predictions - y_train

        theta_weight = np.dot(x_train.T, loss) / n_samples
        theta_bias = np.mean(loss, axis=0)
        weights -= alpha * theta_weight
        biases -= alpha * theta_bias

        for name, (x_split, y_split) in datasets.items():
            y_hat = sigmoid(np.dot(x_split, weights) + biases)
            history[name].append(
                float(np.mean(binary_cross_entropy(y_split, y_hat)))
            )

    return weights, biases, history
"""
================================================================================
    Funcion: predict.
    A partir de un modelo entrenado obtiene la prediccion mayor para cada clase
    objetivo del dataset.

    @param x -> matrix: matriz con los valores de las variables independientes
    @param weights -> matrix: pesos por variable aprendidos durante train.
    @param biases -> matrix: sesgos por variable aprendidos durante train.
    @return tuple: indices de las clases predichas y sus probabilidades.
================================================================================
"""
def predict(x, weights, biases):
    y_hat = sigmoid(np.dot(x, weights) + biases)
    return np.argmax(y_hat, axis=1), y_hat

"""
================================================================================
    Funcion: accuracy.
    Calcula la proporcion de clases correctamente predichas para un conjunto de 
    datos.

    @param y_true -> matrix: matriz con los valores reales de Sex.
    @param y_hat -> np.array: arreglo con el indice de la clase predicha (0,1,2)
    @return num: exactitud del conjunto de datos para predecir la clase
================================================================================
"""
def accuracy(y_true, y_hat):
    true_classes = np.argmax(y_true, axis=1)
    return np.mean(true_classes == y_hat)
"""
================================================================================
    Funcion: hyperparameter_search.
    Prueba todas las combinaciones de epocas y tasa de aprendizaje definidas
    en param_grid. Selecciona primero la mayor exactitud de validacion y usa el
    menor BCE de validacion como criterio de desempate.

    @param x_train -> matrix: variables independientes de entrenamiento.
    @param y_train -> matrix: clases one hot de entrenamiento.
    @param x_val -> matrix: variables independientes de validacion.
    @param y_val -> matrix: clases one hot de validacion.
    @param param_grid -> Dictionary: valores de hiperparametros a probar.
    @return tuple: mejores parametros, mejores metricas y todos los resultados.
================================================================================
"""
def hyperparameter_search(x_train, y_train, x_val, y_val, param_grid):
    search_results = []
    best_params = None
    best_validation_accuracy = -np.inf
    best_validation_bce = np.inf

    for epochs in param_grid["epochs"]:
        for learning_rate in param_grid["learning_rate"]:
            # Durante la busqueda no se guarda el historial para evitar
            # calculos innecesarios. El conjunto test no participa aqui.
            weights, biases, _ = train(
                x_train,
                y_train,
                datasets={},
                epochs=epochs,
                alpha=learning_rate,
            )
            validation_predictions, validation_probabilities = predict(
                x_val,
                weights,
                biases,
            )
            validation_accuracy = float(
                accuracy(y_val, validation_predictions)
            )
            validation_bce = float(
                np.mean(
                    binary_cross_entropy(y_val, validation_probabilities)
                )
            )

            result = {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "validation_accuracy": validation_accuracy,
                "validation_bce": validation_bce,
            }
            search_results.append(result)

            same_accuracy = np.isclose(
                validation_accuracy,
                best_validation_accuracy,
            )
            better_result = validation_accuracy > best_validation_accuracy
            if same_accuracy:
                better_result = validation_bce < best_validation_bce

            if better_result:
                best_params = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                }
                best_validation_accuracy = validation_accuracy
                best_validation_bce = validation_bce

    return (
        best_params,
        best_validation_accuracy,
        best_validation_bce,
        search_results,
    )
"""
================================================================================
    Funcion: confusion_matrix.
    Muestra la matriz de confusion con el resultado de las predicciones contra
    los valores reales.

    @param y_true -> matrix: matriz con los valores reales de Sex.
    @param y_hat -> np.array: arreglo con el indice de la clase predicha (0,1,2)
    @param class_names -> list<String>: lista con los nombres de las clases
================================================================================
"""
def confusion_matrix(y_true, y_hat, class_names):
    # Indice de la clase verdadera.
    true_classes = np.argmax(y_true, axis=1)
    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    np.add.at(confusion, (true_classes, y_hat), 1)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=True,  # Muestra los valores en la matriz de confusion.
        fmt="d",  # Muestra los valores como enteros.
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Matriz de confusion")
    plt.xlabel("Prediccion")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.show()
"""
================================================================================
    Funcion: logistic_regression_analysis()
    Realiza una regresion logistica, primero identifica las variables
    dependientes e independientes y las convierte a arreglos de numpy para
    efectuar los calculos. Separa los datos en train, val y test. Se
    estandarizan los valores porque Rings maneja valores muy grandes respecto a
    las demas medidas. Entrena al modelo utilizando regresion logistica con
    sigmoide, binary cross entropy, gradiente descendente y one vs all. Despues
    realiza las predicciones correspondientes. Finalmente imprime los resultados
    en una matriz de confusion y grafica los errores del modelo.

    @param df -> dataFrame: datos transformados con one hot encoding.
================================================================================
"""
def logistic_regression_analysis(df):
    class_names = ["I", "M", "F"]
    target_columns = ["Sex_I", "Sex_M", "Sex_F"]
    feature_columns = [
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        "Rings",
    ]

    x = df[feature_columns].to_numpy(dtype=float)
    y = df[target_columns].to_numpy(dtype=float)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = distributed_split(
        x, y, seed=1
    )

    # Estandarizacion calculada exclusivamente con train.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    (
        best_params,
        best_validation_accuracy,
        best_validation_bce,
        search_results,
    ) = hyperparameter_search(
        x_train,
        y_train,
        x_val,
        y_val,
        LOGISTIC_REGRESSION_PARAM_GRID,
    )

    datasets = {
        "Train": (x_train, y_train),
        "Validacion": (x_val, y_val),
        "Test": (x_test, y_test),
    }

    weights, biases, history = train(
        x_train,
        y_train,
        datasets,
        best_params["epochs"],
        best_params["learning_rate"],
    )

    predictions_train, _ = predict(x_train, weights, biases)
    predictions_val, _ = predict(x_val, weights, biases)
    predictions_test, _ = predict(x_test, weights, biases)
    print("\nResumen de regresion logistica")
    print("\nResultados de la busqueda de hiperparametros:")
    print(
        f"  {'Epocas':>6}  {'Learning rate':>13}  "
        f"{'Accuracy val':>12}  {'BCE val':>10}"
    )
    for result in search_results:
        print(
            f"  {result['epochs']:>6}  "
            f"{result['learning_rate']:>13.4g}  "
            f"{result['validation_accuracy']:>12.4f}  "
            f"{result['validation_bce']:>10.6f}"
        )

    print("\nMejores hiperparametros:")
    print(f"  epochs: {best_params['epochs']}")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  Accuracy de validacion: {best_validation_accuracy:.4f}")
    print(f"  BCE de validacion: {best_validation_bce:.6f}")

    print("\nTamaños de los conjuntos:")
    print(f"  Entrenamiento: {len(x_train)}")
    print(f"  Validacion: {len(x_val)}")
    print(f"  Prueba: {len(x_test)}")
    print("\nBCE final:")
    for name, losses in history.items():
        print(f"  {name}: {losses[-1]:.6f}")
    # Este resumen se imprime en consola; no se agrega al heatmap de correlacion.
    print("\nExactitud por conjunto:")
    print(f"  Entrenamiento: {accuracy(y_train, predictions_train):.4f}")
    print(f"  Validacion: {accuracy(y_val, predictions_val):.4f}")
    print(f"  Test: {accuracy(y_test, predictions_test):.4f}")
    confusion_matrix(y_test, predictions_test, class_names)

    plt.figure(figsize=(9, 5))
    for name, losses in history.items():
        plt.plot(losses, label=name)
    plt.title("Regresion logistica One-vs-All")
    plt.xlabel("Epocas")
    plt.ylabel("Error BCE")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
