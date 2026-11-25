# Deep Learning — Evaluaciones Parciales

Cuadernos de las evaluaciones de la asignatura **DLY0100 — Deep Learning**. Trabajo de Ortega y Gamboa.

| Entrega | Tema | Cuaderno | Colab |
|---|---|---|---|
| **EV1** | Clasificación de imágenes (Fashion-MNIST) | [`EV1_DeepLearningOrtega_Gamboa.ipynb`](./EV1_DeepLearningOrtega_Gamboa.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ealejandro23/DeepLearning/blob/main/EV1_DeepLearningOrtega_Gamboa.ipynb) |
| **EV3.1** | Modelo de lenguaje con RNNs | [`ev3/Entregable_3.1.ipynb`](./ev3/Entregable_3.1.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ealejandro23/DeepLearning/blob/main/ev3/Entregable_3.1.ipynb) |
| **EV3.2** | Transformer para NLP | [`ev3/Entregable_3.2.ipynb`](./ev3/Entregable_3.2.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ealejandro23/DeepLearning/blob/main/ev3/Entregable_3.2.ipynb) |

---

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| [`EV1_DeepLearningOrtega_Gamboa.ipynb`](./EV1_DeepLearningOrtega_Gamboa.ipynb) | Cuaderno completo: 62 celdas (31 de texto y 31 de código), con los resultados y gráficos ya guardados. |
| [`ev3/Entregable_3.1.ipynb`](./ev3/Entregable_3.1.ipynb) | Modelo del lenguaje a nivel de carácter con RNNs entrenado sobre *El Quijote*, con experimentos y generación de texto. |
| [`ev3/Don_Quijote.txt`](./ev3/Don_Quijote.txt) | Corpus en texto plano usado por EV3.1 (~2 MB). |
| [`ev3/Entregable_3.2.ipynb`](./ev3/Entregable_3.2.ipynb) | Transformer encoder-decoder para generación de respuestas sobre DailyDialog, con ablaciones y búsqueda de hiperparámetros. |
| [`ev3/dataset_3ra_evaluacion.zip`](./ev3/dataset_3ra_evaluacion.zip) | CSVs de `train`, `validation` y `test` de DailyDialog usados por EV3.2 (~2,5 MB). |
| [`requirements.txt`](./requirements.txt) | Dependencias necesarias para ejecutar los cuadernos fuera de Colab. |

---

## Evaluación 1 — Fashion-MNIST (EV1)

Clasificación de prendas de vestir sobre **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)** mediante una red neuronal densa (**MLP**) construida con **TensorFlow / Keras**.

### Cómo ejecutarlo

**En Google Colab (recomendado)** — pulsar el badge de arriba y luego
`Entorno de ejecución → Ejecutar todo`.

- Activar la GPU (`Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU`) acelera bastante el entrenamiento.
- **No hace falta subir ningún archivo**: el dataset se descarga desde `keras.datasets`.
- El orden de las celdas importa: cada experimento depende de los anteriores.

**En local:**

```bash
git clone https://github.com/ealejandro23/DeepLearning.git
cd DeepLearning
pip install -r requirements.txt
jupyter notebook EV1_DeepLearningOrtega_Gamboa.ipynb
```

### Estructura del cuaderno

El cuaderno está organizado en siete secciones:

| # | Sección | Contenido |
|---|---|---|
| 1 | Introducción | Problema, objetivo y dataset |
| 2 | Carga y preprocesamiento | Librerías, descarga, control de calidad, normalización, validación y evidencia empírica de que escalar importa (2.1 – 2.6) |
| 3 | Definición del modelo | Arquitectura de la red y las tres funciones clave: activación, pérdida y salida (3.1 – 3.2) |
| 4 | Entrenamiento y ajuste de hiperparámetros | Funciones auxiliares, modelo base y los cinco experimentos controlados (4.1 – 4.7) |
| 5 | Evaluación del modelo | Accuracy, precision, recall, F1 y matriz de confusión |
| 6 | Análisis y mejora | Regularización, reentrenamiento y tabla consolidada (6.1 – 6.3) |
| 7 | Conclusiones | Resultados y mejoras propuestas |

#### Experimentos controlados

Cada experimento altera **un solo factor** manteniendo fija la semilla, de modo que
las diferencias se atribuyen al parámetro estudiado y no al azar.

| | Factor | Valores comparados |
|---|---|---|
| **A** | Función de activación | ReLU · tanh · sigmoide |
| **B** | Función de pérdida | entropía cruzada · MSE |
| **C** | Learning rate | 0.1 · 0.01 · 0.001 · 0.0001 |
| **D** | Tamaño del batch | 16 · 64 · 256 |
| **E** | Optimizador | SGD · SGD+Momentum · RMSProp · Adam |

---

## Evaluación 3 — Modelos de secuencia (EV3)

Los cuadernos de esta entrega están en la carpeta [`ev3/`](./ev3/) junto con sus datos.

### 3.1 — Modelo del lenguaje con RNNs

Modelo a nivel de **carácter** entrenado sobre el texto de *Don Quijote de la Mancha*:
codificación one-hot, ventanas de 30 caracteres, comparación de arquitecturas LSTM y GRU,
y generación de texto a partir de una semilla (con muestreo por temperatura). Incluye una tabla de decisiones
y la justificación de cada hiperparámetro.

### 3.2 — Transformer para procesamiento de lenguaje natural

Implementación de un **Transformer encoder-decoder** (Pre-LN, multi-head attention,
positional encoding y máscara de padding) sobre **DailyDialog**: dado el turno anterior
de una conversación, el modelo genera la respuesta. Incluye EDA del corpus, pipeline
`tf.data`, loss enmascarada, ablaciones y búsqueda de hiperparámetros.

> **Modo de ejecución:** ambos cuadernos detectan si hay GPU. Con GPU usan la configuración
> completa; sin GPU reducen epochs y tamaño de los datos para poder terminar en CPU.

### Cómo ejecutar EV3

**En local:**

```bash
git clone https://github.com/ealejandro23/DeepLearning.git
cd DeepLearning/ev3
pip install -r ../requirements.txt
jupyter notebook Entregable_3.1.ipynb   # o Entregable_3.2.ipynb
```

**En Google Colab:** abrir el cuaderno con el badge de la tabla inicial y subir los archivos
de datos cuando el cuaderno lo pida (o montar Drive con la carpeta `ev3/`).

Los cuadernos buscan los archivos de forma recursiva en la carpeta actual, `Downloads`
y `/content`, y descomprimen automáticamente el ZIP de DailyDialog. Conviene dejar los
cuatro archivos de `ev3/` juntos.

---

## Referencias

- Xiao, Rasul & Vollgraf (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)
- Sutskever, Vinyals & Le (2014). *Sequence to Sequence Learning with Neural Networks*. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
- Vaswani et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- [Documentación oficial de Keras](https://keras.io/)
- [Tutoriales de TensorFlow](https://www.tensorflow.org/tutorials)
- [Métricas de evaluación en scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html)
