# Deep Learning — Evaluación Parcial N°1

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ealejandro23/DeepLearning/blob/main/EV1_DeepLearningOrtega_Gamboa.ipynb)

Clasificación de prendas de vestir sobre **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)** mediante una red neuronal densa (**MLP**) construida con **TensorFlow / Keras**.

> Asignatura **DLY0100 — Deep Learning** · Trabajo de Ortega y Gamboa.

---

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| [`EV1_DeepLearningOrtega_Gamboa.ipynb`](./EV1_DeepLearningOrtega_Gamboa.ipynb) | Cuaderno completo: 62 celdas (31 de texto y 31 de código), con los resultados y gráficos ya guardados. |
| [`requirements.txt`](./requirements.txt) | Dependencias necesarias para ejecutarlo fuera de Colab. |

## Cómo ejecutarlo

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

## Estructura del cuaderno

El cuaderno está organizado en tres bloques, cada uno con su expositor:

| # | Sección | Contenido | Bloque |
|---|---|---|---|
| 1 | Introducción | Problema, objetivo y dataset | 🎤 Persona 1 |
| 2 | Carga y preprocesamiento | Librerías, descarga, control de calidad, normalización, validación y evidencia empírica de que escalar importa (2.1 – 2.6) | 🎤 Persona 1 |
| 3 | Definición del modelo | Arquitectura de la red y las tres funciones clave: activación, pérdida y salida (3.1 – 3.2) | 🎤 Persona 1 |
| 4 | Entrenamiento y ajuste de hiperparámetros | Funciones auxiliares, modelo base y los cinco experimentos controlados (4.1 – 4.7) | 🎤 Persona 2 |
| 5 | Evaluación del modelo | Accuracy, precision, recall, F1 y matriz de confusión | 🎤 Persona 3 |
| 6 | Análisis y mejora | Regularización, reentrenamiento y tabla consolidada (6.1 – 6.3) | 🎤 Persona 3 |
| 7 | Conclusiones | Resultados y mejoras propuestas | 🎤 Persona 3 |

### Experimentos controlados

Cada experimento altera **un solo factor** manteniendo fija la semilla, de modo que
las diferencias se atribuyen al parámetro estudiado y no al azar.

| | Factor | Valores comparados |
|---|---|---|
| **A** | Función de activación | ReLU · tanh · sigmoide |
| **B** | Función de pérdida | entropía cruzada · MSE |
| **C** | Learning rate | 0.1 · 0.01 · 0.001 · 0.0001 |
| **D** | Tamaño del batch | 16 · 64 · 256 |
| **E** | Optimizador | SGD · SGD+Momentum · RMSProp · Adam |

## Referencias

- Xiao, Rasul & Vollgraf (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)
- [Documentación oficial de Keras](https://keras.io/)
- [Tutoriales de TensorFlow](https://www.tensorflow.org/tutorials)
- [Métricas de evaluación en scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html)
