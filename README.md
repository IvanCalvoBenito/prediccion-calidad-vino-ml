# Predicción de la Calidad del Vino mediante Técnicas de Machine Learning

> **Trabajo académico** — Doble Grado en Ingeniería Informática y Estadística, Universidad de Salamanca (USAL)
> **Asignatura:** Técnicas Estadísticas en Minería de Datos (3º curso)
> **Autor:** Iván Calvo Benito

Este repositorio contiene el trabajo final de la asignatura *Técnicas Estadísticas en Minería de Datos*, cuyo objetivo es realizar un Análisis Exploratorio de Datos (EDA) y construir un modelo de Machine Learning capaz de predecir la calidad del vino a partir de sus características fisicoquímicas.

---

## 📋 Descripción del proyecto

El conjunto de datos utilizado (`wine_quality.csv`) combina los dos datasets originales de *Wine Quality* del repositorio UCI (vino blanco y vino tinto), con 6497 observaciones y 12 variables fisicoquímicas. La variable respuesta, `quality`, toma valores enteros entre 3 y 9, por lo que el problema se aborda como una **clasificación multiclase**.

El trabajo cubre el flujo completo de un proyecto de ciencia de datos:

1. **EDA** — limpieza de duplicados, distribución de la variable respuesta, análisis de outliers (IQR), matriz de correlación y contraste no paramétrico de Kruskal-Wallis por niveles de calidad.
2. **Preprocesado** — imputación por mediana, estandarización (`StandardScaler`) y selección de variables (`SelectKBest`), todo integrado en un `Pipeline` de scikit-learn para evitar fuga de información.
3. **Modelado** — comparación de cuatro algoritmos supervisados: Regresión Logística, KNN, SVM (kernel RBF) y Random Forest.
4. **Optimización** — validación cruzada anidada (5 folds externos / 3 internos) con `RandomizedSearchCV`, usando **F1-macro** como métrica principal por el fuerte desbalanceo de clases.
5. **Evaluación** — predicciones *out-of-fold* (`cross_val_predict`) con la configuración modal de hiperparámetros, matriz de confusión y curvas Precision-Recall y ROC micro-promediadas con punto óptimo de funcionamiento.

## 🏆 Resultados principales

| Métrica | Valor |
|---|---|
| Modelo seleccionado | Random Forest |
| F1-macro (Nested CV) | 0.2870 |
| Accuracy (Nested CV) | 0.5034 |
| F1-macro (Out-of-Fold) | 0.2851 |
| Accuracy (Out-of-Fold) | 0.4818 |
| Average Precision (micro) | 0.4897 |
| AUC ROC (micro) | 0.8787 |

**Variables seleccionadas** (estables en los 5 folds externos): `volatile_acidity`, `chlorides`, `free_sulfur_dioxide`, `density`, `alcohol`.

Random Forest se seleccionó por obtener el mayor F1-macro medio, la métrica más adecuada dado el marcado desbalanceo de la variable `quality` (concentrada en las calidades 5 y 6). El modelo predice mejor las clases centrales (5, 6, 7) y presenta más dificultades en las clases extremas (3, 9) debido a su escasa representación.

## 📁 Estructura del repositorio

```
prediccion-calidad-vino-ml/
├── README.md
├── notebook/
│   └── Trabajo_final.ipynb      # Notebook completo: EDA + modelado + evaluación
├── data/
│   └── wine_quality.csv         # Dataset (UCI Wine Quality, blanco + tinto)
└── memoria/
    └── memoria.pdf              # Memoria del trabajo en formato PDF
```

## 🛠️ Tecnologías y librerías

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (`Pipeline`, `SelectKBest`, `StratifiedKFold`, `RandomizedSearchCV`, `cross_val_predict`)
- scipy (`stats.kruskal`, `stats.t`)

## ▶️ Cómo ejecutar el notebook

El notebook se desarrolló y ejecutó originalmente en **Google Colab**, aunque también puede ejecutarse en local con Jupyter.

### Opción A — Google Colab (recomendada)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IvanCalvoBenito/prediccion-calidad-vino-ml/blob/main/notebook/Trabajo_final.ipynb)

1. Abre el notebook con el botón de arriba (o desde Colab: *Archivo → Abrir notebook → GitHub* y busca este repositorio).
2. Colab ya trae preinstaladas pandas, numpy, matplotlib, seaborn, scikit-learn y scipy, así que no hace falta instalar nada.
3. Sube `wine_quality.csv` a la sesión de Colab (icono de carpeta → *Subir*) o cárgalo directamente desde el repositorio con:
   ```python
   import pandas as pd
   url = "https://raw.githubusercontent.com/IvanCalvoBenito/prediccion-calidad-vino-ml/main/data/wine_quality.csv"
   wine = pd.read_csv(url)
   ```
   Ajusta la celda de carga de datos (`DATA_PATH`) según la opción elegida.

### Opción B — Entorno local con Jupyter

```bash
git clone https://github.com/IvanCalvoBenito/prediccion-calidad-vino-ml.git
cd prediccion-calidad-vino-ml
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
jupyter notebook notebook/Trabajo_final.ipynb
```

> En local, el notebook está preparado para ejecutarse con `wine_quality.csv` accesible desde la ruta configurada en la celda de carga de datos (por defecto, `data/wine_quality.csv`).

## 📚 Fuente de los datos

- UCI Machine Learning Repository — [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.

El fichero `wine_quality.csv` corresponde a la versión proporcionada por el profesorado de la asignatura, que une los dos conjuntos originales de UCI (vino blanco y vino tinto).

## 👤 Autor

**Iván Calvo Benito**
Doble Grado en Ingeniería Informática y Estadística — Universidad de Salamanca
[GitHub](https://github.com/IvanCalvoBenito) · [LinkedIn](https://www.linkedin.com/in/iván-calvo-benito-8ab7a4431)

---

*Este repositorio se publica con fines académicos y de portfolio personal.*
