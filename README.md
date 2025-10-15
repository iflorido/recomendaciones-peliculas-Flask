# 🍿 Sistema de Recomendación de Contenido de Netflix en Flask

## 📝 Descripción del Proyecto
* **Ignacio Florido**

Este proyecto es un pequeño **Sistema de Recomendación basado en Contenido de Netflix** utilizando el conjunto de datos de títulos de Netflix. 

El enfoque central es la **similitud semántica** (Procesamiento de Lenguaje Natural o NLP) aplicada a la columna de descripciones (`description`). El modelo permite sugerir títulos al usuario midiendo qué tan parecido es el significado de su consulta de búsqueda al de las sinopsis de las películas y series.

El trabajo se desarrolla completamente en el *Flask* y está diseñado para ser la base de una futura aplicación web.

***

## ✨ Características y Análisis Clave

El proyecto cubre las siguientes áreas:

### 1. Motor de Recomendación Semántica
* **Vectorización (Embeddings):** Uso del modelo **Sentence-Transformers (S-BERT)** para transformar cada descripción en un vector numérico de alta dimensionalidad, capturando su significado semántico.
* **Similitud Cosenoidal:** Implementación de una función que calcula la **similitud del coseno** entre la consulta del usuario y todos los *embeddings* pregenerados para encontrar los títulos más relevantes.
* **Optimización:** El proyecto incluye la funcionalidad para generar y guardar la matriz de *embeddings* (`description_embeddings.npy`) en un archivo, permitiendo que una aplicación web (Flask/Django) cargue el modelo instantáneamente al iniciar, evitando la costosa recodificación en cada petición.
* **Generación de description_embeddings :** El proyecto se nutre del archivo generado en otro proyecto https://github.com/iflorido/sugerencias-Netfilx-jupyter-notebook.git Aquí se carga el primer csv con toda la imformación, se pocesa sacando algunas gráficas y por último generando un .npy que utilizaremos en esta aplicación. Esto en un futuro se puede automatizar cada cierto tiempo para que esté actualizado por ejemplo cada 2 días.



***

## 🛠️ Tecnologías y Requerimientos

* **Python 3.11**
* **Flask**
* **Pandas & NumPy:** Manipulación de datos y matrices.
* **Seaborn & Matplotlib:** Visualización de resultados.
* **Sentence-Transformers:** Generación de *embeddings* de texto.
* **Scikit-learn:** Funciones de métricas de similitud (Similitud del Coseno).

### Instalación de Dependencias

```bash
pip install Flask pandas numpy seaborn matplotlib scikit-learn sentence-transformers


