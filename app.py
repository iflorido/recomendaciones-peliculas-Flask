from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN Y CARGA DE RECURSOS GLOBALES (Igual que antes)
# ----------------------------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Rutas a los archivos de recursos
ARCHIVO_EMBEDDINGS = os.path.join(BASE_DIR, 'description_embeddings.npy') 
ARCHIVO_DATOS = os.path.join(BASE_DIR, 'netflix_titles.csv') 

# Rutas a los archivos de recursos es español
ARCHIVO_EMBEDDINGS_ES = os.path.join(BASE_DIR, 'description_embeddings_es.npy') 
ARCHIVO_DATOS_ES = os.path.join(BASE_DIR, 'netflix_titles_es.csv') 

global_description_embeddings = None
global_df_base = None
global_model = None

def cargar_recursos():
    """Carga el modelo de NLP, el DataFrame y los embeddings pre-calculados."""
    global global_description_embeddings
    global global_df_base
    global global_model
    
    print("Iniciando la carga de recursos...")
    
    try:
        # --- DIAGNÓSTICO ---
        print(f"Buscando CSV en: {ARCHIVO_DATOS}")
        print(f"Buscando NPY en: {ARCHIVO_EMBEDDINGS}")
        # --- FIN DIAGNÓSTICO ---
        
        # Cargar el modelo de S-BERT
        global_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') 
        
        # Cargar el DataFrame base
        df_base = pd.read_csv(ARCHIVO_DATOS)
        global_df_base = df_base.dropna(subset=['description']).reset_index(drop=True)
        
        # Cargar el archivo .npy
        if os.path.exists(ARCHIVO_EMBEDDINGS):
            global_description_embeddings = np.load(ARCHIVO_EMBEDDINGS)
            print("Recursos cargados exitosamente.")
        else:
            # Aquí es donde el error es más específico
            raise FileNotFoundError(f"El archivo de embeddings no existe en: {ARCHIVO_EMBEDDINGS}")

    except Exception as e:
        # Imprimirá un error más específico
        print(f"🚨 Error al cargar recursos: {type(e).__name__} - {e}")
        # Retorna None en las variables globales si falla (o mejor, no las modifica)

cargar_recursos() 

# ----------------------------------------------------------------------
# 2. FUNCIÓN DE LÓGICA DE RECOMENDACIÓN (Campos actualizados)
# ----------------------------------------------------------------------

def obtener_recomendaciones(input_usuario, df_base, embeddings_base, model, n_sugerencias=5):
    """
    Toma una consulta de texto y devuelve el Top N de títulos más similares 
    con todos los campos necesarios.
    """
    # 1. Codificar la consulta del usuario
    consulta_embedding = model.encode([input_usuario])[0]
    
    # 2. Calcular la similitud cosenoidal
    similaridades = cosine_similarity([consulta_embedding], embeddings_base)[0]
    
    # 3. Obtener los índices de los títulos más similares (Top N)
    indices_similares = np.argsort(similaridades)[::-1]
    top_indices = indices_similares[:n_sugerencias]
    
    # 4. Devolver los títulos correspondientes con los campos requeridos
    campos_requeridos = ['title', 'director', 'cast', 'duration', 'description', 'type', 'release_year']
    sugerencias_df = df_base.loc[top_indices, campos_requeridos]
    
    # Convertir el DataFrame a una lista de diccionarios para la plantilla Jinja
    return sugerencias_df.to_dict('records')

# ----------------------------------------------------------------------
# 3. RUTAS DE FLASK (Lógica de POST implementada)
# ----------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    recomendaciones = None
    consulta_buscada = ""
    num_recs = 5
    error = None

    if request.method == 'POST':
        # 1. Obtener los inputs del formulario
        consulta_buscada = request.form.get('consulta')
        try:
            num_recs = int(request.form.get('num_recs'))
        except (ValueError, TypeError):
            num_recs = 5 

        # 2. Verificar que los recursos estén cargados
        if global_description_embeddings is None or global_df_base is None or global_model is None:
            error = "Error al iniciar la aplicación. Asegúrate de que los archivos .npy y .csv existan."
            
        # 3. Llamar a la función de recomendación si no hay errores
        if not error:
            recomendaciones = obtener_recomendaciones(
                input_usuario=consulta_buscada,
                df_base=global_df_base,
                embeddings_base=global_description_embeddings,
                model=global_model,
                n_sugerencias=num_recs
            )
        
    # Renderiza la plantilla HTML, pasando todas las variables
    return render_template(
        'index.html', 
        recomendaciones=recomendaciones,
        consulta_buscada=consulta_buscada,
        num_recs=num_recs,
        error=error
    )


if __name__ == '__main__':
    app.run(debug=True)