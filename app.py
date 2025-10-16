from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN Y CARGA DE RECURSOS GLOBALES (Igual que antes).
# ----------------------------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Rutas a los archivos de recursos
ARCHIVO_EMBEDDINGS_EN = os.path.join(BASE_DIR, 'description_embeddings.npy') 
ARCHIVO_DATOS_EN = os.path.join(BASE_DIR, 'netflix_titles.csv') 

# Rutas a los archivos de recursos es español
ARCHIVO_EMBEDDINGS_ES = os.path.join(BASE_DIR, 'description_embeddings_es.npy') 
ARCHIVO_DATOS_ES = os.path.join(BASE_DIR, 'netflix_titles_es.csv') 

global_resources = {
    'en': {'embeddings': None, 'df': None, 'model': None},
    'es': {'embeddings': None, 'df': None, 'model': None}
}


def cargar_recursos():
    """Carga los modelos, DataFrames y embeddings pre-calculados para ambos idiomas."""
    print("Iniciando la carga de recursos...")
    
    # Carga para el idioma INGLÉS (en)
    try:
        global_resources['en']['model'] = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') 
        df_en = pd.read_csv(ARCHIVO_DATOS_EN)
        global_resources['en']['df'] = df_en.dropna(subset=['description']).reset_index(drop=True)
        
        if os.path.exists(ARCHIVO_EMBEDDINGS_EN):
            global_resources['en']['embeddings'] = np.load(ARCHIVO_EMBEDDINGS_EN)
            print("Recursos en inglés cargados exitosamente.")
        else:
            print(f"🚨 ADVERTENCIA: El archivo de embeddings en inglés '{ARCHIVO_EMBEDDINGS_EN}' no se encontró.")
    except Exception as e:
        print(f"🚨 Error al cargar recursos en inglés: {type(e).__name__} - {e}")
    
    # Carga para el idioma ESPAÑOL (es)
    try:
        global_resources['es']['model'] = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') 
        df_es = pd.read_csv(ARCHIVO_DATOS_ES)
        global_resources['es']['df'] = df_es.dropna(subset=['description']).reset_index(drop=True)
        if os.path.exists(ARCHIVO_EMBEDDINGS_ES):
            global_resources['es']['embeddings'] = np.load(ARCHIVO_EMBEDDINGS_ES)
            print("Recursos en español cargados exitosamente.")
        else:
            print(f"🚨 ADVERTENCIA: El archivo de embeddings en español '{ARCHIVO_EMBEDDINGS_ES}' no se encontró.")
    except Exception as e:
        print(f"🚨 Error al cargar recursos en español: {type(e).__name__} - {e}")


print("Archivos encontrados:")
print(os.listdir(BASE_DIR))
print(" Fin Archivos encontrados:")
for archivo in [ARCHIVO_EMBEDDINGS_EN, ARCHIVO_DATOS_EN, ARCHIVO_EMBEDDINGS_ES, ARCHIVO_DATOS_ES]:
    if not os.path.exists(archivo):
        print(f"⚠️ Archivo no encontrado: {archivo}")
    else:
        print(f"✅ Archivo encontrado: {archivo}")
        
        
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
    idioma_seleccionado = 'en' # Idioma predeterminado
    
    if request.method == 'POST':
        consulta_buscada = request.form.get('consulta')
        idioma_seleccionado = request.form.get('idioma')
        
        try:
            num_recs = int(request.form.get('num_recs'))
        except (ValueError, TypeError):
            num_recs = 5 
        
        # Seleccionar los recursos según el idioma
        recursos = global_resources.get(idioma_seleccionado)

        if recursos and recursos['embeddings'] is not None and recursos['df'] is not None and recursos['model'] is not None:
            recomendaciones = obtener_recomendaciones(
                input_usuario=consulta_buscada,
                df_base=recursos['df'],
                embeddings_base=recursos['embeddings'],
                model=recursos['model'],
                n_sugerencias=num_recs
            )
        else:
            error = f"Error: Los recursos para el idioma '{idioma_seleccionado}' no están disponibles. Asegúrate de que los archivos existan."
    
    return render_template(
        'index.html', 
        recomendaciones=recomendaciones,
        consulta_buscada=consulta_buscada,
        num_recs=num_recs,
        error=error,
        idioma_seleccionado=idioma_seleccionado
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)