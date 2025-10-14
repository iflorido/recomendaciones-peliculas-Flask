import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator

# Archivo de entrada y salida
input_file = "netflix_titles.csv"
output_file = "netflix_titles_es.csv"

# Cargar el CSV
df = pd.read_csv(input_file)

# Función para dividir texto largo en fragmentos
def dividir_texto(texto, max_len=4000):
    frases = texto.split('. ')
    fragmentos = []
    actual = ''
    for f in frases:
        if len(actual) + len(f) < max_len:
            actual += f + '. '
        else:
            fragmentos.append(actual.strip())
            actual = f + '. '
    if actual:
        fragmentos.append(actual.strip())
    return fragmentos

# Función de traducción segura
def traducir_por_fragmentos(texto):
    if not isinstance(texto, str) or texto.strip() == "":
        return texto
    try:
        fragmentos = dividir_texto(texto)
        traducido = []
        for frag in fragmentos:
            frag_es = GoogleTranslator(source='en', target='es').translate(frag)
            traducido.append(frag_es)
        return " ".join(traducido)
    except Exception as e:
        print(f"Error traduciendo: {e}")
        return texto  # Devuelve el texto original si hay un error

# Traducir con barra de progreso
tqdm.pandas(desc="Traduciendo descripciones")
df["description"] = df["description"].progress_apply(traducir_por_fragmentos)

# Guardar nuevo CSV
df.to_csv(output_file, index=False)
print(f"\n✅ Traducción completada. Archivo guardado como: {output_file}")
