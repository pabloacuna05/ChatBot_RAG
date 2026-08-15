"""
Modulo RAG: extraccion de texto de PDFs, troceado, generacion de
embeddings con la API de Gemini, y recuperacion por similitud coseno.

El indice se actualiza de forma incremental: solo se vuelven a embeber los
PDFs nuevos o modificados, reutilizando los vectores ya calculados del
resto. Los embeddings se guardan ya normalizados, de modo que la similitud
coseno en cada consulta es un simple producto matriz-vector.
"""

import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from pypdf import PdfReader
from google.genai.errors import APIError

DOCS_DIR = Path(__file__).parent / "docs"
INDEX_DIR = Path(__file__).parent / "rag_index"

EMBEDDING_MODEL = "gemini-embedding-001"
# Dimension de los embeddings. gemini-embedding-001 admite 768, 1536 y 3072.
# Menos dimensiones = menos disco y RAM y busquedas mas rapidas, a cambio de
# algo de precision. OJO: el valor debe ser el mismo al indexar y al
# consultar, por eso se guarda en el manifest y se regenera el indice solo
# si cambia.
EMBEDDING_DIM = 768
CHUNK_SIZE = 900  # caracteres por trozo
CHUNK_OVERLAP = 150  # caracteres solapados entre trozos consecutivos
# OCR para PDFs escaneados (los que son imagenes y no llevan texto). Va
# desactivado porque necesita dos dependencias de Python y, ademas, tener
# Tesseract y Poppler instalados en el sistema:
#   pip install pytesseract pdf2image
OCR_IDIOMA = "spa"  # codigo de idioma de Tesseract: spa, eng, fra...
USAR_OCR = False

EMBEDDING_BATCH_SIZE = 10
MAX_REINTENTOS = 3
ESPERA_BASE_SEGUNDOS = 5
# Lotes de embeddings en paralelo durante el indexado. Subirlo acelera la
# primera indexacion, pero pasado cierto punto solo genera 429 y el backoff
# lo acaba pagando. Si se acumulan 429 la concurrencia baja sola.
WORKERS_EMBEDDING = 4

# Modo de recuperacion:
#   "densa"   solo embeddings. Entiende sinonimos y parafrasis, pero falla
#             con codigos de producto, referencias y cifras exactas.
#   "lexica"  solo BM25. Lo contrario: clava las coincidencias literales y
#             no entiende que "calzado" y "zapatilla" son lo mismo. No gasta
#             embedding de consulta.
#   "hibrida" ambas, fusionadas con Reciprocal Rank Fusion. Es el defecto.
MODO_BUSQUEDA = "hibrida"
# Candidatos que pide cada buscador antes de fusionar. Cuantos mas, mas
# posibilidades de que un buen fragmento entre por alguna de las dos vias.
TOP_K_CANDIDATOS = 20
# Constante de RRF. 60 es el valor del paper original; amortigua el peso de
# las primeras posiciones para que un unico buscador no domine la fusion.
RRF_K = 60

# Umbral de relevancia. Un corte fijo es fragil: el rango de similitudes
# depende del corpus y del modelo de embeddings, asi que el mismo 0.5 cuela
# ruido en un corpus y descarta contexto bueno en otro. En su lugar el corte
# es relativo al mejor resultado de cada pregunta.
#
# MARGEN_RELATIVO: que fraccion del mejor score hay que alcanzar para
#   entrar. Subelo (0.9-0.95) si se cuelan fragmentos tangenciales; bajalo
#   (0.7-0.8) si las respuestas se quedan cortas por falta de contexto.
# UMBRAL_MINIMO: suelo absoluto. Existe para el caso en que no hay nada
#   relevante: sin el, el mejor de un monton de fragmentos malos pasaria
#   igualmente por ser el mejor. Subelo si el bot responde cosas
#   inventadas a preguntas fuera del corpus; bajalo si dice "no tengo esa
#   informacion" a preguntas que si estan cubiertas.
MARGEN_RELATIVO = 0.85
UMBRAL_MINIMO = 0.3

# Backend de busqueda vectorial:
#   "auto"  numpy hasta UMBRAL_CHUNKS_HNSW chunks, hnswlib por encima.
#   "numpy" siempre exacto, sin dependencias. Un producto contra toda la
#           matriz va sobrado hasta decenas de miles de chunks.
#   "hnsw"  siempre aproximado. Requiere "pip install hnswlib". Escala
#           mucho mejor, a cambio de poder perder algun vecino (recall < 1).
BACKEND_VECTORIAL = "auto"
UMBRAL_CHUNKS_HNSW = 50000
# Parametros de HNSW. ef_construction y M afectan a la calidad del grafo al
# construirlo; ef_busqueda al recall en cada consulta (mas alto = mas recall
# y mas lento).
HNSW_EF_CONSTRUCTION = 200
HNSW_M = 16
HNSW_EF_BUSQUEDA = 64

# Reranking: se recuperan TOP_K_CANDIDATOS y se reordenan para quedarse con
# los mejores top_k. Sube la precision del contexto, pero añade una llamada
# por pregunta. Medido con gemini-flash-lite sobre 20 fragmentos, ronda
# 0,6-1,5 s; el backend de Cohere suele quedarse por debajo de 300 ms.
# Viene desactivado para no encarecer cada turno por defecto: activalo y
# comprueba con evaluar.py si te compensa en tu corpus.
USAR_RERANKER = False
RERANKER_BACKEND = "llm"  # "llm" (sin API key extra) | "cohere"
MODELO_RERANKER_LLM = "gemini-flash-lite-latest"
MODELO_RERANKER_COHERE = "rerank-multilingual-v3.0"

# Latencia de la ultima llamada al reranker, en segundos. La lee evaluar.py
# para poder ponerle numero al coste de activarlo.
ULTIMA_LATENCIA_RERANK = 0.0

PROMPT_RERANK = """\
Ordena los FRAGMENTOS por su utilidad real para responder a la PREGUNTA.

Devuelve UNICAMENTE un array JSON, sin texto alrededor ni bloques de
codigo, con un objeto por fragmento util, del mas util al menos util:
[{{"id": 3, "puntuacion": 9}}, {{"id": 1, "puntuacion": 6}}]

- "id" es el numero del fragmento tal y como aparece abajo.
- "puntuacion" va de 0 a 10.
- Omite los fragmentos que no aporten nada a la pregunta.

PREGUNTA:
{pregunta}

FRAGMENTOS:
{fragmentos}
"""

# Version del formato del indice. Hay que subirla cuando un cambio invalida
# los embeddings ya guardados (normalizacion, dimension, modelo, troceado).
# Un indice con version distinta se regenera entero en vez de dar
# resultados incorrectos en silencio.
INDEX_VERSION = 3

# Rutas por defecto, derivadas de INDEX_DIR. Las funciones aceptan otro
# index_dir para poder trabajar sobre indices alternativos (y para los tests).
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npz"
MANIFEST_PATH = INDEX_DIR / "manifest.json"


def _rutas_indice(index_dir=INDEX_DIR):
    """Devuelve (chunks_path, embeddings_path, manifest_path) para un
    directorio de indice dado."""
    index_dir = Path(index_dir)
    return (
        index_dir / "chunks.json",
        index_dir / "embeddings.npz",
        index_dir / "manifest.json",
    )


def _leer_paginas_pdf(pdf_path):
    """Devuelve una lista de (numero_pagina, texto) para un PDF."""
    lector = PdfReader(str(pdf_path))
    paginas = []
    for i, pagina in enumerate(lector.pages, start=1):
        texto = (pagina.extract_text() or "").strip()
        if texto:
            paginas.append((i, texto))

    if not paginas:
        # Un PDF escaneado son imagenes: pypdf no saca nada y, sin aviso, el
        # bot acabaria diciendo "no tengo esa informacion" a todo sin que
        # nadie entienda por que.
        if USAR_OCR:
            return _ocr_pdf(pdf_path)
        print(
            f"  Aviso: de '{Path(pdf_path).name}' no se pudo extraer texto. "
            "Probablemente sea un PDF escaneado (imagenes en vez de texto). "
            "Activa USAR_OCR en rag.py para intentar leerlo con OCR."
        )
    return paginas


def _ocr_pdf(pdf_path):
    """OCR de un PDF escaneado. Dependencias opcionales y externas, por eso
    va detras de una constante y avisa con detalle si falta algo."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:
        print(
            f"  Aviso: OCR activado pero falta una dependencia ({e}). "
            "Instala: pip install pytesseract pdf2image"
        )
        return []

    try:
        imagenes = convert_from_path(str(pdf_path))
    except Exception as e:
        print(
            f"  Aviso: no se pudieron rasterizar las paginas de "
            f"'{Path(pdf_path).name}' ({e}). pdf2image necesita Poppler "
            "instalado en el sistema."
        )
        return []

    paginas = []
    for i, imagen in enumerate(imagenes, start=1):
        try:
            texto = (pytesseract.image_to_string(imagen, lang=OCR_IDIOMA) or "").strip()
        except Exception as e:
            print(
                f"  Aviso: fallo el OCR ({e}). Tesseract tiene que estar "
                "instalado en el sistema, no basta con el paquete de Python."
            )
            return []
        if texto:
            paginas.append((i, texto))

    if paginas:
        print(f"  OCR de '{Path(pdf_path).name}': {len(paginas)} pagina(s) leidas.")
    return paginas


def _leer_texto_plano(ruta):
    """.txt y .md. Se trata todo el archivo como una unica 'pagina'."""
    texto = Path(ruta).read_text(encoding="utf-8", errors="replace").strip()
    return [(1, texto)] if texto else []


class _ExtractorHTML(HTMLParser):
    """Saca el texto visible de un HTML sin añadir dependencias."""

    IGNORAR = {"script", "style", "head", "meta", "link"}
    BLOQUE = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__()
        self.partes = []
        self._saltando = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.IGNORAR:
            self._saltando += 1
        elif tag in self.BLOQUE:
            self.partes.append("\n")

    def handle_endtag(self, tag):
        if tag in self.IGNORAR and self._saltando:
            self._saltando -= 1
        elif tag in self.BLOQUE:
            self.partes.append("\n")

    def handle_data(self, data):
        if not self._saltando and data.strip():
            self.partes.append(data.strip())

    def texto(self):
        return re.sub(r"\n{3,}", "\n\n", " ".join(self.partes)).strip()


def _leer_html(ruta):
    parser = _ExtractorHTML()
    parser.feed(Path(ruta).read_text(encoding="utf-8", errors="replace"))
    texto = parser.texto()
    return [(1, texto)] if texto else []


def _leer_docx(ruta):
    """Un .docx es un zip con XML dentro. Se lee con la stdlib para no
    añadir python-docx solo por extraer texto."""
    espacio = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        with zipfile.ZipFile(ruta) as documento:
            xml = documento.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as e:
        print(f"  Aviso: no se pudo leer '{Path(ruta).name}' como .docx ({e}).")
        return []

    raiz = ElementTree.fromstring(xml)
    parrafos = []
    for parrafo in raiz.iter(f"{espacio}p"):
        trozos = [nodo.text for nodo in parrafo.iter(f"{espacio}t") if nodo.text]
        if trozos:
            parrafos.append("".join(trozos).strip())

    texto = "\n\n".join(p for p in parrafos if p)
    return [(1, texto)] if texto else []


# Interfaz comun: cada extractor recibe una ruta y devuelve [(pagina, texto)],
# de modo que el resto del pipeline (troceado, embeddings, busqueda) no sabe
# ni le importa de que formato viene el contenido.
EXTRACTORES = {
    ".pdf": _leer_paginas_pdf,
    ".txt": _leer_texto_plano,
    ".md": _leer_texto_plano,
    ".markdown": _leer_texto_plano,
    ".html": _leer_html,
    ".htm": _leer_html,
    ".docx": _leer_docx,
}


def _leer_documento(ruta):
    extractor = EXTRACTORES.get(Path(ruta).suffix.lower())
    if extractor is None:
        return []
    try:
        return extractor(ruta)
    except Exception as e:
        print(f"  Aviso: fallo al leer '{Path(ruta).name}' ({e}), lo omito.")
        return []


def _trocear_texto(texto, chunk_size=None, overlap=None):
    """Troceado a lo bruto por numero de caracteres. Solo se usa como ultimo
    recurso, cuando ni siquiera una frase suelta cabe en un chunk.

    Los valores por defecto se leen aqui dentro, no en la firma, para que
    cambiar rag.CHUNK_SIZE en caliente (lo hace el barrido de evaluar.py)
    tenga efecto de verdad."""
    chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = CHUNK_OVERLAP if overlap is None else overlap

    trozos = []
    inicio = 0
    longitud = len(texto)
    while inicio < longitud:
        fin = min(inicio + chunk_size, longitud)
        trozo = texto[inicio:fin].strip()
        if trozo:
            trozos.append(trozo)
        if fin == longitud:
            break
        inicio = fin - overlap
    return trozos


def _concatenar_paginas(paginas):
    """Une las paginas en un unico texto y devuelve los offsets de cada una.

    Trocear pagina a pagina perdia todo lo que cruzaba de una a otra: una
    tabla o un parrafo partido por el salto de pagina quedaba cortado en dos
    fragmentos sin sentido. Concatenando y guardando los offsets se puede
    seguir asignando la pagina correcta a cada chunk."""
    partes = []
    offsets = []  # [(offset_de_inicio, numero_de_pagina), ...]
    posicion = 0
    for numero, texto in paginas:
        offsets.append((posicion, numero))
        partes.append(texto)
        posicion += len(texto) + 2  # +2 por el "\n\n" que los une
    return "\n\n".join(partes), offsets


def _pagina_de_offset(offsets, offset):
    """Pagina a la que pertenece una posicion del texto concatenado."""
    pagina = offsets[0][1] if offsets else 1
    for inicio, numero in offsets:
        if inicio <= offset:
            pagina = numero
        else:
            break
    return pagina


# Titulos numerados tipo "3", "3.2" o "3.2.1)" seguidos de texto.
_RE_SECCION_NUMERADA = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+\S")


def _es_titulo_de_seccion(linea):
    """Heuristica deliberadamente conservadora: mas vale no detectar una
    seccion que etiquetar mal medio documento."""
    linea = linea.strip()
    if not linea or len(linea) > 80:
        return False
    if _RE_SECCION_NUMERADA.match(linea):
        return True
    letras = [c for c in linea if c.isalpha()]
    return len(letras) >= 3 and all(c.isupper() for c in letras)


def _secciones_con_offset(texto):
    secciones = []
    posicion = 0
    for linea in texto.splitlines(keepends=True):
        if _es_titulo_de_seccion(linea):
            secciones.append((posicion, linea.strip()))
        posicion += len(linea)
    return secciones


def _seccion_de_offset(secciones, offset):
    """Titulo de seccion mas cercano por encima de esa posicion."""
    titulo = None
    for posicion, texto in secciones:
        if posicion <= offset:
            titulo = texto
        else:
            break
    return titulo


def _partir_con_offsets(texto, patron, base=0):
    """Parte el texto por un separador conservando el offset absoluto de
    cada trozo, que es lo que permite saber luego su pagina y su seccion."""
    partes = []
    inicio = 0
    for separador in re.finditer(patron, texto):
        segmento = texto[inicio : separador.start()]
        if segmento.strip():
            desplazamiento = len(segmento) - len(segmento.lstrip())
            partes.append((segmento.strip(), base + inicio + desplazamiento))
        inicio = separador.end()
    segmento = texto[inicio:]
    if segmento.strip():
        desplazamiento = len(segmento) - len(segmento.lstrip())
        partes.append((segmento.strip(), base + inicio + desplazamiento))
    return partes


_RE_PARRAFO = re.compile(r"\n\s*\n")
_RE_FRASE = re.compile(r"(?<=[.!?…])\s+")


def _unidades_con_offset(texto, chunk_size):
    """Trocea jerarquicamente: primero parrafos, los parrafos que no quepan
    en frases, y solo si una frase sigue sin caber se corta a lo bruto."""
    unidades = []
    for parrafo, offset_parrafo in _partir_con_offsets(texto, _RE_PARRAFO):
        if len(parrafo) <= chunk_size:
            unidades.append((parrafo, offset_parrafo))
            continue
        for frase, offset_frase in _partir_con_offsets(
            parrafo, _RE_FRASE, base=offset_parrafo
        ):
            if len(frase) <= chunk_size:
                unidades.append((frase, offset_frase))
                continue
            for i in range(0, len(frase), chunk_size):
                unidades.append((frase[i : i + chunk_size], offset_frase + i))
    return unidades


def _empaquetar_unidades(unidades, chunk_size, overlap):
    """Agrupa unidades consecutivas hasta llenar un chunk, arrastrando las
    ultimas como solapamiento para no perder el hilo entre fragmentos."""
    grupos = []
    actual = []
    longitud = 0

    for unidad in unidades:
        texto_unidad = unidad[0]
        if actual and longitud + len(texto_unidad) + 1 > chunk_size:
            grupos.append(actual)
            solapamiento = []
            acumulado = 0
            for previa in reversed(actual):
                if acumulado + len(previa[0]) > overlap:
                    break
                solapamiento.insert(0, previa)
                acumulado += len(previa[0]) + 1
            actual = list(solapamiento)
            longitud = acumulado
        actual.append(unidad)
        longitud += len(texto_unidad) + 1

    if actual:
        grupos.append(actual)
    return grupos


def _trocear_estructurado(paginas, nombre_documento, chunk_size=None, overlap=None):
    """Trocea un documento respetando su estructura y devuelve chunks con
    metadatos.

    Nota sobre unidades: se mide en caracteres, no en tokens, para no
    depender de un tokenizador. En español la equivalencia ronda los 3,5-4
    caracteres por token, asi que CHUNK_SIZE=900 son unos 230-260 tokens.
    Textos con muchas cifras, tablas o codigos gastan mas tokens por
    caracter, asi que ahi el chunk real sera algo mayor de lo estimado."""
    chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = CHUNK_OVERLAP if overlap is None else overlap

    texto, offsets = _concatenar_paginas(paginas)
    if not texto.strip():
        return []

    secciones = _secciones_con_offset(texto)
    unidades = _unidades_con_offset(texto, chunk_size)

    chunks = []
    for grupo in _empaquetar_unidades(unidades, chunk_size, overlap):
        cuerpo = "\n".join(u[0] for u in grupo).strip()
        if not cuerpo:
            continue
        offset = grupo[0][1]
        seccion = _seccion_de_offset(secciones, offset)

        # La cabecera va dentro del texto que se embebe (le da contexto al
        # fragmento: sin ella, "pesa 215 g" no dice de que producto habla) y
        # ademas se guarda aparte por si se quiere mostrar o filtrar por ella.
        # La cabecera incluye siempre la seccion detectada, aunque el cuerpo
        # ya empiece por ese titulo: repetirla cuesta unos pocos tokens, y a
        # cambio "cabecera" y "seccion" siempre dicen lo mismo, que es lo que
        # permite fiarse de esos campos para mostrar o filtrar.
        cabecera = f"{nombre_documento} > {seccion}" if seccion else nombre_documento

        chunks.append(
            {
                "text": f"[{cabecera}]\n{cuerpo}",
                "cuerpo": cuerpo,
                "cabecera": cabecera,
                "seccion": seccion,
                "source": nombre_documento,
                "page": _pagina_de_offset(offsets, offset),
            }
        )
    return chunks


def _hash_documento(ruta):
    """Hash sha256 del contenido del archivo, para detectar cambios."""
    hasher = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(8192), b""):
            hasher.update(bloque)
    return hasher.hexdigest()


def _documentos_en(docs_dir=DOCS_DIR):
    """Todos los archivos de docs_dir cuyo formato sabemos leer."""
    directorio = Path(docs_dir)
    if not directorio.exists():
        return []
    return sorted(
        ruta
        for ruta in directorio.iterdir()
        if ruta.is_file() and ruta.suffix.lower() in EXTRACTORES
    )


def _estado_actual_docs(docs_dir=DOCS_DIR):
    """Mapa {nombre_archivo: hash} de los documentos presentes en docs_dir."""
    return {ruta.name: _hash_documento(ruta) for ruta in _documentos_en(docs_dir)}


def _cargar_manifest(manifest_path=MANIFEST_PATH):
    if Path(manifest_path).exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _guardar_manifest(estado_docs, num_chunks_por_doc, manifest_path=MANIFEST_PATH):
    """Guarda el manifest con la configuracion con la que se genero el
    indice y, por cada documento, su hash y cuantos chunks aporto."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": INDEX_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "documentos": {
            nombre: {
                "hash": hash_doc,
                "num_chunks": num_chunks_por_doc.get(nombre, 0),
            }
            for nombre, hash_doc in estado_docs.items()
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _manifest_compatible(manifest):
    """True si el manifest tiene el formato actual y se genero con la misma
    configuracion. Si algo no cuadra (formato antiguo, corrupto, o cambio de
    modelo/dimension/troceado) hay que reconstruir el indice entero."""
    if not isinstance(manifest, dict):
        return False
    if manifest.get("version") != INDEX_VERSION:
        return False
    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        return False
    if manifest.get("embedding_dim") != EMBEDDING_DIM:
        return False
    if manifest.get("chunk_size") != CHUNK_SIZE:
        return False
    if manifest.get("chunk_overlap") != CHUNK_OVERLAP:
        return False
    return isinstance(manifest.get("documentos"), dict)


def _hashes_del_manifest(manifest):
    """Mapa {nombre: hash} extraido de un manifest ya validado."""
    return {
        nombre: datos.get("hash")
        for nombre, datos in manifest.get("documentos", {}).items()
    }


def indice_desactualizado(docs_dir=DOCS_DIR, index_dir=INDEX_DIR):
    """True si faltan PDFs por indexar, se han modificado/eliminado, el
    indice no existe todavia, o se genero con otra configuracion. Compara
    por hash de contenido, no por fecha, para que copiar/mover el PDF no
    dispare un reindexado innecesario."""
    chunks_path, _, manifest_path = _rutas_indice(index_dir)
    estado_actual = _estado_actual_docs(docs_dir)
    if not chunks_path.exists():
        return True, estado_actual
    manifest = _cargar_manifest(manifest_path)
    if not _manifest_compatible(manifest):
        return True, estado_actual
    if _hashes_del_manifest(manifest) != estado_actual:
        return True, estado_actual
    return False, estado_actual


def actualizar_si_es_necesario(client, docs_dir=DOCS_DIR, index_dir=INDEX_DIR):
    """Comprueba si los PDFs de docs_dir han cambiado desde la ultima vez
    (por hash de contenido: alta, baja o modificacion de algun PDF) y, si es
    asi, actualiza el indice automaticamente. Pensado para llamarse al
    arrancar el chatbot: el usuario final solo tiene que colocar/actualizar
    los PDFs en docs/, todo lo demas (troceado, embeddings, indexado) es
    automatico. Devuelve (se_reconstruyo, num_chunks_actuales)."""
    desactualizado, _ = indice_desactualizado(docs_dir, index_dir)
    if not desactualizado:
        chunks, _ = cargar_indice(index_dir)
        return False, len(chunks)

    print("Detectados cambios en los documentos, actualizando la base de conocimiento...")
    num_chunks, pdfs = construir_indice(client, docs_dir=docs_dir, index_dir=index_dir)
    if pdfs:
        print(f"Indice actualizado: {num_chunks} fragmentos de {len(pdfs)} PDF(s).\n")
    else:
        print(
            "No hay ningun PDF en la carpeta docs/ todavia. Añade tus PDFs "
            "ahi y vuelve a arrancar el chatbot.\n"
        )
    return True, num_chunks


def _chunks_de_documento(ruta):
    """Chunks de un unico documento, troceados respetando su estructura."""
    paginas = _leer_documento(ruta)
    return _trocear_estructurado(paginas, Path(ruta).name)


def extraer_chunks_de_docs(docs_dir=DOCS_DIR):
    """Recorre todos los documentos de docs_dir y devuelve (chunks, nombres)."""
    chunks = []
    rutas = _documentos_en(docs_dir)
    for ruta in rutas:
        chunks.extend(_chunks_de_documento(ruta))
    return chunks, [r.name for r in rutas]


class _ControlDeRitmo:
    """Vigila los 429 entre hilos y baja la concurrencia cuando se acumulan.

    Sin esto, ante un limite de cuota los N hilos se limitan a reintentar a
    la vez, con lo que se sigue golpeando la API al mismo ritmo y solo se
    consigue agotar los reintentos. Aqui, cada tanda de 429 reduce a la
    mitad el numero de hilos que pueden estar llamando a la vez."""

    def __init__(self, workers):
        self._lock = threading.Lock()
        self._semaforo = threading.Semaphore(workers)
        self._permitidos = workers
        self._429_seguidos = 0

    def __enter__(self):
        self._semaforo.acquire()
        return self

    def __exit__(self, *_):
        self._semaforo.release()
        return False

    def registrar_429(self):
        """Devuelve True si ha reducido la concurrencia."""
        with self._lock:
            self._429_seguidos += 1
            if self._429_seguidos < 2 or self._permitidos <= 1:
                return False
            # Nos quedamos permanentemente con la mitad de los permisos.
            a_retirar = self._permitidos - max(1, self._permitidos // 2)
            for _ in range(a_retirar):
                self._semaforo.acquire()
            self._permitidos -= a_retirar
            self._429_seguidos = 0
            print(f"  Demasiados 429, bajo la concurrencia a {self._permitidos}.")
            return True

    def registrar_exito(self):
        with self._lock:
            self._429_seguidos = 0


def _embed_con_reintento(client, textos, task_type, control=None):
    from google.genai import types

    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            if control is not None:
                with control:
                    resultado = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=textos,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=EMBEDDING_DIM,
                        ),
                    )
                control.registrar_exito()
            else:
                resultado = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=textos,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=EMBEDDING_DIM,
                    ),
                )
            return [emb.values for emb in resultado.embeddings]
        except APIError as e:
            if e.code == 429 and intento < MAX_REINTENTOS:
                if control is not None:
                    control.registrar_429()
                # Jitter: sin el, todos los hilos que reciben un 429 a la vez
                # reintentarian a la vez y volverian a chocar en bloque.
                espera = ESPERA_BASE_SEGUNDOS * intento * (0.5 + random.random())
                time.sleep(espera)
                continue
            raise


def _barra_progreso(hechos, total, ancho=28):
    """Barra de progreso minima, sin dependencias. El indexado inicial pasa
    a ser algo que el usuario ve, asi que conviene que sepa que avanza."""
    if not total:
        return
    proporcion = hechos / total
    llenos = int(ancho * proporcion)
    barra = "#" * llenos + "-" * (ancho - llenos)
    sys.stdout.write(f"\r  [{barra}] {hechos}/{total} lotes")
    sys.stdout.flush()
    if hechos == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _embed_textos(client, textos, task_type, mostrar_progreso=False):
    """Genera embeddings en lotes paralelos.

    El orden de salida corresponde exactamente al de entrada: cada lote se
    escribe en su hueco por indice, no segun termine. Un desorden aqui
    asignaria a cada chunk el vector de otro, en silencio."""
    lotes = [
        textos[i : i + EMBEDDING_BATCH_SIZE]
        for i in range(0, len(textos), EMBEDDING_BATCH_SIZE)
    ]
    if not lotes:
        return []

    if len(lotes) == 1 or WORKERS_EMBEDDING <= 1:
        vectores = []
        for lote in lotes:
            vectores.extend(_embed_con_reintento(client, lote, task_type))
        return vectores

    control = _ControlDeRitmo(WORKERS_EMBEDDING)
    resultados = [None] * len(lotes)
    completados = 0
    lock_progreso = threading.Lock()

    def procesar(indice):
        nonlocal completados
        vectores = _embed_con_reintento(client, lotes[indice], task_type, control)
        resultados[indice] = vectores
        if mostrar_progreso:
            with lock_progreso:
                completados += 1
                _barra_progreso(completados, len(lotes))

    with ThreadPoolExecutor(max_workers=WORKERS_EMBEDDING) as executor:
        # list() fuerza a que se propague la primera excepcion que ocurra.
        list(executor.map(procesar, range(len(lotes))))

    return [vector for lote in resultados for vector in lote]


def _normalizar_filas(matriz):
    """Normaliza cada fila a norma 1. Con output_dimensionality reducida los
    vectores de Gemini no vienen normalizados, asi que hay que hacerlo
    explicitamente."""
    if matriz.size == 0:
        return matriz
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return matriz / normas


def _embed_normalizado(client, textos, task_type, mostrar_progreso=False):
    """Embeddings ya normalizados, como matriz float32."""
    vectores = _embed_textos(client, textos, task_type, mostrar_progreso)
    matriz = np.array(vectores, dtype=np.float32)
    return _normalizar_filas(matriz)


def _indice_previo_por_documento(chunks_path, embeddings_path, manifest):
    """Agrupa el indice ya guardado por documento de origen, devolviendo
    {nombre: (chunks, matriz)}. Devuelve {} si el indice no existe o si los
    chunks y los embeddings no cuadran en numero, porque en ese caso
    reutilizarlos daria fragmentos equivocados de forma silenciosa."""
    chunks_path = Path(chunks_path)
    embeddings_path = Path(embeddings_path)
    if not chunks_path.exists() or not embeddings_path.exists():
        return {}

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        matriz = np.load(embeddings_path)["embeddings"]
    except (json.JSONDecodeError, OSError, KeyError, ValueError):
        return {}

    if not isinstance(chunks, list) or len(chunks) != matriz.shape[0]:
        return {}
    if matriz.ndim != 2 or matriz.shape[1] != EMBEDDING_DIM:
        return {}

    por_documento = {}
    for posicion, chunk in enumerate(chunks):
        por_documento.setdefault(chunk.get("source"), []).append(posicion)

    # Solo es seguro reutilizar un documento si el numero de chunks que hay
    # en el indice coincide con el que registro el manifest.
    resultado = {}
    documentos = manifest.get("documentos", {})
    for nombre, posiciones in por_documento.items():
        esperado = documentos.get(nombre, {}).get("num_chunks")
        if esperado != len(posiciones):
            continue
        resultado[nombre] = (
            [chunks[i] for i in posiciones],
            matriz[posiciones],
        )
    return resultado


def construir_indice(client, docs_dir=DOCS_DIR, index_dir=INDEX_DIR):
    """Actualiza el indice de docs_dir de forma incremental: reutiliza los
    chunks y embeddings de los PDFs que no han cambiado, genera solo los de
    los nuevos y modificados, y descarta los de los eliminados. Devuelve
    (num_chunks, lista_de_pdfs)."""
    chunks_path, embeddings_path, manifest_path = _rutas_indice(index_dir)
    estado_actual = _estado_actual_docs(docs_dir)
    manifest = _cargar_manifest(manifest_path)

    if _manifest_compatible(manifest):
        previos = _indice_previo_por_documento(chunks_path, embeddings_path, manifest)
        hashes_previos = _hashes_del_manifest(manifest)
    else:
        # Formato antiguo, corrupto o configuracion distinta: desde cero.
        previos = {}
        hashes_previos = {}

    reutilizados = []
    pendientes = []
    for nombre, hash_actual in estado_actual.items():
        if nombre in previos and hashes_previos.get(nombre) == hash_actual:
            reutilizados.append(nombre)
        else:
            pendientes.append(nombre)

    eliminados = [n for n in hashes_previos if n not in estado_actual]
    if reutilizados or pendientes or eliminados:
        print(
            f"  Documentos: {len(reutilizados)} sin cambios, "
            f"{len(pendientes)} por (re)indexar, {len(eliminados)} eliminado(s)."
        )

    # Se recorre en orden alfabetico para que el indice sea reproducible.
    chunks_finales = []
    matrices = []
    num_chunks_por_doc = {}
    docs_dir = Path(docs_dir)

    for nombre in sorted(estado_actual):
        if nombre in previos and nombre in reutilizados:
            chunks_doc, matriz_doc = previos[nombre]
        else:
            chunks_doc = _chunks_de_documento(docs_dir / nombre)
            if chunks_doc:
                print(f"  Indexando {nombre} ({len(chunks_doc)} fragmentos)...")
                matriz_doc = _embed_normalizado(
                    client,
                    [c["text"] for c in chunks_doc],
                    task_type="RETRIEVAL_DOCUMENT",
                    mostrar_progreso=True,
                )
            else:
                matriz_doc = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        num_chunks_por_doc[nombre] = len(chunks_doc)
        if chunks_doc:
            chunks_finales.extend(chunks_doc)
            matrices.append(matriz_doc)

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_finales, f, ensure_ascii=False, indent=2)

    if matrices:
        matriz_total = np.vstack(matrices).astype(np.float32)
        np.savez_compressed(embeddings_path, embeddings=matriz_total)
    elif embeddings_path.exists():
        embeddings_path.unlink()

    _guardar_manifest(estado_actual, num_chunks_por_doc, manifest_path)

    return len(chunks_finales), sorted(estado_actual)


def indice_disponible(index_dir=INDEX_DIR):
    chunks_path, _, _ = _rutas_indice(index_dir)
    return chunks_path.exists()


def cargar_indice(index_dir=INDEX_DIR):
    """Carga los chunks y embeddings guardados. Devuelve (chunks, matriz).
    Si no hay indice o esta vacio, devuelve ([], None)."""
    chunks_path, embeddings_path, _ = _rutas_indice(index_dir)
    if not chunks_path.exists():
        return [], None
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks or not embeddings_path.exists():
        return chunks, None
    datos = np.load(embeddings_path)
    matriz = datos["embeddings"]
    return chunks, matriz


_PUNTUACION = re.compile(r"[^\w\s]", re.UNICODE)


def _normalizar_para_busqueda(texto):
    """Minusculas, sin tildes y sin puntuacion. Solo se usa para construir
    el indice lexico: el texto original de los chunks no se toca, porque es
    lo que acaba viendo el modelo."""
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return _PUNTUACION.sub(" ", texto.lower())


def _tokenizar(texto):
    return _normalizar_para_busqueda(texto).split()


class IndiceBM25:
    """BM25 Okapi sobre los chunks. Implementado a mano (son unas 40 lineas)
    para no añadir rank_bm25 como dependencia.

    Se reconstruye al cargar en vez de persistirse: con ~5000 chunks tarda
    del orden de decimas de segundo, muy por debajo de lo que cuesta la
    llamada de embedding de la consulta, y evita otro artefacto que mantener
    sincronizado con chunks.json."""

    def __init__(self, documentos_tokenizados, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.num_docs = len(documentos_tokenizados)
        self.longitudes = [len(d) for d in documentos_tokenizados]
        self.longitud_media = (
            sum(self.longitudes) / self.num_docs if self.num_docs else 0.0
        )

        frecuencia_documental = {}
        self.postings = {}  # token -> [(indice_doc, frecuencia), ...]
        for indice, tokens in enumerate(documentos_tokenizados):
            frecuencias = {}
            for token in tokens:
                frecuencias[token] = frecuencias.get(token, 0) + 1
            for token, frecuencia in frecuencias.items():
                frecuencia_documental[token] = frecuencia_documental.get(token, 0) + 1
                self.postings.setdefault(token, []).append((indice, frecuencia))

        # Variante del idf que nunca sale negativa, para que un termino muy
        # comun no reste puntuacion.
        self.idf = {
            token: math.log(1 + (self.num_docs - n + 0.5) / (n + 0.5))
            for token, n in frecuencia_documental.items()
        }

    def buscar(self, tokens_consulta, top_n=TOP_K_CANDIDATOS):
        """Devuelve [(indice_chunk, score), ...] ordenado de mas a menos."""
        if not self.num_docs or not self.longitud_media:
            return []

        puntuaciones = {}
        for token in tokens_consulta:
            idf = self.idf.get(token)
            if idf is None:
                continue  # termino que no aparece en el corpus
            for indice, frecuencia in self.postings[token]:
                normalizacion = 1 - self.b + self.b * (
                    self.longitudes[indice] / self.longitud_media
                )
                aporte = idf * (frecuencia * (self.k1 + 1)) / (
                    frecuencia + self.k1 * normalizacion
                )
                puntuaciones[indice] = puntuaciones.get(indice, 0.0) + aporte

        return sorted(puntuaciones.items(), key=lambda par: -par[1])[:top_n]


def _fusion_rrf(rankings, k=RRF_K):
    """Reciprocal Rank Fusion: combina varias listas ordenadas usando solo
    la posicion, no la puntuacion. Es lo que permite mezclar coseno y BM25,
    que viven en escalas distintas y no son comparables directamente."""
    puntuaciones = {}
    for ranking in rankings:
        for posicion, indice in enumerate(ranking, start=1):
            puntuaciones[indice] = puntuaciones.get(indice, 0.0) + 1.0 / (k + posicion)
    ordenados = sorted(puntuaciones.items(), key=lambda par: -par[1])
    return [indice for indice, _ in ordenados], puntuaciones


class IndiceVectorial:
    """Interfaz comun de busqueda vectorial. Existe para poder cambiar de
    exacto a aproximado sin tocar el resto del pipeline."""

    nombre = "abstracto"

    def buscar(self, vector, n):
        """Devuelve [(indice_chunk, similitud_coseno), ...] de mas a menos."""
        raise NotImplementedError


class IndiceNumpy(IndiceVectorial):
    """Busqueda exacta: producto de la consulta contra toda la matriz.
    Sin dependencias y con recall perfecto por definicion."""

    nombre = "numpy"

    def __init__(self, matriz):
        self.matriz = matriz

    def buscar(self, vector, n):
        similitudes = self.matriz @ vector
        orden = np.argsort(-similitudes)[:n]
        return [(int(i), float(similitudes[i])) for i in orden]


class IndiceHNSW(IndiceVectorial):
    """Busqueda aproximada con hnswlib. Dependencia opcional: solo se
    importa si se usa este backend."""

    nombre = "hnsw"

    def __init__(self, matriz, ruta=None):
        import hnswlib  # import perezoso

        self.num_elementos = matriz.shape[0]
        self.dimension = matriz.shape[1]
        self.indice = hnswlib.Index(space="cosine", dim=self.dimension)

        # Se reutiliza el grafo guardado solo si cuadra con el numero de
        # chunks actual; si no, se reconstruye (es la unica forma barata de
        # detectar que el corpus cambio por debajo).
        if ruta and Path(ruta).exists():
            try:
                self.indice.load_index(str(ruta), max_elements=self.num_elementos)
                if self.indice.get_current_count() == self.num_elementos:
                    self.indice.set_ef(HNSW_EF_BUSQUEDA)
                    return
            except Exception:
                pass

        self.indice.init_index(
            max_elements=self.num_elementos,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
        )
        self.indice.add_items(matriz, np.arange(self.num_elementos))
        self.indice.set_ef(HNSW_EF_BUSQUEDA)
        if ruta:
            try:
                Path(ruta).parent.mkdir(parents=True, exist_ok=True)
                self.indice.save_index(str(ruta))
            except Exception as e:
                print(f"Aviso: no se pudo guardar el indice HNSW ({e}).")

    def buscar(self, vector, n):
        n = min(n, self.num_elementos)
        etiquetas, distancias = self.indice.knn_query(vector, k=n)
        # En espacio "cosine", hnswlib devuelve distancia = 1 - coseno.
        return [
            (int(etiqueta), float(1.0 - distancia))
            for etiqueta, distancia in zip(etiquetas[0], distancias[0])
        ]


def crear_indice_vectorial(matriz, backend=None, ruta_hnsw=None):
    """Elige el backend. En "auto" se queda en numpy mientras el corpus sea
    manejable, porque exacto y sin dependencias siempre es preferible."""
    if matriz is None or matriz.size == 0:
        return None

    backend = backend or BACKEND_VECTORIAL
    if backend == "auto":
        backend = "hnsw" if matriz.shape[0] > UMBRAL_CHUNKS_HNSW else "numpy"

    if backend == "hnsw":
        try:
            return IndiceHNSW(matriz, ruta_hnsw)
        except ImportError:
            print(
                "Aviso: hnswlib no esta instalado ('pip install hnswlib'), "
                "uso la busqueda exacta con numpy."
            )
        except Exception as e:
            print(f"Aviso: no se pudo crear el indice HNSW ({e}), uso numpy.")
    return IndiceNumpy(matriz)


class Buscador:
    """Reune lo necesario para recuperar: los chunks, la matriz densa ya
    normalizada, el indice vectorial y, si el modo lo pide, el indice
    lexico. Se construye una vez al arrancar y se reutiliza en cada
    pregunta."""

    def __init__(self, chunks, matriz, modo=None, backend=None, index_dir=INDEX_DIR):
        self.chunks = chunks or []
        self.matriz = matriz
        self.modo = modo or MODO_BUSQUEDA
        if self.modo not in ("densa", "lexica", "hibrida"):
            raise ValueError(f"MODO_BUSQUEDA no valido: {self.modo}")

        self.indice_vectorial = None
        if self.modo in ("densa", "hibrida"):
            ruta = Path(index_dir) / "hnsw.bin" if index_dir else None
            self.indice_vectorial = crear_indice_vectorial(matriz, backend, ruta)

        self.bm25 = None
        self.tiempo_construccion_bm25 = 0.0
        if self.modo in ("lexica", "hibrida") and self.chunks:
            inicio = time.perf_counter()
            self.bm25 = IndiceBM25([_tokenizar(c["text"]) for c in self.chunks])
            self.tiempo_construccion_bm25 = time.perf_counter() - inicio

    @property
    def vacio(self):
        return not self.chunks or (self.modo != "lexica" and self.matriz is None)


def crear_buscador(chunks, matriz, modo=None, backend=None, index_dir=INDEX_DIR):
    return Buscador(chunks, matriz, modo, backend, index_dir)


def _similitud_coseno(consulta, matriz):
    """La matriz del indice ya se guarda normalizada, asi que basta con
    normalizar la consulta y hacer un producto matriz-vector."""
    consulta = consulta / (np.linalg.norm(consulta) + 1e-10)
    return matriz @ consulta


def _rerank_llm(client, pregunta, candidatos, top_k):
    """Una sola llamada al modelo con los fragmentos numerados. No necesita
    ninguna API key aparte de la que ya se usa para chatear."""
    from google.genai import types

    fragmentos = "\n\n".join(
        f"[{i}] {c['text']}" for i, c in enumerate(candidatos)
    )
    instruccion = PROMPT_RERANK.format(pregunta=pregunta, fragmentos=fragmentos)
    salida = client.models.generate_content(
        model=MODELO_RERANKER_LLM,
        contents=[types.Content(role="user", parts=[types.Part(text=instruccion)])],
    )

    crudo = (salida.text or "").strip()
    crudo = re.sub(r"^```(?:json)?|```$", "", crudo, flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", crudo, flags=re.DOTALL)
    if not match:
        raise ValueError("el reranker no devolvio un array JSON")

    orden = json.loads(match.group(0))
    reordenados = []
    vistos = set()
    for entrada in orden:
        indice = entrada.get("id") if isinstance(entrada, dict) else entrada
        if not isinstance(indice, int) or not 0 <= indice < len(candidatos):
            continue
        if indice in vistos:
            continue
        vistos.add(indice)
        candidato = dict(candidatos[indice])
        if isinstance(entrada, dict) and "puntuacion" in entrada:
            candidato["score_rerank"] = entrada["puntuacion"]
        reordenados.append(candidato)

    if not reordenados:
        raise ValueError("el reranker descarto todos los fragmentos")
    return reordenados[:top_k]


def _rerank_cohere(client, pregunta, candidatos, top_k):
    """Backend de Cohere. Requiere COHERE_API_KEY en el .env y el paquete
    cohere instalado (no viene en requirements.txt: es opcional)."""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("falta COHERE_API_KEY en el .env")

    import cohere  # import perezoso: solo se paga si se usa este backend

    co = cohere.Client(api_key)
    respuesta = co.rerank(
        model=MODELO_RERANKER_COHERE,
        query=pregunta,
        documents=[c["text"] for c in candidatos],
        top_n=top_k,
    )

    reordenados = []
    for resultado in respuesta.results:
        candidato = dict(candidatos[resultado.index])
        candidato["score_rerank"] = getattr(resultado, "relevance_score", None)
        reordenados.append(candidato)
    if not reordenados:
        raise ValueError("Cohere no devolvio ningun resultado")
    return reordenados


_BACKENDS_RERANK = {"llm": _rerank_llm, "cohere": _rerank_cohere}


def rerankear(client, pregunta, candidatos, top_k, backend=None):
    """Reordena los candidatos por relevancia real y se queda con top_k.

    Ante cualquier fallo (backend caido, JSON ilegible, API key ausente)
    devuelve los candidatos en su orden original: el reranking es una
    mejora, nunca un punto de fallo que tumbe la conversacion."""
    global ULTIMA_LATENCIA_RERANK

    nombre = backend or RERANKER_BACKEND
    funcion = _BACKENDS_RERANK.get(nombre)
    if funcion is None:
        print(f"Aviso: backend de reranker desconocido ({nombre}), lo omito.")
        return candidatos, 0.0

    inicio = time.perf_counter()
    try:
        reordenados = funcion(client, pregunta, candidatos, top_k)
    except Exception as e:
        ULTIMA_LATENCIA_RERANK = time.perf_counter() - inicio
        print(f"Aviso: el reranker fallo ({e}), uso el orden original.")
        return candidatos, ULTIMA_LATENCIA_RERANK

    ULTIMA_LATENCIA_RERANK = time.perf_counter() - inicio
    return reordenados, ULTIMA_LATENCIA_RERANK


def benchmark_vectorial(matriz, consultas=None, top_k=4, repeticiones=3):
    """Compara numpy y hnswlib sobre el corpus actual: latencia por consulta
    y recall@k del aproximado tomando el exacto como verdad.

    Devuelve un dict por backend. Sirve para decidir con datos si merece la
    pena el indice aproximado, en vez de asumirlo."""
    if matriz is None or matriz.size == 0:
        return {}

    if consultas is None:
        # A falta de consultas reales, se usan vectores del propio corpus
        # con ruido: son el tipo de consulta mas exigente para el grafo.
        generador = np.random.default_rng(0)
        muestra = generador.choice(matriz.shape[0], size=min(50, matriz.shape[0]), replace=False)
        consultas = matriz[muestra] + generador.normal(0, 0.05, (len(muestra), matriz.shape[1]))
        consultas = _normalizar_filas(consultas.astype(np.float32))

    resultados = {}
    exacto = IndiceNumpy(matriz)

    verdad = []
    inicio = time.perf_counter()
    for _ in range(repeticiones):
        verdad = [[i for i, _ in exacto.buscar(v, top_k)] for v in consultas]
    latencia_exacta = (time.perf_counter() - inicio) / (repeticiones * len(consultas))
    resultados["numpy"] = {
        "latencia_ms": latencia_exacta * 1000,
        f"recall@{top_k}": 1.0,
        "chunks": int(matriz.shape[0]),
    }

    try:
        aproximado = IndiceHNSW(matriz)
    except ImportError:
        resultados["hnsw"] = {"error": "hnswlib no instalado (pip install hnswlib)"}
        return resultados
    except Exception as e:
        resultados["hnsw"] = {"error": str(e)}
        return resultados

    inicio = time.perf_counter()
    for _ in range(repeticiones):
        obtenidos = [[i for i, _ in aproximado.buscar(v, top_k)] for v in consultas]
    latencia_aprox = (time.perf_counter() - inicio) / (repeticiones * len(consultas))

    aciertos = sum(
        len(set(esperado) & set(obtenido))
        for esperado, obtenido in zip(verdad, obtenidos)
    )
    resultados["hnsw"] = {
        "latencia_ms": latencia_aprox * 1000,
        f"recall@{top_k}": aciertos / (len(verdad) * top_k),
        "chunks": int(matriz.shape[0]),
    }
    return resultados


def _filtrar_por_umbral(candidatos, umbral=None, margen=None, suelo=None):
    """Descarta los candidatos poco relevantes.

    Con umbral=None (lo normal) el corte es adaptativo: se toma el mejor
    score de esta pregunta y se admite lo que quede dentro de MARGEN_RELATIVO
    de el, nunca por debajo de UMBRAL_MINIMO. Asi una pregunta bien cubierta
    conserva sus varios fragmentos buenos, y una pregunta fuera del corpus
    no cuela el "menos malo" solo por ser el primero.

    Pasando un umbral numerico se vuelve al corte fijo de toda la vida, que
    es lo que usa el barrido de evaluar.py para poder compararlos."""
    con_score = [c for c in candidatos if c.get("score") is not None]
    if not con_score:
        return candidatos

    if umbral is not None:
        return [c for c in con_score if c["score"] >= umbral]

    margen = MARGEN_RELATIVO if margen is None else margen
    suelo = UMBRAL_MINIMO if suelo is None else suelo
    mejor = max(c["score"] for c in con_score)
    corte = max(suelo, mejor * margen)
    return [c for c in con_score if c["score"] >= corte]


def _candidatos_densos(client, pregunta, buscador, n):
    """Devuelve (indices_ordenados, scores_por_indice, vector_consulta).

    Los scores llegan como diccionario, no como array completo, porque un
    indice aproximado solo conoce los n vecinos que ha devuelto. Para
    cualquier otro candidato (los que aporte la via lexica) el score se
    calcula despues, uno a uno, contra la matriz."""
    vector = _embed_textos(client, [pregunta], task_type="RETRIEVAL_QUERY")[0]
    vector = np.array(vector, dtype=np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-10)
    resultados = buscador.indice_vectorial.buscar(vector, n)
    return [i for i, _ in resultados], dict(resultados), vector


def _candidatos_lexicos(pregunta, buscador, n):
    if buscador.bm25 is None:
        return [], {}
    resultados = buscador.bm25.buscar(_tokenizar(pregunta), top_n=n)
    return [indice for indice, _ in resultados], dict(resultados)


def recuperar_contexto(
    client,
    pregunta,
    buscador,
    top_k=4,
    umbral_similitud=None,
    top_k_candidatos=None,
    usar_reranker=None,
):
    """Devuelve los chunks mas relevantes para la pregunta.

    El pipeline es: candidatos (densos, lexicos o ambos fusionados con RRF)
    -> filtro por relevancia -> reranking opcional -> top_k.

    El filtro se aplica sobre la similitud coseno, que es la unica escala
    con significado absoluto; en modo puramente lexico no hay coseno que
    comparar, asi que no se filtra (BM25 no tiene un valor de corte
    universal: depende del corpus).

    umbral_similitud=None usa el corte adaptativo, que es lo recomendado.
    Un valor numerico fuerza un corte fijo."""
    if buscador is None or buscador.vacio:
        return []

    n_candidatos = top_k_candidatos or max(TOP_K_CANDIDATOS, top_k)
    scores = None
    vector = None

    if buscador.modo == "densa":
        orden, scores, vector = _candidatos_densos(
            client, pregunta, buscador, n_candidatos
        )
    elif buscador.modo == "lexica":
        orden, _ = _candidatos_lexicos(pregunta, buscador, n_candidatos)
    else:
        densos, scores, vector = _candidatos_densos(
            client, pregunta, buscador, n_candidatos
        )
        lexicos, _ = _candidatos_lexicos(pregunta, buscador, n_candidatos)
        orden, _ = _fusion_rrf([densos, lexicos])

    candidatos = []
    for indice in orden[:n_candidatos]:
        candidato = dict(buscador.chunks[indice])
        if scores is not None:
            score = scores.get(indice)
            # Un candidato que llego solo por la via lexica no viene con
            # score denso: se calcula aqui para que el filtro por umbral
            # pueda juzgarlo con la misma vara que a los demas.
            if score is None and buscador.matriz is not None:
                score = float(buscador.matriz[indice] @ vector)
            candidato["score"] = score
        candidatos.append(candidato)

    if scores is not None:
        candidatos = _filtrar_por_umbral(candidatos, umbral_similitud)
    if not candidatos:
        return []

    # Si tras el filtro ya caben todos, reordenar no cambiaria que
    # fragmentos llegan al modelo: nos ahorramos la llamada.
    activar = USAR_RERANKER if usar_reranker is None else usar_reranker
    if activar and len(candidatos) > top_k:
        candidatos, _ = rerankear(client, pregunta, candidatos, top_k)

    return candidatos[:top_k]
