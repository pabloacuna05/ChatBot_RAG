"""
Modulo RAG: extraccion de texto de PDFs, troceado, generacion de
embeddings con la API de Gemini, y recuperacion por similitud coseno.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from google.genai.errors import APIError

DOCS_DIR = Path(__file__).parent / "docs"
INDEX_DIR = Path(__file__).parent / "rag_index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npz"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

EMBEDDING_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 900  # caracteres por trozo
CHUNK_OVERLAP = 150  # caracteres solapados entre trozos consecutivos
EMBEDDING_BATCH_SIZE = 10
MAX_REINTENTOS = 3
ESPERA_BASE_SEGUNDOS = 5


def _leer_paginas_pdf(pdf_path):
    """Devuelve una lista de (numero_pagina, texto) para un PDF."""
    lector = PdfReader(str(pdf_path))
    paginas = []
    for i, pagina in enumerate(lector.pages, start=1):
        texto = (pagina.extract_text() or "").strip()
        if texto:
            paginas.append((i, texto))
    return paginas


def _trocear_texto(texto, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Trocea un texto largo en fragmentos con solapamiento."""
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


def _hash_pdf(pdf_path):
    """Hash sha256 del contenido del PDF, para detectar cambios."""
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for bloque in iter(lambda: f.read(8192), b""):
            hasher.update(bloque)
    return hasher.hexdigest()


def _estado_actual_docs(docs_dir=DOCS_DIR):
    """Mapa {nombre_pdf: hash} de todos los PDFs presentes en docs_dir."""
    pdfs = sorted(Path(docs_dir).glob("*.pdf"))
    return {p.name: _hash_pdf(p) for p in pdfs}


def _cargar_manifest():
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _guardar_manifest(estado_docs):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(estado_docs, f, ensure_ascii=False, indent=2)


def indice_desactualizado(docs_dir=DOCS_DIR):
    """True si faltan PDFs por indexar, se han modificado/eliminado, o el
    indice no existe todavia. Compara por hash de contenido, no por fecha,
    para que copiar/mover el PDF no dispare un reindexado innecesario."""
    estado_actual = _estado_actual_docs(docs_dir)
    if not CHUNKS_PATH.exists():
        return True, estado_actual
    manifest = _cargar_manifest()
    if manifest != estado_actual:
        return True, estado_actual
    return False, estado_actual


def actualizar_si_es_necesario(client, docs_dir=DOCS_DIR, index_dir=INDEX_DIR):
    """Comprueba si los PDFs de docs_dir han cambiado desde la ultima vez
    (por hash de contenido: alta, baja o modificacion de algun PDF) y, si es
    asi, reconstruye el indice automaticamente. Pensado para llamarse al
    arrancar el chatbot: el usuario final solo tiene que colocar/actualizar
    los PDFs en docs/, todo lo demas (troceado, embeddings, indexado) es
    automatico. Devuelve (se_reconstruyo, num_chunks_actuales)."""
    desactualizado, _ = indice_desactualizado(docs_dir)
    if not desactualizado:
        chunks, _ = cargar_indice()
        return False, len(chunks)

    print("Detectados cambios en los documentos, actualizando la base de conocimiento...")
    num_chunks, pdfs = construir_indice(client, docs_dir=docs_dir, index_dir=index_dir)
    if pdfs:
        print(f"Indice actualizado: {num_chunks} fragmentos de {len(pdfs)} PDF(s).\n")
    else:
        print(
            "No hay ningun PDF en la carpeta docs/ todavia. Añade el PDF de "
            "NNormal ahi y vuelve a arrancar el chatbot.\n"
        )
    return True, num_chunks


def extraer_chunks_de_docs(docs_dir=DOCS_DIR):
    """Recorre todos los PDFs de docs_dir y devuelve una lista de chunks
    con metadatos: {"text", "source", "page"}."""
    chunks = []
    pdfs = sorted(Path(docs_dir).glob("*.pdf"))
    for pdf_path in pdfs:
        paginas = _leer_paginas_pdf(pdf_path)
        for numero_pagina, texto_pagina in paginas:
            for trozo in _trocear_texto(texto_pagina):
                chunks.append(
                    {
                        "text": trozo,
                        "source": pdf_path.name,
                        "page": numero_pagina,
                    }
                )
    return chunks, [p.name for p in pdfs]


def _embed_con_reintento(client, textos, task_type):
    from google.genai import types

    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            resultado = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=textos,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [emb.values for emb in resultado.embeddings]
        except APIError as e:
            if e.code == 429 and intento < MAX_REINTENTOS:
                espera = ESPERA_BASE_SEGUNDOS * intento
                print(
                    f"Limite de la API alcanzado generando embeddings, "
                    f"reintentando en {espera}s... ({intento}/{MAX_REINTENTOS})"
                )
                time.sleep(espera)
                continue
            raise


def _embed_textos(client, textos, task_type):
    """Genera embeddings en lotes para no exceder limites de la API."""
    vectores = []
    for i in range(0, len(textos), EMBEDDING_BATCH_SIZE):
        lote = textos[i : i + EMBEDDING_BATCH_SIZE]
        vectores.extend(_embed_con_reintento(client, lote, task_type))
    return vectores


def construir_indice(client, docs_dir=DOCS_DIR, index_dir=INDEX_DIR):
    """Procesa los PDFs de docs_dir y guarda el indice (chunks + embeddings)
    en index_dir, junto con el manifest de hashes usado para detectar
    cambios futuros. Devuelve (num_chunks, lista_de_pdfs)."""
    chunks, pdfs = extraer_chunks_de_docs(docs_dir)
    estado_actual = _estado_actual_docs(docs_dir)

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    if chunks:
        textos = [c["text"] for c in chunks]
        vectores = _embed_textos(client, textos, task_type="RETRIEVAL_DOCUMENT")
        matriz = np.array(vectores, dtype=np.float32)
        np.savez_compressed(EMBEDDINGS_PATH, embeddings=matriz)
    elif EMBEDDINGS_PATH.exists():
        EMBEDDINGS_PATH.unlink()

    _guardar_manifest(estado_actual)

    return len(chunks), pdfs


def indice_disponible():
    return CHUNKS_PATH.exists()


def cargar_indice():
    """Carga los chunks y embeddings guardados. Devuelve (chunks, matriz).
    Si no hay indice o esta vacio, devuelve ([], None)."""
    if not indice_disponible():
        return [], None
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks or not EMBEDDINGS_PATH.exists():
        return chunks, None
    datos = np.load(EMBEDDINGS_PATH)
    matriz = datos["embeddings"]
    return chunks, matriz


def _similitud_coseno(consulta, matriz):
    consulta = consulta / (np.linalg.norm(consulta) + 1e-10)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True) + 1e-10
    matriz_normalizada = matriz / normas
    return matriz_normalizada @ consulta


def recuperar_contexto(client, pregunta, chunks, matriz, top_k=4, umbral_similitud=0.5):
    """Devuelve los chunks mas relevantes para la pregunta, por encima del
    umbral de similitud coseno."""
    if chunks is None or matriz is None or len(chunks) == 0:
        return []

    vector_pregunta = _embed_textos(client, [pregunta], task_type="RETRIEVAL_QUERY")[0]
    vector_pregunta = np.array(vector_pregunta, dtype=np.float32)

    similitudes = _similitud_coseno(vector_pregunta, matriz)
    indices_ordenados = np.argsort(-similitudes)[:top_k]

    resultados = []
    for idx in indices_ordenados:
        score = float(similitudes[idx])
        if score >= umbral_similitud:
            resultado = dict(chunks[idx])
            resultado["score"] = score
            resultados.append(resultado)
    return resultados
