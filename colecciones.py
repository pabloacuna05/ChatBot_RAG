"""
Gestion de colecciones: cada una es un corpus independiente con sus propios
documentos y su propio indice, bajo data/{nombre}/.

Incluye la cache LRU de buscadores. Cargar el indice y construir el BM25 en
cada peticion seria inviable, asi que se cachean por coleccion y se
invalidan cuando esa coleccion se reindexa.
"""

import re
import shutil
import threading
from collections import OrderedDict
from pathlib import Path

import rag

DATA_DIR = Path(__file__).parent / "data"

# Nombres permitidos. La lista blanca es deliberadamente estricta: el nombre
# acaba siendo un componente de ruta, y aceptar cosas como "../.." o rutas
# absolutas dejaria leer y escribir fuera de data/.
RE_NOMBRE_VALIDO = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# Cuantas colecciones se mantienen cargadas en memoria a la vez.
MAX_COLECCIONES_EN_CACHE = 4


class NombreInvalido(ValueError):
    pass


class ColeccionNoEncontrada(LookupError):
    pass


def validar_nombre(nombre):
    """Devuelve el nombre si es valido; si no, explota. Se llama SIEMPRE
    antes de construir cualquier ruta con el."""
    if not isinstance(nombre, str) or not RE_NOMBRE_VALIDO.match(nombre):
        raise NombreInvalido(
            "El nombre de la coleccion solo admite letras, numeros, guion y "
            "guion bajo (1-64 caracteres), y debe empezar por letra o numero."
        )
    return nombre


def rutas(nombre, data_dir=None):
    """(docs_dir, index_dir) de una coleccion."""
    validar_nombre(nombre)
    base = Path(data_dir or DATA_DIR) / nombre
    return base / "docs", base / "rag_index"


def crear(nombre, data_dir=None):
    docs_dir, index_dir = rutas(nombre, data_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir, index_dir


def existe(nombre, data_dir=None):
    try:
        docs_dir, _ = rutas(nombre, data_dir)
    except NombreInvalido:
        return False
    return docs_dir.exists()


def listar(data_dir=None):
    """Colecciones existentes, con cuantos documentos tiene cada una."""
    base = Path(data_dir or DATA_DIR)
    if not base.exists():
        return []

    resultado = []
    for carpeta in sorted(base.iterdir()):
        if not carpeta.is_dir() or not RE_NOMBRE_VALIDO.match(carpeta.name):
            continue
        docs_dir = carpeta / "docs"
        documentos = rag._documentos_en(docs_dir) if docs_dir.exists() else []
        chunks_path, _, _ = rag._rutas_indice(carpeta / "rag_index")
        resultado.append(
            {
                "nombre": carpeta.name,
                "documentos": [r.name for r in documentos],
                "indexada": chunks_path.exists(),
            }
        )
    return resultado


def eliminar(nombre, data_dir=None):
    docs_dir, _ = rutas(nombre, data_dir)
    base = docs_dir.parent
    if not base.exists():
        raise ColeccionNoEncontrada(nombre)
    shutil.rmtree(base)


def reindexar(nombre, client, data_dir=None):
    """Actualiza el indice de una coleccion. Devuelve (num_chunks, docs)."""
    docs_dir, index_dir = rutas(nombre, data_dir)
    if not docs_dir.exists():
        raise ColeccionNoEncontrada(nombre)
    return rag.construir_indice(client, docs_dir=docs_dir, index_dir=index_dir)


class CacheBuscadores:
    """Cache LRU de buscadores por coleccion.

    Es una cache de objetos caros (matriz de embeddings en memoria + indice
    BM25 reconstruido), no de respuestas: dos preguntas distintas a la misma
    coleccion comparten buscador."""

    def __init__(self, maxsize=MAX_COLECCIONES_EN_CACHE, data_dir=None):
        self.maxsize = maxsize
        self.data_dir = data_dir
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def obtener(self, nombre):
        """Buscador de la coleccion, cargandolo del disco si no esta ya."""
        validar_nombre(nombre)
        with self._lock:
            if nombre in self._cache:
                self._cache.move_to_end(nombre)
                return self._cache[nombre]

        # La carga se hace fuera del lock: es lenta y no queremos bloquear
        # las peticiones a otras colecciones mientras tanto.
        docs_dir, index_dir = rutas(nombre, self.data_dir)
        if not docs_dir.exists():
            raise ColeccionNoEncontrada(nombre)
        chunks, matriz = rag.cargar_indice(index_dir)
        buscador = rag.crear_buscador(chunks, matriz, index_dir=index_dir)

        with self._lock:
            # Otro hilo puede haberla cargado mientras tanto; nos quedamos
            # con la suya para no tener dos copias vivas del mismo indice.
            if nombre in self._cache:
                self._cache.move_to_end(nombre)
                return self._cache[nombre]
            self._cache[nombre] = buscador
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        return buscador

    def invalidar(self, nombre):
        with self._lock:
            self._cache.pop(nombre, None)

    def vaciar(self):
        with self._lock:
            self._cache.clear()

    @property
    def cargadas(self):
        with self._lock:
            return list(self._cache)
