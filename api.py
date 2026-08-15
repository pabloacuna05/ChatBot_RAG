"""
API HTTP del chatbot RAG: multiusuario (historial por session_id) y
multicoleccion (cada corpus con sus documentos e indice propios).

Arrancar:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload

La logica de conversacion NO vive aqui: se importa de conversacion.py, la
misma que usa la CLI. Este archivo es solo el transporte HTTP.

Endpoints:
    GET  /colecciones                        lista las colecciones
    POST /colecciones/{nombre}/documentos    sube un documento
    POST /colecciones/{nombre}/reindexar     reconstruye su indice
    POST /chat                               conversa
"""

import os
import sys
from contextlib import asynccontextmanager

import bootstrap

bootstrap.asegurar_dependencias()

from dotenv import load_dotenv

try:
    from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
    from pydantic import BaseModel, Field
except ImportError:
    print(
        "Error: la API necesita FastAPI.\n"
        "  pip install fastapi uvicorn python-multipart"
    )
    sys.exit(1)

from google import genai

import colecciones
import conversacion
import rag
from historial import HistorialSQLite

load_dotenv()

API_KEY = os.getenv("API_KEY", "").strip()
RUTA_BD_HISTORIAL = os.getenv("RUTA_BD_HISTORIAL", "").strip() or str(
    colecciones.DATA_DIR / "historial.db"
)
# Tope de tamaño por subida. Sin el, una peticion puede agotar la memoria
# del servidor antes de que lleguemos a validar nada.
MAX_BYTES_SUBIDA = 25 * 1024 * 1024


@asynccontextmanager
async def ciclo_de_vida(app):
    if not API_KEY:
        print(
            "AVISO: API_KEY no esta configurada en el .env, la API acepta "
            "cualquier peticion. No la expongas asi fuera de local."
        )
    yield
    _cache.vaciar()


app = FastAPI(
    title="ChatBot RAG",
    description="Chatbot RAG multiusuario y multicoleccion.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)

_cliente = None
_cache = colecciones.CacheBuscadores()
_almacen = HistorialSQLite(RUTA_BD_HISTORIAL)
_config = conversacion.cargar_configuracion()


def cliente_gemini():
    """Un solo cliente para todo el proceso."""
    global _cliente
    if _cliente is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(500, "No hay GEMINI_API_KEY configurada en el servidor.")
        _cliente = genai.Client(api_key=api_key)
    return _cliente


def verificar_api_key(x_api_key: str = Header(default="")):
    """Autenticacion por cabecera X-API-Key.

    Si API_KEY no esta configurada, la API queda abierta y se avisa al
    arrancar: es comodo en local, pero nunca debe quedarse asi expuesto."""
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(401, "API key invalida o ausente.")


class PeticionChat(BaseModel):
    mensaje: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=128)
    coleccion: str = Field(min_length=1, max_length=64)


class RespuestaChat(BaseModel):
    respuesta: str
    fuentes: list[dict]
    session_id: str
    coleccion: str


def _traducir_errores(e):
    if isinstance(e, colecciones.NombreInvalido):
        return HTTPException(400, str(e))
    if isinstance(e, colecciones.ColeccionNoEncontrada):
        return HTTPException(404, f"La coleccion '{e}' no existe.")
    return None


@app.get("/colecciones", dependencies=[Depends(verificar_api_key)])
def listar_colecciones():
    return {
        "colecciones": colecciones.listar(),
        "en_memoria": _cache.cargadas,
    }


@app.post("/colecciones/{nombre}/documentos", dependencies=[Depends(verificar_api_key)])
async def subir_documento(nombre: str, archivo: UploadFile = File(...)):
    try:
        colecciones.validar_nombre(nombre)
    except colecciones.NombreInvalido as e:
        raise HTTPException(400, str(e))

    # El nombre del archivo tambien viene del cliente: nos quedamos solo con
    # el nombre base para que no pueda escribir fuera de la carpeta.
    nombre_archivo = os.path.basename(archivo.filename or "")
    if not nombre_archivo:
        raise HTTPException(400, "Falta el nombre del archivo.")

    extension = os.path.splitext(nombre_archivo)[1].lower()
    if extension not in rag.EXTRACTORES:
        raise HTTPException(
            400,
            f"Formato no soportado ({extension}). Admitidos: "
            f"{', '.join(sorted(rag.EXTRACTORES))}",
        )

    contenido = await archivo.read()
    if len(contenido) > MAX_BYTES_SUBIDA:
        raise HTTPException(413, "El archivo supera el tamaño maximo permitido.")
    if not contenido:
        raise HTTPException(400, "El archivo esta vacio.")

    docs_dir, _ = colecciones.crear(nombre)
    destino = docs_dir / nombre_archivo
    destino.write_bytes(contenido)

    # El indice deja de reflejar los documentos: fuera de la cache para que
    # nadie siga respondiendo con la version anterior.
    _cache.invalidar(nombre)

    return {
        "coleccion": nombre,
        "archivo": nombre_archivo,
        "bytes": len(contenido),
        "aviso": "Llama a /colecciones/{nombre}/reindexar para indexarlo.",
    }


@app.post("/colecciones/{nombre}/reindexar", dependencies=[Depends(verificar_api_key)])
def reindexar_coleccion(nombre: str):
    try:
        num_chunks, documentos = colecciones.reindexar(nombre, cliente_gemini())
    except Exception as e:
        http = _traducir_errores(e)
        if http:
            raise http
        raise HTTPException(500, f"Fallo al reindexar: {e}")

    _cache.invalidar(nombre)
    return {
        "coleccion": nombre,
        "fragmentos": num_chunks,
        "documentos": documentos,
    }


@app.post(
    "/chat", response_model=RespuestaChat, dependencies=[Depends(verificar_api_key)]
)
def chat(peticion: PeticionChat):
    try:
        buscador = _cache.obtener(peticion.coleccion)
    except Exception as e:
        http = _traducir_errores(e)
        if http:
            raise http
        raise HTTPException(500, f"No se pudo cargar la coleccion: {e}")

    historial = conversacion.recortar_historial(
        _almacen.cargar(peticion.session_id, limite=conversacion.MAX_MENSAJES_HISTORIAL)
    )
    system_prompt = conversacion.construir_system_prompt(
        _config["bot_name"], _config["bot_topic"], _config["mostrar_fuentes"]
    )

    texto, contexto = conversacion.responder(
        cliente_gemini(),
        peticion.mensaje,
        historial,
        buscador,
        system_prompt,
        mostrar_fuentes=_config["mostrar_fuentes"],
    )
    if not texto:
        raise HTTPException(502, "El modelo no devolvio ninguna respuesta.")

    _almacen.añadir_turno(peticion.session_id, "user", peticion.mensaje)
    _almacen.añadir_turno(peticion.session_id, "model", texto)

    fuentes = [
        {
            "n": i,
            "documento": c.get("source"),
            "pagina": c.get("page"),
            "seccion": c.get("seccion"),
        }
        for i, c in enumerate(contexto, start=1)
    ]
    return RespuestaChat(
        respuesta=texto,
        fuentes=fuentes,
        session_id=peticion.session_id,
        coleccion=peticion.coleccion,
    )


@app.delete("/sesiones/{session_id}", dependencies=[Depends(verificar_api_key)])
def borrar_sesion(session_id: str):
    _almacen.borrar(session_id)
    return {"session_id": session_id, "borrada": True}


@app.get("/salud")
def salud():
    """Sin autenticacion: la usan los health checks del orquestador."""
    return {"estado": "ok"}
