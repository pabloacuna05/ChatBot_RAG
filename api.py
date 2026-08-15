"""
API HTTP del chatbot RAG: multiusuario (sesion firmada por cookie) y
multicoleccion (cada corpus con sus documentos e indice propios).

Arrancar:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app

La logica de conversacion NO vive aqui: se importa de conversacion.py, la
misma que usa la CLI. Este archivo es transporte HTTP y control de acceso.

Antes de exponerlo a internet, lee la checklist del README: hay cosas que
este codigo no puede resolver por ti (HTTPS, limites de recursos del
contenedor, y las advertencias sobre extraccion del corpus y privacidad).

Endpoints:
    POST   /sesiones                         abre sesion ligada a una coleccion
    POST   /chat                             conversa (usa la cookie de sesion)
    DELETE /sesiones/actual                  cierra la sesion en curso
    GET    /colecciones                      lista las colecciones (admin)
    POST   /colecciones/{nombre}/documentos  sube un documento (admin)
    POST   /colecciones/{nombre}/reindexar   reconstruye su indice (admin)
    GET    /salud                            health check, sin autenticacion
"""

import hmac
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as TimeoutFuturo
from contextlib import asynccontextmanager

import bootstrap

bootstrap.asegurar_dependencias()

from dotenv import load_dotenv

try:
    from fastapi import (
        Depends,
        FastAPI,
        File,
        Header,
        HTTPException,
        Request,
        Response,
        UploadFile,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    print(
        "Error: la API necesita FastAPI.\n"
        "  pip install fastapi uvicorn python-multipart"
    )
    sys.exit(1)

from google import genai
from google.genai import types

import colecciones
import conversacion
import rag
import seguridad
from historial import AlmacenSesiones, HistorialSQLite

load_dotenv()


def _entero_env(nombre, por_defecto):
    try:
        return int(os.getenv(nombre, "").strip() or por_defecto)
    except ValueError:
        return por_defecto


# --- Configuracion -----------------------------------------------------

# Clave de administracion: protege subir documentos y reindexar. Los
# endpoints de conversacion NO la usan (los protege la sesion firmada).
ADMIN_API_KEY = os.getenv("API_KEY", "").strip()

# Clave con la que se firman las cookies de sesion. Si no se configura se
# genera una efimera: comodo en local, pero al reiniciar caducan todas las
# sesiones y con varios workers cada uno firmaria distinto.
SECRET_KEY_EFIMERA = not os.getenv("SECRET_KEY", "").strip()
SECRET_KEY = os.getenv("SECRET_KEY", "").strip() or seguridad.generar_clave_secreta()

RUTA_BD = os.getenv("RUTA_BD_HISTORIAL", "").strip() or str(
    colecciones.DATA_DIR / "historial.db"
)

# Origenes permitidos para CORS, separados por comas. Vacio = ninguno. El
# comodin "*" se ignora a proposito: con cookies de sesion nunca es correcto.
ORIGENES_CORS = [
    o.strip()
    for o in os.getenv("ORIGENES_CORS", "").split(",")
    if o.strip() and o.strip() != "*"
]

# Cookies solo por HTTPS. Se puede desactivar para probar en local sin TLS.
COOKIES_SEGURAS = conversacion.flag_env("COOKIES_SEGURAS", True)
NOMBRE_COOKIE = "rag_sesion"

MAX_CHARS_PREGUNTA = _entero_env("MAX_CHARS_PREGUNTA", 2000)
MAX_MENSAJES_POR_SESION = _entero_env("MAX_MENSAJES_POR_SESION", 100)
SEGUNDOS_EXPIRACION_SESION = _entero_env("SEGUNDOS_EXPIRACION_SESION", 24 * 3600)
TIMEOUT_MODELO_SEGUNDOS = _entero_env("TIMEOUT_MODELO_SEGUNDOS", 60)
TIMEOUT_INDEXADO_SEGUNDOS = _entero_env("TIMEOUT_INDEXADO_SEGUNDOS", 300)
MAX_BYTES_SUBIDA = _entero_env("MAX_BYTES_SUBIDA", 25 * 1024 * 1024)

# Dos ventanas: una corta contra rafagas, otra diaria contra el goteo.
LIMITE_MINUTO = _entero_env("LIMITE_PETICIONES_MINUTO", 10)
LIMITE_DIA = _entero_env("LIMITE_PETICIONES_DIA", 200)

LOG_NIVEL = os.getenv("LOG_NIVEL", "").strip() or "INFO"
# Por defecto NO se registra el texto de las preguntas: son datos del
# usuario y acabarian en disco y en el agregador de logs.
LOG_PREGUNTAS = conversacion.flag_env("LOG_PREGUNTAS", False)

seguridad.configurar_logging(LOG_NIVEL)

_cliente = None
_cache = colecciones.CacheBuscadores()
_almacen = HistorialSQLite(RUTA_BD)
_sesiones = AlmacenSesiones(RUTA_BD)
_config = conversacion.cargar_configuracion()
_limitador_sesion = seguridad.LimitadorPeticiones(
    [(LIMITE_MINUTO, 60), (LIMITE_DIA, 86400)]
)
# El limite por IP es mas holgado que el de sesion: detras de una misma IP
# puede haber varias personas (una oficina, una red movil).
_limitador_ip = seguridad.LimitadorPeticiones(
    [(LIMITE_MINUTO * 3, 60), (LIMITE_DIA * 3, 86400)]
)
# Las llamadas bloqueantes (modelo, indexado) van aqui para poder ponerles
# un timeout y que una peticion colgada no retenga un worker para siempre.
_ejecutor = ThreadPoolExecutor(max_workers=8)


@asynccontextmanager
async def ciclo_de_vida(app):
    if SECRET_KEY_EFIMERA:
        print(
            "AVISO: SECRET_KEY no configurada. Se usa una clave efimera: al "
            "reiniciar caducaran todas las sesiones, y con varios workers "
            "cada uno firmaria distinto. Configurala antes de desplegar."
        )
    if not ADMIN_API_KEY:
        print(
            "AVISO: API_KEY no configurada. Los endpoints de administracion "
            "(subir documentos, reindexar) estan ABIERTOS."
        )
    if not ORIGENES_CORS:
        print(
            "AVISO: ORIGENES_CORS vacio. Ningun navegador de otro origen "
            "podra usar la API (las llamadas server-to-server si funcionan)."
        )

    # Limpieza de arranque: sesiones inactivas y su historial.
    caducadas = _sesiones.caducar(SEGUNDOS_EXPIRACION_SESION)
    for sid in caducadas:
        _almacen.borrar(sid)
    if caducadas:
        seguridad.registrar("sesiones_caducadas", cuantas=len(caducadas))

    yield
    _cache.vaciar()
    _ejecutor.shutdown(wait=False)


DOCS_HABILITADOS = conversacion.flag_env("DOCS_HABILITADOS", True)

app = FastAPI(
    title="ChatBot RAG",
    description="Chatbot RAG multiusuario y multicoleccion.",
    version="1.1.0",
    lifespan=ciclo_de_vida,
    docs_url="/docs" if DOCS_HABILITADOS else None,
    redoc_url="/redoc" if DOCS_HABILITADOS else None,
    openapi_url="/openapi.json" if DOCS_HABILITADOS else None,
)

if ORIGENES_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ORIGENES_CORS,
        allow_credentials=True,  # necesario para la cookie de sesion
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


# La documentacion interactiva (Swagger/ReDoc) carga su JS y su CSS de un
# CDN, asi que con la CSP estricta de la API se veria en blanco. Se le
# afloja la politica SOLO a esas rutas, y se pueden desactivar del todo
# poniendo DOCS_HABILITADOS=false (recomendable en produccion: describen
# toda la superficie de la API a cualquiera que la visite).
RUTAS_DOCUMENTACION = {"/docs", "/redoc", "/docs/oauth2-redirect", "/openapi.json"}
CSP_API = "default-src 'none'; frame-ancestors 'none'"
CSP_DOCUMENTACION = (
    "default-src 'none'; "
    "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "img-src 'self' https://fastapi.tiangolo.com data:; "
    "font-src https://cdn.jsdelivr.net; "
    "connect-src 'self'; "
    "frame-ancestors 'none'"
)


@app.middleware("http")
async def cabeceras_de_seguridad(request: Request, call_next):
    respuesta = await call_next(request)
    respuesta.headers["X-Content-Type-Options"] = "nosniff"
    respuesta.headers["X-Frame-Options"] = "DENY"
    respuesta.headers["Referrer-Policy"] = "no-referrer"
    respuesta.headers["Cache-Control"] = "no-store"
    respuesta.headers["Content-Security-Policy"] = (
        CSP_DOCUMENTACION
        if request.url.path in RUTAS_DOCUMENTACION
        else CSP_API
    )
    if COOKIES_SEGURAS:
        respuesta.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return respuesta


# --- Errores -----------------------------------------------------------


class ErrorInterno(Exception):
    """Fallo del servidor cuyo detalle no debe salir al cliente."""


@app.exception_handler(Exception)
async def manejador_global(request: Request, exc: Exception):
    """Nada de tracebacks ni mensajes de terceros hacia fuera.

    Un error de la API de Gemini puede llevar dentro la URL de la peticion o
    datos del proyecto; devolverlo tal cual seria filtrar informacion del
    servidor a cualquiera capaz de provocar un fallo."""
    correlacion = seguridad.id_correlacion()
    seguridad.registrar(
        "error_no_controlado",
        correlacion=correlacion,
        ruta=request.url.path,
        tipo=type(exc).__name__,
        detalle=str(exc)[:500],  # el detalle real se queda en el log
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "No se pudo procesar la peticion.",
            "correlacion": correlacion,
        },
    )


def _fallo(codigo, mensaje, **contexto):
    """HTTPException con id de correlacion, para poder cruzar el 'me falla
    esto' del usuario con el error real del log sin filtrar detalles."""
    correlacion = seguridad.id_correlacion()
    seguridad.registrar(
        "peticion_rechazada",
        correlacion=correlacion,
        codigo=codigo,
        motivo=mensaje,
        **contexto,
    )
    return HTTPException(
        status_code=codigo, detail={"error": mensaje, "correlacion": correlacion}
    )


def _limite_superado(espera, ambito):
    correlacion = seguridad.id_correlacion()
    seguridad.registrar("limite_superado", correlacion=correlacion, ambito=ambito)
    return HTTPException(
        status_code=429,
        detail={
            "error": "Demasiadas peticiones. Intentalo mas tarde.",
            "correlacion": correlacion,
        },
        headers={"Retry-After": str(espera)},
    )


# --- Dependencias ------------------------------------------------------


def cliente_gemini():
    """Un solo cliente para todo el proceso, con timeout configurado."""
    global _cliente
    if _cliente is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # El mensaje no menciona el valor de la key en ningun caso.
            raise ErrorInterno("falta GEMINI_API_KEY en el servidor")
        _cliente = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=TIMEOUT_MODELO_SEGUNDOS * 1000),
        )
    return _cliente


def _ip_cliente(request: Request):
    """IP real detras de un proxy inverso.

    Solo se mira X-Forwarded-For si CONFIAR_EN_PROXY esta activo: si no,
    cualquiera podria falsear la cabecera y saltarse el limite por IP."""
    if conversacion.flag_env("CONFIAR_EN_PROXY", False):
        reenviada = request.headers.get("x-forwarded-for", "")
        if reenviada:
            return reenviada.split(",")[0].strip()
    return request.client.host if request.client else "desconocida"


def verificar_admin(x_api_key: str = Header(default="")):
    """Autenticacion de los endpoints de administracion."""
    if not ADMIN_API_KEY:
        return
    if not hmac.compare_digest(x_api_key or "", ADMIN_API_KEY):
        raise _fallo(401, "Credenciales invalidas.")


def sesion_actual(request: Request):
    """Sesion a partir de la cookie firmada.

    El cliente no elige su id: si se aceptara un session_id arbitrario,
    cualquiera podria leer la conversacion de otro sondeando identificadores."""
    cookie = request.cookies.get(NOMBRE_COOKIE, "")
    try:
        session_id = seguridad.verificar_sesion(cookie, SECRET_KEY)
    except seguridad.FirmaInvalida:
        raise _fallo(401, "Sesion no valida. Abre una sesion en POST /sesiones.")

    datos = _sesiones.obtener(session_id)
    if not datos:
        raise _fallo(401, "La sesion ha caducado. Abre una nueva.")

    if time.time() - datos["vista_en"] > SEGUNDOS_EXPIRACION_SESION:
        _sesiones.borrar(session_id)
        _almacen.borrar(session_id)
        raise _fallo(401, "La sesion ha caducado por inactividad.")

    if datos["mensajes"] >= MAX_MENSAJES_POR_SESION:
        raise _fallo(429, "Esta sesion ha alcanzado su limite de mensajes.")

    return datos


def _comprobar_limites(request, session_id=None):
    permitido, espera = _limitador_ip.comprobar(f"ip:{_ip_cliente(request)}")
    if not permitido:
        raise _limite_superado(espera, ambito="ip")
    if session_id:
        permitido, espera = _limitador_sesion.comprobar(f"s:{session_id}")
        if not permitido:
            raise _limite_superado(espera, ambito="sesion")


# --- Modelos -----------------------------------------------------------


class PeticionSesion(BaseModel):
    coleccion: str = Field(min_length=1, max_length=64)


class PeticionChat(BaseModel):
    mensaje: str = Field(min_length=1, max_length=MAX_CHARS_PREGUNTA)


class RespuestaChat(BaseModel):
    respuesta: str
    fuentes: list[dict]
    coleccion: str


# --- Endpoints ---------------------------------------------------------


@app.get("/salud")
def salud():
    """Sin autenticacion y sin datos internos: solo para health checks."""
    return {"estado": "ok"}


@app.post("/sesiones")
def abrir_sesion(peticion: PeticionSesion, request: Request, respuesta: Response):
    """Crea una sesion ligada a UNA coleccion y devuelve la cookie firmada.

    La coleccion se fija aqui y no se puede cambiar despues: asi una sesion
    no puede saltar a otro corpus a mitad de conversacion."""
    _comprobar_limites(request)

    if not colecciones.existe(peticion.coleccion):
        raise _fallo(404, "La coleccion solicitada no existe.")

    session_id = seguridad.nuevo_session_id()
    _sesiones.crear(session_id, peticion.coleccion)

    respuesta.set_cookie(
        key=NOMBRE_COOKIE,
        value=seguridad.firmar_sesion(session_id, SECRET_KEY),
        httponly=True,  # inaccesible desde JavaScript
        secure=COOKIES_SEGURAS,
        samesite="lax",
        max_age=SEGUNDOS_EXPIRACION_SESION,
        path="/",
    )
    seguridad.registrar(
        "sesion_abierta",
        sesion=seguridad.hash_para_log(session_id),
        coleccion=peticion.coleccion,
    )
    return {"coleccion": peticion.coleccion, "expira_en": SEGUNDOS_EXPIRACION_SESION}


@app.post("/chat", response_model=RespuestaChat)
def chat(peticion: PeticionChat, request: Request, sesion=Depends(sesion_actual)):
    session_id = sesion["session_id"]
    coleccion = sesion["coleccion"]  # sale de la sesion, NO del cuerpo
    _comprobar_limites(request, session_id)

    mensaje = peticion.mensaje.strip()
    if not mensaje:
        raise _fallo(400, "El mensaje esta vacio.")
    # Pydantic ya corta por max_length. Esto lo deja explicito y cubre el
    # caso de que alguien toque el modelo: se rechaza ANTES de gastar API.
    if len(mensaje) > MAX_CHARS_PREGUNTA:
        raise _fallo(400, f"El mensaje supera los {MAX_CHARS_PREGUNTA} caracteres.")

    inicio = time.perf_counter()
    try:
        buscador = _cache.obtener(coleccion)
    except colecciones.ColeccionNoEncontrada:
        raise _fallo(404, "La coleccion asociada a la sesion ya no existe.")

    historial = conversacion.recortar_historial(
        _almacen.cargar(session_id, limite=conversacion.MAX_MENSAJES_HISTORIAL)
    )
    system_prompt = conversacion.construir_system_prompt(
        _config["bot_name"], _config["bot_topic"], _config["mostrar_fuentes"]
    )

    futuro = _ejecutor.submit(
        conversacion.responder,
        cliente_gemini(),
        mensaje,
        historial,
        buscador,
        system_prompt,
        mostrar_fuentes=_config["mostrar_fuentes"],
    )
    try:
        texto, contexto = futuro.result(timeout=TIMEOUT_MODELO_SEGUNDOS)
    except TimeoutFuturo:
        futuro.cancel()
        raise _fallo(504, "El servicio ha tardado demasiado. Intentalo de nuevo.")

    if not texto:
        raise _fallo(502, "No se pudo generar una respuesta.")

    _almacen.añadir_turno(session_id, "user", mensaje)
    _almacen.añadir_turno(session_id, "model", texto)
    _sesiones.registrar_uso(session_id)

    campos = {
        "sesion": seguridad.hash_para_log(session_id),
        "coleccion": coleccion,
        "latencia_ms": round((time.perf_counter() - inicio) * 1000),
        "chars_pregunta": len(mensaje),
        "chars_respuesta": len(texto),
        "fragmentos": len(contexto),
        "con_contexto": bool(contexto),
    }
    if LOG_PREGUNTAS:
        campos["pregunta"] = mensaje
    seguridad.registrar("chat", **campos)

    fuentes = [
        {
            "n": i,
            "documento": c.get("source"),
            "pagina": c.get("page"),
            "seccion": c.get("seccion"),
        }
        for i, c in enumerate(contexto, start=1)
    ]
    return RespuestaChat(respuesta=texto, fuentes=fuentes, coleccion=coleccion)


@app.delete("/sesiones/actual")
def cerrar_sesion(respuesta: Response, sesion=Depends(sesion_actual)):
    session_id = sesion["session_id"]
    _almacen.borrar(session_id)
    _sesiones.borrar(session_id)
    _limitador_sesion.olvidar(f"s:{session_id}")
    respuesta.delete_cookie(NOMBRE_COOKIE, path="/")
    seguridad.registrar("sesion_cerrada", sesion=seguridad.hash_para_log(session_id))
    return {"cerrada": True}


@app.get("/colecciones", dependencies=[Depends(verificar_admin)])
def listar_colecciones():
    return {"colecciones": colecciones.listar(), "en_memoria": _cache.cargadas}


@app.post("/colecciones/{nombre}/documentos", dependencies=[Depends(verificar_admin)])
async def subir_documento(nombre: str, archivo: UploadFile = File(...)):
    try:
        colecciones.validar_nombre(nombre)
    except colecciones.NombreInvalido:
        raise _fallo(400, "Nombre de coleccion no valido.")

    contenido = await archivo.read()
    if len(contenido) > MAX_BYTES_SUBIDA:
        raise _fallo(413, "El archivo supera el tamaño maximo permitido.")

    # El tipo se decide por el contenido real, no por la extension ni por el
    # Content-Type: los dos los elige quien sube el archivo.
    try:
        extension = seguridad.validar_contenido(
            archivo.filename or "", contenido, set(rag.EXTRACTORES)
        )
    except seguridad.ArchivoRechazado as e:
        raise _fallo(400, f"Archivo rechazado: {e}")

    docs_dir, _ = colecciones.crear(nombre)
    # Nombre generado por el servidor: el del usuario es entrada no confiable
    # y el pipeline no necesita conservarlo.
    destino = docs_dir / seguridad.nombre_seguro(extension)
    destino.write_bytes(contenido)

    _cache.invalidar(nombre)
    seguridad.registrar(
        "documento_subido", coleccion=nombre, bytes=len(contenido), extension=extension
    )
    return {
        "coleccion": nombre,
        "documento": destino.name,
        "bytes": len(contenido),
        "aviso": "Llama a /colecciones/{nombre}/reindexar para indexarlo.",
    }


@app.post("/colecciones/{nombre}/reindexar", dependencies=[Depends(verificar_admin)])
def reindexar_coleccion(nombre: str):
    try:
        colecciones.validar_nombre(nombre)
    except colecciones.NombreInvalido:
        raise _fallo(400, "Nombre de coleccion no valido.")

    # Parsear un archivo malformado puede consumir recursos sin control, asi
    # que se le pone techo de tiempo. El de memoria no se puede garantizar
    # dentro del proceso: lo tiene que imponer el contenedor (ver README).
    futuro = _ejecutor.submit(colecciones.reindexar, nombre, cliente_gemini())
    try:
        num_chunks, documentos = futuro.result(timeout=TIMEOUT_INDEXADO_SEGUNDOS)
    except TimeoutFuturo:
        futuro.cancel()
        raise _fallo(504, "El indexado ha tardado demasiado.")
    except colecciones.ColeccionNoEncontrada:
        raise _fallo(404, "La coleccion solicitada no existe.")

    _cache.invalidar(nombre)
    seguridad.registrar("reindexado", coleccion=nombre, fragmentos=num_chunks)
    return {"coleccion": nombre, "fragmentos": num_chunks, "documentos": documentos}
