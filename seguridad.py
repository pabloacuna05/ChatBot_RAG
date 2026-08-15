"""
Piezas de seguridad de la API: firma de sesiones, limitacion de peticiones,
validacion real de archivos subidos y utilidades de log.

Todo lo de aqui es logica pura, sin FastAPI, para poder probarlo sin montar
un servidor. Se usa SOLO desde api.py: la CLI no lo necesita.
"""

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections import deque

logger = logging.getLogger("chatbot_rag")

# --- Firma de sesiones -------------------------------------------------

# El id de sesion identifica una conversacion. Si se aceptara tal cual del
# cliente, cualquiera podria leer la conversacion de otro probando ids. Por
# eso el servidor lo genera y lo firma: un id sin firma valida no vale.
_SEPARADOR_FIRMA = "."
RE_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")


class FirmaInvalida(ValueError):
    pass


def generar_clave_secreta():
    """Clave efimera para desarrollo. En produccion se pone SECRET_KEY en el
    .env: sin ella, reiniciar el servidor invalida todas las sesiones."""
    return secrets.token_hex(32)


def nuevo_session_id():
    return uuid.uuid4().hex


def firmar_sesion(session_id, clave):
    firma = hmac.new(
        clave.encode("utf-8"), session_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{session_id}{_SEPARADOR_FIRMA}{firma}"


def verificar_sesion(cookie, clave):
    """Devuelve el session_id si la cookie es autentica. Si no, explota.

    La comparacion es en tiempo constante: comparar firmas con == filtra
    informacion por el tiempo de respuesta."""
    if not cookie or _SEPARADOR_FIRMA not in cookie:
        raise FirmaInvalida("cookie de sesion ausente o mal formada")

    session_id, _, firma = cookie.partition(_SEPARADOR_FIRMA)
    if not RE_SESSION_ID.match(session_id):
        raise FirmaInvalida("identificador de sesion mal formado")

    esperada = hmac.new(
        clave.encode("utf-8"), session_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(firma, esperada):
        raise FirmaInvalida("firma de sesion invalida")
    return session_id


def hash_para_log(valor):
    """Identificador estable pero no reversible, para poder seguir una
    sesion en los logs sin dejar ahi el id real."""
    if not valor:
        return "-"
    return hashlib.sha256(str(valor).encode("utf-8")).hexdigest()[:12]


def id_correlacion():
    """Id corto que se le da al cliente y se escribe en el log, para poder
    cruzar 'me ha fallado esto' con el error real sin filtrar detalles."""
    return uuid.uuid4().hex[:12]


# --- Limitacion de peticiones -----------------------------------------


class LimitadorPeticiones:
    """Ventanas deslizantes en memoria.

    Aviso importante: el estado vive en el proceso. Con varios workers de
    uvicorn cada uno lleva su propia cuenta, asi que los limites efectivos
    se multiplican por el numero de workers. Para un despliegue serio esto
    debe ir a Redis; la interfaz esta pensada para poder sustituirlo."""

    def __init__(self, limites):
        """limites: lista de (peticiones, segundos_de_ventana)."""
        self.limites = sorted(limites, key=lambda l: l[1])
        self._eventos = {}
        self._lock = threading.Lock()

    def _purgar(self, cola, ahora):
        ventana_mayor = self.limites[-1][1] if self.limites else 0
        while cola and ahora - cola[0] > ventana_mayor:
            cola.popleft()

    def comprobar(self, clave, ahora=None):
        """Devuelve (permitido, segundos_hasta_reintentar)."""
        if not self.limites:
            return True, 0
        ahora = time.time() if ahora is None else ahora

        with self._lock:
            cola = self._eventos.setdefault(clave, deque())
            self._purgar(cola, ahora)

            for maximo, ventana in self.limites:
                recientes = [t for t in cola if ahora - t <= ventana]
                if len(recientes) >= maximo:
                    espera = ventana - (ahora - recientes[0])
                    return False, max(1, int(espera) + 1)

            cola.append(ahora)
            return True, 0

    def olvidar(self, clave):
        with self._lock:
            self._eventos.pop(clave, None)

    def limpiar(self, ahora=None):
        """Descarta claves sin actividad reciente, para que la memoria no
        crezca sin limite con el tiempo."""
        ahora = time.time() if ahora is None else ahora
        ventana_mayor = self.limites[-1][1] if self.limites else 0
        with self._lock:
            vacias = []
            for clave, cola in self._eventos.items():
                self._purgar(cola, ahora)
                if not cola:
                    vacias.append(clave)
            for clave in vacias:
                del self._eventos[clave]


# --- Validacion de archivos subidos ------------------------------------

# Firmas de contenido. Fiarse de la extension o del Content-Type que manda
# el cliente no sirve de nada: los dos los elige el atacante.
FIRMAS = {
    ".pdf": [b"%PDF-"],
    ".docx": [b"PK\x03\x04"],  # es un zip
}
EXTENSIONES_DE_TEXTO = {".txt", ".md", ".markdown", ".html", ".htm"}


class ArchivoRechazado(ValueError):
    pass


def validar_contenido(nombre_archivo, contenido, extensiones_permitidas):
    """Comprueba que el contenido se corresponde de verdad con el tipo que
    dice la extension. Devuelve la extension normalizada."""
    extension = os.path.splitext(nombre_archivo)[1].lower()
    if extension not in extensiones_permitidas:
        raise ArchivoRechazado(f"formato no admitido: {extension or 'sin extension'}")
    if not contenido:
        raise ArchivoRechazado("el archivo esta vacio")

    firmas = FIRMAS.get(extension)
    if firmas:
        if not any(contenido.startswith(f) for f in firmas):
            raise ArchivoRechazado(
                f"el contenido no se corresponde con un archivo {extension}"
            )
        return extension

    if extension in EXTENSIONES_DE_TEXTO:
        # Un ejecutable renombrado a .txt no debe colarse: se exige que sea
        # texto decodificable y sin bytes nulos.
        if b"\x00" in contenido[:8192]:
            raise ArchivoRechazado("el archivo parece binario, no texto")
        try:
            contenido[:8192].decode("utf-8")
        except UnicodeDecodeError:
            try:
                contenido[:8192].decode("latin-1")
            except UnicodeDecodeError:
                raise ArchivoRechazado("no se pudo interpretar el archivo como texto")
        return extension

    raise ArchivoRechazado(f"formato no admitido: {extension}")


def nombre_seguro(extension):
    """Nombre generado por el servidor. No se conserva el del usuario: es
    entrada no confiable y no aporta nada al pipeline, que identifica los
    documentos por su ruta."""
    return f"{uuid.uuid4().hex}{extension}"


# --- Log estructurado --------------------------------------------------


def configurar_logging(nivel="INFO"):
    """Una linea JSON por evento, para que se pueda agregar y consultar."""
    manejador = logging.StreamHandler()
    manejador.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [manejador]
    logger.setLevel(getattr(logging, nivel.upper(), logging.INFO))
    logger.propagate = False
    return logger


def registrar(evento, **campos):
    """Escribe un evento estructurado. Nunca metas aqui texto del usuario ni
    secretos: lo que entra aqui acaba en disco y en el agregador de logs."""
    campos["evento"] = evento
    campos["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    logger.info(json.dumps(campos, ensure_ascii=False, default=str))
