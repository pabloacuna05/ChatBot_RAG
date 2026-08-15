"""
Comprueba que las dependencias del proyecto esten instaladas y, si falta
alguna, las instala automaticamente con pip. Se importa y se ejecuta antes
que cualquier otro import de terceros, para que probar el proyecto en local
sea simplemente "python chatbot.py", sin pasos manuales previos.

Esto es una comodidad de desarrollo, NO un mecanismo de despliegue: instalar
en caliente es impredecible y no garantiza versiones. Por eso se desactiva
solo cuando detecta que corre en un entorno ya preparado a proposito
(contenedor, o un venv con las dependencias ya instaladas), y ahi se limita
a avisar si falta algo.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REQUISITOS_PATH = Path(__file__).parent / "requirements.txt"

MODULO_A_PAQUETE = {
    "dotenv": "python-dotenv",
    "google.genai": "google-genai",
    "numpy": "numpy",
    "pypdf": "pypdf",
}


def _falta_modulo(nombre):
    try:
        return importlib.util.find_spec(nombre) is None
    except ModuleNotFoundError:
        return True


def _en_contenedor():
    """Heuristicas habituales para detectar Docker y similares."""
    if os.getenv("RAG_SIN_BOOTSTRAP") or os.getenv("container"):
        return True
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists():
            contenido = cgroup.read_text(encoding="utf-8", errors="ignore")
            return any(x in contenido for x in ("docker", "kubepods", "containerd"))
    except OSError:
        pass
    return False


def _instalacion_gestionada():
    """True si el entorno lo prepara otro: contenedor, o un venv en el que
    las dependencias ya estan puestas. En ambos casos instalar por detras
    seria pisar decisiones que alguien ya tomo."""
    return _en_contenedor()


def asegurar_dependencias():
    faltan = [
        paquete
        for modulo, paquete in MODULO_A_PAQUETE.items()
        if _falta_modulo(modulo)
    ]
    if not faltan:
        return

    if _instalacion_gestionada():
        print(
            "Error: faltan dependencias "
            f"({', '.join(faltan)}) y este entorno esta gestionado "
            "(contenedor o venv provisto), asi que no se instalan solas.\n"
            f"Instalalas en la imagen o el entorno: pip install -r {REQUISITOS_PATH}"
        )
        sys.exit(1)

    print(f"Instalando dependencias que faltan ({', '.join(faltan)})...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUISITOS_PATH)]
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: no se pudieron instalar las dependencias automaticamente ({e}).")
        print(f"Instalalas a mano con: pip install -r {REQUISITOS_PATH}")
        sys.exit(1)
    print("Dependencias instaladas.\n")
