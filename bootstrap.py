"""
Comprueba que las dependencias del proyecto esten instaladas y, si falta
alguna, las instala automaticamente con pip. Se importa y se ejecuta antes
que cualquier otro import de terceros, para que el usuario final pueda
simplemente ejecutar "python chatbot.py" sin pasos manuales previos.
"""

import importlib.util
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


def asegurar_dependencias():
    faltan = [
        paquete
        for modulo, paquete in MODULO_A_PAQUETE.items()
        if _falta_modulo(modulo)
    ]
    if not faltan:
        return

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
