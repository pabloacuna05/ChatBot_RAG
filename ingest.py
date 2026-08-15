"""
Utilidad manual para (re)generar el indice RAG a partir de los PDFs en
docs/. No hace falta ejecutarla a mano en el uso normal: chatbot.py detecta
cambios en los PDFs y actualiza el indice el solo al arrancar. Este script
es util para pre-generar el indice antes de desplegar.

    python ingest.py            actualizacion incremental (solo lo que cambio)
    python ingest.py --forzar   borra el indice y lo reconstruye entero
"""

import os
import shutil
import sys

import bootstrap

bootstrap.asegurar_dependencias()

from dotenv import load_dotenv
from google import genai

import rag


def main():
    forzar = "--forzar" in sys.argv[1:]

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: no se encontro GEMINI_API_KEY en el .env")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    if forzar and rag.INDEX_DIR.exists():
        # Sin indice previo que reutilizar, construir_indice reembebe todo.
        shutil.rmtree(rag.INDEX_DIR)
        print("Indice anterior borrado, se reconstruira desde cero.")

    print(f"Procesando PDFs en {rag.DOCS_DIR}...")
    num_chunks, pdfs = rag.construir_indice(client)

    if not pdfs:
        print(f"No se encontro ningun PDF en {rag.DOCS_DIR}.")
        sys.exit(1)

    print(f"Indice generado: {num_chunks} fragmentos de {len(pdfs)} PDF(s):")
    for nombre in pdfs:
        print(f"  - {nombre}")


if __name__ == "__main__":
    main()
