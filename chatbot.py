"""
Cliente de terminal del chatbot RAG: responde solo con informacion de los
documentos que el usuario coloque en docs/, con memoria de conversacion.

Aqui solo vive lo propio de la terminal (bucle de entrada, impresion,
comandos). La logica de conversacion y el system prompt estan en
conversacion.py, compartidos con la API, para que el bot se comporte igual
por los dos caminos.
"""

import os
import sys
from pathlib import Path

import bootstrap

bootstrap.asegurar_dependencias()

from dotenv import load_dotenv
from google import genai
from google.genai import types

import conversacion
import rag
from historial import HistorialJSON

# Se reexportan para que se puedan seguir usando como chatbot.X y no
# romper el codigo (y los tests) que ya dependian de estos nombres.
from conversacion import (  # noqa: F401
    BOT_NAME_POR_DEFECTO,
    BOT_TOPIC_POR_DEFECTO,
    ESPERA_BASE_SEGUNDOS,
    MAX_MENSAJES_HISTORIAL,
    MAX_REINTENTOS,
    MODEL_NAME,
    PROMPT_REESCRITURA,
    REGLA_CON_FUENTES,
    REGLA_SIN_FUENTES,
    SYSTEM_PROMPT_TEMPLATE,
    TOP_K_CONTEXTO,
    TURNOS_PARA_REESCRITURA,
    USAR_STREAMING,
    construir_mensaje_con_contexto,
    construir_system_prompt,
    enviar_mensaje_con_reintento,
    enviar_mensaje_en_streaming,
    flag_env as _flag_env,
    formatear_leyenda_fuentes,
    historial_a_contents,
    recortar_historial,
    reescribir_consulta,
)
from google.genai.errors import APIError  # noqa: F401  (lo parchean los tests)

HISTORIAL_PATH = Path(__file__).parent / "historial.json"


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: no se encontro GEMINI_API_KEY en el .env")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    config = conversacion.cargar_configuracion()
    bot_name = config["bot_name"]
    mostrar_fuentes = config["mostrar_fuentes"]
    debug_reescritura = config["debug_reescritura"]
    system_prompt = conversacion.construir_system_prompt(
        bot_name, config["bot_topic"], mostrar_fuentes
    )

    # El usuario final solo tiene que colocar/actualizar documentos en docs/:
    # el indice se genera o se refresca solo si alguno es nuevo, se ha
    # modificado o se ha eliminado.
    rag.actualizar_si_es_necesario(client)
    chunks, matriz = rag.cargar_indice()
    buscador = rag.crear_buscador(chunks, matriz)

    almacen = HistorialJSON(HISTORIAL_PATH)
    historial = conversacion.recortar_historial(almacen.cargar())

    print(f"{bot_name} - pregunta lo que quieras sobre {config['bot_topic']}.")
    print("Escribe /reset para borrar el historial o /salir para terminar.\n")

    try:
        while True:
            try:
                entrada = input("Tu: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nHasta la proxima!")
                break

            if not entrada:
                continue

            if entrada.lower() in ("/salir", "/exit", "/quit"):
                print("Hasta la proxima!")
                break

            if entrada.lower() == "/reset":
                historial = []
                almacen.borrar()
                print("Historial borrado. Empezamos de cero.\n")
                continue

            # Para buscar se usa la pregunta reescrita; al modelo y al
            # historial les llega siempre la original.
            consulta = conversacion.reescribir_consulta(
                client, historial, entrada, debug=debug_reescritura
            )
            contexto = rag.recuperar_contexto(
                client, consulta, buscador, top_k=conversacion.TOP_K_CONTEXTO
            )
            mensaje_aumentado = conversacion.construir_mensaje_con_contexto(
                entrada, contexto, mostrar_fuentes
            )
            contents = conversacion.historial_a_contents(historial) + [
                types.Content(role="user", parts=[types.Part(text=mensaje_aumentado)])
            ]

            if conversacion.USAR_STREAMING:
                texto_respuesta = conversacion.enviar_mensaje_en_streaming(
                    client, contents, system_prompt, prefijo=f"{bot_name}: "
                )
            else:
                respuesta = conversacion.enviar_mensaje_con_reintento(
                    client, contents, system_prompt
                )
                texto_respuesta = respuesta.text if respuesta else None
                if texto_respuesta:
                    print(f"{bot_name}: {texto_respuesta}\n")

            if not texto_respuesta:
                print(
                    f"{bot_name}: no pude generar una respuesta a eso "
                    "(puede que el contenido haya sido bloqueado). Prueba a "
                    "reformular tu mensaje.\n"
                )
                continue

            if mostrar_fuentes and contexto:
                print(conversacion.formatear_leyenda_fuentes(contexto) + "\n")

            historial.append({"role": "user", "text": entrada})
            historial.append({"role": "model", "text": texto_respuesta})
            historial = conversacion.recortar_historial(historial)
    finally:
        almacen.guardar(None, historial)


if __name__ == "__main__":
    main()
