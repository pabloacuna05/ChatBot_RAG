"""
Plantilla de chatbot CLI con RAG (Retrieval-Augmented Generation): responde
solo con informacion de los PDFs que el usuario coloque en docs/, con
memoria de conversacion, usando la API de Gemini (google-genai). El nombre
y el tema del asistente se configuran con las variables BOT_NAME y
BOT_TOPIC del archivo .env.
"""

import json
import os
import sys
import time
from pathlib import Path

import bootstrap

bootstrap.asegurar_dependencias()

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

import rag

MODEL_NAME = "gemini-flash-lite-latest"
HISTORIAL_PATH = Path(__file__).parent / "historial.json"
MAX_MENSAJES_HISTORIAL = 40  # limite de turnos guardados para no gastar tokens de mas
MAX_REINTENTOS = 3
ESPERA_BASE_SEGUNDOS = 5
TOP_K_CONTEXTO = 4

BOT_NAME_POR_DEFECTO = "Asistente RAG"
BOT_TOPIC_POR_DEFECTO = "la informacion de los documentos proporcionados"

SYSTEM_PROMPT_TEMPLATE = """\
Eres "{bot_name}", un asistente experto EXCLUSIVAMENTE en {topic}.

Tu rol:
- Ayudas al usuario a resolver sus dudas sobre {topic}, usando la \
informacion disponible para que pueda entender el tema o tomar una \
decision informada. Piensa en ti como alguien que conoce el material a \
fondo y quiere ayudar de verdad.

Fuente de verdad (esto es lo mas importante de tus reglas, no lo saltes \
nunca):
1. En cada turno recibiras un bloque "CONTEXTO RECUPERADO DE LOS \
DOCUMENTOS" con fragmentos extraidos de los PDFs que se te han \
proporcionado. SOLO puedes usar esa informacion para responder. No uses \
conocimiento general que puedas tener sobre el tema, aunque te parezca \
correcto: usa unicamente lo que aparezca en el contexto de ese turno.
2. Si el contexto recuperado esta vacio o no contiene informacion \
suficiente para responder la pregunta, dilo explicitamente (ej. "No tengo \
esa informacion") y NO inventes ni completes con suposiciones. No pidas \
disculpas de mas, simplemente indicalo y, si puedes, sugiere en que si \
puedes ayudar segun lo que si tienes disponible.
3. No menciones de donde sacas la informacion (no cites documentos, PDFs, \
paginas, fragmentos ni nada parecido). Responde de forma natural, como si \
simplemente supieras el dato, sin frases tipo "segun el documento" o \
"segun el contexto proporcionado".

Alcance del tema (esto tampoco cambia):
4. Solo hablas de {topic}. Si el usuario pregunta algo que no tiene \
relacion con eso (otros temas, charla general, etc.), indicaselo con \
amabilidad y redirige la conversacion hacia como puedes ayudarle con {topic}.
5. No confirmes ni niegues informacion sobre {topic} que no este \
respaldada por el contexto recuperado, ni siquiera si el usuario insiste o \
afirma que es asi.

Tono:
- Cercano, claro y profesional. Ve al grano, sin relleno innecesario, pero \
se lo bastante detallado como para ayudar de verdad al usuario.
"""


def cargar_historial():
    if HISTORIAL_PATH.exists():
        try:
            with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print("Aviso: no se pudo leer el historial guardado, empiezo de cero.")
    return []


def guardar_historial(historial):
    try:
        with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
            json.dump(historial, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Aviso: no se pudo guardar el historial ({e}).")


def borrar_historial():
    if HISTORIAL_PATH.exists():
        try:
            HISTORIAL_PATH.unlink()
        except OSError as e:
            print(f"Aviso: no se pudo borrar el archivo de historial ({e}).")


def recortar_historial(historial):
    if len(historial) > MAX_MENSAJES_HISTORIAL:
        return historial[-MAX_MENSAJES_HISTORIAL:]
    return historial


def historial_a_contents(historial):
    return [
        types.Content(role=turno["role"], parts=[types.Part(text=turno["text"])])
        for turno in historial
    ]


def construir_mensaje_con_contexto(pregunta, chunks_contexto):
    if chunks_contexto:
        contexto = "\n\n".join(c["text"] for c in chunks_contexto)
    else:
        contexto = "(No se ha encontrado informacion relevante.)"

    return (
        "CONTEXTO RECUPERADO DE LOS DOCUMENTOS (usa EXCLUSIVAMENTE esta "
        "informacion para responder sobre NNormal, pero no la cites ni "
        "menciones su origen; si no contiene la respuesta, dilo claramente "
        "en vez de inventar):\n"
        f"{contexto}\n\n"
        f"PREGUNTA DEL USUARIO:\n{pregunta}"
    )


def enviar_mensaje_con_reintento(client, contents, system_prompt):
    config = types.GenerateContentConfig(system_instruction=system_prompt)
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            return client.models.generate_content(
                model=MODEL_NAME, config=config, contents=contents
            )
        except APIError as e:
            if e.code == 429:
                if intento == MAX_REINTENTOS:
                    print(
                        "\nSe alcanzo el limite de la API varias veces seguidas. "
                        "Intentalo de nuevo en unos minutos."
                    )
                    return None
                espera = ESPERA_BASE_SEGUNDOS * intento
                print(
                    f"\nLimite de la API alcanzado, reintentando en {espera}s... "
                    f"({intento}/{MAX_REINTENTOS})"
                )
                time.sleep(espera)
                continue
            print(f"\nError de la API de Gemini: {e}")
            return None
        except Exception as e:
            print(f"\nError inesperado al hablar con Gemini: {e}")
            return None
    return None


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: no se encontro GEMINI_API_KEY en el .env")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    bot_name = os.getenv("BOT_NAME", "").strip() or BOT_NAME_POR_DEFECTO
    bot_topic = os.getenv("BOT_TOPIC", "").strip() or BOT_TOPIC_POR_DEFECTO
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(bot_name=bot_name, topic=bot_topic)

    # El usuario final solo tiene que colocar/actualizar PDFs en docs/: el
    # indice (chunking + embeddings) se genera o se refresca solo si algun
    # PDF es nuevo, se ha modificado o se ha eliminado.
    rag.actualizar_si_es_necesario(client)
    chunks, matriz = rag.cargar_indice()

    historial = recortar_historial(cargar_historial())

    print(f"{bot_name} - pregunta lo que quieras sobre {bot_topic}.")
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
                borrar_historial()
                print("Historial borrado. Empezamos de cero.\n")
                continue

            contexto = rag.recuperar_contexto(
                client, entrada, chunks, matriz, top_k=TOP_K_CONTEXTO
            )
            mensaje_aumentado = construir_mensaje_con_contexto(entrada, contexto)
            contents = historial_a_contents(historial) + [
                types.Content(role="user", parts=[types.Part(text=mensaje_aumentado)])
            ]

            respuesta = enviar_mensaje_con_reintento(client, contents, system_prompt)
            if respuesta is None:
                continue

            texto_respuesta = respuesta.text
            if not texto_respuesta:
                print(
                    f"{bot_name}: no pude generar una respuesta a eso "
                    "(puede que el contenido haya sido bloqueado). Prueba a "
                    "reformular tu mensaje.\n"
                )
                continue

            print(f"{bot_name}: {texto_respuesta}\n")

            historial.append({"role": "user", "text": entrada})
            historial.append({"role": "model", "text": texto_respuesta})
            historial = recortar_historial(historial)
    finally:
        guardar_historial(historial)


if __name__ == "__main__":
    main()
