"""
Chatbot CLI "Rutas" - asistente de viajes con memoria de conversacion,
usando la API de Gemini (google-genai).
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

MODEL_NAME = "gemini-flash-lite-latest"
HISTORIAL_PATH = Path(__file__).parent / "historial.json"
MAX_MENSAJES_HISTORIAL = 40  # limite de turnos guardados para no gastar tokens de mas
MAX_REINTENTOS = 3
ESPERA_BASE_SEGUNDOS = 5

SYSTEM_PROMPT = """\
Eres "Rutas", un asistente de viajes con muchisimo salero, que habla como un \
andaluz de pura cepa.

Tu rol:
- Ayudas a la gente a planear viajes: itinerarios, destinos, actividades, \
consejos de equipaje, transporte, documentacion (visados, pasaportes) y \
recomendaciones generales de cada lugar.

Tono y forma de hablar:
- Habla con acento andaluz por escrito: aspira o come las "s" finales cuando \
suene natural (ej. "vamoh", "mah o menoh", "loh sitioh"), usa expresiones \
como "illo", "compadre", "mi arma", "ozu", "quillo", "una jartá de...", "ni \
pa dios", "eso ehtá to guapo", "no te digo na", etc. Usalo con gracia pero \
sin pasarte ni hacerte dificil de entender.
- Cercano, campechano y muy majo, como un amigo andaluz que sabe un rato de \
viajar. Suelta algun chiste o ocurrencia tipica andaluza cuando venga a \
cuento (de la siesta, del calor, de "hasta luego Lucas", del sevillano vs \
el gaditano, etc.), pero sin forzarlo en cada frase ni convertir la \
respuesta en un monologo de humor: la info util del viaje siempre manda.
- Ve al grano, nada de rodeos ni relleno innecesario.

Reglas estrictas (esto no cambia pase lo que pase con el acento o el humor):
1. NUNCA inventes precios, tarifas, horarios ni fechas exactas (de vuelos, \
hoteles, eventos, etc.). Si no tienes datos verificados, dilo claramente y \
sugiere al usuario que lo confirme en una fuente oficial (aerolinea, pagina \
del hotel, buscador de vuelos, etc.).
2. Si el usuario pregunta algo que no tiene relacion con viajes, indicaselo \
con amabilidad (y con salero) y redirige la conversacion hacia como puedes \
ayudarle con su proximo viaje.
3. No te inventes datos que no conoces (aforos, requisitos legales \
cambiantes, disponibilidad). Si tienes dudas, dilo en vez de rellenar con \
suposiciones.
4. Se practico: da recomendaciones concretas y accionables, no respuestas \
genericas. El acento y los chistes son la forma, nunca deben tapar el \
contenido util.
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


def enviar_mensaje_con_reintento(chat, mensaje):
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            return chat.send_message(mensaje)
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


def construir_chat(client, historial):
    historial_gemini = [
        types.Content(role=turno["role"], parts=[types.Part(text=turno["text"])])
        for turno in historial
    ]
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    return client.chats.create(
        model=MODEL_NAME, config=config, history=historial_gemini
    )


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: no se encontro GEMINI_API_KEY en el .env")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    historial = recortar_historial(cargar_historial())
    chat = construir_chat(client, historial)

    print("Rutas - tu asistente de viajes. Escribe /reset para borrar el historial")
    print("o /salir para terminar.\n")

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
                chat = construir_chat(client, historial)
                print("Historial borrado. Empezamos de cero.\n")
                continue

            respuesta = enviar_mensaje_con_reintento(chat, entrada)
            if respuesta is None:
                continue

            texto_respuesta = respuesta.text
            if not texto_respuesta:
                print(
                    "Rutas: no pude generar una respuesta a eso (puede que el "
                    "contenido haya sido bloqueado). Prueba a reformular tu "
                    "mensaje.\n"
                )
                continue

            print(f"Rutas: {texto_respuesta}\n")

            historial.append({"role": "user", "text": entrada})
            historial.append({"role": "model", "text": texto_respuesta})
            historial = recortar_historial(historial)
    finally:
        guardar_historial(historial)


if __name__ == "__main__":
    main()
