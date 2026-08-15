"""
Evaluacion del RAG: mide si un cambio de configuracion (troceado, top_k,
umbral, modo de busqueda...) mejora o empeora, en vez de decidirlo a ojo.

    python evaluar.py                     evalua con la configuracion actual
    python evaluar.py --sin-juez          solo metricas de recuperacion (no gasta
                                          llamadas de generacion)
    python evaluar.py --csv salida.csv    añade la corrida a un CSV comparable
    python evaluar.py --barrido           prueba varias combinaciones y ordena
    python evaluar.py --detalle           imprime caso a caso lo que fallo

Los casos viven en eval/preguntas.json. Cada caso positivo declara un
fragmento literal que debe aparecer en el contexto recuperado, y
opcionalmente el documento y la pagina correctos. Los casos negativos son
preguntas fuera del corpus: el acierto consiste en que el bot reconozca que
no tiene la informacion.

Metricas:
- recall@k: fraccion de casos positivos en los que algun chunk recuperado
  es el correcto. Es el techo de calidad del sistema: lo que no se recupera,
  el modelo no lo puede responder.
- MRR: media de 1/posicion del primer chunk correcto. Distingue "lo encontro
  el primero" de "lo encontro el ultimo".
- fidelidad / responde: juez LLM sobre la respuesta final.
- rechazo: fraccion de casos negativos en los que el bot NO se invento nada.
"""

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import bootstrap

bootstrap.asegurar_dependencias()

from dotenv import load_dotenv
from google import genai
from google.genai import types

import chatbot
import rag

EVAL_DIR = Path(__file__).parent / "eval"
PREGUNTAS_PATH = EVAL_DIR / "preguntas.json"

MODELO_JUEZ = "gemini-flash-lite-latest"

# Combinaciones que prueba --barrido. Solo parametros de consulta: cambiarlos
# no obliga a reindexar, asi que el barrido no gasta embeddings de documento.
# Para barrer CHUNK_SIZE hay que reindexar, ver --barrido-troceado.
BARRIDO_CONSULTA = {
    "modo": ["densa", "lexica", "hibrida"],
    "top_k": [3, 4, 6, 8],
    # None = corte adaptativo (relativo al mejor score de cada pregunta).
    # Los numericos son cortes fijos, para poder compararlos con el.
    "umbral": [None, 0.3, 0.4, 0.5, 0.6],
}
BARRIDO_TROCEADO = {
    "chunk_size": [600, 900, 1400],
    "chunk_overlap": [100, 150],
}

PROMPT_JUEZ = """\
Eres un evaluador estricto de un sistema de preguntas y respuestas sobre
documentos. Te doy el CONTEXTO que se recupero, la PREGUNTA y la RESPUESTA
que dio el sistema.

Evalua dos cosas por separado:
- "fiel": true si TODO lo que afirma la respuesta esta respaldado por el
  contexto. false si añade datos que no aparecen ahi, aunque suenen
  plausibles o sean ciertos en el mundo real.
- "responde": true si la respuesta contesta de verdad a la pregunta. Si el
  sistema dice que no tiene la informacion, "responde" es false.

Devuelve UNICAMENTE un objeto JSON, sin texto alrededor ni bloques de
codigo:
{{"fiel": true/false, "responde": true/false, "nota": 0-10, "motivo": "..."}}

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}

RESPUESTA:
{respuesta}
"""

# Marcadores de que el bot ha reconocido no tener la informacion. Se usan
# como red de seguridad cuando el juez esta desactivado.
MARCADORES_RECHAZO = (
    "no tengo esa informacion",
    "no tengo informacion",
    "no dispongo",
    "no puedo ayudarte con eso",
    "no aparece",
    "no se menciona",
    "no encuentro",
    "no tengo datos",
)


def normalizar(texto):
    """Minusculas, sin tildes y con los espacios colapsados, para comparar
    fragmentos sin que un acento o un salto de linea rompa la comparacion."""
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto).strip().lower()


def cargar_casos(ruta=PREGUNTAS_PATH):
    if not Path(ruta).exists():
        print(f"Error: no existe {ruta}. Crea ahi tus casos de evaluacion.")
        sys.exit(1)
    with open(ruta, "r", encoding="utf-8") as f:
        casos = json.load(f)
    if not casos:
        print(f"Error: {ruta} no contiene ningun caso.")
        sys.exit(1)
    return casos


def _chunk_es_relevante(chunk, caso):
    """Un chunk cuenta como correcto si cumple TODOS los criterios que el
    caso declara: el fragmento esperado y, si se indican, documento y
    pagina."""
    esperado = caso.get("esperado")
    if esperado and normalizar(esperado) not in normalizar(chunk.get("text", "")):
        return False
    documento = caso.get("documento")
    if documento and chunk.get("source") != documento:
        return False
    pagina = caso.get("pagina")
    if pagina is not None and chunk.get("page") != pagina:
        return False
    return bool(esperado or documento or pagina is not None)


def _posicion_primer_acierto(contexto, caso):
    """Posicion (1-indexada) del primer chunk correcto, o None."""
    for posicion, chunk in enumerate(contexto, start=1):
        if _chunk_es_relevante(chunk, caso):
            return posicion
    return None


def _parece_rechazo(respuesta):
    normalizada = normalizar(respuesta)
    return any(marcador in normalizada for marcador in MARCADORES_RECHAZO)


def juzgar(client, pregunta, contexto, respuesta):
    """Juez LLM. Devuelve un dict con fiel/responde/nota/motivo, o None si
    la llamada falla o no devuelve JSON parseable."""
    texto_contexto = "\n\n".join(c["text"] for c in contexto) or "(vacio)"
    instruccion = PROMPT_JUEZ.format(
        contexto=texto_contexto, pregunta=pregunta, respuesta=respuesta
    )
    try:
        salida = client.models.generate_content(
            model=MODELO_JUEZ,
            contents=[types.Content(role="user", parts=[types.Part(text=instruccion)])],
        )
        crudo = (salida.text or "").strip()
    except Exception:
        return None

    # El modelo a veces envuelve el JSON en ```json ... ```
    crudo = re.sub(r"^```(?:json)?|```$", "", crudo, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", crudo, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def evaluar_caso(client, caso, buscador, config, system_prompt, con_juez):
    """Ejecuta un caso completo y devuelve sus metricas."""
    pregunta = caso["pregunta"]
    es_negativo = caso.get("tipo") == "negativo"

    # En la config el umbral se guarda como "adaptativo" para que se lea en
    # la tabla y en el CSV; recuperar_contexto espera None para ese caso.
    umbral = config["umbral"]
    umbral = None if umbral == "adaptativo" else umbral

    inicio = time.perf_counter()
    contexto = rag.recuperar_contexto(
        client,
        pregunta,
        buscador,
        top_k=config["top_k"],
        umbral_similitud=umbral,
    )
    latencia_recuperacion = time.perf_counter() - inicio

    posicion = None if es_negativo else _posicion_primer_acierto(contexto, caso)

    resultado = {
        "id": caso.get("id", pregunta[:40]),
        "tipo": caso.get("tipo", "positivo"),
        "pregunta": pregunta,
        "num_contexto": len(contexto),
        "posicion": posicion,
        "acierto_recuperacion": posicion is not None,
        "latencia_recuperacion": latencia_recuperacion,
        "respuesta": None,
        "fiel": None,
        "responde": None,
        "nota": None,
        "rechaza": None,
    }

    if not con_juez:
        return resultado

    mensaje = chatbot.construir_mensaje_con_contexto(pregunta, contexto)
    contents = [types.Content(role="user", parts=[types.Part(text=mensaje)])]
    inicio = time.perf_counter()
    respuesta = chatbot.enviar_mensaje_con_reintento(client, contents, system_prompt)
    resultado["latencia_respuesta"] = time.perf_counter() - inicio

    texto = (respuesta.text if respuesta else None) or ""
    resultado["respuesta"] = texto
    resultado["rechaza"] = _parece_rechazo(texto)

    veredicto = juzgar(client, pregunta, contexto, texto)
    if veredicto:
        resultado["fiel"] = bool(veredicto.get("fiel"))
        resultado["responde"] = bool(veredicto.get("responde"))
        resultado["nota"] = veredicto.get("nota")
        resultado["motivo"] = veredicto.get("motivo", "")
        # Para un caso negativo, el acierto es NO responder.
        resultado["rechaza"] = not resultado["responde"]

    return resultado


def agregar(resultados):
    """Resume los resultados por caso en las metricas de la corrida."""
    positivos = [r for r in resultados if r["tipo"] != "negativo"]
    negativos = [r for r in resultados if r["tipo"] == "negativo"]

    def media(valores):
        valores = [v for v in valores if v is not None]
        return statistics.mean(valores) if valores else None

    recall = media([1.0 if r["acierto_recuperacion"] else 0.0 for r in positivos])
    mrr = media([1.0 / r["posicion"] if r["posicion"] else 0.0 for r in positivos])
    fidelidad = media([1.0 if r["fiel"] else 0.0 for r in positivos if r["fiel"] is not None])
    responde = media(
        [1.0 if r["responde"] else 0.0 for r in positivos if r["responde"] is not None]
    )
    nota = media([r["nota"] for r in positivos if isinstance(r["nota"], (int, float))])
    rechazo = media(
        [1.0 if r["rechaza"] else 0.0 for r in negativos if r["rechaza"] is not None]
    )

    return {
        "casos": len(resultados),
        "positivos": len(positivos),
        "negativos": len(negativos),
        "recall": recall,
        "mrr": mrr,
        "fidelidad": fidelidad,
        "responde": responde,
        "nota_media": nota,
        "rechazo": rechazo,
        "latencia_recuperacion": media([r["latencia_recuperacion"] for r in resultados]),
    }


def _fmt(valor, decimales=3):
    if valor is None:
        return "-"
    if isinstance(valor, float):
        return f"{valor:.{decimales}f}"
    return str(valor)


def imprimir_tabla(filas, columnas):
    """Tabla de ancho fijo, sin dependencias."""
    anchos = [
        max(len(str(col)), max((len(str(f.get(col, ""))) for f in filas), default=0))
        for col in columnas
    ]
    separador = "-+-".join("-" * a for a in anchos)
    print(" | ".join(str(c).ljust(a) for c, a in zip(columnas, anchos)))
    print(separador)
    for fila in filas:
        print(" | ".join(str(fila.get(c, "")).ljust(a) for c, a in zip(columnas, anchos)))


def config_actual(top_k=None, umbral=None, modo=None):
    return {
        "modo": modo or rag.MODO_BUSQUEDA,
        "top_k": top_k if top_k is not None else chatbot.TOP_K_CONTEXTO,
        "umbral": "adaptativo" if umbral is None else umbral,
        "chunk_size": rag.CHUNK_SIZE,
        "chunk_overlap": rag.CHUNK_OVERLAP,
        "embedding_dim": rag.EMBEDDING_DIM,
        "embedding_model": rag.EMBEDDING_MODEL,
        "modelo": chatbot.MODEL_NAME,
    }


def guardar_csv(ruta, config, metricas):
    """Añade la corrida al CSV, con la configuracion al lado de las metricas
    para poder comparar ejecuciones distintas."""
    ruta = Path(ruta)
    fila = {"fecha": datetime.now().isoformat(timespec="seconds")}
    fila.update(config)
    fila.update({k: _fmt(v) for k, v in metricas.items()})

    existe = ruta.exists()
    with open(ruta, "a", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(fila))
        if not existe:
            escritor.writeheader()
        escritor.writerow(fila)
    print(f"\nCorrida añadida a {ruta}")


def ejecutar(client, casos, buscador, config, system_prompt, con_juez, detalle):
    resultados = []
    for i, caso in enumerate(casos, start=1):
        print(f"  [{i}/{len(casos)}] {caso.get('id', caso['pregunta'][:40])}", end="\r")
        resultados.append(
            evaluar_caso(client, caso, buscador, config, system_prompt, con_juez)
        )
    print(" " * 70, end="\r")

    if detalle:
        print("\nCaso a caso:")
        filas = [
            {
                "id": r["id"],
                "tipo": r["tipo"],
                "pos": _fmt(r["posicion"], 0) if r["posicion"] else "-",
                "ctx": r["num_contexto"],
                "fiel": _fmt(r["fiel"]),
                "responde": _fmt(r["responde"]),
                "nota": _fmt(r["nota"], 1),
            }
            for r in resultados
        ]
        imprimir_tabla(filas, ["id", "tipo", "pos", "ctx", "fiel", "responde", "nota"])

        fallos = [r for r in resultados if r["tipo"] != "negativo" and not r["acierto_recuperacion"]]
        if fallos:
            print("\nCasos donde no se recupero el fragmento esperado:")
            for r in fallos:
                print(f"  - {r['id']}: {r['pregunta']}")

    return resultados


def main():
    parser = argparse.ArgumentParser(description="Evaluacion del RAG")
    parser.add_argument("--sin-juez", action="store_true", help="solo recuperacion")
    parser.add_argument("--csv", help="archivo CSV donde acumular las corridas")
    parser.add_argument("--barrido", action="store_true", help="barre top_k y umbral")
    parser.add_argument(
        "--barrido-troceado",
        action="store_true",
        help="barre tambien CHUNK_SIZE (reindexa, gasta llamadas a la API)",
    )
    parser.add_argument("--detalle", action="store_true", help="imprime caso a caso")
    parser.add_argument("--casos", default=str(PREGUNTAS_PATH), help="ruta del json")
    parser.add_argument(
        "--benchmark-indice",
        action="store_true",
        help="compara latencia y recall@4 entre numpy y hnswlib, y sale",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: no se encontro GEMINI_API_KEY en el .env")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    casos = cargar_casos(args.casos)

    bot_name = os.getenv("BOT_NAME", "").strip() or chatbot.BOT_NAME_POR_DEFECTO
    bot_topic = os.getenv("BOT_TOPIC", "").strip() or chatbot.BOT_TOPIC_POR_DEFECTO
    system_prompt = chatbot.construir_system_prompt(bot_name, bot_topic)

    rag.actualizar_si_es_necesario(client)
    chunks, matriz = rag.cargar_indice()
    if not chunks:
        print("No hay indice. Coloca tus documentos en docs/ y vuelve a intentarlo.")
        sys.exit(1)

    if args.benchmark_indice:
        print(f"Benchmark del indice vectorial sobre {len(chunks)} fragmentos:\n")
        resultado = rag.benchmark_vectorial(matriz)
        filas = [{"backend": nombre, **datos} for nombre, datos in resultado.items()]
        columnas = sorted({c for f in filas for c in f})
        imprimir_tabla(
            [{c: _fmt(f.get(c, "-")) for c in columnas} for f in filas], columnas
        )
        return

    con_juez = not args.sin_juez
    print(f"{len(casos)} casos sobre {len(chunks)} fragmentos.")
    if con_juez:
        print("Con juez LLM (usa --sin-juez para evaluar solo la recuperacion).\n")
    else:
        print("Solo metricas de recuperacion.\n")

    if not args.barrido and not args.barrido_troceado:
        config = config_actual()
        buscador = rag.crear_buscador(chunks, matriz)
        resultados = ejecutar(
            client, casos, buscador, config, system_prompt, con_juez, args.detalle
        )
        metricas = agregar(resultados)

        print("\nResultado:")
        imprimir_tabla(
            [{k: _fmt(v) for k, v in metricas.items()}], list(metricas)
        )
        if args.csv:
            guardar_csv(args.csv, config, metricas)
        return

    # Modo barrido. Los parametros de troceado son el bucle exterior porque
    # cambiarlos obliga a reindexar; los de consulta se prueban en caliente
    # sobre el indice ya construido.
    if args.barrido_troceado:
        troceados = [
            {"chunk_size": cs, "chunk_overlap": co}
            for cs in BARRIDO_TROCEADO["chunk_size"]
            for co in BARRIDO_TROCEADO["chunk_overlap"]
            if co < cs
        ]
        print(
            f"Aviso: se reindexara el corpus {len(troceados)} veces, una por "
            "combinacion de troceado. Esto gasta llamadas a la API.\n"
        )
    else:
        troceados = [{"chunk_size": rag.CHUNK_SIZE, "chunk_overlap": rag.CHUNK_OVERLAP}]

    filas = []
    chunk_size_original = rag.CHUNK_SIZE
    chunk_overlap_original = rag.CHUNK_OVERLAP
    try:
        for troceado in troceados:
            chunks_actual, matriz_actual = chunks, matriz
            temporal = None
            buscador = None

            if args.barrido_troceado:
                rag.CHUNK_SIZE = troceado["chunk_size"]
                rag.CHUNK_OVERLAP = troceado["chunk_overlap"]
                temporal = Path(tempfile.mkdtemp(prefix="eval_indice_"))
                print(
                    f"Reindexando con chunk_size={troceado['chunk_size']} "
                    f"overlap={troceado['chunk_overlap']}..."
                )
                rag.construir_indice(client, index_dir=temporal)
                chunks_actual, matriz_actual = rag.cargar_indice(temporal)

            try:
                # El buscador se reconstruye por modo porque el indice
                # lexico solo existe en los modos que lo usan.
                for modo in BARRIDO_CONSULTA["modo"]:
                    buscador = rag.crear_buscador(chunks_actual, matriz_actual, modo)
                    for top_k in BARRIDO_CONSULTA["top_k"]:
                        for umbral in BARRIDO_CONSULTA["umbral"]:
                            config = config_actual(
                                top_k=top_k, umbral=umbral, modo=modo
                            )
                            config["chunk_size"] = troceado["chunk_size"]
                            config["chunk_overlap"] = troceado["chunk_overlap"]
                            print(
                                f"  chunk={troceado['chunk_size']} modo={modo} "
                                f"top_k={top_k} umbral={umbral}"
                            )
                            resultados = ejecutar(
                                client,
                                casos,
                                buscador,
                                config,
                                system_prompt,
                                con_juez,
                                False,
                            )
                            metricas = agregar(resultados)
                            filas.append(
                                {
                                    "chunk": troceado["chunk_size"],
                                    "overlap": troceado["chunk_overlap"],
                                    "modo": modo,
                                    "top_k": top_k,
                                    "umbral": umbral,
                                    "recall": _fmt(metricas["recall"]),
                                    "mrr": _fmt(metricas["mrr"]),
                                    "fidelidad": _fmt(metricas["fidelidad"]),
                                    "rechazo": _fmt(metricas["rechazo"]),
                                    "_orden": metricas["mrr"] or 0.0,
                                }
                            )
                            if args.csv:
                                guardar_csv(args.csv, config, metricas)
            finally:
                if temporal:
                    shutil.rmtree(temporal, ignore_errors=True)
    finally:
        rag.CHUNK_SIZE = chunk_size_original
        rag.CHUNK_OVERLAP = chunk_overlap_original

    filas.sort(key=lambda f: f["_orden"], reverse=True)
    for fila in filas:
        fila.pop("_orden")

    print("\nBarrido ordenado por MRR:")
    imprimir_tabla(
        filas,
        [
            "chunk",
            "overlap",
            "modo",
            "top_k",
            "umbral",
            "recall",
            "mrr",
            "fidelidad",
            "rechazo",
        ],
    )


if __name__ == "__main__":
    main()
