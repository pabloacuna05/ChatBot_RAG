# ChatBot_RAG

Plantilla para montar tu propio chatbot con RAG (Retrieval-Augmented
Generation) sobre los documentos que tu quieras. El asistente responde
EXCLUSIVAMENTE con informacion que aparezca en tus documentos y no habla de
ningun otro tema: si algo no esta ahi, lo dice en vez de inventarselo.

Sirve para cualquier documentacion propia — un catalogo de producto, un
manual, unos apuntes de clase, normativa interna, un TFG... — sin tocar
codigo: solo pones tu API key, tus documentos y el tema del que hablara.

Requisitos: Python 3.9 o superior y una API key de Gemini (el nivel gratuito
es suficiente para probarlo).

## Probarlo en local

1. Consigue tu API key en [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Copia `.env.example` a `.env` y rellenalo:
   - `GEMINI_API_KEY` (obligatorio): la key del paso anterior.
   - `BOT_NAME` (opcional): como se llama tu asistente, ej. `Asistente de Cocina`.
   - `BOT_TOPIC` (opcional): el tema del que puede hablar, escrito como
     complemento de la frase "experto en...", ej. `las recetas del libro`.
   - `MOSTRAR_FUENTES` (opcional): a `true`, el bot cita con `[1]`, `[2]` y
     al final se imprime de que documento y pagina sale cada cosa.
   - `DEBUG_REESCRITURA` (opcional): imprime la consulta reescrita que se
     usa para buscar. Util para entender por que una pregunta no encuentra
     contexto.
3. Coloca tus documentos en la carpeta `docs/`. Se admiten **PDF, DOCX, TXT,
   MD y HTML**, y puedes mezclar varios.
4. Arranca el chatbot:
   ```
   python chatbot.py
   ```

No hace falta instalar nada a mano: al arrancar, `bootstrap.py` detecta las
dependencias que falten y las instala solo. Si prefieres hacerlo tu, o vas a
desplegar, usa `pip install -r requirements.txt` (versiones fijadas).

## Como funciona en el dia a dia

Al arrancar, el chatbot comprueba (por hash del contenido) si los documentos
de `docs/` son nuevos, se han modificado o se han borrado, y si es asi
actualiza el indice el solo. La actualizacion es **incremental**: solo se
vuelven a embeber los documentos que han cambiado, asi que añadir uno a una
carpeta grande cuesta una fraccion de llamadas a la API.

En cada pregunta:

1. **Se reescribe tu pregunta** para que se entienda sola. "¿Y eso cuanto
   pesa?" se convierte en "¿Cuanto pesa la Kjerag?", porque una pregunta
   llena de pronombres no tiene con que buscar. Al modelo y al historial les
   llega siempre tu pregunta original.
2. **Se buscan los fragmentos relevantes** combinando dos vias: embeddings
   (entiende sinonimos y parafrasis) y BM25 (clava codigos, referencias y
   cifras exactas), fusionadas con Reciprocal Rank Fusion.
3. **Se filtra por relevancia** con un corte relativo al mejor resultado de
   esa pregunta, no con un umbral fijo, mas un suelo absoluto para que una
   pregunta fuera del corpus no cuele el "menos malo".
4. **Se responde** solo con eso.

La conversacion se guarda en `historial.json`, asi que puedes cerrar y
retomar. Comandos dentro del chat:

- `/reset` - borra el historial y empieza de cero.
- `/salir` - termina el chat.

## Medir antes de tocar

`evaluar.py` existe para no tunear a ojo. Escribe tus casos en
`eval/preguntas.json` (pregunta, fragmento esperado y, si quieres, documento
y pagina; incluye tambien preguntas fuera del corpus que el bot deba
rechazar) y ejecuta:

```
python evaluar.py                  # recall@k, MRR, fidelidad y tasa de rechazo
python evaluar.py --sin-juez       # solo recuperacion, sin gastar generacion
python evaluar.py --detalle        # que caso concreto ha fallado
python evaluar.py --barrido        # prueba modos, top_k y umbrales, ordenado por MRR
python evaluar.py --csv runs.csv   # acumula corridas para comparar
python evaluar.py --benchmark-indice   # latencia y recall@4: numpy vs hnswlib
```

`recall@k` es el techo del sistema: lo que no se recupera, el modelo no lo
puede responder por bien que escriba.

## Desplegarlo

Con Docker:

```
docker compose up --build
```

`docs/` se monta en solo lectura y el indice persiste en un volumen, para no
repetir los embeddings en cada arranque. La imagen corre como usuario sin
privilegios y desactiva `bootstrap.py` (las dependencias las fija la imagen).

Como API HTTP multiusuario y multicoleccion:

```
pip install fastapi uvicorn python-multipart
uvicorn api:app --reload
```

| Metodo | Ruta | Que hace |
|---|---|---|
| `GET` | `/colecciones` | lista las colecciones y cuales estan en memoria |
| `POST` | `/colecciones/{nombre}/documentos` | sube un documento |
| `POST` | `/colecciones/{nombre}/reindexar` | reconstruye su indice |
| `POST` | `/chat` | conversa (`mensaje`, `session_id`, `coleccion`) |
| `DELETE` | `/sesiones/{session_id}` | borra el historial de una sesion |
| `GET` | `/salud` | health check, sin autenticacion |

Cada coleccion es un corpus independiente en `data/{nombre}/`. El historial
va por `session_id` en SQLite, y los indices se cargan de forma perezosa con
una cache LRU, no uno por peticion. Protege la API poniendo `API_KEY` en el
`.env`: los clientes la mandan en la cabecera `X-API-Key`. **Si la dejas
vacia, la API acepta cualquier peticion**; vale para local, no para exponerla.

Documentacion interactiva en `http://localhost:8000/docs`.

## Estructura

- `.env.example` - plantilla de configuracion. Copiala a `.env`; `.env` no
  se sube a git.
- `docs/` - tus documentos fuente. En el uso normal es lo unico que tocas.
- `chatbot.py` - cliente de terminal: bucle de chat, comandos e impresion.
- `conversacion.py` - logica de conversacion y system prompt, **compartidos**
  por la CLI y la API para que el bot se comporte igual por ambos caminos.
- `rag.py` - extraccion de texto, troceado, embeddings, busqueda hibrida,
  reranking e indices vectoriales.
- `historial.py` - almacenes de historial (JSON para la CLI, SQLite por
  sesion para la API) tras una interfaz comun.
- `colecciones.py` - colecciones de la API: rutas, validacion de nombres y
  cache LRU de buscadores.
- `api.py` - servidor FastAPI. Solo transporte HTTP.
- `evaluar.py` + `eval/` - set de evaluacion y metricas.
- `ingest.py` - pre-generar el indice antes de desplegar. Con `--forzar` lo
  reconstruye entero.
- `bootstrap.py` - instala dependencias que falten al arrancar en local; se
  desactiva solo dentro de un contenedor.
- `rag_index/`, `data/`, `historial.json` - generados automaticamente, no se
  suben a git.
- `test_*.py` - 176 tests. `python -m unittest discover`. No gastan API.

## Personalizacion avanzada

Cambiando solo el `.env` y los documentos ya tienes un bot distinto. Si
necesitas ir mas alla, todo esta en constantes al principio de cada archivo:

**Comportamiento** (`conversacion.py`)
- `SYSTEM_PROMPT_TEMPLATE` - tono, formato y restricciones. Las dos
  variantes de la regla de citas se inyectan, no se duplica el prompt.
- `MODEL_NAME`, `TOP_K_CONTEXTO`, `USAR_STREAMING`.
- `REESCRIBIR_CONSULTA`, `TURNOS_PARA_REESCRITURA` - desactivarla ahorra una
  llamada por turno, a costa de que las preguntas de seguimiento recuperen peor.

**Recuperacion** (`rag.py`)
- `MODO_BUSQUEDA` - `"hibrida"` (defecto), `"densa"` o `"lexica"`.
- `MARGEN_RELATIVO` (0.85) - subelo a 0.9-0.95 si se cuelan fragmentos
  tangenciales; bajalo a 0.7-0.8 si las respuestas se quedan cortas.
- `UMBRAL_MINIMO` (0.3) - subelo si el bot responde cosas inventadas a
  preguntas fuera del corpus; bajalo si dice "no tengo esa informacion"
  demasiado a menudo.
- `USAR_RERANKER`, `RERANKER_BACKEND` - mas precision a cambio de una
  llamada extra (~0,6-1,5 s con el backend `llm`; Cohere baja de 300 ms).

**Indexado** (`rag.py`)
- `CHUNK_SIZE` / `CHUNK_OVERLAP` - se mide en caracteres; en español un
  token ronda 3,5-4 caracteres, asi que 900 son unos 230-260 tokens.
- `EMBEDDING_DIM` (768, 1536 o 3072), `WORKERS_EMBEDDING`.
- `BACKEND_VECTORIAL` - `"auto"` usa numpy hasta 50 000 fragmentos y
  hnswlib por encima (`pip install hnswlib`).
- `USAR_OCR` - para PDFs escaneados. Necesita `pytesseract` y `pdf2image`
  mas Tesseract y Poppler en el sistema.

No hace falta borrar nada a mano al tocar estos parametros: el manifest
guarda el modelo, la dimension y el troceado con los que se genero el
indice, y si alguno cambia se regenera solo en el siguiente arranque.
