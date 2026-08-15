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
uvicorn api:app
```

| Metodo | Ruta | Quien | Que hace |
|---|---|---|---|
| `POST` | `/sesiones` | publico | abre sesion ligada a una coleccion, devuelve cookie |
| `POST` | `/chat` | sesion | conversa (`mensaje`); la coleccion sale de la sesion |
| `DELETE` | `/sesiones/actual` | sesion | cierra la sesion y borra su historial |
| `GET` | `/colecciones` | admin | lista las colecciones |
| `POST` | `/colecciones/{nombre}/documentos` | admin | sube un documento |
| `POST` | `/colecciones/{nombre}/reindexar` | admin | reconstruye su indice |
| `GET` | `/salud` | publico | health check |

Cada coleccion es un corpus independiente en `data/{nombre}/`. Los indices se
cargan de forma perezosa con una cache LRU, no uno por peticion.

**El cliente no elige su `session_id`.** El servidor lo genera (UUID4), lo
firma con HMAC y lo entrega en una cookie `HttpOnly`/`Secure`/`SameSite=Lax`.
Si se aceptara un id arbitrario, cualquiera podria leer la conversacion de
otro sondeando identificadores. La sesion queda ligada a una coleccion al
crearse y no puede cambiarla despues.

Documentacion interactiva en `http://localhost:8000/docs`.

### Antes de exponerlo a internet

- [ ] **`SECRET_KEY`** puesta en el `.env`
      (`python -c "import secrets; print(secrets.token_hex(32))"`). Sin ella
      se usa una efimera: las sesiones caducan en cada reinicio y con varios
      workers cada uno firmaria distinto.
- [ ] **`API_KEY`** puesta, o los endpoints de administracion (subir
      documentos, reindexar) quedan abiertos a cualquiera.
- [ ] **HTTPS** delante. El codigo asume un proxy inverso (nginx, Caddy,
      Traefik) que termina TLS; sin el, `COOKIES_SEGURAS=true` impide que la
      cookie viaje y nada funcionara.
- [ ] **`CONFIAR_EN_PROXY=true`** solo si ese proxy fija `X-Forwarded-For`.
      Activarlo sin proxy permite falsear la cabecera y saltarse el limite
      por IP.
- [ ] **`ORIGENES_CORS`** con tus dominios reales. Vacio = ningun navegador
      de otro origen. El comodin `*` se ignora a proposito: con cookies de
      sesion nunca es correcto.
- [ ] **Limites de recursos en el contenedor** (`mem_limit`, `cpus`). Los
      timeouts del codigo acotan el tiempo, pero un archivo malformado puede
      disparar la memoria durante el parseo y eso solo lo frena el runtime.
- [ ] **Un solo worker**, o rate limiting externo. El limitador vive en
      memoria del proceso: con N workers los limites efectivos se multiplican
      por N. Para varios workers, ponlo en Redis o en el proxy.
- [ ] **Copia de seguridad de `data/`**, que contiene indices, documentos
      subidos e historiales.
- [ ] **Revisa tus logs**: por defecto no guardan el texto de las preguntas
      (`LOG_PREGUNTAS=false`). Si lo activas para depurar, estaras
      almacenando datos de tus usuarios.

Lo que ya viene hecho: rate limiting por sesion e IP con dos ventanas y
`Retry-After`, tope de longitud de mensaje aplicado antes de gastar API,
expiracion de sesiones inactivas, timeout en las llamadas al modelo, errores
genericos con id de correlacion (el detalle solo va al log), handler global
que evita tracebacks, validacion de archivos por contenido real y no por
extension, nombres de archivo generados por el servidor, cabeceras de
seguridad y log estructurado con el id de sesion hasheado.

### Dos cosas que el codigo no puede resolver

**Un usuario insistente puede extraer buena parte del corpus.** El bot esta
diseñado para responder con el contenido de tus documentos: preguntando de
forma sistematica se puede reconstruir una porcion importante de ellos. Los
limites de peticiones lo hacen mas lento y mas caro, no imposible. **No
pongas en `docs/` nada que no estarias dispuesto a publicar**, y si el
corpus es sensible, pon autenticacion real de usuarios delante en vez de
dejar `/sesiones` abierto.

**Las conversaciones se envian a Google.** Cada pregunta, junto con los
fragmentos recuperados de tus documentos, viaja a la API de Gemini. Eso tiene
implicaciones de privacidad y, segun que datos manejes, tambien legales
(RGPD si hay datos personales). Revisa las condiciones de uso de la API que
tengas contratada, informa a tus usuarios de que sus mensajes se procesan en
un tercero, y no metas datos personales o confidenciales en el corpus sin
haber comprobado antes que puedes hacerlo.

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
- `seguridad.py` - firma de sesiones, rate limiting, validacion real de
  archivos subidos y log estructurado. Sin FastAPI, para poder probarlo suelto.
- `api.py` - servidor FastAPI: transporte HTTP y control de acceso.
- `evaluar.py` + `eval/` - set de evaluacion y metricas.
- `ingest.py` - pre-generar el indice antes de desplegar. Con `--forzar` lo
  reconstruye entero.
- `bootstrap.py` - instala dependencias que falten al arrancar en local; se
  desactiva solo dentro de un contenedor.
- `rag_index/`, `data/`, `historial.json` - generados automaticamente, no se
  suben a git.
- `test_*.py` - 232 tests. `python -m unittest discover`. No gastan API.

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
