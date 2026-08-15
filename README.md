# ChatBot_RAG

Plantilla para montar tu propio chatbot con RAG (Retrieval-Augmented
Generation) sobre los documentos que tu quieras. El asistente responde
EXCLUSIVAMENTE con informacion que aparezca en los PDFs que le proporciones
y no habla de ningun otro tema: si algo no esta en tus documentos, lo dice
en vez de inventarselo.

Sirve para cualquier documentacion propia — un catalogo de producto, un
manual, unos apuntes de clase, normativa interna, un TFG... — sin tocar
codigo: solo pones tu API key, tus PDFs y el tema del que hablara.

Requisitos: Python 3.9 o superior y una API key de Gemini (el nivel gratuito
es suficiente para probarlo).

## Puesta en marcha

1. Consigue tu API key en [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Copia `.env.example` a `.env` y rellenalo:
   - `GEMINI_API_KEY` (obligatorio): la key del paso anterior.
   - `BOT_NAME` (opcional): como se llama tu asistente, ej. `Asistente de Cocina`.
   - `BOT_TOPIC` (opcional): el tema del que puede hablar, escrito como
     complemento de la frase "experto en...", ej. `las recetas del libro`.
     Si lo dejas vacio, respondera de forma generica sobre el contenido de
     tus PDFs.
3. Coloca tu(s) PDF(s) dentro de la carpeta `docs/`. Puedes poner varios.
4. Arranca el chatbot:
   ```
   python chatbot.py
   ```

No hace falta instalar nada a mano: al arrancar, `bootstrap.py` detecta las
dependencias que falten y las instala solo. Si prefieres hacerlo tu:

```
pip install -r requirements.txt
```

## Como funciona en el dia a dia

Al arrancar, el chatbot comprueba (por hash del contenido) si los PDFs de
`docs/` son nuevos, se han modificado o se han borrado desde la ultima vez
y, si es asi, regenera el indice el solo — troceado + embeddings — antes de
empezar la conversacion. Tu unico paso manual es colocar o actualizar los
PDFs; todo lo demas es automatico.

En cada pregunta busca los fragmentos mas parecidos entre tus documentos, se
los pasa al modelo como contexto y le obliga a responder solo con eso.

La conversacion se guarda en `historial.json`, asi que puedes cerrar y
retomar el chat donde lo dejaste. Comandos dentro del chat:

- `/reset` - borra el historial y empieza de cero.
- `/salir` - termina el chat.

## Estructura

- `.env.example` - plantilla de configuracion (API key, nombre y tema del
  asistente). Copiala a `.env` y rellena tus datos; `.env` no se sube a git.
- `docs/` - carpeta donde colocas tus PDFs fuente. En el uso normal es lo
  unico que tocas.
- `chatbot.py` - bucle de chat: construye el system prompt a partir de
  `BOT_NAME`/`BOT_TOPIC`, actualiza el indice si hace falta, recupera
  contexto relevante en cada pregunta y responde solo en base a el.
- `rag.py` - extraccion de texto de los PDFs, troceado, embeddings (Gemini
  `gemini-embedding-001`) y recuperacion por similitud coseno.
- `bootstrap.py` - instala automaticamente las dependencias que falten al
  arrancar.
- `ingest.py` - utilidad opcional para forzar la regeneracion completa del
  indice (util para pre-generarlo antes de desplegar). No es necesaria en el
  uso normal.
- `rag_index/` - indice generado automaticamente (fragmentos, embeddings y
  manifest de hashes). No se edita a mano ni se sube a git.
- `historial.json` - memoria de la conversacion. Tampoco se sube a git.

## Personalizacion avanzada

Cambiando solo el `.env` y los PDFs ya tienes un bot distinto. Si necesitas
ir mas alla, todo esta en constantes al principio de cada archivo:

- Reglas de comportamiento (tono, formato de respuesta, restricciones
  adicionales): plantilla `SYSTEM_PROMPT_TEMPLATE` en `chatbot.py`.
- Modelo de chat: `MODEL_NAME` en `chatbot.py`.
- Cuantos fragmentos se le pasan al modelo por pregunta: `TOP_K_CONTEXTO` en
  `chatbot.py`.
- Tamaño de los fragmentos y solapamiento: `CHUNK_SIZE` y `CHUNK_OVERLAP` en
  `rag.py`. Fragmentos mas grandes dan mas contexto por trozo pero menos
  precision en la busqueda.
- Exigencia de la busqueda: `umbral_similitud` en `recuperar_contexto`
  (`rag.py`). Subelo si cuela contexto poco relevante; bajalo si responde
  "no tengo esa informacion" demasiado a menudo.

Si cambias el modelo de embeddings o el troceado, borra la carpeta
`rag_index/` para que se regenere desde cero.
