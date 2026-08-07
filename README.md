# ChatBot_RAG

Chatbot especializado en zapatillas de running de la marca **NNormal**. Usa RAG
(Retrieval-Augmented Generation): solo responde con informacion que aparezca en
los PDFs que se le proporcionen, y no habla de ningun otro tema.

## Uso

1. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```
2. Configura tu `GEMINI_API_KEY` en el archivo `.env`.
3. Coloca el/los PDF(s) con la informacion de NNormal dentro de la carpeta `docs/`.
4. Arranca el chatbot:
   ```
   python chatbot.py
   ```

Eso es todo. Al arrancar, el chatbot comprueba automaticamente si los PDFs de
`docs/` son nuevos o han cambiado desde la ultima vez y, si es asi, regenera el
indice (troceado + embeddings) el solo antes de empezar la conversacion — no
hace falta ejecutar ningun paso manual aparte de colocar/actualizar los PDFs.

Comandos dentro del chat:
- `/reset` - borra el historial de conversacion.
- `/salir` - termina el chat.

## Estructura

- `docs/` - carpeta donde se colocan los PDFs fuente (el usuario final solo
  toca esto).
- `rag.py` - extraccion de texto de PDFs, troceado, embeddings (Gemini
  `text-embedding-004`) y recuperacion por similitud coseno.
- `rag_index/` - indice generado automaticamente (chunks, embeddings y
  manifest de hashes para detectar cambios). No se edita a mano.
- `ingest.py` - utilidad manual opcional para forzar la regeneracion completa
  del indice (por ejemplo, antes de desplegar). No es necesaria en el uso
  normal.
- `chatbot.py` - bucle de chat: actualiza el indice si hace falta, recupera
  contexto relevante por cada pregunta y responde solo en base a el.
