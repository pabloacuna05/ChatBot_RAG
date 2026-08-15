# Imagen slim: no hace falta nada del sistema salvo que actives el OCR.
FROM python:3.12-slim

# Sin .pyc y con la salida sin bufferear, para que los logs del contenedor
# aparezcan en tiempo real en vez de a bloques.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_SIN_BOOTSTRAP=1

WORKDIR /app

# Las dependencias van en su propia capa: mientras requirements.txt no
# cambie, reconstruir la imagen no vuelve a instalarlas.
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

COPY *.py ./

# Usuario sin privilegios. Los directorios se crean y se ceden antes de
# cambiar de usuario, para que pueda escribir el indice al montarlos.
RUN useradd --create-home --uid 1000 rag \
    && mkdir -p /app/docs /app/rag_index /app/data \
    && chown -R rag:rag /app
USER rag

# docs/ se monta con tus documentos y rag_index/ persiste el indice de la
# CLI. data/ es el de la API: colecciones, indices e historiales por sesion.
VOLUME ["/app/docs", "/app/rag_index", "/app/data"]

CMD ["python", "chatbot.py"]
