# One image for the API, the ingest worker and the UI; compose picks the command.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# The sentence splitter fetches tokenizer/NLTK data on first use and caches it
# inside site-packages. Do that now, as root, so the non-root runtime user
# never needs network access or write access there.
RUN python -c "from llama_index.core.node_parser import SentenceSplitter; \
SentenceSplitter(chunk_size=64, chunk_overlap=8).split_text('Warm the caches. Then stop.')"

COPY aka ./aka
COPY eval ./eval
COPY app.py ./

RUN useradd --create-home --uid 10001 aka \
 && mkdir -p /app/data /app/var/trash /app/var/quarantine /app/chroma_db \
 && chown -R aka:aka /app
USER aka

EXPOSE 8080 8501
CMD ["uvicorn", "aka.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
