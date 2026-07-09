FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    ZOOM_SEARCH_DEMO_MODE=true

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY zoom_search ./zoom_search

RUN pip install --no-cache-dir ".[mcp]"

CMD ["zoom-search-mcp"]
