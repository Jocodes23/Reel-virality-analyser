FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Minimal system deps (Pillow, numpy wheels are self-contained; bash for compose cmd).
RUN apt-get update && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY reels_trend_intel ./reels_trend_intel

# Install with the Postgres + analytics extras (GPU/audio/OCR extras are opt-in and
# large; add them in a derived image if you want the real ML stack in-container).
RUN pip install ".[postgres,analytics]"

EXPOSE 8000
CMD ["rti", "serve"]
