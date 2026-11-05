FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data ./data
RUN useradd --create-home app && chown -R app /app
USER app
RUN rag ingest --docs data/docs --index build/index
ENTRYPOINT ["rag"]
CMD ["eval", "--index", "build/index"]
