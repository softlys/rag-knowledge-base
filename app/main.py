import logging

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from app.chunking import chunk_text
from app.config import settings
from app.document_loader import UnsupportedFileTypeError, extract_text
from app.llm_client import LLMUnavailableError, answer_question
from app.schemas import AskRequest, AskResponse, HealthResponse, SourceChunk, SourcesListResponse, UploadResponse
from app.vector_store import add_chunks, list_sources, query

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RAG Knowledge Base",
    description="Загружаешь PDF/TXT, задаёшь вопросы — ответ строится по содержимому документов, с указанием источников.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/sources", response_model=SourcesListResponse)
def sources() -> SourcesListResponse:
    return SourcesListResponse(sources=list_sources())


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile) -> UploadResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Файл пустой")

    try:
        text = extract_text(file.filename, content)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Не удалось извлечь текст из файла (возможно, это скан без текстового слоя)",
        )

    chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    added = add_chunks(chunks, source=file.filename)

    return UploadResponse(filename=file.filename, chunks_added=added)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    found = query(request.question, top_k=settings.top_k)

    try:
        answer = answer_question(request.question, found)
    except LLMUnavailableError as e:
        raise HTTPException(status_code=503, detail=f"LLM сервис временно недоступен: {e}") from e

    return AskResponse(
        answer=answer,
        sources=[SourceChunk(source=c["source"], text=c["text"]) for c in found],
    )
