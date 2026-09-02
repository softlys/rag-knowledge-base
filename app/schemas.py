from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class SourceChunk(BaseModel):
    source: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class HealthResponse(BaseModel):
    status: str = "ok"


class SourcesListResponse(BaseModel):
    sources: list[str]
