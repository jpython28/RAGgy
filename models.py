from pydantic import BaseModel

class Document(BaseModel):
    text: str

class IngestionResponse(BaseModel):
    document_id: str
    num_chunks: int
    status: str

class Query(BaseModel):
    chat: list[dict[str, str]]

class QueryResponse(BaseModel):
    chat: list[dict[str, str]]
    chunks_used: int

class HealthSummary(BaseModel):
    server_status: str
    chroma_status: str