from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from rag_core import generate_answer, init_knowledge_base

app = FastAPI(title="Atguigu 客服知识库 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class SourceItem(BaseModel):
    chunk_id: int | str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        result = generate_answer(request.question, top_k=request.top_k)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {e}")


@app.post("/init")
async def init():
    try:
        n = init_knowledge_base()
        return {"message": "知识库初始化完成", "chunks": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)