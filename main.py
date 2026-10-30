import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 제미나이 클라이언트 초기화
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

# 서버 시작 시 docs 폴더 내용을 메모리에 캐싱
DOCS_CACHE = {}

def load_docs_into_cache():
    docs_dir = Path("docs")
    if not docs_dir.exists():
        return
    for md_file in docs_dir.glob("*.md"):
        try:
            DOCS_CACHE[md_file.name] = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

@app.on_event("startup")
async def startup_event():
    load_docs_into_cache()

def get_all_docs_context() -> str:
    """캐시된 모든 문서 내용을 통합하여 반환합니다 (제미나이가 알아서 맥락을 파악합니다)."""
    context = ""
    for filename, content in DOCS_CACHE.items():
        context += f"\n\n=== [문서 파일명: {filename}] ===\n{content}\n"
    return context

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_msg = request.message
        
        # 모든 문서 내용을 가져옴
        retrieved_docs = get_all_docs_context()
        
        prompt = f"""
        당신은 사내 문서 기반 친절한 AI 어시스턴트입니다.
        아래의 [사내 전체 참고 문서] 내용을 바탕으로 사용자의 질문에 정확하고 친절하게 답변해주세요.
        만약 참고 문서 어디에도 관련 내용이 없다면, "제공된 문서에서 관련 내용을 찾을 수 없습니다."라고 정직하게 답해주세요.

        [사내 전체 참고 문서]
        {retrieved_docs if retrieved_docs else "등록된 문서가 없습니다."}

        [사용자 질문]
        {user_msg}
        """
        
        # 스트리밍 응답 제너레이터
        def generate():
            response_stream = client.models.generate_content_stream(
                model='gemini-3.8-flash',
                contents=prompt,
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Render 잠자기 방지용 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "alive"}
