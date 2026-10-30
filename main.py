import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
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

# 제미나이 클라이언트 초기화 (Render 환경 변수의 GEMINI_API_KEY 자동 연동)
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

# 루트 주소("/")로 접속 시 index.html 파일을 화면에 직접 띄워줌
@app.get("/", response_class=HTMLResponse)
async def read_index():
    html_path = Path("index.html")
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "index.html 파일을 찾을 수 없습니다."

def search_relevant_docs(query: str) -> str:
    """질문 키워드와 매칭되는 문서 내용만 스마트하게 추출합니다."""
    context = ""
    keywords = query.split()
    for filename, content in DOCS_CACHE.items():
        matched = False
        for kw in keywords:
            if kw in content or kw in filename:
                matched = True
                break
        if matched:
            context += f"\n\n=== [관련 문서: {filename}] ===\n{content}\n"
            
    # 키워드 매칭이 없을 경우 안전하게 문서 앞부분 일부 제공
    if not context and DOCS_CACHE:
        for filename, content in DOCS_CACHE.items():
            context += f"\n\n=== [문서: {filename}] ===\n{content[:1000]}...\n"
    return context

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_msg = request.message
        retrieved_docs = search_relevant_docs(user_msg)
        
        prompt = f"""
        당신은 사내 문서 기반 친절한 AI 어시스턴트입니다.
        아래의 [참고 문서] 내용을 바탕으로 사용자의 질문에 정확하고 친절하게 답변해주세요.
        만약 참고 문서에 내용이 없다면, "제공된 문서에서 관련 내용을 찾을 수 없습니다."라고 정직하게 답해주세요.

        [참고 문서]
        {retrieved_docs if retrieved_docs else "관련 문서 없음"}

        [사용자 질문]
        {user_msg}
        """
        
        # 실시간 스트리밍 제너레이터
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
@app.get("/debug/docs")
async def debug_docs():
    return {
        "cached_file_names": list(DOCS_CACHE.keys()),
        "total_files": len(DOCS_CACHE)
    }
