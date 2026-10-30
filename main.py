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

def search_relevant_docs(query: str) -> str:
    """사용자 질문에 키워드가 포함된 문서만 골라내어 토큰 낭비를 방지합니다."""
    context = ""
    # 질문을 공백 단위로 쪼개어 키워드 추출
    keywords = query.split()
    
    for filename, content in DOCS_CACHE.items():
        # 파일 이름이나 내용에 키워드가 하나라도 포함되어 있는지 확인
        matched = False
        for kw in keywords:
            if kw in content or kw in filename:
                matched = True
                break
        
        # 매칭된 문서만 컨텍스트에 추가
        if matched:
            context += f"\n\n=== [관련 문서 파일명: {filename}] ===\n{content}\n"
            
    # 만약 키워드 매칭이 하나도 안 되면, 안전하게 전체 문서의 앞부분 일부나 요약본 제공
    if not context and DOCS_CACHE:
        for filename, content in DOCS_CACHE.items():
            context += f"\n\n=== [문서 파일명: {filename}] ===\n{content[:1000]}...\n" # 너무 길면 일부만
            
    return context

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_msg = request.message
        
        # 키워드 기반 관련 문서만 스마트하게 검색
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
