import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

app = FastAPI()

# CORS 설정 (어떤 프론트엔드 주소에서든 API를 호출할 수 있도록 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 제미나이 클라이언트 초기화 (Render의 환경 변수에서 API 키 자동 로드)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

def search_docs(query: str) -> str:
    """docs 폴더 내 마크다운 파일들을 검색하여 관련 내용을 반환합니다."""
    docs_dir = Path("docs")
    if not docs_dir.exists():
        return ""
    
    context = ""
    for md_file in docs_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            # 간단한 키워드 매칭 또는 전체 내용 포함
            if query in content or md_file.stem in query:
                context += f"\n--- 문서 출처: {md_file.name} ---\n{content}\n"
        except Exception:
            continue
    return context

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_msg = request.message
        
        # 1. docs 폴더에서 관련 문서(RAG) 검색
        retrieved_docs = search_docs(user_msg)
        
        # 2. 제미나이에게 전달할 프롬프트 구성
        prompt = f"""
        당신은 사내 문서 기반 친절한 AI 어시스턴트입니다.
        아래의 [참고 문서] 내용을 바탕으로 사용자의 질문에 정확하고 친절하게 답변해주세요.
        만약 참고 문서에 내용이 없다면, "제공된 문서에서 관련 내용을 찾을 수 없습니다."라고 정직하게 답해주세요.

        [참고 문서]
        {retrieved_docs if retrieved_docs else "관련 문서 없음"}

        [사용자 질문]
        {user_msg}
        """
        
        # 3. 제미나이 모델 호출 (gemini-2.5-flash 또는 최신 플래시 모델 사용)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
