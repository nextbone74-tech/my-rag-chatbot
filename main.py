import os
from pathlib import Path
import numpy as np
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

# 제미나이 클라이언트 초기화
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

# 문서 캐시 및 벡터 저장소
DOCS_CACHE = {}
DOCS_EMBEDDINGS = {}

def get_embedding(text: str) -> list[float]:
    """Gemini 임베딩 모델을 사용하여 텍스트를 벡터로 변환합니다."""
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )
    return response.embedding.values

def cosine_similarity(a, b):
    """두 벡터 간의 코사인 유사도를 계산합니다."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_and_vectorize_docs():
    """서버 시작 시 docs 폴더의 파일을 읽고 벡터화(Embedding)합니다."""
    docs_dir = Path("docs")
    if not docs_dir.exists():
        print("❌ [경고] docs 폴더를 찾을 수 없습니다.")
        return
        
    for md_file in docs_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            DOCS_CACHE[md_file.name] = content
            # 문서 내용을 벡터화하여 저장
            DOCS_EMBEDDINGS[md_file.name] = get_embedding(content)
            print(f"✅ [벡터화 완료] {md_file.name}")
        except Exception as e:
            print(f"❌ [에러] {md_file.name} 벡터화 실패: {e}")

@app.on_event("startup")
async def startup_event():
    load_and_vectorize_docs()

# 루트 주소로 index.html 서빙
@app.get("/", response_class=HTMLResponse)
async def read_index():
    html_path = Path("index.html")
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "index.html 파일을 찾을 수 없습니다."

def search_relevant_docs_by_vector(query: str) -> str:
    """사용자 질문 벡터와 문서 벡터를 비교하여 가장 유사한 문서를 찾습니다."""
    if not DOCS_CACHE or not DOCS_EMBEDDINGS:
        return ""
        
    try:
        query_vector = get_embedding(query)
        best_match_file = None
        highest_similarity = -1.0
        
        for filename, doc_vector in DOCS_EMBEDDINGS.items():
            sim = cosine_similarity(query_vector, doc_vector)
            if sim > highest_similarity:
                highest_similarity = sim
                best_match_file = filename
                
        # 유사도가 일정 수준 이상이거나 가장 높은 문서 반환
        if best_match_file:
            content = DOCS_CACHE[best_match_file]
            return f"\n\n=== [가장 관련 높은 문서: {best_match_file} (유사도: {highest_similarity:.2f})] ===\n{content}\n"
    except Exception as e:
        print(f"벡터 검색 중 에러 발생: {e}")
        
    # 벡터 검색 실패 시 전체 문서 통합 반환
    fallback = ""
    for filename, content in DOCS_CACHE.items():
        fallback += f"\n\n=== [문서: {filename}] ===\n{content}\n"
    return fallback

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_msg = request.message
        # 의미 기반 벡터 검색 수행
        retrieved_docs = search_relevant_docs_by_vector(user_msg)
        
        prompt = f"""
        당신은 사내 문서 기반 친절한 AI 어시스턴트입니다.
        아래의 [참고 문서] 내용을 바탕으로 사용자의 질문에 정확하고 친절하게 답변해주세요.
        만약 참고 문서에 관련 내용이 전혀 없다면, "제공된 문서에서 관련 내용을 찾을 수 없습니다."라고 정직하게 답해주세요.

        [참고 문서]
        {retrieved_docs if retrieved_docs else "관련 문서 없음"}

        [사용자 질문]
        {user_msg}
        """
        
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

@app.get("/health")
async def health_check():
    return {"status": "alive"}

@app.get("/debug/docs")
async def debug_docs():
    return {
        "cached_file_names": list(DOCS_CACHE.keys()),
        "vectorized_files": list(DOCS_EMBEDDINGS.keys()),
        "total_files": len(DOCS_CACHE)
    }
